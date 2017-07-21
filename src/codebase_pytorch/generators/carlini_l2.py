import pdb
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from src.codebase.utils.utils import log as log
from src.codebase_pytorch.utils.dataset import Dataset

class CarliniL2Generator(nn.Module):
    '''
    Class for generating noise using method in Carlini and Wagner, '17

    TODO
        - deal with discretization
        - multiple starting point gradient descent
    '''

    def __init__(self, args, n_ins):
        '''

        '''
        super(CarliniL2Generator, self).__init__()
        self.use_cuda = not args.no_cuda
        self.targeted = args.target != 'none'
        self.c = args.generator_opt_const
        self.k = -1. * args.generator_confidence
        self.n_ins = n_ins # expected test size
        self.lr = args.generator_learning_rate

    def forward(self, x, w, labels, model, step):
        '''
        Function to optimize

        Labels should be one-hot
        '''
        corrupt_im = .5 * F.tanh(w + x)
        logits = model(corrupt_im)
        target_logit = torch.sum(logits * labels, dim=1)
        second_logit = torch.max(logits*(1.-labels)-(labels*10000), dim=1)[0]
        if self.targeted:
            class_loss = torch.clamp(second_logit - target_logit, min=self.k)
        else:
            class_loss = torch.clamp(target_logit - second_logit, min=self.k)
        dist_loss = torch.sum(torch.pow(corrupt_im - .5*F.tanh(x), 2).view(self.n_ins, -1), dim=1)
        #dist_loss = torch.sum(torch.pow(torch.norm(corrupt_im - x, p=2, dim=1), 2), dim=1)
        if not (step % 100):
            pass
            #print "***STEP %d***" % step
            #print class_loss[:20], dist_loss[:20]
        return torch.sum(dist_loss + self.c * class_loss)

    def generate(self, data, model, args, fh):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - data: tuple of (ins, outs) or Dataset class
            - model: a model class
            - args: argparse object with training parameters
            - fh: file handle for logging progress
        outputs:
            - noise: n_ims x im_size x im_size x n_channels
        '''

        if isinstance(data, tuple):
            ins = torch.FloatTensor(data[0]) if not isinstance(data[0], torch.FloatTensor) else data[0]
            outs = torch.LongTensor(data[1]) if not isinstance(data[1], torch.LongTensor) else data[1]
        elif isinstance(data, Dataset):
            ins = data.ins
            outs = data.outs
        else:
            raise TypeError("Invalid data format")

        ins = torch.FloatTensor(np.arctanh(ins.numpy() * 1.999999))

        # make targs one-hot
        one_hot_targs = np.zeros((outs.size()[0], args.n_classes))
        one_hot_targs[np.arange(outs.size()[0]), outs.numpy().astype(int)] = 1
        one_hot_targs = torch.FloatTensor(one_hot_targs.astype(int))

        w = torch.zeros(data.ins.size())

        if self.use_cuda:
            ins, one_hot_targs, w = ins.cuda(), one_hot_targs.cuda(), w.cuda()
        ins, one_hot_targs, w = Variable(ins), Variable(one_hot_targs), Variable(w, requires_grad=True)

        # optimizer
        params = [w]
        if args.generator_optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
        elif args.generator_optimizer == 'adam':
            optimizer = optim.Adam(params, lr=args.lr)
        elif args.generator_optimizer == 'adagrad':
            optimizer = optim.Adagrad(params, lr=args.lr())
        else:
            raise NotImplementedError

        for i in xrange(args.n_generator_steps):
            optimizer.zero_grad()
            outs = self(ins, w, one_hot_targs, model, i)
            obj_val = outs.data[0]
            outs.backward()
            optimizer.step()
            noise = .5*(torch.tanh(w + ins) - torch.tanh(ins))
            if not (i % (args.n_generator_steps / 10.)) and i:
                log(fh, '\t\tStep %d: objective: %.4f, avg noise magnitude: %.7f' %
                        (i, obj_val, torch.mean(torch.abs(noise)).data[0]))

        return noise.data.cpu().numpy()
