import pdb
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from src.codebase.utils.utils import log as log
from src.codebase_pytorch.utils.scheduler import ReduceLROnPlateau
from src.codebase_pytorch.utils.dataset import Dataset

class CarliniL2Generator(nn.Module):
    '''
    Class for generating noise using method in Carlini and Wagner, '17

    TODO
        - deal with discretization
        - multiple starting point gradient descent
    '''

    def __init__(self, args, n_ins, early_abort=True):
        '''

        '''
        super(CarliniL2Generator, self).__init__()
        self.use_cuda = not args.no_cuda
        self.targeted = args.target != 'none'
        self.early_abort = early_abort
        self.c = args.generator_opt_const
        self.k = -1. * args.generator_confidence
        self.n_ins = n_ins # expected test size

    def forward(self, x, w, labels, model):
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
            optimizer = optim.SGD(params, lr=args.generator_lr, momentum=args.momentum)
        elif args.generator_optimizer == 'adam':
            optimizer = optim.Adam(params, lr=args.generator_lr)
        elif args.generator_optimizer == 'adagrad':
            optimizer = optim.Adagrad(params, lr=args.generator_lr)
        else:
            raise NotImplementedError
        scheduler = ReduceLROnPlateau(optimizer, 
                'min', factor=.5, patience=3, epsilon=1e-3)

        start_time = time.time()
        prev_val = self(ins, w, one_hot_targs, model).data[0]
        for i in xrange(args.n_generator_steps):
            if not (i % args.n_generator_steps / 10.):
                log(fh, '\t\tStep %d \tLearning rate: %.3f' % (i, scheduler.get_lr()[0]))
            optimizer.zero_grad()
            outs = self(ins, w, one_hot_targs, model)
            obj_val = outs.data[0]
            outs.backward()
            optimizer.step()
            noise = .5*(torch.tanh(w + ins) - torch.tanh(ins))
            if not (i % (args.n_generator_steps / 10.)) and i:
                log(fh, '\t\t\tobjective: %.3f, avg noise magnitude: %.7f \t(%.3f s)' % (obj_val, torch.mean(torch.abs(noise)).data[0], time.time() - start_time))
                scheduler.step(obj_val, i)
                if obj_val > prev_val and self.early_abort:
                    log(fh, '\t\t\tAborted search because stuck')
                    break
                prev_val =  obj_val
                start_time = time.time()

        return noise.data.cpu().numpy()
