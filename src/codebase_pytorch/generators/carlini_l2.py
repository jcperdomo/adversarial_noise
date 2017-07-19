import pdb
import time
import torch
import torch.nn as nn
import numpy as np
from src.codebase_pytorch.utils.utils import log as log
from src.codebase_pytorch.utils.dataset import Dataset

class CarliniL2Generator(nn.Module):
    '''
    Class for generating noise using method in Carlini and Wagner, '17

    TODO
        - deal with discretization
        - multiple starting point gradient descent
    '''

    def __init__(self, args, model, n_ins):
        '''

        '''
        super(CarliniL2Generator, self).__init__()
        self.use_cuda = not args.no_cuda
        self.c = args.generator_opt_const
        self.k = -args.generator_confidence
        self.lr = args.generator_learning_rate
        self.model = model # can I throw this into forward()?
        self.ws = Variable(torch.randn(n_ins, args.im_size, args.im_size, args.n_channels).type('float'), requires_grad=True)

    def forward(self, x, labels):
        '''
        Function to optimize

        Labels should be one-hot
        '''
        corrupt_im = .5 * (nn.tanh(self.ws) + 1)
        logits = self.model(corrupt_im) # need to get the actual logits
        target_logit = torch.sum(logits * labels)
        second_logit = torch.max(logits * (1. - labels))
        class_loss = torch.max(second_logit - target_logit, -self.k)
        dist_loss = nn.sum(nn.square(corrupt_im - x))
        return dist_loss + self.c * class_loss

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

        # make outs one-hot
        one_hot_outs = np.zeros((outs.size()[0], args.n_classes))
        one_hot_outs[np.arange(outs.size()[0]), outs.numpy().astype(int)] = 1

        # optimizer
        if args.generator_optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.generator_optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=args.lr)
        elif args.generator_optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=args.lr())
        else:
            raise NotImplementedError

        for i in xrange(args.n_generator_steps):
            if self.use_cuda:
                ins, targs = ins.cuda(), targs.cuda()
            ins, targs = Variable(ins), Variable(targs)
            optimizer.zero_grad()
            outs = self(ins, one_hot_outs)
            obj_val = outs.data[0]
            noise = .5*(torch.nn.tanh(self.ws) + 1) - ins
            outs.backward()
            optimizer.step()
            if not (i % (args.n_generator_steps / 10.)) and i:
                log(fh, '\t\tStep %d: objective: %.4f, avg noise magnitude: %.7f' %
                        (i, obj_val, np.mean(noise)))

        return noise
