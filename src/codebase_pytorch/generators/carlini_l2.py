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

    def __init__(self, args, batch_size, early_abort=True):
        '''

        '''
        super(CarliniL2Generator, self).__init__()
        self.use_cuda = not args.no_cuda
        self.targeted = args.target != 'none'
        self.k = -1. * args.generator_confidence
        self.binary_search_steps = args.n_binary_search_steps
        self.init_const = args.generator_init_opt_const
        self.early_abort = early_abort
        self.batch_size = batch_size # rename to batch_size

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
        return torch.sum(dist_loss + self.c * class_loss), dist_loss, corrupt_im, logits

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

        TODO
            - handle non batch_size inputs (smaller and larger)
        '''

        def compare(x, y):
            if not isinstance(x (float, int, np.int32)):
                x = np.copy(x)
                x[y] += self.k # confidence
                x = np.argmax(x)
            if self.targeted:
                return x == y
            else:
                return x != y

        if isinstance(data, tuple):
            ins = torch.FloatTensor(data[0]) if not isinstance(data[0], torch.FloatTensor) else data[0]
            outs = torch.LongTensor(data[1]) if not isinstance(data[1], torch.LongTensor) else data[1]
        elif isinstance(data, Dataset):
            ins = data.ins
            outs = data.outs
        else:
            raise TypeError("Invalid data format")

        batch_size = self.batch_size

        # convert to arctanh space, following Carlini code
        ins = torch.FloatTensor(np.arctanh(ins.numpy() * 1.999999))

        # make targs one-hot
        one_hot_targs = np.zeros((outs.size()[0], args.n_classes))
        one_hot_targs[np.arange(outs.size()[0]), outs.numpy().astype(int)] = 1
        one_hot_targs = torch.FloatTensor(one_hot_targs.astype(int))

        # variable to optimize
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
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=.5, patience=3, threshold=1e-3)

        lower_bounds = np.zeroes(batch_size)
        upper_bounds = np.ones(batch_size)
        opt_consts = np.ones(batch_size) * self.init_const

        overall_best_ims = np.zeroes(data.ins.size())
        overall_best_dists = [1e10] * batch_size
        overall_best_classes = [-1] * batch_size # class of the corresponding best im, doesn't seem totally necessary to track

        start_time = time.time()
        for b_step in xrange(self.binary_search_steps):
            # TODO reset optimizer parameters

            best_dists = [1e10] * batch_size
            best_classes = [-1] * batch_size

            # repeat binary search one more time?

            prev_loss = 1e6
            for step in xrange(args.n_generator_steps):
                optimizer.zero_grad()
                obj, dists, corrupt_ims, logits = self(ins, w, one_hot_targs, model)
                total_loss = obj.data[0]
                total_loss.backward()
                optimizer.step()

                if not (i % (args.n_generator_steps / 10.)) and i:
                    # TODO logging
                    log(fh, '\t\t\tobjective: %.3f, avg noise magnitude: %.7f \t(%.3f s)' % \
                            (total_loss, torch.mean(torch.abs(noise)).data[0], time.time() - start_time))

                    # Check for early abort
                    if self.early_abort and total_loss > prev_loss*.9999:
                        log(fh, '\t\t\tAborted search because stuck')
                        break
                    prev_loss = total_loss
                    scheduler.step(obj_val, i)

                # bookkeeping
                for e, (dist, logit, im) in enumerate(zip(dists, logits, corrupt_ims)):
                    if not compare(logit, outs[e]): # if not the targeted class, continue
                        continue
                    if dist < best_l2[e]: # if smaller noise within the binary search step
                        best_dists[e] = dist
                        best_classes[e] = np.argmax(logit)
                    if dist < overall_best_dists[e]: # if smaller noise overall
                        overall_best_dists[e] = dist
                        overall_best_classes[e] = np.argmax(logit)
                        overall_best_ims[e] = im

            # binary search stuff
            for e in xrange(batch_size):
                if compare(best_classes[e], outs[e]) and best_classes[e] != -1: # success; looking for lower c
                    upper_bounds[e] = min(upper_bounds[e], consts[e])
                    if upper_bounds[e] < 1e9:
                        consts[e] = (lower_bounds[e] + upper_bounds[e]) / 2
                else: # failure, search with greater c
                    lower_bound[e] = max(lower_bound[e], consts[e])
                    if upper_bound[e] < 1e9:
                        consts[e] = (lower_bounds[e] + upper_bounds[e]) / 2
                    else:
                        consts[e] *= 10

        noise = overall_best_ims - .5*torch.tanh(ins) #.5*(torch.tanh(w + ins) - torch.tanh(ins))
        return noise.data.cpu().numpy()
