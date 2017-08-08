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
from src.codebase_pytorch.utils.timer import Timer

class EnsembleGenerator(nn.Module):
    '''
    Optimization-based generator using an ensemble of models
    '''

    def __init__(self, args, normalize=(None,None)):
        '''

        '''
        super(EnsembleGenerator, self).__init__()
        self.use_cuda = args.cuda
        self.targeted = args.target != 'none'
        self.k = -1. * args.generator_confidence
        self.binary_search_steps = args.n_binary_search_steps
        self.init_const = args.generator_init_opt_const
        self.early_abort = args.early_abort
        self.batch_size = args.generator_batch_size
        self.mean = normalize[0]
        self.std = normalize[1]
        self.n_models = args.n_models

    def forward(self, tanh_x, x, w, c, labels, models, mean=None, std=None):
        '''
        Function to optimize
            - Labels should be one-hot
            - tanh_x should be inputs in tanh space
            - x should be images in [0,1]^d (unnormalized)

        TODO: ensemble weights
        '''
        corrupt_im = .5 * F.tanh(w + tanh_x) + .5 # puts into [0,1]
        input_im = corrupt_im
        if mean is not None: # if model expects normalized input
            input_im = (corrupt_im - mean) / std
        total_class_loss = 0 # will get cast into tensor of correct size
        total_pred_dstrb = 0
        pred_dstrbs = []
        for model in models:
            logits = model(input_im)
            target_logit = torch.sum(logits * labels, dim=1)
            second_logit = torch.max(logits*(1.-labels)-(labels*10000), dim=1)[0]
            if self.targeted:
                class_loss = torch.clamp(second_logit-target_logit, min=self.k)
            else:
                class_loss = torch.clamp(target_logit-second_logit, min=self.k)
            total_class_loss += class_loss
            pred_dstrb = F.softmax(logits)
            pred_dstrbs.append(pred_dstrb)
            total_pred_dstrb += pred_dstrb
        dist_loss = torch.sum(torch.pow(corrupt_im - x, 2).view(self.batch_size, -1), dim=1)
        return torch.sum(dist_loss + c * total_class_loss), dist_loss, \
                input_im, pred_dstrbs, total_pred_dstrb / self.n_models

    def generate(self, data, models, args, fh):
        adv_ims = []
        for i in xrange(data.n_batches):
            adv_ims.append(self.generate_batch(data[i], models, args, fh))
        return np.vstack(adv_ims)

    def generate_batch(self, batch, models, args, fh):
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

        def compare(x, y):
            '''
            Check if predicted class is target class
                or not targeted class if untargeted
            '''
            return x == y if self.targeted else x != y

        ins, outs = batch[0].numpy(), batch[1].numpy()
        batch_size = ins.shape[0]

        # convert inputs to arctanh space
        if self.mean is not None:
            ins = (ins * self.std) + self.mean
        for i in xrange(ins.shape[0]):
            ins[i] = ins[i] - ins[i].min()
            ins[i] = ins[i] / ins[i].max()
        assert ins.max() <= 1.0 and ins.min() >= 0.0 # in [0, 1]
        tanh_ins = 1.999999 * (ins - .5) # in (-1, 1)
        tanh_ins = torch.FloatTensor(np.arctanh(tanh_ins)) # in tanh space
        ins = torch.FloatTensor(ins)

        # make targs one-hot
        one_hot_targs = np.zeros((outs.shape[0], args.n_classes))
        one_hot_targs[np.arange(outs.shape[0]), outs.astype(int)] = 1
        one_hot_targs = torch.FloatTensor(one_hot_targs)
        outs = torch.LongTensor(outs)

        lower_bounds = torch.zeros(batch_size)
        upper_bounds = torch.ones(batch_size) * 1e10
        opt_cs = torch.ones((batch_size,1)) * self.init_const

        overall_best_ims = np.zeros(ins.size())
        overall_best_dists = [1e10] * batch_size

        # variable to optimize
        # TODO
        # - random initialization
        # - multiple restarts?
        w = torch.zeros(ins.size())

        if self.use_cuda:
            tanh_ins, ins, one_hot_targs, w, opt_cs = \
                tanh_ins.cuda(), ins.cuda(), one_hot_targs.cuda(), \
                w.cuda(), opt_cs.cuda()
        tanh_ins, ins, one_hot_targs, w, opt_cs = \
            Variable(tanh_ins), Variable(ins), Variable(one_hot_targs), \
            Variable(w, requires_grad=True), Variable(opt_cs)

        if self.mean is not None:
            mean = torch.FloatTensor(self.mean)
            std = torch.FloatTensor(self.std)
            if self.use_cuda:
                mean, std = mean.cuda(), std.cuda()
            mean = Variable(mean.expand_as(ins))
            std = Variable(std.expand_as(ins))
        else:
            mean, std = None, None

        start_time = time.time()
        for b_step in xrange(self.binary_search_steps):
            log(fh, ('\tBinary search step %d \tavg const: %s' 
                     '\tmin const: %s \tmax const: %s') % 
                     (b_step+1, opt_cs.mean().data[0], 
                     opt_cs.min().data[0], opt_cs.max().data[0]))

            #w.data.zero_()
            #w.data.uniform_(-.1, .1)
            w.data.normal_(0, .1)
            # lazy way to reset optimizer parameters
            if args.generator_optimizer == 'sgd':
                optimizer = optim.SGD([w], lr=args.generator_lr, momentum=args.momentum)
            elif args.generator_optimizer == 'adam':
                optimizer = optim.Adam([w], lr=args.generator_lr)
            elif args.generator_optimizer == 'adagrad':
                optimizer = optim.Adagrad([w], lr=args.generator_lr)
            else:
                raise NotImplementedError

            best_dists = [1e10] * batch_size
            best_classes = [-1] * batch_size

            # repeat binary search one more time?

            prev_loss = 1e6
            for step in xrange(args.n_generator_steps):
                optimizer.zero_grad()
                obj, dists, corrupt_ims, pred_dstrbs, total_pred_dstrb = \
                    self(tanh_ins, ins, w, opt_cs, \
                        one_hot_targs, models, mean, std)
                total_loss = obj.data[0]
                obj.backward()
                optimizer.step()

                if not (step % (args.n_generator_steps / 10.)) and step:
                    # Logging every 1/10
                    _, preds = total_pred_dstrb.topk(1, 1, True, True)
                    n_correct = torch.sum(torch.eq(preds.data.cpu(), outs))

                    n_consensus = 0
                    for i in xrange(outs.size()[0]):
                        if self.targeted:
                            n_consensus += int(sum([outs[i] == dstrb.topk(1, 1, True, True)[1][i].data.cpu()[0] for dstrb in pred_dstrbs]) == self.n_models)
                        else:
                            n_consensus += int(sum([outs[i] != dstrb.topk(1, 1, True, True)[1][i].data.cpu()[0] for dstrb in pred_dstrbs]) == self.n_models)
                    if not self.targeted:
                        n_correct = batch_size - n_correct
                    log(fh, ('\t\tStep %d \tobjective: %06.3f '
                             '\tavg dist: %06.3f \tn targeted class: %02d'
                             '\tn consensus exs: %d \t(%.3f s)') % 
                             (step, total_loss, torch.mean(dists.data), 
                                 n_correct, n_consensus, 
                                 time.time() - start_time))

                    # Check for early abort
                    if self.early_abort and total_loss > prev_loss*.9999:
                        log(fh, '\t\tAborted search because stuck')
                        break
                    prev_loss = total_loss
                    #scheduler.step(total_loss, step)

                # bookkeeping
                for i, (dist, im) in enumerate(zip(dists, corrupt_ims)):

                    if self.targeted:
                        n_voters = sum([outs[i] == dstrb.topk(1, 1, True, True)[1][i].data[0] for dstrb in pred_dstrbs])
                    else:
                        n_voters = sum([outs[i] != dstrb.topk(1, 1, True, True)[1][i].data[0] for dstrb in pred_dstrbs])
                    if n_voters != self.n_models:
                        continue

                    if dist < best_dists[i]: 
                        best_dists[i] = dist
                        best_classes[i] = n_voters
                    if dist < overall_best_dists[i]:
                        overall_best_dists[i] = dist
                        overall_best_ims[i] = im.data.cpu().numpy()

            # binary search stuff
            for i in xrange(batch_size):
                if best_classes[i] != -1: # success; search w/ smaller c
                    assert best_classes[i] == self.n_models
                    upper_bounds[i] = min(upper_bounds[i], opt_cs.data[i,0])
                    if upper_bounds[i] < 1e9:
                        opt_cs[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                else: # failure, search with greater c
                    lower_bounds[i] = max(lower_bounds[i], opt_cs.data[i,0])
                    if upper_bounds[i] < 1e9:
                        opt_cs[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                    else:
                        opt_cs[i] = opt_cs[i] * 10

        # DEBUGGING STUFF
        t_ins = Variable(torch.FloatTensor(overall_best_ims))
        t_outs = F.softmax(models[0](t_ins.cuda()))
        probs, preds = t_outs.topk(1,1,True,True)
        n_correct = torch.sum(torch.eq(preds.data.cpu(), outs))

        return overall_best_ims
