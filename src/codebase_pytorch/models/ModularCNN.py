import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from src.codebase.utils.utils import log

'''

Modular CNN

'''

class ModularCNN(nn.Module):
    def __init__(self, args):
        '''
        Model variables

        TODO
            - init scales
            - saving and loading models
            - get gradient of loss wrt input
            - get prediction (probability distribution)
            - get logits
        '''
        super(ModularCNN, self).__init__()
        self.weights = []
        for i in xrange(args.n_modules):
            if not i:
                conv = nn.Conv2d(args.n_channels, args.n_kerns, 
                        kernel_size=args.kern_size)
            else:
                conv = nn.Conv2d(args.n_kerns, args.n_kerns, 
                        kernel_size=args.kern_size)
            bn = nn.BatchNorm2d(args.n_kerns)
            self.weights.append((conv, bn))
        self.fc = nn.Linear(args.n_kerns, args.n_classes)

    def forward(self, x):
        '''
        Model definition

        Following the modular CNN from Vinyals, each module is
            conv -> bn -> relu -> max_pool
        The tutorial seems to suggest
            conv -> max_pool -> relu
        '''
        for i in xrange(len(self.weights)):
            #x = F.relu(F.max_pool2d(self.weights[i][0](x), 2))
            x = F.max_pool2d(F.relu(self.weights[i][1](self.weights[i][0](x))), 2)
        x = self.fc(x)
        return F.log_softmax(x)

    def train_model(self, args, tr_data, val_data, fh):
        '''
        Train the model according to the parameters specified in args.
        '''
        self.train()
        optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)
        start_time = time.time()
        val_loss, val_acc = self.evaluate(val_data)
        best_loss, best_acc = val_loss, val_acc
        last_acc = val_acc
        log(fh, "\tInitial val loss: \t %.3f, val acc: %.2f \t(%.3f s)" %
                (val_loss, val_acc, time.time() - start_time()))

        for epoch in xrange(args.n_epochs):
            log(fh, "\tEpoch %d, \tlearning rate: %.3f" % (i+1, args.lr))
            total_loss = 0.
            start_time = time.time()
            for batch_idx in xrange(tr_data.n_batches): # TODO use PyTorch data class
                ins, targs = tr_data[j]
                if args.cuda:
                    ins, targs = ins.cuda(), targs.cuda()
                ins, targs = Variable(ins), Variable(targs)
                optimizer.zero_grad()
                outs = self(ins)
                loss = F.nll_loss(outs, targs, size_average=False)
                total_loss += loss.data[0]
                loss.backward()
                optimizer.step()

            _, val_acc = self.evaluate(val_data)
            log(fh, "\t\tTraining loss: %.2f \tValidation accuracy: %.2f \t(%.3f s)"
                    % (epoch, total_loss / tr_data.n_ins, val_acc, 
                        time.time()-start_time))
            if val_acc > best_acc:
                # TODO save model
                best_acc = val_acc
            if val_acc <= last_acc:
                # TODO degrade learning rate
                pass
            last_acc = val_acc

        # TODO load best model
        _, val_acc = self.evaluate(val_data)
        log(fh, "\tFinished training in %.3f s, \tValidation accuracy: %.2f" % 
                (time.time() - start_time, val_acc))
        return

    def evaluate(self, data):
        '''
        Evaluate model on data, usually either validation or test
        '''
        self.eval()
        total_loss, total_correct = 0., 0.
        for batch_idx in xrange(data.n_batches):
            ins, targs = data[batch_idx]
            if args.cuda:
                ins, targs = ins.cuda(), targs.cuda()
            ins, targs = Variable(ins, volatile=True), Variable(targs)
            outs = self(ins)
            total_loss += F.nll_loss(outs, targs, size_average=False).data[0]
            preds = outs.data.max(1)[1]
            total_correct += preds.eq(targs.data).cpu().sum()
        return total_loss / len(data.dataset), \
                100. * total_correct / len(data.dataset)