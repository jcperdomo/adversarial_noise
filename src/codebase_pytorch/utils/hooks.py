import pdb
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src.codebase.utils.utils import log

def print_outputs(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print '\t\tInside ' + self.__class__.__name__ + ' forward'
    print '\t\t\tweight norm:', self.weight.data.norm()
    print '\t\t\tweight mean:', self.weight.data.mean()
    print '\t\t\tinput norm:', input[0].data.norm()
    print '\t\t\tinput mean:', input[0].data.mean()
    print '\t\t\toutput norm:', output.data.norm()
    print '\t\t\toutput mean:', output.data.mean()

def print_grads(self, grad_input, grad_output):
    print '\t\tInside ' + self.__class__.__name__ + ' backward'
    print '\t\t\tgrad_params norm: ', self.weight.grad.data.norm()
    print '\t\t\tgrad_params mean: ', self.weight.grad.data.mean()
    print '\t\t\tgrad_input norm: ', grad_input[0].data.norm()
    print '\t\t\tgrad_output norm: ', grad_output[0].data.norm()

