import pdb
import numpy as np
from src.codebase.utils.dataset import Dataset

class FastGradientGenerator:
    '''
    Class for generating noise using fast gradient method (Goodfellow et al., 2015)
    Also allows for random initialization of the noise, which was shown to
        improve performance (Tramer et al., 2017)

    '''

    def __init__(self, args):
        self.eps = args.eps
        self.alpha = args.alpha
        self.n_iters = args.n_iters

    def generate(self, data, model):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - images: n_images x im_size x im_size x n_channels
            - model: a model class
        outputs:
            - adversaries: n_images x im_size x im_size x n_channels
            - noise: n_ims x im_size x im_size x n_channels
        '''

        if type(data) is tuple:
            ins = data[0]
            outs = data[1]
        elif type(data) is Dataset:
            ins = data.ins
            outs = data.outs
        else:
            raise NotImplementedError("Invalid data format")

        for _ in xrange(self.n_iters):
            if self.alpha:
                random_noise = np.random.normal(0, 1, size=ins.shape)
                ins = ins + self.alpha * random_noise
                gradients = model.get_gradient(ins, outs)
                adv_noise = self.alpha * random_noise + \
                    (self.eps - self.alpha) * np.sign(gradients)
            else:
                gradients = model.get_gradient(ins, outs)
                adv_noise = self.eps * np.sign(gradients)
        return adv_noise
