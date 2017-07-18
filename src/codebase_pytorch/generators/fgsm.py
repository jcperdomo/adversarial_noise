import pdb
import numpy as np
from src.codebase.utils.dataset import Dataset

class FastGradientGenerator:
    '''
    Class for generating noise using fast gradient method (Goodfellow et al., 2015)
    Also allows for random initialization of the noise, which was shown to
        improve performance (Tramer et al., 2017)

    TODO
        - super generator class
        - iterated FGM
        - random FGM

    '''

    def __init__(self, args):
        self.eps = args.eps
        self.alpha = args.alpha
        self.targeted = (args.target != 'none')

    def generate(self, data, model, args, fh=None):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - images: n_images x im_size x im_size x n_channels
            - model: a model class
        outputs:
            - adversaries: n_images x im_size x im_size x n_channels
            - noise: n_ims x im_size x im_size x n_channels
        '''

        if isinstance(data, tuple):
            ins = data[0]
            outs = data[1]
        elif isinstance(data, Dataset):
            ins = data.ins
            outs = data.outs
        else:
            raise NotImplementedError("Invalid data format")

        for _ in xrange(args.n_generator_steps):
            if self.alpha:
                random_noise = self.alpha * \
                        np.sign(np.random.normal(0, 1, size=ins.shape))
                ins = ins + random_noise
                gradients = model.get_gradient(ins, outs)
                # TODO deal with targeted version
                adv_noise = random_noise + \
                    (self.eps - self.alpha) * np.sign(gradients)
            else:
                gradients = model.get_gradient(ins, outs)
                if self.targeted:
                    gradients *= -1.
                adv_noise = self.eps * np.sign(gradients)
        return adv_noise
