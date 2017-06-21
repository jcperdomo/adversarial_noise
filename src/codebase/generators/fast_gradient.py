

class FastGradientGenerator:
    '''
    Class for generating noise using fast gradient method (Goodfellow et al., 2015)
    '''

    def __init__(self, args):
        self.eps = args.eps

    def generate(self, ins, outs, model):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - images: n_images x im_size x im_size x n_channels
            - model: a model class
        outputs:
            - adversaries: n_images x im_size x im_size x n_channels
        '''
        gradients = model.get_gradient(ins, outs) # get gradient of model's loss wrt images
        grad_signs = np.sign(gradients)
        return self.eps*grad_signs
