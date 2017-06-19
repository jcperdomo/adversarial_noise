

class fast_gradient_generator:
    '''
    Class for generating noise using fast gradient method (Goodfellow et al., 2015)
    '''

    def __init__(self, opt):
        self.eps = opt.eps

    def generate(self, images, model):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - images: n_images x im_size x im_size x n_channels
            - model: a model class
        outputs:
            - adversaries: n_images x im_size x im_size x n_channels
        '''
        gradients = model.gradient(images) # get gradient of model's loss wrt images
        grad_signs = np.sign(gradients)
        return self.eps*grad_signs
