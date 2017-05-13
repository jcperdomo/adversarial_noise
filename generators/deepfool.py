# TODO package imports

class deepfool_generator:
    '''
    Class for generating noise using DeepFool (Moosavi-Dezfooli et al., 2015)
    '''

    def __init__(self, opt):
        '''

        TODO: pass only relevant arguments, not entire opt object
        '''
        self.norm_type = opt.norm_type
        self.max_iter = opt.max_iter

    def generate(self, images, model):
        '''
        Generate adversarial image using DeepFool method

        inputs:
            - images: n_images x im_size x im_size x n_channels
            - model: a model class
        outputs:
            - noises: n_images x im_size x im_size x n_channels

        TODO
            - add max iterations
        '''
        noise = np.zeros(images.shape())
        true_classes = np.argmax(model.predict(images), axis=1)
        false_classes = np.argmax(model.predict(images+noise), axis=1)
        while true_classes == false_classes:
            dists = model.predict(images+noise)
            gradients = model.gradient(images+noise)

            dists - dists[true_classes]
            gradients - gradients[true_classes]
        return noise
