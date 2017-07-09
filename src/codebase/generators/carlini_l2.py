import pdb
import time
import numpy as np
import tensorflow as tf
from src.codebase.utils.utils import log as log
from src.codebase.utils.dataset import Dataset

class CarliniL2Generator(): # TODO superclass
    '''
    Class for generating noise using method in Carlini and Wagner, '17

    TODO
        - deal with discretization
        - multiple starting point gradient descent
    '''

    def __init__(self, args, model):
        self.c = args.generator_opt_const
        self.k = -args.generator_confidence
        self.lr = args.generator_learning_rate

        with model.graph.as_default(), model.sess.as_default():
            # placeholders
            self.ws_ph = tf.placeholder(tf.float32, shape=[None, args.im_size, args.im_size, args.n_channels])
            self.ins_ph = model.input_ph #tf.placeholder(tf.float32, shape=shape)
            self.outs_ph = tf.placeholder(tf.float32, shape=[None, args.n_classes])
            self.lr_ph = lr_ph = tf.placeholder(tf.float32, shape=[])
            # TODO: sometimes outs_ph is a TARGET class

            # targets
            self.obf_im = tf.scalar_mul(.5, tf.tanh(ws_ph) + 1)
            self.noise = self.obf_im -  self.im_ph

            # objective function
            logits = model.logits
            label_score = tf.reduce_sum(self.outs_ph * logits) # borrowed from C
            second_score = tf.reduce_max((1. - self.outs_ph) * logits)
            class_score = tf.maximum(second_score - label_score, -self.k)

            self.objective = tf.reduce_sum(tf.square(self.noise)) + \
                    tf.scalar_mul(self.c, class_score)

            # optimizer
            if args.generator_optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(lr_ph).minimize(self.objective)
            elif args.generator_optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(self.objective)
            elif args.generator_optimizer == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(lr_ph).minimize(self.objective)
            else:
                raise NotImplementedError

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

        if type(data) is tuple:
            ins = data[0]
            outs = data[1]
        elif type(data) is Dataset:
            ins = data.ins
            outs = data.outs
        else:
            raise TypeError("Invalid data format")

        # make outs one-hot
        one_hot_outs = np.zeroes((outs.shape[0], args.n_classes))
        one_hot_outs[np.arange(outs.shape[0]), outs] = 1

        with model.graph.as_default(), model.sess.as_default():
            # initialize w
            ws = tf.Variable(np.zeros(ins.shape[0], args.im_size, args.im_size, args.n_channels))
            tf.global_variables_initializer().run()

            for i in xrange(args.n_generator_steps):
                f_dict = {self.ws_ph: ws, self.ins_ph:ins, 
                        self.out_ph:one_hot_outs, self.lr_ph: self.lr, 
                        model.phase_ph:True}
                _, obj_val, noise = self.session.run(
                        [self.optimizer, self.objective, self.noise], 
                        feed_dict=f_dict)
                if not (i % 10) and i:
                    log(fh, '\tStep %d: objective: %.3f, avg noise magnitude: %.3f' %
                            (i+1, obj_val, np.mean(noise)))

        return noise
