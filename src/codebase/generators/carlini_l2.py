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

    def __init__(self, args, model, n_ins):
        self.c = args.generator_opt_const
        self.k = -args.generator_confidence
        self.lr = args.generator_learning_rate

        with model.graph.as_default(), model.session.as_default():
            # placeholders
            self.ins_ph = ins_ph = model.input_ph 
            self.outs_ph = outs_ph = \
                    tf.placeholder(tf.float32, shape=[None, args.n_classes])
            self.lr_ph = lr_ph = tf.placeholder(tf.float32, shape=[])
            self.ws = tf.Variable(np.zeros((n_ins, args.im_size, args.im_size, args.n_channels), dtype=np.float32))

            # stuff we care about
            self.obf_im = tf.scalar_mul(.5, tf.tanh(self.ws) + 1)
            self.noise = self.obf_im - ins_ph

            # objective function
            logits = model.logits # v borrowed from Carlini
            label_score = tf.reduce_sum(outs_ph * logits) 
            second_score = tf.reduce_max((1. - outs_ph) * logits)
            class_score = tf.maximum(second_score - label_score, -self.k)

            self.objective = tf.reduce_sum(tf.square(self.noise)) + \
                    tf.scalar_mul(self.c, class_score)

            # optimizer
            if args.generator_optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(lr_ph, name='optimizer').minimize(self.objective)
            elif args.generator_optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph, name='optimizer').minimize(self.objective)
            elif args.generator_optimizer == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(lr_ph, name='optimizer').minimize(self.objective)
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

        if isinstance(data, tuple):
            ins = data[0]
            outs = data[1]
        elif isinstance(data, Dataset):
            ins = data.ins
            outs = data.outs
        else:
            raise TypeError("Invalid data format")

        # make outs one-hot
        one_hot_outs = np.zeros((outs.shape[0], args.n_classes))
        one_hot_outs[np.arange(outs.shape[0]), outs.astype(int)] = 1

        with model.graph.as_default(), model.session.as_default():
            global_vars = tf.global_variables()
            is_initialized = model.session.run([tf.is_variable_initialized(var) for var in global_vars])
            uninitialized = [v for (v, f) in zip(global_vars, is_initialized) if not f]
            tf.variables_initializer(uninitialized).run()

            for i in xrange(args.n_generator_steps):
                f_dict = {self.ins_ph:ins, self.outs_ph:one_hot_outs, 
                        self.lr_ph:self.lr, model.phase_ph:True}
                _, obj_val, noise = model.session.run(
                        [self.optimizer, self.objective, self.noise], 
                        feed_dict=f_dict)
                if not (i % 10) and i:
                    log(fh, '\tStep %d: objective: %.3f, avg noise magnitude: %.3f' %
                            (i, obj_val, np.mean(noise)))

        return noise
