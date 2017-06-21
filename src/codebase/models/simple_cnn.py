import pdb
import time
import numpy as np
import tensorflow as tf
from utils.utils import log as log

class SimpleCNN:
    '''
    Class for a simple CNN (four convolutional modules)

    TODO
        - super CNN class?
        - variable kernel sizes
        - batch normalization
        - other nonlinearities
        - consistent naming convenations
        - reuse tf.Session()?
    '''

    def __init__(self, args):
        # Build computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_ph = tf.placeholder(tf.float32, \
                    shape=[None, args.im_size, args.im_size, args.n_channels])
            self.targ_ph = tf.placeholder(tf.int32, shape=[None])
            self.phase_ph = tf.placeholder(tf.bool) # for BN
            curr_layer = self.input_ph

            self.weights = []
            for i in xrange(args.n_modules):
                if not i:
                    weight_init = tf.truncated_normal(
                            [args.kern_size, args.kern_size, args.n_channels, args.n_kernels],
                            stddev=args.init_scale)
                else:
                    weight_init = tf.truncated_normal(
                            [args.kern_size, args.kern_size, args.n_kernels, args.n_kernels],
                            stddev=args.init_scale)
                weight = tf.Variable(weight_init, name='weights_%d' % i)
                conv = tf.nn.conv2d(curr_layer, weight, 
                        [1,1,1,1], 'SAME', name='conv_%d' % i)
                bn = tf.layers.batch_normalization(conv, 
                        training=self.phase_ph,
                        name='bn_%d' % i)
                nonlin = tf.nn.relu(bn, name='relu_%d' % i)
                pool = tf.nn.max_pool(nonlin, [1,2,2,1], 
                        [1,2,2,1], 'VALID', name='pad_%d' % i)
                curr_layer = pool
                self.weights.append(weight)

            weight_init = tf.truncated_normal([args.n_kernels, args.n_classes], stddev=args.init_scale)
            bias_init = tf.truncated_normal([args.n_classes], stddev=args.init_scale*.01)
            weight = tf.Variable(weight_init, name='fc_weight')
            bias = tf.Variable(bias_init, name='fc_bias')
            self.weights += [weight, bias]
            logits = tf.matmul(tf.squeeze(curr_layer), weight) #+ bias

            # Loss and gradients
            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.targ_ph))
            self.gradient = tf.gradients(self.loss, [self.input_ph])[0]
            self.predictions = tf.nn.softmax(logits)

    def train(self, tr_data, val_data, args, fh):
        '''
        Train the model using data

        TODO
            - training options: n_epochs, halving learning rate, etc.
            - logging
            - model saving
        '''


        with tf.Session(graph=self.graph) as session:

            # Setup
            self.learning_rate_ph = learning_rate_ph = tf.placeholder(tf.float32, shape=[])
            learning_rate = args.learning_rate
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if args.optimizer == 'sgd':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(self.loss)
                elif args.optimizer == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph).minimize(self.loss)
                elif args.optimizer == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate_ph).minimize(self.loss)
                else:
                    raise NotImplementedError

            # Initial stuff
            tf.global_variables_initializer().run()
            initial_time = time.time()
            val_loss, val_acc = self.validate(val_data, session)
            best_loss, best_acc = val_loss, val_acc
            last_acc = val_acc
            log(fh, "\tInitial val loss: %.3f, val acc: %.3f (%.3f s)" % 
                    (val_loss, val_acc, time.time() - initial_time))

            # Training loop
            for i in xrange(args.n_epochs):
                log(fh, "\tEpoch %d, learning rate: %.3f" % (i+1, learning_rate))
                total_loss = 0.
                total_correct = 0.
                n_ins = 0.
                start_time = time.time()
                for j in xrange(tr_data.n_batches):
                    inputs, outputs = tr_data[j]
                    f_dict = {self.input_ph:inputs, self.targ_ph:outputs,
                            self.learning_rate_ph:learning_rate,
                            self.phase_ph:True}
                    _, loss, preds = session.run(
                            [self.optimizer, self.loss, self.predictions], 
                            feed_dict=f_dict)
                    total_loss += loss
                    total_correct += np.sum(
                            np.equal(outputs, np.argmax(preds, axis=1)))
                    n_ins += inputs.shape[0]

                val_loss, val_acc = self.validate(val_data, session)
                log(fh, "\t\ttraining loss: %.3f, accuracy: %.3f" % 
                        (total_loss/tr_data.n_batches, total_correct/n_ins))
                log(fh, "\t\tval loss: %.3f, val acc: %.3f (%.3f s)" % 
                        (val_loss, val_acc, time.time()-start_time))
                if val_acc <= last_acc:
                    learning_rate *= .5
                    log(fh, "\t\tLearning rate halved to %.3f" % learning_rate)
                    if val_acc == last_acc:
                        pdb.set_trace()
                else:
                    best_acc = val_acc
                last_acc = best_acc

        log(fh, "\tFinished training in %.3f s" % (time.time() - initial_time))

    def validate(self, data, session=None):
        if session is None:
            session = tf.Session(graph=self.graph) # this won't really work because I need to close it
        total_loss = 0.
        n_correct = 0.
        n_ins = 0.
        for i in xrange(data.n_batches):
            inputs, outputs = data[i]
            f_dict = {self.input_ph:inputs, self.targ_ph:outputs,
                    self.phase_ph:False}
            l, preds = session.run(
                    [self.loss, self.predictions], 
                    feed_dict=f_dict)
            total_loss += l
            n_correct += np.sum(np.equal(outputs, np.argmax(preds, axis=1)))
            n_ins += inputs.shape[0]
        return total_loss/data.n_batches, 100.* n_correct/n_ins

    def get_predictions(self, inputs):
        '''
        Return class probability distributions for inputs
        '''
        with tf.Session(graph=self.graph) as session:
            f_dict = {self.input_ph:inputs}
            preds = session.run([self.predictions], feed_dict=f_dict)
        return preds

    def get_gradient(self, inputs, outputs):
        '''
        Return gradient of loss wrt inputs
        '''
        with tf.Session(graph=self.graph) as session:
            f_dict = {self.input_ph:inputs, self.targ_ph:outputs, 
                    self.phase_ph:False}
            gradients = session.run([self.gradient], feed_dict=f_dict)
        return gradients
