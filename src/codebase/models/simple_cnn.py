import pdb
import numpy as np
import tensorflow as tf

class Simple_CNN:
    '''
    Class for a simple CNN (four convolutional modules)

    TODO
        - super CNN class
        - variable kernel sizes
        - batch normalization
        - other nonlinearities
        - consistent naming convenations
        - reuse tf.Session()?
    '''

    def __init__(self, args):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_ph = tf.placeholder(tf.float32, \
                    shape=[None, args.im_size, args.im_size, args.n_channels])
            self.targ_ph = tf.placeholder(tf.int32, shape=[None])
            curr_layer = self.input_ph

            self.weights = []
            for i in xrange(6):
                if not i:
                    weight_init = tf.truncated_normal(
                            [args.kern_size, args.kern_size, args.n_channels, args.n_kernels],
                            stddev=.1)
                else:
                    weight_init = tf.truncated_normal(
                            [args.kern_size, args.kern_size, args.n_kernels, args.n_kernels],
                            stddev=.1)
                weight = tf.Variable(weight_init, name='weights_%d' % i)
                conv = tf.nn.conv2d(curr_layer, weight, 
                        [1,1,1,1], 'SAME', name='conv_%d' % i)
                pool = tf.nn.max_pool(conv, [1,2,2,1], 
                        [1,2,2,1], 'VALID', name='pad_%d' % i)
                curr_layer = tf.squeeze(
                        tf.nn.relu(pool, name='relu_%d' % i))
                self.weights.append(weight)

            weight_init = tf.truncated_normal([args.n_kernels, args.n_classes], stddev=.1)
            bias_init = tf.truncated_normal([args.n_classes], stddev=.1)
            weight = tf.Variable(weight_init, name='fc_weight')
            bias = tf.Variable(bias_init, name='fc_bias')
            logits = tf.matmul(curr_layer, weight) + bias

            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.targ_ph))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            self.gradient = tf.gradients(self.loss, [self.input_ph])[0]
            self.predictions = tf.nn.softmax(logits)

    def train(self, tr_data, val_data, args):
        '''
        Train the model using data

        TODO
            - training options: n_epochs, halving learning rate, etc.
            - logging
            - model saving
        '''
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            for i in xrange(args.n_epochs):
                loss = 0.
                accuracy = 0.
                n_ins = 0.
                for j in xrange(tr_data.n_batches):
                    inputs, outputs = tr_data[i]
                    f_dict = {self.input_ph:inputs, self.targ_ph:outputs}
                    _, l, preds = session.run(
                            [self.optimizer, self.loss, self.predictions], 
                            feed_dict=f_dict)
                    loss += l
                    accuracy += np.sum(np.equal(outputs, np.argmax(preds)))
                    n_ins += inputs.shape[0]
                print("Epoch %d, loss: %.3f, accuracy: %.3f" % 
                        (i, loss/tr_data.n_batches, accuracy/ins))

    def predict(self, inputs):
        '''
        Return class probability distributions for inputs
        '''
        with tf.Session(graph=self.graph) as session:
            f_dict = {self.input_ph:inputs}
            preds = session.run([self.predictions], feed_dict=f_dict)
        return preds

    def gradient(self, inputs, outputs):
        '''
        Return gradient of loss wrt inputs
        '''
        with tf.Session(graph=self.graph) as session:
            f_dict = {self.input_ph:inputs, self.targ_ph:outputs}
            gradients = session.run([self.gradient], feed_dict=f_dict)
        return gradients
