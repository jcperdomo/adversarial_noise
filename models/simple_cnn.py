import tensorflow as tf

class simple_cnn:
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

    def __init__(self, opt):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_ph = tf.placeholder(tf.float32, \
                    shape=[None, opt.im_size, opt.im_size, opt.n_channels])
            self.targ_ph = tf.placeholder(tf.int32, shape=[None, 1])
            curr_layer = self.input_ph

            self.weights = []
            for i in xrange(opt.n_modules):
                if not i:
                    weight_init = tf.truncated_normal(
                            [opt.kern_size, opt.kern_size, opt.n_channels, opt.n_kernels],
                            stddev=.1)
                else:
                    weight_init = tf.truncated_normal(
                            [opt.kern_size, opt.kern_size, opt.n_kernels, opt.n_kernels],
                            stddev=.1)
                weight = tf.Variable(weight_init, name='weights_%d' % i)
                conv = tf.nn.conv2d(curr_layer, weight, 
                        [1,1,1,opt.n_channels], 'SAME', name='conv_%d' % i)
                pool = tf.nn.max_pool(conv, [1,2,2,opt.n_channels], 
                        [1,2,2,opt.n_channels], 'SAME', name='pad_%d' % i)
                curr_layer = tf.nn.relu(pool, name='relu_%d' % i)
                self.weights.append(weights)

            # may need to squeeze here
            weight_init = tf.truncated_normal([opt.n_kernels, opt.n_classes], stddev=.1)
            bias_init = tf.truncated_normal([opt.n_classes], stddev=.1)
            weight = tf.Variable(weight_init, name='fc_weight')
            bias = tf.Variable(bias_init, name='fc_bias')
            logits = tf.matmul(curr_layer, weight) + bias

            self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, self.targ_ph))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            self.gradient = tf.gradients(self.loss, [self.input_ph])[0]
            self.predictions = tf.nn.softmax(logits)

    def train(self, data_path):
        '''
        Train the model using data

        TODO
            - training options: n_epochs, halving learning rate, etc.
            - logging
            - model saving
        '''
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            for i in xrange(n_epochs):
                loss = 0
                inputs, outputs = data[i]
                f_dict = {self.input_ph:inputs, self.targ_ph:outputs}
                _, l, preds = session.run([self.optimizer, self.loss, self.distribution], 
                        feed_dict=f_dict)

                loss += l
                print("Epoch %d. Loss: %.3f" % (i, loss))

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
