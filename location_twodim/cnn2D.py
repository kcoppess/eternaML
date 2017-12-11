import tensorflow as tf
import numpy as np
import string
import random

rows = 80
columns = 80
depths = 2
l = 80 # number of positions
check = 1
directory = './results/loc1D'

# initializes weights
def weight_variable(shape, name='weights'):
    initial = tf.truncated_normal(shape, stddev = 0.1, name=name)
    return tf.Variable(initial)

# initializes bias terms
def bias_variable(shape, name='biases'):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

# convolution operation
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # zero padding, stride = 1

#pooling operation (2x2)
def max_pool_2x2(x, layer_name):
    with tf.name_scope(layer_name):
        return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(x, w_shape, b_shape, layer_name):
    with tf.name_scope(layer_name):
        weights = weight_variable(w_shape)
        biases = bias_variable(b_shape)
        return weights, conv2d(x, weights) + biases

def fully_connected_layer(x, w_shape, b_shape, layer_name):
    with tf.name_scope(layer_name):
        weights = weight_variable(w_shape)
        biases = bias_variable(b_shape)
        return weights, tf.matmul(x, weights) + biases

def setup_conv_layers(x_input, w_shapes, b_shapes, N):
    x = tf.reshape(x_input, [-1, columns, rows, depths])
    weights = []
    weight, conv = conv_layer(x, w_shapes[0], b_shapes[0], 'conv1')
    act = tf.nn.sigmoid(conv)
    act_pool = max_pool_2x2(act, 'conv1')
    weights.append(weight)
    for i in range(1, N):
        weight, conv = conv_layer(act_pool, w_shapes[i], b_shapes[i], 'conv{}'.format(i+1))
        act = tf.nn.sigmoid(conv)
        act_pool = max_pool_2x2(act, 'conv{}'.format(i+1))
        weights.append(weight)
    return weights, act_pool

def setup_fc_layers(x_input, w_shapes, b_shapes, N):
    shape = tf.shape(x_input)
    x = tf.reshape(x_input, [-1, shape[1]*shape[2]*shape[3]])
    weights = []
    weight, z = fully_connected_layer(x, w_shapes[0], b_shapes[0], 'fc1')
    act = tf.nn.sigmoid(z)
    weights.append(weight)
    for i in range(1, N):
        weight, z = fully_connected_layer(act, w_shapes[i], b_shapes[i], 'fc{}'.format(i+1))
        act = tf.nn.sigmoid(z)
        weights.append(weight)
    return weights, act

class CNN():
    def __init__(self, conv_w_shapes, conv_b_shapes, fc_w_shapes, fc_b_shapes, learning_rate = 1e-5, name=''):
        self.name = name + '_' + ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
        self.sess = tf.Session()
        self.construction(conv_w_shapes, conv_b_shapes, fc_w_shapes, fc_b_shapes, learning_rate)
        self.sess.run(tf.global_variables_initializer())
        self.init_writer()

        self.i = 0

    def init_writer(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('stats', self.sess.graph)

    def construction(self, conv_w_shapes, conv_b_shapes, fc_w_shapes, fc_b_shapes, learning_rate):
        self.pairmap = tf.placeholder(tf.float32, shape=[None, rows*columns*depths], name='pairmap')
        self.loop = tf.placeholder(tf.float32, shape=[None,l], name='loop')

        conv_weights, conv_res = setup_conv_layers(self.pairmap, conv_w_shapes, conv_b_shapes, len(conv_b_shapes))
        fc_weights, result = setup_fc_layers(conv_res, fc_w_shapes, fc_b_shapes, len(fc_b_shapes))

        self.weight = conv_weights + fc_weights
        self.loop_pred = result

        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.loop, logits=result)
        self.loss = tf.reduce_mean(self.loss)
        self.train_summ = tf.summary.scalar("CE loss", self.loss)

        correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(self.loop, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)
        self.train_summ = tf.summary.scalar("Accuracy", self.accuracy)

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train(self, all_pairmaps, all_loops, iters=10000, batch_size=100):
        # NOTE figure out how to manipulate this for epochs
        loss = np.zeros(int(np.ceil(iters/float(check))))
        accuracy = np.zeros(int(np.ceil(iters/float(check))))
        for i in range(iters):
            batch_i = np.random.choice(np.shape(all_loops)[0], batch_size)
            if i%check == 0:
                result = self.sess.run([self.loss, self.accuracy, self.train_summ], feed_dict={self.pairmap: all_pairmaps, self.loop: all_loops})
                loss[i/check] = result[0]
                accuracy[i/check] = result[1]
                self.writer.add_summary(result[2], i)
                print 'step %d: loss %g, accuracy %g' % (i, loss[i/check], accuracy[i/check])

            self.sess.run([self.train_step], feed_dict={self.pairmap: np.take(all_pairmaps, batch_i, axis=0), self.loop: np.take(all_loops, batch_i, axis=0)})
        return loss, accuracy

    def test(self, pairmap):
        return self.loop_pred.eval(session=self.sess,feed_dict={self.pairmap: pairmap})

    def restore(self, filename):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, filename)

    def save(self, suffix=''):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, '%s/%s%s' % (directory, self.name, suffix), global_step=self.i)

    def __del__(self):
        self.writer.close()
