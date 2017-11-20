from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pairmaps as pm
import numpy as np
import random

# initializes weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# initializes bias terms
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution operation
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # zero padding, stride = 1

#pooling operation (2x2)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME')

def cnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5,5,1,2])
        b_conv1 = bias_variable([2])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5,5,2,4])
        b_conv2 = bias_variable([4])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5,5,4,8])
        b_conv3 = bias_variable([8])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)
    
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([80*80*8,1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 80*80*8])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 640])
        b_fc2 = bias_variable([640])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

pMap = np.array(pm.pairmaps)
loops = np.array(pm.loops)

features = pMap[:90]
labels = loops[:90]
test_features = pMap[90:]
test_label = pMap[90:]
'''
data = tf.data.Dataset.from_tensor_slices((training_features, training_label))
iterator = data.make_one_shot_iterator()
features, labels = iterator.get_next()

test_data = tf.data.Dataset.from_tensor_slices((test_features, test_label))
test_iterator = test_data.make_one_shot_iterator()
test_features, test_labels = test_iterator.get_next()
'''
x = tf.placeholder(tf.float32, shape=[None, 640*640])
y_ = tf.placeholder(tf.float32, shape=[None, 640])

y_conv, keep_prob = cnn(x)

with tf.name_scope('loss'):
    l2 = tf.nn.l2_loss(y_ - y_conv)

l2 = tf.reduce_mean(l2)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(l2)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.round(y_conv), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:features, y_:labels, keep_prob:1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: features, y_: labels, keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_data.test_features, y_: test_data.test_label, keep_prob: 1.0}))
# game plan: two convolution layers for small number of examples; figure out diagnostics; error analysis
# output of network: 640 x 6 matrix
