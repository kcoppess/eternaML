from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pairmaps as pm
import numpy as np
import random
import time

def summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var);
        tf.summary.scalar('mean', mean);
        tf.summary.histogram('histogram', var);

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
def max_pool_2x2(x, layer_name):
    with tf.name_scope(layer_name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(x, kernal_dim1, kernal_dim2, num_input_channels, num_features, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([kernal_dim1, kernal_dim2, num_input_channels, num_features])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([num_features])
            variable_summaries(biases)
        with tf.name_scope('convWx_plus_b'):
            preactivate = conv2d(x, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

def fully_connected_layer(x, x_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([x_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(x, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activations')
        tf.summary.histogram('activations', activations)
        return activations


def cnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, 1])
    
    h_conv1 = conv_layer(x_image, 5, 5, 1, 4, 'conv1')
    h_pool1 = max_pool_2x2(h_conv1, 'pool1')

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5,5,4,16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        summaries(W_conv2);
        summaries(b_conv2);
    
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5,5,16,32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        summaries(W_conv3);
        summaries(b_conv3);
    
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)
    
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([80*80*32,1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 80*80*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        summaries(W_fc1);
        summaries(b_fc1);

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 640])
        b_fc2 = bias_variable([640])
        summaries(W_fc2);
        summaries(b_fc2);

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

pMap = np.array(pm.pairmaps)
loops = np.array(pm.loops)

features = pMap[:90]
labels = loops[:90]
test_features = pMap[90:]
test_label = loops[90:]
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
tf.summary.scalar('l2', l2)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(l2)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.round(y_conv), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    test_writer = tf.summary.FileWriter('test')
    
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        start = time.clock()
        if i%1 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:features, y_:labels, keep_prob:1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        summary, _ = sess.run([merged, train_step], feed_dict={x: features, y_: labels, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        end = time.clock()
        print(end-start)

    print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_features, y_: test_label, keep_prob: 1.0}))
    prediction = tf.argmax(loops - y_conv,1)
    print(prediction.eval(feed_dict={x: pMap, keep_prob: 1.0}))

    train_writer.close()
    test_writer.close()
# output of network: 640 x 6 matrix
