from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import random
import time

def variable_summaries(var):
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

def fully_connected_layer(x, x_dim, output_dim, layer_name, act=tf.nn.sigmoid):
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
        x_image = tf.reshape(x, [-1, 640, 640, 2])
    
    h_conv1 = conv_layer(x_image, 10, 10, 2, 32, 'conv1')
    h_pool1 = max_pool_2x2(h_conv1, 'pool1')
    h_conv2 = conv_layer(h_pool1, 10, 10, 32, 64, 'conv2')
    h_pool2 = max_pool_2x2(h_conv2, 'pool2')
    h_conv3 = conv_layer(h_pool2, 10, 10, 64, 128, 'conv3')
    h_pool3 = max_pool_2x2(h_conv3, 'pool3')
    
    h_pool3_flat = tf.reshape(h_pool3, [-1, 80*80*128])
    h_fc1 = fully_connected_layer(h_pool3_flat, 80*80*128, 1024, 'fc1')
    h_fc2 = fully_connected_layer(h_fc1, 1024, 2048, 'fc2')
    h_fc3 = fully_connected_layer(h_fc2, 2048, 4096, 'fc3')
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
    
    y_conv = fully_connected_layer(h_fc3_drop, 4096, 6, 'final')

    return y_conv, keep_prob

length = len(loops)
traintest_boundary = int(0.7*length)

pMap = np.load('pairmap2D.csv', delimiter=',')
loops = np.load('loops.csv', delimiter=',')

features = pMap[:traintest_boundary]
labels = loops[:traintest_boundary]
test_features = pMap[traintest_boundary:]
test_label = loops[traintest_boundary:]

x = tf.placeholder(tf.float32, shape=[None, 2*640*640])
y_ = tf.placeholder(tf.float32, shape=[None, 6])

y_conv, keep_prob = cnn(x)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-7).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('indiv_train', sess.graph)
    test_writer = tf.summary.FileWriter('indiv_test')
    
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        start = time.clock()
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:features, y_:labels, keep_prob:1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        summary, _ = sess.run([merged, train_step], feed_dict={x: features, y_: labels, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        end = time.clock()
        print(end-start)

    print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_features, y_: test_label, keep_prob: 1.0}))
    prediction = tf.round(y_conv) #tf.argmax(loops - y_conv,1)
    pred = prediction.eval(feed_dict={x: pMap, keep_prob: 1.0})
    pred1 = np.asarray(pred)
    pred1 = pred1.astype(int)
    np.savetxt("prediction.csv", pred1, delimiter=",")
    
    train_writer.close()
    test_writer.close()
# output of network: 640 x 6 matrix
