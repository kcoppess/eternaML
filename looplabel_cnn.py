import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from cnn import *
import pandas as pd

restore = None #put in file name of restoring
learning_rate = 1e-4
iters = 1
Ntrain = 1
Ntest = 2
batch_size = 1
save = True
filename = './results'

conv_w_shapes = [ [10, 10, 1, 4], \
        [10, 10, 4, 4], \
        [10, 10, 4, 4] ]

conv_b_shapes = [ [4], \
        [4], \
        [4] ]

fc_w_shapes = [ [80*80*4, 1024], \
        [1024, 6] ]

fc_b_shapes = [ [1024], \
        [6] ]

print 'loading data...'
inputs = pd.read_csv('pairmap2D.csv', delimiter=',', header=None, nrows=Ntrain)
labels = pd.read_csv('loops.csv', delimiter=',', header = None, nrows=Ntrain)

test_inputs = pd.read_csv('pairmap2D.csv', delimiter=',', header=None, skiprows = Ntrain, nrows=Ntest)
test_labels = pd.read_csv('loops.csv', delimiter=',', header=None, skiprows=Ntrain, nrows=Ntest)


print 'construction in progess... {} convolution layers and {} fully connected layers, learning_rate = {}, {} iterations'.format(len(conv_w_shapes), len(fc_w_shapes), learning_rate, iters)
name = '%d_conv_%d_fc_%.2e' % (len(conv_b_shapes),len(fc_b_shapes), learning_rate)
net = CNN(conv_w_shapes, conv_b_shapes, fc_w_shapes, fc_b_shapes, learning_rate, name)

if restore is not None:
    print "restoration..."
    net.restore(restore)
else:
    print "training..."
    loss, accuracy = net.train(inputs, labels, iters, batch_size)
    np.savetxt('%s/%s.loss' % ('./results', net.name), loss, delimiter='\t')
    np.savetxt('%s/%s.accuracy' % ('./results', net.name), accuracy, delimiter='\t')

print "testing..."

test_pred = np.asarray(net.test(test_inputs))
pred = np.argmax(test_pred, axis=1)
truth = np.argmax(np.asarray(test_labels), axis=1)
predictions = np.concatenate((np.array([pred]),np.array([truth]))).transpose()
print predictions
correct = np.equal(truth, pred)
test_accuracy= correct.astype(float)
acc = np.mean(test_accuracy)
np.savetxt('%s/%s.predictions' % ('./results', net.name), predictions, delimiter='\t')
print "final dev-test error: "+str(acc)

if save:
    print "saving model..."
    net.save('_testerr%f' % acc)
