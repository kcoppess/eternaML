import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from cnn2D import *
import pandas as pd

restore = None #put in file name of restoring
learning_rate = 1e-5
iters = 10
Ntrain = 1
Ntest = 2
batch_size = 1
save = True
filename = './results/loc1D'

conv_w_shapes = [ [5, 5, 2, 96], \
        [5, 5, 96, 96], \
        [5, 5, 96, 96], \
        [5, 5, 96, 96] ]

conv_b_shapes = [ [96], \
        [96], \
        [96], \
        [96] ]

fc_w_shapes = [ [5*5*96, 1024], \
        [1024, 2048], \
        [2048, 4096], \
        [4096, 8192], \
        [8192, 80] ]

fc_b_shapes = [ [1024], \
        [2048], \
        [4096], \
        [8192], \
        [80] ]

print 'loading data...'
all_inputs = np.loadtxt('/Users/kcoppess/yggdrasil/eternaML/single_playout_pid6502997/twolocation_features.csv', delimiter=',') #pd.read_csv('pairmap2D.csv', delimiter=',', header=None, nrows=Ntrain)
all_labels = np.loadtxt('/Users/kcoppess/yggdrasil/eternaML/single_playout_pid6502997/onedim_locations.csv', delimiter=',') #pd.read_csv('loops.csv', delimiter=',', header = None, nrows=Ntrain)

inputs = all_inputs[:Ntrain]
labels = all_labels[:Ntrain]
test_inputs = all_inputs[Ntrain:Ntest] #pd.read_csv('pairmap2D.csv', delimiter=',', header=None, skiprows = Ntrain, nrows=Ntest)
test_labels = all_labels[Ntrain:Ntest] #pd.read_csv('loops.csv', delimiter=',', header=None, skiprows=Ntrain, nrows=Ntest)


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
correct = np.equal(truth, pred)
test_accuracy= correct.astype(float)
acc = np.mean(test_accuracy)
np.savetxt('%s/%s.predictions' % (filename, net.name), predictions, delimiter='\t')
print "final dev-test error: "+str(acc)

if save:
    print "saving model..."
    net.save('_testerr%f' % acc)
