# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:31:39 2017

@author: hzxieshukun
"""

import cPickle
import gzip

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

GPU = False
if GPU:
    print "Trying to run under a GPU. If this is not desired, then modify" +\
    " Network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU. If this is not desired, then the modify "+\
    " Network3.py to set\nthe GPU flag to True."

#加载数据并把数据导入theano内存
def load_data_shared(filename = "../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype = theano.config.floatX), borrow = True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype = theano.config.floatX), borrow = True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), share(test_data)]
            
class Network(object):
    
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size) #这个对吗？两个x
        for j in xrange