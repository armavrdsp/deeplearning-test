# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 20:40:34 2017

@author: hzxieshukun
"""

import Network
import Network2
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network2.Network2([784, 30, 10], cost=Network2.CrossEntropyCost)
evaluation_cost, evaluation_accuracy, \
training_cost, training_accuracy = net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, \
                                           evaluation_data = validation_data, \
                                           monitor_evaluation_accuracy = True, \
                                           monitor_evaluation_cost = True, \
                                           monitor_training_accuracy = True, \
                                           monitor_training_cost = True)
