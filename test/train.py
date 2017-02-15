# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:48:50 2017

@author: hzxieshukun
"""

import load_double_ball
import network


###train###
red_ball_train, red_ball_test, blue_ball_train, blue_ball_test \
                                = load_double_ball.load_traintest_data_wrapper() 

#red ball
net = network.Network([1650, 30, 33], cost=network.CrossEntropyCost, color = "red")
evaluation_cost, evaluation_accuracy, \
training_cost, training_accuracy = net.SGD(red_ball_train, 100, 10, 0.01, lmbda = 0, \
                                           evaluation_data = red_ball_test, \
                                           monitor_evaluation_accuracy = True, \
                                           monitor_evaluation_cost = True, \
                                           monitor_training_accuracy = True, \
                                           monitor_training_cost = True)

net.save("model/red_ball_model")

#blue ball
net = network.Network([800, 100, 16], cost=network.CrossEntropyCost, color = "blue")
evaluation_cost, evaluation_accuracy, \
training_cost, training_accuracy = net.SGD(blue_ball_train, 50, 10, 0.1, lmbda = 0, \
                                           evaluation_data = blue_ball_test, \
                                           monitor_evaluation_accuracy = True, \
                                           monitor_evaluation_cost = True, \
                                           monitor_training_accuracy = True, \
                                           monitor_training_cost = True)
net.save("model/bule_ball_model")
