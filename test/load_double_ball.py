# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:08:47 2017

@author: hzxieshukun
"""

import pickle
import numpy as np

def load_traintest_data():
    pkl_file = open('data/train_test_data.pkl', 'rb')
    red_ball_train = pickle.load(pkl_file)
    red_ball_test = pickle.load(pkl_file)
    blue_ball_train = pickle.load(pkl_file)
    blue_ball_test = pickle.load(pkl_file)
    pkl_file.close()
    return red_ball_train, red_ball_test, blue_ball_train, blue_ball_test

def load_predict_data():
    pkl_file = open('data/predict_data.pkl', 'rb')
    red_ball_preditct_data = pickle.load(pkl_file)
    blue_ball_predict_data = pickle.load(pkl_file)
    pkl_file.close()
    return red_ball_preditct_data, blue_ball_predict_data
    
def load_traintest_data_wrapper():
    red_ball_train, red_ball_test, blue_ball_train, blue_ball_test = load_traintest_data()
    
    red_ball_train_inputs = [np.reshape(x, (1650, 1)) for x in red_ball_train[0]]
    red_ball_train_tags = [red_ball_vectorized_result(t) for t in red_ball_train[1]]
    red_ball_train_data = zip(red_ball_train_inputs, red_ball_train_tags)
    red_ball_test_inputs = [np.reshape(x, (1650, 1)) for x in red_ball_test[0]]
    red_ball_test_tags = red_ball_test[1]
    red_ball_test_data = zip(red_ball_test_inputs, red_ball_test_tags)
    
    blue_ball_train_inputs = [np.reshape(x, (800, 1)) for x in blue_ball_train[0]]
    blue_ball_train_tags = [blue_ball_vectorized_result(t) for t in blue_ball_train[1]]
    blue_ball_train_data = zip(blue_ball_train_inputs, blue_ball_train_tags)
    blue_ball_test_inputs = [np.reshape(x, (800, 1)) for x in blue_ball_test[0]]
    blue_ball_test_tags = blue_ball_test[1]
    blue_ball_test_data = zip(blue_ball_test_inputs, blue_ball_test_tags)
    return red_ball_train_data, red_ball_test_data, blue_ball_train_data, blue_ball_test_data
    
def load_predict_data_wrapper():
    red_ball_preditct_data, blue_ball_predict_data = load_predict_data()
    
    red_ball_predict_inputs = [np.reshape(x, (1650, 1)) for x in red_ball_preditct_data[0]]
    red_ball_predict_tags = red_ball_preditct_data[1]
    red_ball_predict_data = zip(red_ball_predict_inputs, red_ball_predict_tags)
    
    blue_ball_predict_inputs = [np.reshape(x, (800, 1)) for x in blue_ball_predict_data[0]]
    blue_ball_predict_tags = blue_ball_predict_data[1]
    blue_ball_predict_data = zip(blue_ball_predict_inputs, blue_ball_predict_tags)
    return red_ball_predict_data, blue_ball_predict_data
    
def red_ball_vectorized_result(t):
    """Return a 34-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...33) into a corresponding desired output from the neural
    network."""
    e = np.zeros((33, 1))
    for j in t:
        e[j - 1] = 1.0
    return e

def blue_ball_vectorized_result(t):
    """Return a 17-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...33) into a corresponding desired output from the neural
    network."""
    e = np.zeros((16, 1))
    for j in t:
        e[j - 1] = 1.0
    return e