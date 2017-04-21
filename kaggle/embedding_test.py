# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:15:52 2017

@author: hzxieshukun
"""
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
import numpy as np

a = np.zeros((10, 2))
a[0] = np.array([0,0])
a[1] = np.array([1,1])
a[2] = np.array([2,2])
a[3] = np.array([3,3])
a[4] = np.array([4,4])
a[5] = np.array([5,5])
a[6] = np.array([6,6])
a[7] = np.array([7,7])
a[8] = np.array([8,8])
a[9] = np.array([9,9])
model = Sequential()
model.add(Embedding(10, 2, weights = [a], input_length=10))
input_array = np.random.randint(10, size = (10, 10))
print("input_array:\n", input_array)
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print("output_array length:%s" % (len(output_array)))
print("output_array[0][0] length:%s" % (len(output_array[0][0])))
print("output_array:\n", output_array)