# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:17:19 2017

@author: hzxieshukun
"""

import load_double_ball
import network

red_ball_predict_data, blue_ball_predict_data = load_double_ball.load_predict_data_wrapper()
red_ball_model = network.load("model/red_ball_model")
blue_ball_model = network.load("model/bule_ball_model")
red_result = red_ball_model.predict(red_ball_predict_data)
blue_result = blue_ball_model.predict(blue_ball_predict_data)
print "red balls:{}, blue ball:{}".format(red_result[0], blue_result[0])