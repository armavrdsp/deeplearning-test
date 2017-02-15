# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:15:59 2017

@author: hzxieshukun
"""

import load_double_ball
import network
import numpy as np


    
red_ball_train, red_ball_test, blue_ball_train, blue_ball_test \
                                = load_double_ball.load_traintest_data_wrapper()

red_ball_model = network.load("model/red_ball_model")
blue_ball_model = network.load("model/bule_ball_model")
red_result = red_ball_model.predict(red_ball_test)
blue_result = blue_ball_model.predict(blue_ball_test)
win = {"6th":0, "5th":0, "4th":0, "3th":0, "2th":0, "1th":0}

def print_win_info(info, i):
    print "======\n{}".format(info)
    print "-{} red:predict:{}, true:{}".format(len(red_result) - i, red_result[i], red_ball_test[i][1])
    print "-{} blue:predict:{}, true:{}".format(len(blue_result) - i, blue_result[i], blue_ball_test[i][1])
    
for i in xrange(len(red_result)):
    red_right_num = len(np.intersect1d(red_result[i], red_ball_test[i][1]))
    blue_right_num = len(np.intersect1d(blue_result[i], blue_ball_test[i][1]))
    if red_right_num < 3 and blue_right_num == 1:
        win["6th"] += 1
        print_win_info("6th", i)
    elif (red_right_num == 4 and blue_right_num == 0) \
            or (red_right_num == 3 and blue_right_num == 1):
        win["5th"] += 1
        print_win_info("5th", i)
    elif (red_right_num == 5 and blue_right_num == 0) \
            or (red_right_num == 4 and blue_right_num == 1):
        win["4th"] += 1
        print_win_info("4th", i)
    elif red_right_num == 5 and blue_right_num == 1:
        win["3th"] += 1
        print_win_info("3th", i)
    elif red_right_num == 6 and blue_right_num == 0:
        win["2th"] += 1
        print_win_info("2th", i)
    elif red_right_num == 6 and blue_right_num == 1:
        win["1th"] += 1
        print_win_info("1th", i)
print "total-{}. win:{}".format(len(red_result), win)


#blue_result = blue_ball_model.predict(blue_ball_test)
#print blue_result
