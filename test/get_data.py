# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:34:16 2017

@author: hzxieshukun
"""

#!/usr/bin/python                                                             
#encoding:utf-8

import sys 
import json
import urllib2
from bs4 import BeautifulSoup
import pickle
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")

base_url = 'http://baidu.lecai.com/lottery/draw/list/50?type=range_date&'

def get_original_data(begin = "2015-01-26", end = "2017-02-14"):
    results = []
    url = "%sstart=%s&end=%s" % (base_url, begin, end)
    print url
    content = urllib2.urlopen(url).read()
    soup = BeautifulSoup(content)
    table = soup.table
    tbody = table.tbody
    for h_tr in tbody.children:
        result = {}
        if h_tr.name != 'tr':continue
        for h_td in h_tr.children:
            if h_td.name != 'td':continue
            # find date
            if '20' in h_td.contents[0]:
                result['date'] = h_td.contents[0][0:10]
            # find term
            if h_td.a != None and '20' in h_td.a.contents[0]:
                result['term'] = h_td.a.contents[0]
            # find red balls and blue ball
            if h_td.get('class') != None and 'balls' == h_td.get('class')[0]:
                m_table = h_td.table
                l_tr = m_table.tr
                l_tds = l_tr.children
                for td in l_tds:
                    if td.name != 'td':continue
                    if td.get('class') != None and 'redBalls' == td.get('class')[0]:
                        for em in td.children:
                            if em.name != 'em':continue
                            if 'redBalls' not in result:
                                result['redBalls'] = []
                            result['redBalls'].append(em.contents[0])
                    if td.get('class') != None and 'blueBalls' == td.get('class')[0]:
                        for em in td.children:
                            if em.name != 'em':continue
                            if 'blueBalls' not in result:
                                result['blueBalls'] = []
                            result['blueBalls'].append(em.contents[0])
        results.append(result)
    print "get the orginal red and blue ball data done.\n======"
    return results

def save_original_data(results):
    with open("data/original_data", "w") as f:
        for k in results:
            f.write("%s\n" % json.dumps(k))
    print "save the original data done.\n======"
            
def save_train_test_data(results, term = 50, ratio = 0.7):
    red_ball_features = []
    red_ball_tags = []
    blue_ball_features = []
    blue_ball_tags = []
    for k in xrange(term, len(results)):
        tmp = results[k - term : k]
        red_ball_feature = []
        blue_ball_feature = []
        for k2 in tmp:
            red_ball_feature.extend(red_ball_vectorized_result(k2['redBalls']))
            blue_ball_feature.extend(blue_ball_vectorized_result(k2['blueBalls']))
        red_ball_features.append(red_ball_feature)
        red_ball_tags.append([int(w) for w in results[k]['redBalls']])
        blue_ball_features.append(blue_ball_feature)
        blue_ball_tags.append([int(w) for w in results[k]['blueBalls']])
    red_ball_train = [red_ball_features[0 : int(len(red_ball_features) * ratio)]]
    red_ball_train.append(red_ball_tags[0 : int(len(red_ball_tags) * ratio)])
    red_ball_test = [red_ball_features[int(len(red_ball_features) * ratio) : ]]
    red_ball_test.append(red_ball_tags[int(len(red_ball_tags) * ratio) : ])
    blue_ball_train = [blue_ball_features[0 : int(len(blue_ball_features) * ratio)]]
    blue_ball_train.append(blue_ball_tags[0 : int(len(blue_ball_tags) * ratio)])
    blue_ball_test = [blue_ball_features[int(len(blue_ball_features) * ratio) : ]]
    blue_ball_test.append(blue_ball_tags[int(len(blue_ball_tags) * ratio) : ])                                 
    output = open('data/train_test_data.pkl', 'wb')
    pickle.dump(red_ball_train, output)
    pickle.dump(red_ball_test, output)
    pickle.dump(blue_ball_train, output)
    pickle.dump(blue_ball_test, output)
    output.close()
    print "red_ball_train length:{}".format(len(red_ball_train[0]))
    print "red_ball_test length:{}".format(len(red_ball_test[0]))
    print "blue_ball_train length:{}".format(len(blue_ball_train[0]))
    print "blue_ball_test length:{}".format(len(blue_ball_test[0]))
    
def save_predict_data(results, term = 50):
    red_ball_features = []
    red_ball_tags = []
    blue_ball_features = []
    blue_ball_tags = []
    red_ball_feature = []
    blue_ball_feature = []
    tmp = results[len(results) - term : ]
    for k in tmp:
        red_ball_feature.extend(red_ball_vectorized_result(k['redBalls']))
        blue_ball_feature.extend(blue_ball_vectorized_result(k['blueBalls']))
    red_ball_features.append(red_ball_feature)
    red_ball_tags.append(np.zeros((6)))
    blue_ball_features.append(blue_ball_feature)
    blue_ball_tags.append(np.zeros((1)))
    red_ball_predict_data = [red_ball_features]
    red_ball_predict_data.append(red_ball_tags)
    blue_ball_predict_data = [blue_ball_features]
    blue_ball_predict_data.append(blue_ball_tags)
    output = open('data/predict_data.pkl', 'wb')
    pickle.dump(red_ball_predict_data, output)
    pickle.dump(blue_ball_predict_data, output)
    print "======\nsave the predict data done."
    output.close()
    
    
def red_ball_vectorized_result(t):
    """Return a 33-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...33) into a corresponding desired output from the neural
    network."""
    e = [0.0 * k for k in range(33)]
    for j in t:
        e[int(j) - 1] = 1.0
    return e

def blue_ball_vectorized_result(t):
    """Return a 16-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...33) into a corresponding desired output from the neural
    network."""
    e = [0.0 * k for k in range(16)]
    for j in t:
        e[int(j) - 1] = 1.0
    return e
    
results = get_original_data()
save_original_data(results)
save_train_test_data(results, 50, 0.6)
save_predict_data(results, 50) 
   
"""
url = 'http://baidu.lecai.com/lottery/draw/list/50?type=latest&num=100'
content = urllib2.urlopen(url).read()
soup = BeautifulSoup(content)
table = soup.table
tbody = table.tbody

result = {}
with open('result_publish', 'w') as f:
    for h_tr in tbody.children:
		if h_tr.name != 'tr':continue
		for h_td in h_tr.children:
			if h_td.name != 'td':continue

			# find date
			if '20' in h_td.contents[0]:
				result['date'] = h_td.contents[0][0:10]
			
			# find term
			if h_td.a != None and '20' in h_td.a.contents[0]:
				result['term'] = h_td.a.contents[0]

			# find red balls and blue ball
			if h_td.get('class') != None and 'balls' == h_td.get('class')[0]:
				m_table = h_td.table
				l_tr = m_table.tr
				l_tds = l_tr.children
				for td in l_tds:
					if td.name != 'td':continue
					if td.get('class') != None and 'redBalls' == td.get('class')[0]:
						for em in td.children:
							if em.name != 'em':continue
							if 'redBalls' not in result:
								result['redBalls'] = []
							result['redBalls'].append(em.contents[0])
					if td.get('class') != None and 'blueBalls' == td.get('class')[0]:
						for em in td.children:
							if em.name != 'em':continue
							if 'blueBalls' not in result:
								result['blueBalls'] = []
							result['blueBalls'].append(em.contents[0])
		f.write("%s\n" % json.dumps(result))
		result = {}
"""