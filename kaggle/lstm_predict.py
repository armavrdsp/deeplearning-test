# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:55:41 2017

@author: hzxieshukun
"""

import csv
import codecs
import numpy as np
import pandas as pd
import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from util import text_to_wordlist

BASE_DIR = 'data/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MODEL_VERSION = 'lstm_1_1_0.00_0.00.h5'
MODEL_FILE = MODEL_VERSION
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
SET_NB_WORDS = 8000
EMBEDDING_DIM = 300

print("start to predict")
## read train data
t1 = time.time()
texts_1 = []
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding = 'utf-8') as f:
    reader = csv.reader(f, delimiter = ',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))


## read test data
test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding = 'utf-8') as f:
    reader = csv.reader(f, delimiter = ',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))
t2 = time.time()
print("load data use %ss" % (t2 - t1))

## transfer words to sequences
t1 = time.time()
tokenizer = Tokenizer(num_words=SET_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
t2 = time.time()
print("transfer words to sequences use %ss" % (t2 - t1))

##unify the sequences length to MAX_SEQUENCE_LENGTH
t1 = time.time()
test_data_1 = pad_sequences(test_sequences_1, maxlen = MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen = MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)
t2 = time.time()
print("unify the sequences length to MAX_SEQUENCE_LENGTH use %ss" % (t2 - t1))

#load model
model = load_model(MODEL_FILE)

##making submission
print('Start making the submission before fine-tuning')
preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2
submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv(MODEL_VERSION+'_submission.csv', index=False)