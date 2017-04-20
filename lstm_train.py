# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:33:46 2017

@author: hzxieshukun
"""

import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import time

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

##set directories and parameters
BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100,150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'

# whether to re-weight classes to fit the 17.5% share in test set
re_weight = True

STAMP = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

##process texts in datasets
print('Processing text dataset')
def text_to_wordlist(text, remove_stopwords = False, stem_words = False):
    #convert words to lower case and split them
    text = text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    ##clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", "have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e -mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    #optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        
    #return a list of words
    return (text)

## read train data
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
t1 = time.time()
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
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
#test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
#test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
test_texts_1 = []
test_texts_2 = []
test_ids = []
word_index = tokenizer.word_index
print('Foud %s unique tokens' % len(word_index))
t2 = time.time()
print('transfer words to sequences use %ss' % (t2 - t1))

##unify the sequences length to MAX_SEQUENCE_LENGTH
data_1 = pad_sequences(sequences_1, maxlen = MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen = MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print("Shape of data tensor:", data_1.shape)
print("Shape of label tensor:", labels.shape)

#test_data_1 = pad_sequences(test_sequences_1, maxlen = MAX_SEQUENCE_LENGTH)
#test_data_2 = pad_sequences(test_sequences_2, maxlen = MAX_SEQUENCE_LENGTH)
#test_ids = np.array(test_ids)


##index word vectors
t1 = time.time()
print("Indexing word vectors")
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary = True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))
t2 = time.time()
print('index word vectors init use %ss' % (t2 - t1))

##prepare embeddings
print('Preparing embedding marrix')
t1 = time.time()
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embddings: %d' % np.sum(np.sum(embedding_matrix, axis = 1) == 0))
t2 = time.time()
print("prepare embeddings use %ss" % (t2 -t1))


##sample train/validation data -- shuffle the data
t1 = time.time()
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val == 0] = 1.309028344
t2 = time.time()
print("data_1_train shape:", data_1_train.shape)
print("data_1_train example:")
print(data_1_train[1])
print("sample train/val data use %ss" % (t2 -t1))
word2vec = None

##define the model structure
embedding_layer = Embedding(input_dim = nb_words,
        output_dim = EMBEDDING_DIM,
        weights = [embedding_matrix],
        input_length = MAX_SEQUENCE_LENGTH,
        trainable = False)
lstm_layer = LSTM(num_lstm, dropout = rate_drop_lstm, recurrent_dropout = rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype = 'int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype = 'int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation = act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation = 'sigmoid')(merged)

###add class weight
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

##train the model
model = Model(inputs=[sequence_1_input, sequence_2_input], outputs = preds)
model.compile(loss = 'binary_crossentropy',
        optimizer = 'nadam',
        metrics = ['acc'])
print(STAMP)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data = ([data_1_val, data_2_val], labels_val, weight_val), \
        epochs = 200, batch_size = 2048, shuffle = True, \
        class_weight = class_weight, callbacks = [early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
print("best_val_score:", bst_val_score)