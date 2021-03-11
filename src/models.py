# -*- coding: utf-8 -*-

"""
"""
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import keras.backend as k
from keras.models import load_model

import os
import tensorflow as tf
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
random.seed(7)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
k.set_session(sess)

class MLP:
    """
    ラベル分類モデル
    """
    def __init__(
        self,
        in_seq,
        out_vec_size,
        dropout_rate=0.2,
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        ):
        self._in_seq = in_seq
        self._out_vec_size = out_vec_size
        self._dropout_rate = dropout_rate
        self._loss = loss
        self._optimizer = optimizer

    def build_model(self):
        model = Sequential()
        model.add(Dense(int(self._in_seq*1.2), input_shape=(self._in_seq,)))
        model.add(Activation('relu'))
        model.add(Dropout(self._dropout_rate))
        model.add(Dense(int(self._in_seq*0.2)))
        model.add(Activation('relu'))
        model.add(Dropout(self._dropout_rate))
        model.add(Dense(self._out_vec_size))
        model.add(Activation('softmax'))
        model.summary()

        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=['accuracy'])

        return model

class CUIMLP:
    """
    CUI分類モデル
    """
    def __init__(
        self,
        in_seq,
        out_vec_size=3,
        dropout_rate=0.2,
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        ):
        self._in_seq = in_seq
        self._out_vec_size = out_vec_size
        self._dropout_rate = dropout_rate
        self._loss = loss
        self._optimizer = optimizer

    def build_model(self):
        model = Sequential()
        model.add(Dense(int(self._in_seq*1.2), input_shape=(self._in_seq,)))
        model.add(Activation('relu'))
        model.add(Dropout(self._dropout_rate))
        model.add(Dense(int(self._in_seq*0.2)))
        model.add(Activation('relu'))
        model.add(Dropout(self._dropout_rate))
        model.add(Dense(self._out_vec_size))
        model.add(Activation('softmax'))
        model.summary()

        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=['accuracy'])

        return model
