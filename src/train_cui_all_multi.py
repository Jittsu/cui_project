# -*- coding: utf-8 -*-

"""
2020-11-16 created
clear, unclear or impossibleの3値分類を行うモデルの訓練用コード
10分割交差検証は行わず、全てで学習
from train_mlp_clearlabeling_multi.py to predict_cui.py or cuipredict4tests/predict_cui.py
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import collections
from keras.utils import np_utils, to_categorical
import keras.backend as k
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import load_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import collections
import gc
from rm_noun_verb_preprocess import rakutenWakati
rw = rakutenWakati()

import os
import tensorflow as tf
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
random.seed(7)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=10,
    inter_op_parallelism_threads=10
)
tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
k.set_session(sess)

exist_dic = pd.read_pickle('./data/except_en/word_dic4test1.pkl')

def oversampling(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    clear = data[data['cui'] == 0]
    unclear = data[data['cui'] == 1]
    impossible = data[data['cui'] == 2]
    length = [len(clear), len(unclear), len(impossible)]
    overed_length = max(length)

    for i, cnt in enumerate(length):
        mag = overed_length // cnt
        if mag >= 2:
            for n in range(mag-1):
                if i == 0:
                    if n == 0:
                        copyed = clear.copy()
                        clear = pd.concat([clear, copyed], axis=0)
                    else:
                        clear = pd.concat([clear, copyed], axis=0)
                elif i == 1:
                    if n == 0:
                        copyed = unclear.copy()
                        unclear = pd.concat([unclear, copyed], axis=0)
                    else:
                        unclear = pd.concat([unclear, copyed], axis=0)
                elif i == 2:
                    if n == 0:
                        copyed = impossible.copy()
                        impossible = pd.concat([impossible, copyed], axis=0)
                    else:
                        impossible = pd.concat([impossible, copyed], axis=0)
                else:
                    print('ERROR')
            del copyed
            gc.collect()
        else:
            pass

    ret_value = clear.copy()
    ret_value = pd.concat([ret_value, unclear], axis=0)
    ret_value = pd.concat([ret_value, impossible], axis=0)
    ret_value = ret_value.sample(frac=1, random_state=42).reset_index(drop=True)

    return ret_value

def embedding(bow):
    feature_vec_list = []
    for one_bow in bow:
        feature_dic = exist_dic.copy()
        for word in one_bow.split(' '):
            if word != '':
                if word in feature_dic:
                    feature_dic[word] = 1
                else:
                    feature_dic['unknown'] = 1

        feature_vec = list(feature_dic.values())
        feature_vec_list.append(feature_vec)

    return feature_vec_list

def get_batch(batch_size):
    """
    batchを取得する関数
    ref: https://aotamasaki.hatenablog.com/entry/2018/08/27/124349
    """
    global x_train, y_train_cate
    SIZE = len(x_train)
    # n_batchs
    n_batchs = SIZE//batch_size
    # for でyield
    i = 0
    while (i < n_batchs):
        print("doing", i, "/", n_batchs)
        y_batch = y_train_cate[(i * batch_size):(i * batch_size + batch_size)]
        
        #あるbatchのfilenameの配列を持っておく
        x_batch = x_train[(i * batch_size):(i * batch_size + batch_size)]
        x_batch = embedding(x_batch)

        i += 1
        yield x_batch, y_batch

def build_model():
    model = Sequential()
    model.add(Dense(750, input_shape=(4072,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def main():
    # normal: pat1, v1: v1(pat2), v2: v2(pat3)
    rmnoun = pd.read_csv('./predicted_rmnoun1_1009_notuseips.csv')
    rmnoun = rmnoun[['Input', 'Category', 'cui']]
    rmverb = pd.read_csv('./predicted_rmverb1_1009_notuseips.csv')
    rmverb = rmverb[['Input', 'Category', 'cui']]
    imp = pd.read_csv('data/except_en/select_qa_over1_ips1009labeled_thisuse_relabeled.csv')
    imp482 = imp[imp['Category'] == 482]
    imp482 = imp482[['Input', 'Category']]
    bow = rw.wakati(list(imp482['Input']))
    imp482['Input'] = bow
    imp_cui = [2 for _ in range(len(imp482))]
    imp482['cui'] = imp_cui
    imp1009 = imp[imp['Category'] == 1009]
    imp_cui = [2 for _ in range(len(imp1009))]
    imp1009 = imp1009[['Input', 'Category']]
    bow = rw.wakati(list(imp1009['Input']))
    imp1009['Input'] = bow
    imp1009['cui'] = imp_cui
    del imp
    gc.collect()

    df = pd.concat([rmnoun, rmverb])
    df = pd.concat([df, imp482])
    df = pd.concat([df, imp1009])

    odf = oversampling(df)
    #with open('pat1_os.txt', 'a') as f:
    #    f.write('Data num: ' + str(len(odf)) + '\n')

    del rmnoun, rmverb, df
    gc.collect()

    global x_train, y_train_cate
    #x_train, x_test, y_train, y_test = train_test_split(odf['Input'].values, odf['cui'].values, test_size=0.1, random_state=42)
    #y_train, y_test = to_categorical(y_train, 3), to_categorical(y_test, 3)
    
    x = odf['Input'].values
    y = odf['cui'].values
    N_EPOCHS = 3
    x_train = x
    y_train = y
    y_train_cate = to_categorical(y_train, 3)
    
    model = build_model()
    for epoch in range(N_EPOCHS):
        print("=" * 50)
        print('Epoch' + str(epoch+1), '/', N_EPOCHS)
        acc = []
        
        # batch_size=1000でHDDからバッチを取得する
        for x_batch, y_batch in get_batch(1000):
            x_batch = np.array(x_batch)
            model.train_on_batch(x_batch, y_batch)
            score = model.evaluate(x_batch, y_batch)
            print('batch accuracy:', score[1])
            acc.append(score[1])
        print('Train accuracy', np.mean(acc))

    model.save('./models/model_os_pat1_rm1_all_1009_notuseips.h5', include_optimizer=False)

if __name__ == '__main__':
    main()
