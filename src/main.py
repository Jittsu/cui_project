# -*- coding: utf-8 -*-

"""
"""

# 標準ライブラリ
import os
import sys
import json
import shutil
from datetime import datetime
import gc
import collections

# 追加ライブラリ
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Input, Embedding, Dense, GaussianNoise, Multiply, Add, Subtract, Lambda
from tensorflow.keras.models import Model
from keras.utils import np_utils
from tensorflow.keras.backend import clear_session, zeros, std, repeat_elements
import tensorflow as tf

# 小分けスクリプト
from preprocess import TokenVectorizer
from models import MLP, CUIMLP

# コンフィグなど
from config import FILE_PATH, STOPWORD, USE_POS, STEMMING, AS_NUMPY, MODEL_PARAMS
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
from version import __version__

# 関数 ---
def get_pred_rank(predicted, num=6):
    use_predicted = predicted.copy()
    ret_values = []
    for _ in range(num):
        max_idx = use_predicted.argmax(axis=-1)
        for i, pred in enumerate(max_idx):
            use_predicted[i][pred] = 0
        ret_values.append(max_idx)
    return np.array(ret_values)

# データ読み込み ---

if 'har' in FILE_PATH:
    df = pd.read_csv(FILE_PATH, sep=';')
    data = df[['scenario', 'intent', 'answer']]
    data = data.dropna(how='any')
    x = np.array(list(data['answer']))
    y = np.array(list(data['intent']))
    del df
    gc.collect()
elif 'oos' in FILE_PATH:
    print()
elif 'snips' in FILE_PATH:
    print()
else:
    sys.exit('ERROR: This file is not supported.')

# ラベル分類モデル作成部 ---

skf = StratifiedKFold(n_splits=MODEL_PARAMS['split_num'], shuffle=True)
split_cnt = 1
y2id = {}
id2y = {}
i = 0
for intent in y:
    if not intent in y2id:
        y2id[intent] = i
        id2y[i] = intent
        i += 1
y_id = np.array([y2id[intent] for intent in y])
y_onehot = np_utils.to_categorical(y_id, len(collections.Counter(y)))
scores = []
cui_scores = []
for train_idx, test_idx in skf.split(x, y):
    tv = TokenVectorizer(list(x[train_idx]), use_pos=[], stopword=True, stemming=True, as_numpy=True)
    train_vec = tv.vectorizer()
    in_seq_size = len(train_vec[0])
    model = MLP(in_seq=in_seq_size, out_vec_size=len(collections.Counter(y)), dropout_rate=0.2, loss=MODEL_PARAMS['loss'], optimizer=MODEL_PARAMS['optimizer'])
    mlp = model.build_model()
    mlp.fit(train_vec, y_onehot[train_idx], batch_size=MODEL_PARAMS['batch_size'], epochs=MODEL_PARAMS['epochs'], verbose=1)
    test_vec = tv.vectorizer(x[test_idx])
    score = mlp.evaluate(test_vec, y_onehot[test_idx])
    scores.append(score[1])
    print(f'SCORE: {score[1]}')
    mlp.save(f'../models/mlp_split{split_cnt}.h5', include_optimizer=False)

    # CUI訓練用データ作成部 ---
    rm_nv_tokens, _ = tv.rm_nv_tokenizer(y_id[train_idx])
    rm_nv_vectors, rm_nv_labels = tv.rm_nv_vectorizer(y_id[train_idx])
    multi_label_dict = tv.create_multi_label(rm_nv_tokens, rm_nv_labels)
    rm_nv_multi_labels = []
    rm_nv_tokens = tv.list2str(rm_nv_tokens)
    for token in rm_nv_tokens:
        multi_label = multi_label_dict[token]
        rm_nv_multi_labels.append(multi_label)

    pred_proba = mlp.predict(rm_nv_vectors)
    pred_ranks = get_pred_rank(pred_proba, num=MODEL_PARAMS["topn"])
    pred_ranks = np.array(pred_ranks)
    predicted_label_ranks = pred_ranks.T

    cui_labels = []
    i = 0
    for p_label, m_label in zip(predicted_label_ranks, rm_nv_multi_labels):
        m_label_l = m_label.split(';')
        m_label_l = [int(ml) for ml in m_label_l]
        p_label = np.array([int(pl) for pl in p_label])
        print(p_label)
        print(m_label_l)
        if p_label[0] in m_label_l:
            cui_labels.append(0)
        elif len(set(p_label) & set(m_label_l)) > 0:
            cui_labels.append(1)
        else:
            cui_labels.append(2)
        print(cui_labels)
        i += 1
        if i == 5:
            break
    
    sys.exit(1)
    
    # CUI分類モデル作成部 ---
    cui_onehot = np_utils.to_categorical(cui_labels, 3)
    model = CUIMLP(in_seq=in_seq_size, dropout_rate=0.2, loss=MODEL_PARAMS['loss'], optimizer=MODEL_PARAMS['optimizer'])
    cuimlp = model.build_model()
    cuimlp.fit(rm_nv_vectors, cui_onehot, batch_size=MODEL_PARAMS['batch_size'], epochs=MODEL_PARAMS['epochs'], verbose=1)
    pred_cui = cuimlp.predict_classes(test_vec)
    pred_class = mlp.predict_classes(test_vec)
    test_label = y[test_idx]
    collect_num = 0
    for p_cui, p_class, t_label in zip(pred_cui, pred_class, test_label):
        if not p_cui == 2:
            if str(p_class) == str(t_label):
                collect_num += 1
            else:
                pass
        else:
            if not str(p_class) == str(t_label):
                collect_num += 1
            else:
                pass
    cuiscore = collect_num/len(test_label)
    cui_scores.append(cuiscore)
    print(f'epoch:{split_cnt}')
    print(f'normal score: {score[1]}')
    print(f'cui score: {cuiscore}')
    cuimlp.save(f'../models/cuimlp_split{split_cnt}.h5', include_optimizer=False)
    split_cnt += 1
    df = pd.DataFrame()
    df['cui'] = pred_cui
    df['pred'] = pred_class
    df['label'] = test_label
    df.to_csv('out.csv', index=False)
    break

scores = np.array(scores)
cui_scores = np.array(cui_scores)
print('final score')
print(f'normal score: {scores.mean()}')
print(f'cui score: {cui_scores.mean()}')