# -*- coding: utf-8 -*-

# データ設定 ---
FILE_PATH = '../data/' + 'har-aug/NLU-Data-Home-Domain-Annotated-All.csv' # ファイルパス ---
STOPWORD = True # nltk.corpus.stopwords.words('english')の利用可否 ---
USE_POS = [] # stopwordをnltk.corpus.stopwords.words('english')から独自指定の品詞に変更する際（利用する品詞を指定） ---
STEMMING = True # nltk.stem.PorterStemmer()での語幹変換の利用可否 ---
AS_NUMPY = True # ベクトルをnp配列にするか否か ---

# model設定 ---
MODEL_PARAMS = {}
MODEL_PARAMS["split_num"] = 10 # N分割交差検証（Noneで分割なし） --- 
MODEL_PARAMS["loss"] = "categorical_crossentropy" # 損失関数 ---
MODEL_PARAMS["optimizer"] = "adam" # 勾配計算 ---
MODEL_PARAMS["batch_size"] = 16 # バッチサイズ ---
MODEL_PARAMS["epochs"] = 3 # エポック数 ---
MODEL_PARAMS["topn"] = 3 # CUIラベル作成時に利用する回答数 ---
