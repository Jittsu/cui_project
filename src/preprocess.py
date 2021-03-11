# -*- coding: utf-8 -*-

"""
"""

import numpy as np
import re
import collections
import gc
import nltk
import sys
import math

class TokenVectorizer:
    """
    NLTKを用いた形態素解析\n
    Input:\n
        \tdata: [str, str, str, ...]
        \tuse_pos: [str, str, str, ...]
        \tas_numpy: bool
    Output:\n
        \ttokenizer: [[str, str, ...], [str, str, ...], ...]
    """
    def __init__(self, data: list, use_pos=[], stopword=True, stemming=True, as_numpy=True):
        self._data = data
        self._use_pos = use_pos
        self._cleaned_text = self._clean()
        self._stop_word = stopword
        self._stemming = stemming
        self._as_numpy = as_numpy
        self.word_dict = self._create_word_dict()

    def _clean(self, data=[]):
        if len(data) == 0:
            d = self._data
        else:
            d = data
        ret_value = []
        for text in d:
            text = re.sub(r',', '', text)
            text = re.sub(r'\.', '', text)
            text = re.sub(r'\(.*?\)', '', text)
            ret_value.append(text)
        return ret_value

    def _flatten_list(self, l):
        for el in l:
            if isinstance(el, list):
                yield from self._flatten_list(el)
            else:
                yield el

    def tokenizer(self, data=[]) -> list:
        if len(data) == 0:
            sentences = self._cleaned_text
        else:
            sentences = self._clean(data)
        stemmer = nltk.stem.PorterStemmer()
        ret_value = []
        for sentence in sentences:
            tokenized = nltk.tokenize.word_tokenize(sentence)
            if self._stop_word:
                if len(self._use_pos) == 0:
                    stop_words = nltk.corpus.stopwords.words('english')
                    if self._stemming:
                        ret_sentence = [stemmer.stem(word) for word in tokenized if word not in stop_words]
                    else:
                        ret_sentence = [word for word in tokenized if word not in stop_words]
                else:
                    pos = nltk.pos_tag(tokenized)
                    if self._stemming:
                        ret_sentence = [stemmer.stem(word[0]) for word in pos if word[1] in self._use_pos]
                    else:
                        ret_sentence = [word[0] for word in pos if word[1] in self._use_pos]
                ret_value.append(ret_sentence)
            else:
                if self._stemming:
                    ret_sentence = [stemmer.stem(word) for word in tokenized]
                else:
                    ret_sentence = tokenized
                ret_value.append(ret_sentence)
        return ret_value

    def _create_word_dict(self):
        tokenized = self.tokenizer()
        flatten = list(self._flatten_list(tokenized))
        word_dict = collections.Counter(flatten)
        for k, v in word_dict.items():
            word_dict[k] = 0
        return word_dict

    def vectorizer(self, data=[]):
        if len(data) == 0:
            tokenized = self.tokenizer()
        else:
            tokenized = self.tokenizer(data)
        vectors = []
        for sentence in tokenized:
            vector_dict = self.word_dict.copy()
            for word in sentence:
                if word in vector_dict:
                    vector_dict[word] = 1
                else:
                    pass
            vectors.append(list(vector_dict.values()))
        if self._as_numpy:
            vectors = np.array(vectors)
        return vectors

    def vectorizer_from_dict(self, added_word_dict: dict, data=[]):
        if len(data) == 0:
            tokenized = self.tokenizer()
        else:
            tokenized = self.tokenizer(data)
        vectors = []
        for sentence in tokenized:
            vector_dict = added_word_dict.copy()
            for word in sentence:
                if word in vector_dict:
                    vector_dict[word] = 1
                else:
                    pass
            vectors.append(list(vector_dict.values()))
        if self._as_numpy:
            vectors = np.array(vectors)
        return vectors

    def rm_nv_tokenizer(self, labels: list, data=[]) -> list:
        if len(data) == 0:
            sentences = self._cleaned_text
        else:
            sentences = self._clean(data)
        stemmer = nltk.stem.PorterStemmer()
        ret_value = []
        ret_labels = []
        for sentence, label in zip(sentences, labels):
            tokenized = nltk.tokenize.word_tokenize(sentence)
            if self._stop_word:
                if len(self._use_pos) == 0:
                    pos = nltk.pos_tag(tokenized)
                    rm_idx = []
                    for i, word in enumerate(pos):
                        for nv in ['NN', 'VB']:
                            if nv in word[1]:
                                rm_idx.append(i)
                                break
                    stop_words = nltk.corpus.stopwords.words('english')
                    for idx in rm_idx:
                        removed = tokenized.copy()
                        _ = removed.pop(idx)
                        if self._stemming:
                            ret_sentence = [stemmer.stem(word) for word in removed if word not in stop_words]
                        else:
                            ret_sentence = [word for word in removed if word not in stop_words]
                        ret_value.append(ret_sentence)
                        ret_labels.append(label)
                else:
                    pos = nltk.pos_tag(tokenized)
                    rm_idx = []
                    for i, word in enumerate(pos):
                        for nv in ['NN', 'VB']:
                            if nv in word[1]:
                                rm_idx.append(i)
                                break
                    for idx in rm_idx:
                        removed = pos.copy()
                        _ = removed.pop(idx)
                        if self._stemming:
                            ret_sentence = [stemmer.stem(word[0]) for word in removed if word[1] in self._use_pos]
                        else:
                            ret_sentence = [word[0] for word in removed if word[1] in self._use_pos]
                        ret_value.append(ret_sentence)
                        ret_labels.append(label)
            else:
                pos = nltk.pos_tag(tokenized)
                rm_idx = []
                for i, word in enumerate(pos):
                    for nv in ['NN', 'VB']:
                        if nv in word[1]:
                            rm_idx.append(i)
                            break
                for idx in rm_idx:
                    removed = tokenized.copy()
                    _ = removed.pop(idx)
                    if self._stemming:
                        ret_sentence = [stemmer.stem(word) for word in removed]
                    else:
                        ret_sentence = removed
                    ret_value.append(ret_sentence)
                    ret_labels.append(label)
        return ret_value, ret_labels

    def rm_nv_vectorizer(self, labels, data=[]):
        if len(data) == 0:
            tokenized, label = self.rm_nv_tokenizer(labels)
        else:
            tokenized, label = self.rm_nv_tokenizer(labels, data)
        vectors = []
        for sentence in tokenized:
            vector_dict = self.word_dict.copy()
            for word in sentence:
                if word in vector_dict:
                    vector_dict[word] = 1
                else:
                    pass
            vectors.append(list(vector_dict.values()))
        if self._as_numpy:
            vectors = np.array(vectors)
        return vectors, label

    def list2str(self, data):
        ret_value = []
        for bow in data:
            str_bow = ''
            for word in bow:
                if str_bow == '':
                    str_bow = word
                else:
                    str_bow = str_bow + ';' + word
            ret_value.append(str_bow)
        return ret_value

    def create_multi_label(self, data, labels) -> dict:
        forMultiDict = {}
        data = self.list2str(data)
        for bow, label in zip(data, labels):
            if not bow in forMultiDict:
                forMultiDict[bow] = str(label)
            else:
                lstr = forMultiDict[bow]
                multi_label_list = lstr.split(';')
                if not label in multi_label_list:
                    multi_label = lstr + ';' + str(label)
                    forMultiDict[bow] = multi_label
                else:
                    pass
        return forMultiDict

def test():
    test_data1 = ['I have a pen.', 'Process is a natural pen.', 'There are pens called "PPAP".']
    test_data2 = ['I like this pen.', 'I am a pen.']
    test_label1 = [1, 1, 2]
    test_label2 = [1, 2]
    print('stop word: True(auto), stemming: True, as_numpy: True')
    tv = TokenVectorizer(test_data1, use_pos=[], stopword=True, stemming=True, as_numpy=True)
    print('words list')
    print(tv.word_dict)
    print('tokenized words')
    tokenized = tv.tokenizer()
    print(tokenized)
    print('tokenized words from added data')
    tokenized_added_data = tv.tokenizer(test_data2)
    print(tokenized_added_data)
    print('vectors')
    vectorized = tv.vectorizer()
    print(vectorized)
    print('vectors from added data')
    vectorized_added_data = tv.vectorizer(test_data2)
    print(vectorized_added_data)
    print('added word dict vector')
    added_word_dict = {'I': 0, 'am': 0, 'pen': 0, 'red': 0}
    vectorized_added_word_dict = tv.vectorizer_from_dict(added_word_dict)
    print(vectorized_added_word_dict)

    print('stop word: True, stemming: False, as_numpy: False')
    pos = ['DT', 'JJ', 'NN', 'NNS', 'NNP', 'NNPS']#, 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    tv = TokenVectorizer(test_data1, use_pos=pos, stopword=True, stemming=False, as_numpy=False)
    print('words list')
    print(tv.word_dict)
    print('tokenized words')
    tokenized = tv.tokenizer()
    print(tokenized)
    print('tokenized words from added data')
    tokenized_added_data = tv.tokenizer(test_data2)
    print(tokenized_added_data)
    print('vectors')
    vectorized = tv.vectorizer()
    print(vectorized)
    print('vectors from added data')
    vectorized_added_data = tv.vectorizer(test_data2)
    print(vectorized_added_data)
    print('added word dict vector')
    added_word_dict = {'I': 0, 'am': 0, 'pen': 0, 'red': 0}
    vectorized_added_word_dict = tv.vectorizer_from_dict(added_word_dict)
    print(vectorized_added_word_dict)

    print('remove noun verb tokenizer')
    tv = TokenVectorizer(test_data1, use_pos=pos, stopword=False, stemming=False, as_numpy=True)
    print('tokenized words')
    tokenized, label = tv.rm_nv_tokenizer(test_label1)
    print(tokenized)
    print(label)
    print('tokenized words from added data')
    tokenized_added_data, label = tv.rm_nv_tokenizer(test_label2, test_data2)
    print(tokenized_added_data)
    print(label)
    print('vectors')
    vectorized, label = tv.rm_nv_vectorizer(test_label1)
    print(vectorized)
    print(label)
    print('vectors from added data')
    vectorized_added_data, label = tv.rm_nv_vectorizer(test_label2, test_data2)
    print(vectorized_added_data)
    print(label)

if __name__ == '__main__':
    test()