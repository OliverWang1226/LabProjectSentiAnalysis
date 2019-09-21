from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import jieba
import random
import re

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import StratifiedKFold
import sklearn.linear_model as LinReg
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn import svm
# 读取文件
def getDataset(path):
    data = pd.read_csv(path, encoding='utf-8')
    return data

# 分词函数
def cutWord(series):
    chopped_words = []
    for line in series:
        chopped_words.append(list(jieba.cut(line)))
    chopped_words = pd.Series(chopped_words)
    return chopped_words

# 创建字典
def getDict(data):
    all_words = []
    for i in data:
        all_words.extend(i)
    dict = pd.DataFrame(pd.Series(all_words).value_counts())
    dict['id'] = list(range(1, len(dict) + 1))
    return dict

# comment向量化
def word2vector(series, dict):
    cmt_vct = pd.Series()
    get_vct = lambda x:list(dict['id'][x])
    cmt_vct = series.apply(get_vct)
    return cmt_vct

# 对齐
def padVector(series, maxlen):
    series = list(sequence.pad_sequences(series, maxlen=maxlen))
    return series

# 从数据集中获取部分，pos 8000条， neg 5000条
def getPartSet(path, ratio, columns):
    data = getDataset(path)
    len = data.shape[0]
    data = np.array(data)
    random.shuffle(data)
    data = data[:int(ratio * len)]
    data = pd.DataFrame(data, columns=columns)
    return data

# 将一个数据集拆分成两个
def split2differentSamples(data, ratio):
    len = data.shape[0]  # 数据集长度
    data = np.array(data)
    data_1 = data[:int(ratio*len)]
    data_2 = data[int(ratio*len):]
    return data_1, data_2

# 将numpy对象转换为dataframe
def np2dataframe(np_data, columns):
    data = pd.DataFrame(np_data, columns=columns)
    return data

# 将dataframe保存为CSV文件
def saveDataframeAsCSV(dataframe, path):
    dataframe.to_csv(path, encoding='utf-8', index=False)
    return

# 去除停用词
def dropPunctuation(data, save_path):
    all_cmmt = pd.DataFrame(columns=['comment', 'sentiment'])
    for index, row in data.iterrows():
        row['comment'] = re.sub(u'[<()（）:\-^_,，。！？：～.!”“?、\n\t…]', '', row['comment'])
        print(row['comment'])
        all_cmmt = all_cmmt.append(row, ignore_index=True)
    all_cmmt = all_cmmt.sample(frac=1.0)  # 打乱顺序
    all_cmmt.to_csv(save_path, encoding='utf-8', index=False)

# 获取训练集，测试集
# columns = ['comment', 'sentiment']
# train_mis, test = split2differentSamples('./dataset/misjudgements.csv', 0.5)
# train_mis = np2dataframe(train_mis, columns)
# test = np2dataframe(test, columns)
# train_pos = getPartSet('pos_data.csv', 0.05, columns)
# train_neg = getPartSet('neg_data.csv', 0.16, columns)

# 连接训练集
# train = pd.DataFrame(columns=['comment', 'sentiment'])
# train = pd.concat([train_pos, train_neg, train_mis], ignore_index=True)
# del train_pos, train_neg, train_mis

# temp: 9/17——测试集:(0.7*shuffled_dataset + train_mis); 训练集:(0.3 * shuffled_dataset + test_mis)
normal_comment = getDataset('./dataset/shuffled_dataset.csv')
misjudgements = getDataset('./dataset/misjudgements.csv')
columns = ['comment', 'sentiment']

train_normal, test_normal = split2differentSamples(normal_comment, 0.7)
train_mis, test_mis = split2differentSamples(misjudgements, 0.5)
del normal_comment, misjudgements

train_normal = np2dataframe(train_normal, columns)
test_normal = np2dataframe(test_normal, columns)
train_mis = np2dataframe(train_mis, columns)
test_mis = np2dataframe(test_mis, columns)

train = pd.DataFrame(columns=['comment', 'sentiment'])
test = pd.DataFrame(columns=['comment', 'sentiment'])
train = pd.concat([train_normal, train_mis], ignore_index=True)
test = pd.concat([test_normal, test_mis], ignore_index=True)


# temp: Normal Dataset Without Misjudgements
# data = getDataset('shuffled_dataset.csv')
# train = []
# test = []
# train, test = split2differentSamples(data, 0.7)
# train = np2dataframe(train, columns)
# test = np2dataframe(train, columns)

# 分词
train['words'] = cutWord(train['comment'])
test['words'] = cutWord(test['comment'])
all_data = pd.DataFrame(columns=['words'])
all_data = pd.concat([train['words'], test['words']], ignore_index=True)

# 获取词典
dict = []
dict = getDict(all_data)

# 向量化
train['vector'] = word2vector(train['words'], dict)
test['vector'] = word2vector(test['words'], dict)

# 对齐
train['vector'] = padVector(train['vector'], 50)
test['vector'] = padVector(test['vector'], 50)

# 划分x, y, xt, yt
x = np.array(list(train['vector']))
y = np.array(list(train['sentiment']))
xt = np.array(list(test['vector']))
yt = np.array(list(test['sentiment']))

# 全集
xa = np.array(list(pd.concat([train, test], ignore_index=True)['vector']))
ya = np.array(list(pd.concat([train, test], ignore_index=True)['sentiment']))
del train, test

# 随机种子
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

# 建立模型
# Keras LR
# print(">>>>>>>>>>Building model")
# model = Sequential()
# model.add(Dense(1, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(x, y)
# score = model.evaluate(xt, yt)
# print("The score is: " + str(score))

# sklearn LR
# lgr = LinReg.LogisticRegression().fit(x, y)
# r2_score_ = lgr.score(xt, yt)
# print(r2_score_)

# sklearn linear SVM
linear_svc = svm.LinearSVC()
linear_svc.fit(x, y)
print(linear_svc.score(xt, yt))

# sklearn SVM
svc = svm.SVC(kernel='poly')
svc.fit(x, y)
print(svc.score(xt, yt))

# K-fold model
# counter = 0
# for train, test in kfold.split(xa, ya):
#     model = Sequential()
#     model.add(Embedding(len(dict) + 1, 128))
#     model.add(LSTM(128))
#     model.add(Dropout(1 - 0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(xa[train], ya[train], batch_size=32, epochs=10)
#     score = model.evaluate(xa[test], ya[test])
#     counter += 1
#     print(score)
#     cvscores.append(score[1])
# for line in cvscores:
#     print(line)


















