from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import jieba
import random
import re
import pickle
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import *

def getDataset(path):
    data = pd.read_csv(path, encoding='utf-8')
    return data

# 读取词典
def readDict():
    with open('./dictionary/dict.file', 'rb') as f:
        d = pickle.load(f)
    return d

# 分词函数
def cutWord(series):
    chopped_words = []
    for line in series:
        chopped_words.append(list(jieba.cut(line)))
    chopped_words = pd.Series(chopped_words)
    return chopped_words

# 分词操作
def splitWords(df, new_row_name, split_row_name):
    df[new_row_name] = cutWord(df[split_row_name])
    return df

# comment向量化
def word2vector(series, dict):
    cmt_vct = pd.Series()
    get_vct = lambda x:list(dict['id'][x])
    cmt_vct = series.apply(get_vct)
    return cmt_vct

# 对齐向量
def padVector(series, maxlen):
    series = list(sequence.pad_sequences(series, maxlen=maxlen))
    return series

# 向量化并对齐
def getPaddedVector(df, vector_row_name, word_row_name, dict, maxlen):
    df[vector_row_name] = word2vector(df[word_row_name], dict)
    df[vector_row_name] = padVector(df[vector_row_name], maxlen)
    return df

# 划分x, y
def getXY(df, x_row_name, y_row_name):
    x = np.array(list(df[x_row_name]))
    y = np.array(list(df[y_row_name]))
    return x, y

model_path = './lstm_model.h5'
test_path = './test.csv'
new_row_name = 'words'
split_row_name = 'comment'
word_row_name = new_row_name
vector_row_name = 'vector'
vec_maxlen = 50
x_row_name = 'vector'
y_row_name = 'sentiment'
result = pd.DataFrame(columns=['comment', 'sentiment', 'predict']) # when testing on data with label
result = pd.DataFrame(columns=['comment', 'predict']) # when testing on data without label

## If test_data saved not as csv file, should preprocess it as follow:
# data = pd.read_excel(file_path, index_col=0)
# data.to_csv(csv_path, encoding='utf-8', header=0)
# data = pd.read_csv(csv_path, encoding='utf-8')
# column = ['comment'] ## Data without label
# column = ['comment', 'sentiment'] ## Data with label
# data.columns = column
# data = data.drop_duplicates(subset=['comment'], keep='first) ## drop repeated data
# test = pd.DataFrame(columns=column) ## save processed data
# for index, row in test.iterrows():
#     row['comment'] = re.sub(u'[<()（）\-^_,，。！？～.!”“?、\n\t…]+[\s+\.\!\/_,$%^*+\"\']+|[+——、~@#￥%……&*；]+|[A-Z]+|[a-z]+|[0-9]', '', row['comment'])
#     test = test.append(row, ignore_index=True)
# del data
# test = test.sample(frac=1.0) ##shuffle the sequence of test_data
# test.to_csv(test_path, encoding='utf-8', index=False)

np.set_printoptions(suppress=True) # 将默认输出从科学计数转为数字
test = getDataset(test_path)
test = splitWords(test, new_row_name, split_row_name)
dict = readDict()
test = getPaddedVector(test, vector_row_name, word_row_name, dict, vec_maxlen)
x, y = getXY(test, x_row_name, y_row_name) # data with label
# x = np.array(list(test[x_row_name])) # data without label
model = load_model(model_path)

# save prediction result
result['comment'] = test['comment']
result['sentiment'] = test['sentiment'] # only when testing on data with label
result['predict'] = model.predict_classes(x)

result.to_csv('predict_result.csv', encoding='utf-8', index=False)




