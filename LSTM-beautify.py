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
from keras.layers import *
from sklearn.model_selection import StratifiedKFold

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

# 对齐向量
def padVector(series, maxlen):
    series = list(sequence.pad_sequences(series, maxlen=maxlen))
    return series

# 从文件中获取部分，pos 8000条， neg 5000条
def getPartSetFromFile(path, ratio, columns):
    data = getDataset(path)
    len = data.shape[0]
    data = np.array(data)
    random.shuffle(data)
    data = data[:int(ratio * len)]
    data = pd.DataFrame(data, columns=columns)
    return data

# 从dataframe中获取部分
def getPartSetFromDF(data, ratio, columns):
    len = data.shape[0]
    data = np.array(data)
    random.shuffle(data)
    data = data[:int(ratio * len)]
    data = pd.DataFrame(data, columns=columns)
    return data

# 将一个数据集拆分成两个
def split2differentSamples(data, ratio, columns):
    len = data.shape[0]  # 数据集长度
    data = np.array(data)
    data_1 = data[:int(ratio*len)]
    data_2 = data[int(ratio*len):]
    data_1 = pd.DataFrame(data_1, columns=columns)
    data_2 = pd.DataFrame(data_2, columns=columns)
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

# 获取mis数据集
def getMisData():
    mis_data = getDataset('./dataset/misjudgements.csv')
    return mis_data

# 获取pos数据集
def getPosData():
    pos_data = getDataset('./dataset/pos_data.csv')
    return pos_data

# 获取neg数据集
def getNegData():
    neg_data = getDataset('./dataset/neg_data.csv')
    return neg_data

# 获取normal数据集
def getNormalData():
    normal_data = getDataset('./dataset/shuffled_dataset.csv')
    return normal_data

# 连接全部分词
def getAllWords(train_df, test_df, word_row_name):
    all_words = pd.DataFrame(columns=[word_row_name])
    all_words = pd.concat([train_df[word_row_name], test_df[word_row_name]],
                          ignore_index=True)
    return all_words

# 获取连接df
def getConcatedDf(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

# 保存词典
def save_dict(dict):
    with open('./dictionary/dict.file', 'wb') as f:
        pickle.dump(d, f)
    return

# 读取词典
def read_dict():
    with open('./dictionary/dict.file', 'rb') as f:
        d = pickle.load(f)
    return d

# 获取指定构成的训练集与测试集
def getTrainTest(columns, pos_num, neg_num, train_pos_num,
                 train_neg_num, train_mis_num):
    mis_data = getMisData()
    pos_data = getPosData()
    neg_data = getNegData()
    # 获取比例
    pos_ratio = pos_num / 8553
    neg_ratio = neg_num / 4904
    train_pos_ratio = train_pos_num / pos_num
    train_neg_ratio = train_neg_num / neg_num
    train_mis_ratio = train_mis_num / 773
    # 获取一定数量的pos neg （train + test)
    pos_data = getPartSetFromDF(pos_data, pos_ratio, columns)
    neg_data = getPartSetFromDF(neg_data, neg_ratio, columns)
    # 将pos, neg, mis 分别划分为 train, test
    train_pos, test_pos = split2differentSamples(pos_data, train_pos_ratio, columns)
    train_neg, test_neg = split2differentSamples(neg_data, train_neg_ratio, columns)
    train_mis, test_mis = split2differentSamples(mis_data, train_mis_ratio, columns)
    # 链接train, test
    train = pd.DataFrame(columns=columns)
    test = pd.DataFrame(columns=columns)
    train = pd.concat([train_pos, train_neg, train_mis], ignore_index=True)
    test = pd.concat([test_pos, test_neg, test_mis], ignore_index=True)
    return train, test

# 分词操作
def splitWords(train_df, test_df, new_row_name, split_row_name):
    train_df[new_row_name] = cutWord(train_df[split_row_name])
    test_df[new_row_name] = cutWord(test_df[split_row_name])
    return train_df, test_df

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

# 建立模型
def setModel(embedding_input_len, LSTM_output_dim,
             dropout_rat, cur_activation, cur_optimizer):
    model = Sequential()
    model.add(Embedding(embedding_input_len, embedding_output_dim))
    model.add(LSTM(LSTM_output_dim))
    model.add(Dropout(1 - dropout_rat))
    model.add(Dense(1, activation=cur_activation))
    model.compile(loss='binary_crossentropy', optimizer=cur_optimizer, metrics=['accuracy'])
    return model

# 设置参数
columns = ['comment', 'sentiment']
new_row_name = 'words'
split_row_name = 'comment'
word_row_name = new_row_name
vector_row_name = 'vector'
x_row_name = 'vector'
y_row_name = 'sentiment'
pos_num = 500
neg_num = 500
train_pos_num = 350
train_neg_num = 350
train_mis_num = 540
vec_maxlen = 50
embedding_output_dim = 128
LSTM_output_dim = 128
dropout_rat = 0.5
cur_activation = 'sigmoid'
cur_optimizer = 'rmsprop'
batch_size_num = 16
epochs_num = 10
circle_num = 0
fold_num = 0
circle_score = 0
epoch_scores = []
fold_scores = []
record = dict()
record_df = pd.DataFrame()
for j in range(0, 13):
    fold_num += 1
    # 10次循环共1个fold, 每次循环为 1 circle = 10 epochs, 得到1个score
    for i in range(0, 10):
        circle_num += 1
        train, test = getTrainTest(columns, pos_num, neg_num, train_pos_num,
                                   train_neg_num, train_mis_num) # 获得训练集, 测试集
        train, test = splitWords(train, test, new_row_name, split_row_name) # 分词
        dict = getDict(getAllWords(train, test, word_row_name)) # 构建词典
        embedding_input_len = len(dict) + 1

        # 向量化并对齐
        train = getPaddedVector(train, vector_row_name, word_row_name, dict, vec_maxlen)
        test = getPaddedVector(test, vector_row_name, word_row_name, dict, vec_maxlen)

        # 划分 x, y
        x, y = getXY(train, x_row_name, y_row_name)
        xt, yt = getXY(test, x_row_name, y_row_name)
        xa, ya = getXY(getConcatedDf(train, test), x_row_name, y_row_name)
        del train, test, dict

        model = setModel(embedding_input_len, LSTM_output_dim, dropout_rat,
                         cur_activation, cur_optimizer) # 设置模型
        model.fit(x, y, batch_size=batch_size_num, epochs=epochs_num) # 训练
        score = model.evaluate(xt, yt) # 测试
        epoch_scores.append(score[1]) # 记录每次epoch分数
        print('Fold ' + str(fold_num) + ", Circle " + str(circle_num) +
              ', epoch score:' + str(score[1]))
    # 一个fold后, 将10次circle的分数取均值
    for line in epoch_scores:
        circle_score += line
    circle_score = circle_score / 10
    print("Circle average score: " + str(circle_score))
    pos_num = int(pos_num * 1.2)
    neg_num = int(neg_num * 1.2)
    train_pos_num = int(train_pos_num * 1.2)
    train_neg_num = int(train_neg_num * 1.2)
    record['pos_num'] = pos_num
    record['train_pos'] = train_pos_num
    record['score'] = circle_score
    record_df = record_df.append(record, ignore_index=True)

    # 将用于单次fold数据记录的变量存储后归零
    fold_scores.append(circle_score)
    epoch_scores = []
    circle_score = 0
    circle_num = 0
    del x, y, xt, yt

for line in fold_scores:
    print(line)

print(record_df)










