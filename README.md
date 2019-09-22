# UNIQLO评论情感分析




## 项目介绍

​	UNIQLO使用词典规则方式，对用户评论信息做情感分析(二分类)

​	该方式在普通评论上效果很好，但在一些特殊评论上效果较差。

​	尝试使用机器学习方法，对用户评论做情感分析，以达到更好的效果。



## 原始数据集

### 普通数据

​	共 **17113** 条已标注的用户评论，正向评论标注为 **1**，负向为 **0**

​	文件名：xlsx

​	数据目录：[./dataset/uniqlo_real_comments.xlsx](./dataset/uniqlo_real_comments.xlsx)

​	数据概览：		![1569124818349](.\dataset\image\normal_data.png)



### 特殊数据

​	共 **806** 条已标注的用户评论，评论真实情感为正向，规则判断为负向

​	特殊数据情感标注均为 **0** 

​	文件格式：xls

​	数据目录：[./dataset/misjudged_comments.xls](.\dataset\misjudged_comments.xls)

​	数据概览：

![1569124749182](.\dataset\image\mis_data.png)

​	



## 技术选型	

### 模型选择

原本利用规则进行二分类的效果不够理想，故尝试使用 <u>深度学习</u> 的方法。

且深度学习模型框架较为成熟，效果大多数情况下比较令人满意

处理的数据均为中文文本评论，选择采用 <u>自然语言处理</u> 的方法。

**RNN, LSTM **及其变体在自然语言处理中的应用较为广泛，其效果均比较明显。

**RNN**在处理 long term memory时存在缺陷（只考虑最近的状态）	

**LSTM**增加了对过去状态的过滤，可以选择那些状态对当前影响更大，而非如**RNN**只选择最近的状态

故采用 **LSTM** 模型对UNIQLO评论进行情感分析

使用<u>线性回归</u>、<u>逻辑</u><u>回归</u>、<u>SVM</u>、<u>GRU</u>方法作为对比



### 框架选择

目前成熟的机器学习框架中，可运行于 **Tensorflow** 上的 **Keras** 句法明晰，文档完善，代码简洁





## 开发过程

### 开发环境

系统：Win10  64位

语言：Anaconda Python3.7

框架：[tensorflow-gpu 1.14.0](https://tensorflow.google.cn/install/pip)；[keras 2.2.5](https://keras.io/)

CUDA：[cuda10.1](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)

CPU：i7-4790K

显卡:  GTX 1060

内存：32G



### 数据预处理

#### 普通数据

去除重复数据，去除标点符号，打乱顺序后保存，并将其再次按照情感分类后分别保存

将第一次处理过的数据按照情感分类，保存为**CSV**文件

正向：[./dataset/pos_data.csv](./dataset/pos_data.csv)

负向：[./dataset/neg_data.csv](./dataset/neg_data.csv)



#### 特殊数据

去除重复数据，去除标点符号，打乱顺序，将情感标签重新编辑为 **1** 并保存

特殊：[./dataset/processed_mis.csv](./dataset/processed_mis.csv)

**处理结果如下表**![reprocessed_result](G:\PycharmProjects\LabProjectUniqlo\dataset\image\reprocessed_result.jpg)

其中评论列名为 **‘comment’**， 情感标签列名为 **’sentiment‘**



### LSTM情感分析步骤

1. 从**csv**文件中读取数据并保存为 **pandas.Dataframe**

2. 使用结巴分词的默认模式（精准）作为分词函数，对 **Dataframe** 的 **’comment'** 列进行分词

3. 分词后结果作为 **‘words’** 列添加到原 **Dataframe** 中

4. 将所有的分词组合成词典，按照出现频率排序编号

5. 按照词典，对**Dataframe**进行向量化，并对齐

6. 向量化结果作为 **‘vector’** 列添加到原 **Dataframe** 中

6. 划分训练集，测试集

7. 训练集 以 **‘vector’** 作为输入**x， ‘sentiment’** 作为 **y**

8. 测试集 以 **‘vector’** 作为输入**xt，‘sentiment’** 作为**yt**

9. 将 训练集馈送到 **LSTM** 模型中，并在测试集上测试

10. 输出精度

    

## 实验结果及分析

初始参数设置：

![param_set_1](G:\PycharmProjects\LabProjectUniqlo\dataset\image\param_set_1.jpg)



### 尝试  1

在普通数据集上测试，按照 7 : 3 比例划分 **train，test**

保证 **train，test** 中正负向评论比例相同

参数设置保持初始值不变

**数据构成：**

![test_1_data](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_1_data.jpg)



**测试精度：**

**0.9415552253295668**



### 尝试  2

将全部<u>普通数据</u>集作为train，所有<u>特殊数据</u>作为test，测试精度 66%

参数设置保持初始值不变

**数据构成：**

![test_2_data](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_2_data.jpg)

**测试精度：**

**0.644243208**

**分析：**

效果很不理想，因为测试集与训练集内容、比例差别都较大







### 尝试  3

将特殊数据拆分成两部分，一部分放在train中，一部分放在**test**中

改变选取的普通数据的数量，使得 **train ：test = 4 : 1**

利用控制变量法，更改参数

**数据构成：**

![test_3_data](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_3_data.jpg)

**测试结果：**

![test-3](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test-3.jpg)

**平均精度：**

**0.9700258398**

**分析：**

结果比较理想，且精度随参数变化，上下波动

但由于每次只是用部分的普通数据，所以可能存在偶然误差，无法确定哪个参数可以影响结果





### 尝试  4

将<a href="#尝试  3">尝试  3</a> 中的 **train，test**连接后做 **K**折交叉验证**（k = 10）**

**测试结果：**

![1569136909409](C:\Users\Olymn\AppData\Roaming\Typora\typora-user-images\1569136909409.png)

**平均精度：**

**0.970252779**

**分析：**

结果比较理想，尝试利用上全部的普通数据



### 尝试  5

将特殊数据平分成两部分，并与全部的正常数据共同构成**train， test**

利用控制变量法，更改参数

**数据构成：**

![test_5_data](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_5_data.jpg)

**测试结果：**

![test-5](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_5.jpg)

分析：

使用 **RMSProp** 可以使精度提高 **(1，6对照)**



使用 **SoftMax** 作为激活函数的效果较差 精度为 **0.66**



### 尝试  6

按照<a href="#尝试  5">尝试  5</a> 中的数据构成，开始对比实验

使用 逻辑回归 进行二分类

**Keras逻辑回归测试结果:**

![test_6_keras_lr](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_6_keras_lr.jpg)

**sklearn.LinReg.LogisticRegression方法测试结果：**

![test_6_sklearn_LinReg](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_6_sklearn_LinReg.jpg)

**sklearn.svm.LinearSVC 及 sklearn.svm.S方法测试结果：**

![test_6_svm](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_6_svm.jpg)



**分析：**

两个包中的逻辑回归方法内部存在差异，导致结果不同，**sklearn**的**LR**方法在该数据集上表现较好

在该数据集 (<a href="#尝试  5">尝试  5</a> ) 上的**SVC**方法表现优于**LinearSVC**方法

综合两种**LR**和两种**SVM**模型的结果可以看出，**LSTM**模型准确率明显更好，体现了深度学习的优势

继续尝试不同的数据划分，以提高精度



### 尝试  7

将<u>特殊数据</u>按照 **7 : 3** 的比例划分成两部分，分别放入 **train, test**

选取 <u>正向数据</u> 和 <u>负向数据</u>  各选取**500**条，同样按 **7 : 3** 的比例划分到 **train，test** 中

**数据划分：**

![test_7_data](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_7_data.jpg)

**测试方法：**

参数采用初始设置

为提高实验的准确性，根据 K折交叉验证的思路，每次从普通数据集中随机选取**600**正，**600**负

下文中用 <u>手动k-fold</u> 表示该方法

每次 <u>手动K-fold</u> 后，将数据划分中的红色部分扩大**1.2**倍并再次训练、测试，共**12**次

**测试结果：**

![test_7_result](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_7_result.jpg)



**分析：**

精度在 **0.94**上下浮动，精度低于 <a href="#尝试  3">尝试  3</a>

这可能是由于train中 <u>负向数据</u> 的比例较低导致的（与尝试3数据构成进行对比）

<u>特殊数据</u>的形式更接近<u>负向数据</u>，因此需要更多样本训练才能加以区分



### 尝试  8

**思路**

在实际应用中，可能从已经<u>标注为错误的数据</u>中区分出<u>特殊数据</u>，

更改**test**组成，使其更接近真实应用场景

尽量将所有的<u>负向数据</u>全部利用上，不断改变**train，test**构成

**测试结果**

(以下结果均使用 <u>手动K-fold</u>)

![test_8_1](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_8_1.jpg)

```Python
# 正向正常数据记为 +

# 负向正常数据记为 -

# 特殊数据记为 S

# num(i) 表示数据集 i 中数据的数量

# 实际情况中，负向数据：特殊数据  比例为 num(-) : num(S)

# 实际情况中，正向数据：负向数据  比例为 num(+) : num(-)

# 训练集中的 +数量记为 train(+)

# 测试集中的 - 数量 test(-)

# train(S) 表示训练集中特殊数据的数量
```

![test_8_sample](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_8_sample.jpg)

**划分依据：**

```Python
# Order 1

train与 尝试3 中的构成相同（内容不一定相同）

test 加入除了 train(-) 外全部的 负向数据

# Order 2 

随机尝试，无依据

# Order 3

train(S) : test(S) = 1 : 1

test(-) : test(S) = num(-) : num(S)

# Order 4

{ train(+) + train(S) } : train (-) = 1 : 1

train(-) : test(-) = train(S) : test(S) = 7 : 3

该次测试 将 train 和 test 中的 S 划分比例设置成了 3 : 7， 在Order 5中改正

# Order 5

{ train(+) + train(S) } : train (-) = 1 : 1

train(-) : test(-) = train(S) : test(S) = 7 : 3

# Order 6

train(-) : test(-) = 1 : 1

test(-) : test(S) = num(-) : num(S)

# Order 7

train(-) : test(-) = train(S) : test(S) = 7 : 3

test(-) : test(S) = num(-) : num(S)

# Order 8

{train(+) + train(S)} : train(-) = num(+) : num(-)  
test(-) : test(S) = num(-) : num(S)
```

**分析**

可以看出精度随数据构成的变化，波动明显



### 尝试  9

**思路**

如<a href="#尝试  8">尝试  8</a>中提到的实际情况，可以尝试只训练模型学习 <u>负向数据</u> 和 <u>特殊数据</u> 的区别即可



**测试结果**

![test_9_data](G:\PycharmProjects\LabProjectUniqlo\dataset\image\test_9_data.jpg)



**分析**

由于使用了全部的 负向数据，故不在使用 手动 k-fold

```
 train(-) + test(-) = num(-)
```

保证测试集中 <u>负向数据</u> 与 <u>特殊数据</u> 比例 符合实际情况:

```
test(-) : test(S) = num(-) : num(S)
```

精度较高，可以看出只用<u>负向数据</u>和<u>特殊数据</u>来训练模型，可以更好将两者区分开



## 总结

相较于 逻辑回归 和 SVM 模型， 深度学习模型在此次文本情感分类问题上表现更好

证明深度学习模型经过训练可以更好区分正向情感和负向情感

由于特殊数据与正向和负向数据差别都比较大，故单纯使用二分类方法将其从所有文本评论中提取出来的精确度不太令人满意（最高为平均0.97）

但 尝试 9 中仅使用了负向数据和特殊数据进行学习和测试，更加符合二分类模型的应用场景，故精度可达到0.99



## 展望

可以试着使用多分类模型来学习区分正向、负向、特殊三种数据

可在 <a href="#尝试  9">尝试 9</a> 的基础上更改LSTM层，Embedding层参数，寻找最优方案

可尝试使用GRU， Bi-LSTM