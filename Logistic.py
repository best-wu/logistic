# -*- coding: utf-8 -*-

import random
import numpy as np

TRAIN_FILE = 'horseColicTraining.txt'
TEST_FILE = 'horseColicTest.txt'
"""
函数说明：sigmoid函数

Parameters:
    inX - 数据
    
Returns:
    sigmoid函数
"""


# logistic回归使用了sigmoid函数
def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))


# 函数中涉及如何将list转化成矩阵的操作：mat()
# 同时还含有矩阵的转置操作：transpose()
# 还有list和array的shape函数
# 在处理矩阵乘法时，要注意的便是维数是否对应

# graAscent函数实现了梯度上升法，隐含了复杂的数学推理
# 梯度上升算法，每次参数迭代时都需要遍历整个数据集
def graAscent0(dataMath, classLabels, maxCycles=500, alpha=0.001):
    # 转换成numpy的mat(矩阵)
    dataMatrix = np.mat(dataMath)
    # 转换成numpy的mat(矩阵)并进行转置
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 移动步长，也就是学习效率，控制更新的幅度
    alpha = alpha
    # 最大迭代次数
    maxCycles = maxCycles
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，返回权重数组
    # mat.getA()将自身矩阵变量转化为ndarray类型变量
    return weights.getA()


# 批量随机梯度上升算法的实现，对于数据量较多的情况下计算量小，但分类效果差
def graAscent1(dataMatrix, matLabel, num=100, alpha=0.001, batch=20):
    m, n = np.shape(dataMatrix)
    matMatrix = np.mat(dataMatrix)
    batch = batch
    w = np.ones((n, 1))
    num = num  # 这里的这个迭代次数对于分类效果影响很大，很小时分类效果很差
    for i in range(num):
        for j in range(int(m / batch)):
            alpha = alpha
            error = sigmoid(matMatrix[batch * j:batch * (j + 1)] * w) - matLabel[batch * j:batch * (j + 1)]
            w = w - alpha * matMatrix[batch * j:batch * (j + 1)].transpose() * error
    return w


# 改进后的随机梯度上升算法
# 从两个方面对随机梯度上升算法进行了改进,正确率确实提高了很多
# 改进一：对于学习率alpha采用非线性下降的方式使得每次都不一样
# 改进二：每次使用一个数据，但是每次随机的选取数据，选过的不在进行选择
def graAscent2(dataMatrix, matLabel, num=100):
    m, n = np.shape(dataMatrix)
    matMatrix = np.mat(dataMatrix)
    w = np.ones((n, 1))
    num = num  # 这里的这个迭代次数对于分类效果影响很大，很小时分类效果很差
    setIndex = set([])
    for i in range(num):
        for j in range(m):
            alpha = 4 / (1 + i + j) + 0.01
            dataIndex = random.randint(0, 100)
            while dataIndex in setIndex:
                setIndex.add(dataIndex)
                dataIndex = random.randint(0, 100)
            error = sigmoid(matMatrix[dataIndex] * w) - matLabel[dataIndex]
            w = w - alpha * matMatrix[dataIndex].transpose() * error
    return w


"""
读取数据

Parameters:
    filename(string)

Returns:
    trainingSet(list),trainingLabel(list)
"""


def load_data(filename):
    # 打开训练集
    frTrain = open(filename)
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    return trainingSet, trainingLabels


"""
函数说明：用python写的Logistic分类器做预测

Parameters:
    None
    
Returns:
    None
"""


def colicTest(trainWeights):
    # 打开测试集
    frTest = open(TEST_FILE)
    # 使用上升梯度训练
    errorCount = 0
    numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:, 0])) != int(currLine[-1]):
            errorCount += 1
    # 错误概率计算
    errorRate = (float(errorCount) / numTestVect) * 100
    print("测试集错误率为：%.2f%%" % errorRate)


"""
函数说明：分类函数

Parameters:
    inX - 特征向量
    weights - 回归系数
    
Returns:
    分类结果
"""


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


if __name__ == '__main__':
    '''
    trainingSet, trainingLabels = load_data(TRAIN_FILE)
    print('梯度上升算法结果：',end=' ')
    trainWeights = graAscent0(np.array(trainingSet), np.array(trainingLabels))
    colicTest(trainWeights)
    print('批量梯度上升算法结果：', end=' ')
    trainWeights = graAscent1(np.array(trainingSet), np.array(trainingLabels))
    colicTest(trainWeights)
    print('随机梯度上升算法结果：', end=' ')
    trainWeights = graAscent2(np.array(trainingSet), np.array(trainingLabels))
    colicTest(trainWeights)
    '''
    trainingSet, trainingLabels = load_data(TRAIN_FILE)
    for num in [100, 300, 500]:
        for batch in [10, 20,30,50]:
            print('batch=%f,num=%f' % (batch, num),end=' ')
            trainWeights = graAscent1(np.array(trainingSet), np.array(trainingLabels),num=num,batch=batch)
            colicTest(trainWeights)
    for num in [100,300,500]:
        print('num = %f'%num,end=' ')
        trainWeights = graAscent2(np.array(trainingSet), np.array(trainingLabels),num=num)
        colicTest(trainWeights)
