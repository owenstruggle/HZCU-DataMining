import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.read_csv('data/telcoPreprocess.csv')
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :41], data.iloc[:, 41:], test_size=0.3,
                                                    random_state=42)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# sigmod函数
def sigmoid(x):
    return 1.0 / (1+math.exp(-x))


# 计算hessian矩阵
def computeHessianMatrix(data, hypothesis):
    hessianMatrix = []
    n = len(data)

    for i in range(n):
        row = []
        for j in range(n):
            row.append(-data[i]*data[j]*(1-hypothesis)*hypothesis)
        hessianMatrix.append(row)
    return hessianMatrix


# 计算两个向量的点积
def computeDotProduct(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    dotProduct = 0
    for i in range(n):
        dotProduct += a[i] * b[i]
    return dotProduct


# 计算两个向量的和
def computeVectPlus(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    sum = []
    for i in range(n):
        sum.append(a[i]+b[i])
    return sum


# 计算某个向量的n倍
def computeTimesVect(vect, n):
    nTimesVect = []
    for i in range(len(vect)):
        nTimesVect.append(*(n * vect[i]))
    return nTimesVect


# 牛顿法
def newtonMethod(dataMat, labelMat, iterNum=10, _lambda=0.1):
    m = len(dataMat)  # 训练集个数
    n = len(dataMat[0])  # 数据特征纬度
    theta = [0.0] * n

    while (iterNum):
        gradientSum = [0.0] * n
        hessianMatSum = [[0.0] * n] * n
        for i in range(m):
            try:
                hypothesis = sigmoid(computeDotProduct(dataMat[i], theta))
            except:
                continue
            error = labelMat[i] - hypothesis
            gradient = computeTimesVect(dataMat[i], error / m) + computeTimesVect(_lambda / m, theta)
            gradientSum = computeVectPlus(gradientSum, gradient)
            hessian = computeHessianMatrix(dataMat[i], hypothesis / m)
            for j in range(n):
                hessianMatSum[j] = computeVectPlus(hessianMatSum[j], hessian[j])

        # 计算hessian矩阵的逆矩阵有可能异常，如果捕获异常则忽略此轮迭代
        try:
            hessianMatInv = np.mat(hessianMatSum).I.tolist()
        except:
            continue
        for k in range(n):
            theta[k] -= computeDotProduct(hessianMatInv[k], gradientSum)

        iterNum -= 1
    return theta


def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i] * theta[i]
    return sigmoid(z)


def Declare_Winner(theta):
    score = 0
    length = len(X_test)
    for i in range(length):
        prediction = round(Hypothesis(X_test[i], theta))
        answer = y_test[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(length)
    print('Your score: ', my_score)


theta = newtonMethod(X_train, y_train)
print(theta)
Declare_Winner(theta)
