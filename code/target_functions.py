import numpy as np
import tensorflow as tf
from BPNN import BPNN
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from NN import NN

# 输入：X(n, d), parameters,(m, d+1)
# d 特征的数量, n样本个数, m种群大小
# 输出：predValue:n*m
def linear_predict(X, parameters):
    new_X = np.ones((X.shape[0], X.shape[1]+1))
    new_X[:, 0:-1] = X
    # new_X 转换为 (n, d+1)
    # 返回n*m矩阵, 其中每一列是一个对应染色体的预测y值
    return np.dot(new_X, np.transpose(parameters)).astype(float)

def get_logistic_predict(X, parameters):
    new_X = np.ones((X.shape[0], X.shape[1] + 1))
    new_X[:, 0:-1] = X
    y = np.dot(new_X, np.transpose(parameters)).astype(float)
    new_y = 1 / (1 + np.exp(-x))
    if new_y >= 0.5:
        new_y = 1
    else:
        new_y = -1
    return np.array(new_y)


def logistic_predict(X, parameters):
    if parameters.ndim == 1:
        return get_logistic_predict(X, parameters)
    predValues = []
    for i in range(parameters.shape[0]):
        param = parameters[i]
        predvalue = get_logistic_predict(X, param)
        predvalue = predvalue.flatten()
        predvalue = predvalue.tolist()
        predValues.append(predvalue)
    return np.array(predValues).T


# 输入：X(n, d), parameters,(1, d)
# d 特征的数量, n样本个数, m种群大小
# 输出：predValue:n*1
def get_bpnn_predict(X, y, param):
    model = BPNN(learning_rate=0.05, num_of_training=100 , input_size=X.shape[1],
                 hidden_n=2, parameters=list(param))
    model.fit(X, y)
    predvalue = model.predict(X)
    return np.array(predvalue)

# 输入：X(n, d), parameters,(m, d)
# d 特征的数量, n样本个数, m种群大小
# 输出：predValue:n*m
def bpnn_predict(X, y, parameters):
    if parameters.ndim == 1:
        return get_bpnn_predict(X, y, parameters)
    predValues = []
    for i in range(parameters.shape[0]):
        param = parameters[i]
        predvalue = get_bpnn_predict(X, y, param)
        predvalue = predvalue.flatten()
        predvalue = predvalue.tolist()
        predValues.append(predvalue)
    return np.array(predValues).T

def get_nn_predict(X, param):
    model = NN(learning_rate=0.05, num_of_training=100 , input_size=X.shape[1],
                 hidden_n=2, parameters=list(param))
    predvalue = model.predict(X)
    return np.array(predvalue)


def nn_predict(X, parameters):
    if parameters.ndim == 1:
        return get_nn_predict(X, parameters)
    predValues = []
    for i in range(parameters.shape[0]):
        param = parameters[i]
        predvalue = get_nn_predict(X, param)
        predvalue = predvalue.flatten()
        predvalue = predvalue.tolist()
        predValues.append(predvalue)
    return np.array(predValues).T


def get_mlp_predict(X, parameters, hidden_n):
    model = MLPRegressor(hidden_layer_sizes=hidden_n, activation='logistic', max_iter=1)
    y = np.random.rand(X.shape[0])
    model.fit(X, y)
    weights = []
    bias = []
    for i in range(X.shape[1]):
        weights.append(parameters[i * hidden_n:(i + 1) * hidden_n])
    for i in range(len(parameters) - hidden_n, len(parameters)):
        bias.append([parameters[i]])
    weights = np.array(weights)
    bias = np.array(bias)
    coefs = [weights, bias]
    model.coefs_ = coefs
    predvalue = model.predict(X)
    return np.array(predvalue)


def mlp_predict(X, parameters):
    hidden_n = 3
    if parameters.ndim == 1:
        return get_mlp_predict(X, parameters, hidden_n)
    predValues = []
    for i in range(parameters.shape[0]):
        param = parameters[i]
        predvalue = get_mlp_predict(X, param, hidden_n)
        predvalue = predvalue.flatten()
        predvalue = predvalue.tolist()
        predValues.append(predvalue)
    return np.array(predValues).T

def get_mlpn_predict(X, parameters, hidden_n):
    model = MLPRegressor(hidden_layer_sizes=hidden_n, activation='logistic', max_iter=1)
    y = np.random.rand(X.shape[0])
    model.fit(X, y)
    weights1 = []
    weights2 = []
    bias1 = []
    bias2 = []
    for i in range(X.shape[1]):
        weights1.append(parameters[i * hidden_n:(i + 1) * hidden_n])
    p_index = X.shape[1] * hidden_n
    bias1.append(parameters[p_index : p_index + hidden_n])
    p_index =  p_index + hidden_n
    for i in range(p_index, p_index+hidden_n):
        weights2.append([parameters[i]])
    p_index = p_index + hidden_n
    bias2.append(parameters[p_index:])
    weights1 = np.array(weights1)
    bias1 = np.array(bias1)
    weights2 = np.array(weights2)
    bias2 = np.array(bias2)
    weights = [weights1, weights2]
    bias = [bias1, bias2]
    model.coefs_ = weights
    model.intercepts_ = bias
    predvalue = model.predict(X)
    return np.array(predvalue)


def mlpn_predict(X, parameters, hidden_n):
    if parameters.ndim == 1:
        return get_mlpn_predict(X, parameters, hidden_n)
    predValues = []
    for i in range(parameters.shape[0]):
        param = parameters[i]
        predvalue = get_mlpn_predict(X, param, hidden_n)
        predvalue = predvalue.flatten()
        predvalue = predvalue.tolist()
        predValues.append(predvalue)
    return np.array(predValues).T




def bubbleSort(Ovector):
    index = []
    for i in range(0, len(Ovector)):
        index.append(i)

    length = len(Ovector)
    for i in range(1, length):
        for j in range(length-i):
            if Ovector[j] > Ovector[j+1]:
                temp = Ovector[j]
                Ovector[j] = Ovector[j+1]
                Ovector[j+1] = temp
                temp_index = index[j]
                index[j] = index[j+1]
                index[j+1] = temp_index
    return index


# 输入：predValue(n*m矩阵), target（长度为n的array)
# 输出：FPAs
# 非常需要注意的地方是,在python中对于列表等，修改值之后，原始值会改变，因此会有影响。

def get_FPA(predvalue, t):
#----------------因此在这里额外赋值一下！！！！
    predValue = predvalue.copy()
    target = t.copy()
    index = bubbleSort(predValue)
    length = len(target)
    targetRe = []
    for i in range(0, length):
        targetRe.append(0)
    totalFaults = 0
    for j in range(0, length):
        totalFaults = totalFaults + target[j]
    i = 0
    while i < length:
        count = 1
        temp = target[index[i]]

        while ((i + count) < length):
            if predValue[index[i]] != predValue[index[i + count]]:
                break
            temp = temp + target[index[i + count]]
            count = count + 1
        for j in range(i, i + count):
            targetRe[j] = temp / count
        i = i + count
    NK = 0
    length = len(targetRe)
    for i in range(0, length):
        NK = NK + (length - i) * targetRe[length - 1 - i]
    if (totalFaults == 0):
        percent = 1
    else:
        percent = NK / (length * totalFaults)
    return percent

# 非常需要注意的地方是,在python中对于列表等，修改值之后，原始值会改变，因此会有影响。
def FPA(predValues, target):
    if predValues.ndim == 1:
        return get_FPA(predValues, target)
    fpas = []
    for t in range(predValues.shape[1]):
        predValue = predValues[:,t]

        percent = get_FPA(predValue, target)
        fpas.append(percent)
    return np.array(fpas).astype(float)


# 输入：predValue(n*m矩阵), target（长度为n的array)
# 输出：AAE

def get_AAE(predvalue, target):
    abssum = 0
    for i in range(len(target)):
        abssum += abs(predvalue[i] - target[i])
    abssum = abssum / len(target)
    return abssum
def AAE(predValues, target):
    if predValues.ndim == 1:
        return get_AAE(predValues, target)
    aaes = []
    for t in range(predValues.shape[1]):
        predValue = predValues[:,t]
        abssum = get_AAE(predValue, target)
        aaes.append(abssum)
    return np.array(aaes).astype(float)

def get_MSE(predvalue, target):
    mse = mean_squared_error(predvalue, target)
    return mse

def MSE(predValues, target):
    if predValues.ndim == 1:
        return get_MSE(predValues, target)
    mses = []
    for t in range(predValues.shape[1]):
        predValue = predValues[:,t]
        abssum = get_MSE(predValue, target)
        mses.append(abssum)
    return np.array(mses).astype(float)



# 输入: parameters,(m, d+1)
# 输出：非零参数的个数
def get_numOfnonZero(parameter):
    nonz = 0
    for j in range(len(parameter)):
        if parameter[j] != 0:
            nonz += 1
    return nonz

def numOfnonZero(parameters):
    nonzs = []
    if parameters.ndim == 1:
        return get_numOfnonZero(parameters)
    for i in range(parameters.shape[0]):
        nonz = get_numOfnonZero(parameters[i])
        nonzs.append(nonz)
    return np.array(nonzs).astype(float)


# 计算L1范数
def get_l1_value(parameter):
    l1 = 0
    for j in range(len(parameter)):
        l1 += abs(parameter[j])
    return l1


def l1_values(parameters):
    l1s = []
    if parameters.ndim == 1:
        return get_l1_value(parameters)
    for i in range(parameters.shape[0]):
        l1 = get_l1_value(parameters[i])
        l1s.append(l1)
    return np.array(l1s).astype(float)

