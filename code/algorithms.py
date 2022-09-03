import random
import numpy as np


'''
=======================================================================================================
=======================================================================================================
本文件所包含的函数, 主要实现一些别的算法所需要的小算法
如 CoDE算法所需要的一些变异函数, 如二进制重组, DE/rand/1/bin, DE/rand/2/bin
============================================================================================= ==========
'''




'''
rand/1/
输入: 三条染色体, r1, r2, r3; F 差分系数
输出: 一条新的染色体
'''
def mutde_1(r1, r2, r3, F):
    new_chrom = []
    for i in range(len(r1)):
        t = r1[i] + F * (r2[i]-r3[i])
        new_chrom.append(t)
    return new_chrom

'''
rand/2/
输入: 三条染色体, r1, r2, r3, r4, r5; F 差分系数
输出: 一条新的染色体
'''
def mutde_2(r1, r2, r3, r4, r5, F):
    new_chrom = []
    F1 = random.uniform(0, 1)
    for i in range(len(r1)):
        t = r1[i] + F1* (r2[i]-r3[i]) + F * (r4[i]-r5[i])
        new_chrom.append(t)
    return new_chrom

'''
bin二进制重组
输入: 两条染色体oldChrom, u)
输出: 一条新的染色体
'''
def binary_recombination(oldChrom, u, Cr, ub, lb):
    # randrange(a,b)---> [a, b-1]
    j = random.randrange(0, len(oldChrom))
    newChrom = []
    for i in range(len(oldChrom)):
        rand = random.uniform(0, 1)
        if rand <= Cr or i == j:
            newChrom.append(u[i])
        else:
            newChrom.append(oldChrom[i])
    return check_for_feasible( oldChrom=newChrom, ub = ub, lb = lb)


'''

'''
def mutde_current_to_rand(oldChrom, r1, r2, r3, F, ub, lb):
    new_chrom = []
    # uniform(a, b)---->[a ,b)
    rand = random.uniform(0, 1)

    for i in range(len(oldChrom)):
        t = oldChrom[i]+ rand*(r1[i] - oldChrom[i]) + F * (r2[i] - r3[i])
        new_chrom.append(t)
    return check_for_feasible(oldChrom=new_chrom, ub=ub, lb=lb)

def check_for_feasible(oldChrom, ub, lb):
    new_chrom = []
    for i in range(len(oldChrom)):
        if oldChrom[i] < lb[i]:
            t1 = 2 * lb[i] - oldChrom[i]
            new_chrom.append(min(ub[i], t1))
        elif oldChrom[i] > ub[i]:
            t1 = 2 * ub[i] - oldChrom[i]
            new_chrom.append(max(lb[i], t1))
        else:
            new_chrom.append(oldChrom[i])
    return new_chrom

def check_toZero_bound(oldChrom, bound ):
    oldChrom[abs(oldChrom) < bound] = 0


def check_toZero_ratio(oldChrom, ratio):
    max_indexes = oldChrom.argmax(axis=1)
    min_indexes = oldChrom.argmin(axis=1)
    for i in range(oldChrom.shape[0]):
        pos_v = abs(oldChrom[i][max_indexes[i]] * ratio)
        neg_v = abs(oldChrom[i][min_indexes[i]] * ratio)
        max_v = pos_v
        if neg_v > max_v:
            max_v = neg_v

        for j in range(oldChrom.shape[1]):
            if abs(oldChrom[i][j]) < max_v:
                oldChrom[i][j] = 0


def check_toZero_ratiolr(oldChrom, ratio):
    max_indexes = oldChrom.argmax(axis=1)
    min_indexes = oldChrom.argmin(axis=1)
    for i in range(oldChrom.shape[0]):
        pos_v = oldChrom[i][max_indexes[i]] * ratio
        neg_v = oldChrom[i][min_indexes[i]] * ratio
        if pos_v < 0:
            pos_v = 0
        if neg_v > 0:
            neg_v = 0

        for j in range(oldChrom.shape[1]):
            if oldChrom[i][j] > 0 and oldChrom[i][j] < pos_v:
                oldChrom[i][j] = 0
            elif oldChrom[i][j] < 0 and oldChrom[i][j] > neg_v:
                oldChrom[i][j] = 0

def check_toZero_random(oldChrom, pz):
    for i in range(oldChrom.shape[0]):
        for j in range(oldChrom.shape[1]):
            # uniform(a, b)---->[a ,b)
            rand = random.uniform(0, 1)
            if rand < pz:
                oldChrom[i][j] = 0



def minmaxScaler(x):
    min_value = np.min(x)
    max_value = np.max(x)
    dis = max_value-min_value
    if dis == 0:
        return np.ones(x.shape, dtype = int)
    else:
        return (x - min_value) / dis

