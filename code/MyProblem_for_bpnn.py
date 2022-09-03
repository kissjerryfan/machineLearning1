import geatpy as ea
import numpy as np
import target_functions as tgf
from BPNN import BPNN

class MyProblem_for_bpnn(ea.Problem):

#---------------------------需要预先指定的参数
# 输入相应的值
    def __init__(self, M, X, y, model):



        name = 'LTR' # 初始化名称multi-learn-to-rank
        self.M = M # 初始化目标维数，三个优化目标
        if self.M == 1:
            maxormins = [-1]
        elif self.M == 2:
            maxormins = [-1, 1] # 最大最小化标记列表，1最小化标记该目标，-1最大化标记该目标
        else:
            maxormins = [-1, 1, 1]
        Dim = X.shape[1] * 5 + 5 # 输入层的系数，以及隐藏层的系数，设置为5
        varTypes = [0] * Dim # 决策变量的类似，0：实数，1：整数

# --------------------------决策变量的上下界需要填
        lb = [-1]* Dim # 决策变量的下界
        ub = [1] * Dim #决策变量的上界
        lbin = [1] * Dim # 下界是否被包含， 0不包含， 1包含
        ubin = [1] * Dim # 上界是否被包含， 0不包含， 1包含

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 添加属性存储X和y
        self.X  = X
        self.y = y

# 输出目标矩阵
# 这个函数还需重写
# ObjV和CV都是Numpy array类型矩阵，且行数等于种群的个体数目。ObjV的每一行对应一个个体，每一列对应一个优化目标。
# CV矩阵的每一行也是对应一个个体，而每一列对应一个约束条件。
# 根据Geatpy数据结构可知，种群对象中的Chrom、ObjV、FitnV、CV和Phen都是Numpy array类型的行数等于种群规模sizes的矩阵，即这些成员属性的每一行都跟种群的每一个个体是一一对应的。
#
    def aimFunc(self, pop):
        # 决策变量是m * (d+1)
        parameters = pop.Phen.astype(float)




        # f1, f2, f3 都是 m*1的
        # print(predvalue)
        if self.M == 1:
            f1 = tgf.FPA(predvalue, self.y)
            pop.ObjV = np.array([f1]).T
        elif self.M == 2:
            f1 = tgf.FPA(predvalue, self.y)
            f2 = tgf.AAE(predvalue, self.y)
           # f2 = tgf.numOfnonZero(parameters)
            pop.ObjV = np.vstack([f1, f2]).T
        else:
            f1 = tgf.FPA(predvalue, self.y)
            f2 = tgf.AAE(predvalue, self.y)
            f3 = tgf.numOfnonZero(parameters)
            pop.ObjV = np.vstack([f1, f2, f3]).T
        #print(f1.shape, f2.shape, f3.shape)
        #print(f1, f2, f3)
      #  pop.ObjV = np.vstack([f1, f2, f3]).T
      #  pop.ObjV = np.vstack([f1, f2]).T
       # pop.ObjV = np.array([f1]).T
        #print(pop.ObjV.shape)
        #print(pop.ObjV)

