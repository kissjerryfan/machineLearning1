import geatpy as ea
import numpy as np
import target_functions as tgf

class MyProblem(ea.Problem):

    #---------------------------需要预先指定的参数
    # 输入相应的值
    '''
        target 的值
            0---FPA
            1---AAE
            2---numOfnonZero
            3---l1
            4---MSE

        model--'linear', 'BPNN'
        l:决策变量的下界
        u:决策变量的上界
    '''
    def __init__(self, target, X, y, model, l, u):

        name = 'LTR' # 初始化名称learn-to-rank
        self.target = target
        self.model = model
        self.M =len(target)
        maxormins = []
        for param in self.target:
            if param == 0:
                maxormins.append(-1)
            elif param == 1:
                maxormins.append(1)
            elif param == 2:
                maxormins.append(1)
            elif param == 3:
                maxormins.append(1)
            elif param == 4:
                maxormins.append(1)

        Dim = 0
        if self.model == 1:
            Dim = X.shape[1] + 1
        elif self.model == 2:
            # X.shape[1] * n_hidden + n_hidden
            Dim = X.shape[1] * 2 + 2
        elif self.model == 3:
            # X.shape[1] * n_hidden + n_hidden
            Dim = X.shape[1] * 2 + 2
        elif self.model == 4:
            Dim = X.shape[1] * 3 + 3
        elif self.model == 5:
            Dim = X.shape[1] * 3 + 3 + 3 + 1
        elif self.model == 6:
            Dim = X.shape[1] * 5 + 5 + 5 + 1
        else:
            print(self.model)
            print('model error!!!!!!')
            #
        varTypes = [0] * Dim # 决策变量的类似，0：实数，1：整数

# --------------------------决策变量的上下界需要填
        lb = [l]* Dim # 决策变量的下界
        ub = [u] * Dim #决策变量的上界
        lbin = [1] * Dim # 下界是否被包含， 0不包含， 1包含
        ubin = [1] * Dim # 上界是否被包含， 0不包含， 1包含

        ea.Problem.__init__(self, name, self.M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
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

        if self.model == 1:
            predvalue = tgf.linear_predict(self.X, parameters)
        elif self.model == 2:
            predvalue = tgf.bpnn_predict(self.X, self.y, parameters)
        elif self.model == 3:
            predvalue = tgf.nn_predict(self.X, parameters)
        elif self.model == 4:
            predvalue = tgf.mlp_predict(self.X, parameters)
        elif self.model == 5:
            predvalue = tgf.mlpn_predict(self.X, parameters, 3)
        elif self.model == 6:
            predvalue = tgf.mlpn_predict(self.X, parameters, 5)
        else:
            print(self.model)
            print('model error!!!!')

        # f1, f2, f3 都是 m*1的
        # print(predvalue)
        if self.M == 1:
            f1 = tgf.FPA(predvalue, self.y)
            pop.ObjV = np.array([f1]).T
        else:
            fs = []
            for param in self.target:
                if param == 0:
                    f1 = tgf.FPA(predvalue, self.y)
                    fs.append(f1)
                elif param == 1:
                    f2 = tgf.AAE(predvalue, self.y)
                    fs.append(f2)
                elif param == 2:
                    f3 = tgf.numOfnonZero(parameters)
                    fs.append(f3)
                elif param == 3:
                    f4 = tgf.l1_values(parameters)
                    fs.append(f4)
                elif param == 4:
                    f5 = tgf.MSE(predvalue, self.y)
                    fs.append(f5)
                else:
                    print('target value error!!')
            if len(fs) == 2:
                pop.ObjV = np.vstack([fs[0], fs[1]]).T
            elif len(fs) == 3:
                pop.ObjV = np.vstack([fs[0], fs[1], fs[2]]).T
            else:
                print('object more than three!!!!')

        #     self.M == 2:
        #     f1 = tgf.FPA(predvalue, self.y)
        #     f2 = tgf.AAE(predvalue, self.y)
        #   #  f2 = tgf.numOfnonZero(parameters)
        #     pop.ObjV = np.vstack([f1, f2]).T
        # else:
        #     f1 = tgf.FPA(predvalue, self.y)
        #     f2 = tgf.AAE(predvalue, self.y)
        #     f3 = tgf.numOfnonZero(parameters)
        #     pop.ObjV = np.vstack([f1, f2, f3]).T
        #print(f1.shape, f2.shape, f3.shape)
        #print(f1, f2, f3)
      #  pop.ObjV = np.vstack([f1, f2, f3]).T
      #  pop.ObjV = np.vstack([f1, f2]).T
       # pop.ObjV = np.array([f1]).T
        #print(pop.ObjV.shape)
        #print(pop.ObjV)

