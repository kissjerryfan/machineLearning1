from single_objective.CoDE import CoDE
from single_objective.CoDE_toZero import CoDE_toZero
from single_objective.CoDE_10p_lr_toZero import CoDE_10p_lr_toZero
from single_objective.CoDE_20p_lr_toZero import  CoDE_20p_lr_toZero
from single_objective.CoDE_10p_toZero import CoDE_10p_toZero
from single_objective.CoDE_20p_toZero import CoDE_20p_toZero
from single_objective.CoDE_random10p_toZero import CoDE_random10p_toZero
from single_objective.CoDE_random20p_toZero import CoDE_random20p_toZero
from single_objective.CoDE_random30p_toZero import CoDE_random30p_toZero

import sys
sys.path.append('..')
from MyProblem import MyProblem
import geatpy as ea
import numpy as np



class SSDP:
    '''
    用来设置单目标目标优化方法的流程
        model用来判断模型的类别
        model == 'linear'说明是线性模型
        model == 'BPNN' 说明是神经网络模型
        drawing--绘图方式的参数，
                 0表示不绘图，
                 1表示绘制结果图，
                 2表示实时绘制目标空间动态图，
                 3表示实时绘制决策空间动态图。
        model--'linear', 'BPNN'
        l:决策变量的下界
        u:决策变量的上界
    '''
    def __init__(self, X, y, model, drawing, l, u, soea):

        '''初始化必要的相关参数'''

        '''===============================实例化问题对象=================================='''
        self.problem = MyProblem(target =[0], X = X, y = y, model = model, l = l, u = u)

        '''===========================种群设置=============================='''
        self.model = model

        Encoding = 'RI'  # 'RI':实整数编码，即实数和整数的混合编码；

        if self.model == 1:
            NIND = 100  # 种群规模
        elif self.model == 2:
            NIND = 10
        elif self.model == 3:
            NIND = 30
        elif self.model == 4:
            NIND = 100
        else:
            print(self.model)
            print('model parameters error!!')
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges, self.problem.borders)  # 创建区域描述器
        self.population = ea.Population(Encoding, Field, NIND)

        '''实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）'''

        """================================算法参数设置============================="""
        if soea == 1:
            self.myAlgorithm = CoDE(self.problem, self.population)  # 实例化一个算法模板对象
        elif soea == 2:
            self.myAlgorithm = ea.soea_DE_rand_1_bin_templet(self.problem, self.population)
        elif soea == 3:
            self.myAlgorithm = CoDE_toZero(self.problem, self.population)
        elif soea == 4:
            self.myAlgorithm = CoDE_10p_toZero(self.problem, self.population)
        elif soea == 5:
            self.myAlgorithm = CoDE_20p_toZero(self.problem, self.population)
        elif soea == 6:
            self.myAlgorithm = CoDE_10p_lr_toZero(self.problem, self.population)
        elif soea == 7:
            self.myAlgorithm = CoDE_20p_lr_toZero(self.problem, self.population)
        elif soea == 8:
            self.myAlgorithm = CoDE_random10p_toZero(self.problem, self.population)
        elif soea == 9:
            self.myAlgorithm = CoDE_random20p_toZero(self.problem, self.population)
        elif soea == 10:
            self.myAlgorithm = CoDE_random30p_toZero(self.problem, self.population)
        else:
            print('error soea number!!!!')
        if self.model == 1:

            self.myAlgorithm.MAXGEN = 100  # 设置最大遗传代数
        elif self.model == 2:
            self.myAlgorithm.MAXGEN = 30
        elif self.model == 3:
            self.myAlgorithm.MAXGEN = 50
        elif self.model == 4:
            self.myAlgorithm.MAXGEN = 100
        else:
            print(self.model)
            print('model parameters error!!')
        self.myAlgorithm.drawing = drawing


    def run(self):
        [self.population, self.obj_trace, self.var_trace] = self.myAlgorithm.run()  # 执行算法模板
        self.best_gen = np.argmin(self.problem.maxormins * self.obj_trace[:, 1])  # 记录最优种群个体是在哪一代
        self.best_ObjV = self.obj_trace[self.best_gen, 1]
        #self.population.save()  # 把最后一代种群的信息保存到文件中
        # 预测函数


    def predict(self, testX):
        pass


    def output(self):

        print('最优的目标函数值为：%s' % (self.best_ObjV))
        print('有效进化代数：%s' % (self.obj_trace.shape[0]))
        print('最优的一代是第 %s 代' % (self.best_gen + 1))
        print('评价次数：%s' % (self.myAlgorithm.evalsNum))
        print('时间已过 %s 秒' % (self.myAlgorithm.passTime))
        for num in self.var_trace[self.best_gen, :]:
            print(chr(int(num)), end='')


