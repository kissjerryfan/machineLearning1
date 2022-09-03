from multi_objective.moea_NSGA2_toZero import moea_NSGA2_toZero
from multi_objective.moea_NSGA2_DE_toZero import moea_NSGA2_DE_toZero
from multi_objective.moea_NSGA2_10p_toZero import  moea_NSGA2_10p_toZero
from multi_objective.moea_NSGA2_20p_toZero import  moea_NSGA2_20p_toZero
from multi_objective.moea_NSGA2_30p_toZero import  moea_NSGA2_30p_toZero
from multi_objective.moea_NSGA2_random10p_toZero import moea_NSGA2_random10p_toZero
from multi_objective.moea_NSGA2_random20p_toZero import moea_NSGA2_random20p_toZero
from multi_objective.moea_NSGA2_random30p_toZero import moea_NSGA2_random30p_toZero

import sys
sys.path.append('..')
from MyProblem import MyProblem
import geatpy as ea

class MSDP:
    # 用来设置关于多目标优化方法的流程
    '''
        target 的值
            0---FPA
            1---AAE
            2---numOfnonZero
            3---L1
            4---MSE

        model--'linear', 'BPNN'
        l:决策变量的下界
        u:决策变量的上界
        moea 的值：
        1--------moea_NSGA2_templet
        2--------moea_NSGA2_DE_templet
        3--------moea_NSGA2_toZero
        4--------moea_NSGA2_DE_toZero
    '''

    def __init__(self, X, y, target, model, l, u, drawing = 1, maxgen = 100, moea = 1):

        '''===============================实例化问题对象=================================='''
        self.problem = MyProblem(target=target, X = X, y = y, model = model, l = l, u = u)

        '''===========================种群设置=============================='''

        Encoding = 'RI' # 'RI':实整数编码，即实数和整数的混合编码；
        NIND = 100 # 种群规模
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges, self.problem.borders) # 创建区域描述器
        self.population = ea.Population(Encoding, Field, NIND)

        '''实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）'''

        """=========================算法参数设置============================"""
        if moea == 1:
            self.myAlgorithm = ea.moea_NSGA2_templet(self.problem, self.population) # 实例化一个算法模板对象
        elif moea == 2:
            self.myAlgorithm = ea.moea_NSGA2_DE_templet(self.problem, self.population)
        elif moea == 3:
            self.myAlgorithm = moea_NSGA2_toZero(self.problem, self.population)
        elif moea == 4:
            self.myAlgorithm = moea_NSGA2_DE_toZero(self.problem, self.population)
        elif moea == 5:
            self.myAlgorithm = ea.moea_NSGA3_templet(self.problem, self.population)
        elif moea == 6:
            self.myAlgorithm = ea.moea_NSGA3_DE_templet(self.problem, self.population)
        elif moea == 7:
            self.myAlgorithm = ea.moea_awGA_templet(self.problem, self.population)
        elif moea == 8:
            self.myAlgorithm = ea.moea_RVEA_templet(self.problem, self.population)
        elif moea == 9:
            self.myAlgorithm = moea_NSGA2_10p_toZero(self.problem, self.population)
        elif moea == 10:
            self.myAlgorithm = moea_NSGA2_20p_toZero(self.problem, self.population)
        elif moea == 11:
            self.myAlgorithm = moea_NSGA2_30p_toZero(self.problem, self.population)
        elif moea == 12:
            self.myAlgorithm = moea_NSGA2_random10p_toZero(self.problem, self.population)
        elif moea == 13:
            self.myAlgorithm = moea_NSGA2_random20p_toZero(self.problem, self.population)
        elif moea == 14:
            self.myAlgorithm = moea_NSGA2_random30p_toZero(self.problem, self.population)
        else:
            print('error moea method!!')


        self.myAlgorithm.MAXGEN = maxgen # 设置最大遗传代数
        self.myAlgorithm.drawing = drawing

    def run(self):
        self.NDSet = self.myAlgorithm.run()
       # print(self.NDSet)
       # self.NDSet.save()

    # 预测函数
    def predict(self, testX):
        # if self.NDSet:
        pass


    def output(self):
        if self.NDSet:
            print('用时：%s 秒' % (self.myAlgorithm.passTime))
            print('非支配个体数：%s 个' % (self.NDSet.sizes))
            print('单位时间找到帕累托前沿点个数：%s 个' % (int(self.NDSet.sizes // self.myAlgorithm.passTime)))
            PF = self.problem.getBest()
            if PF is not None and self.NDSet.sizes != 0:
                GD = ea.indicator.GD(self.NDSet.ObjV, PF)#计算GD指标
                IGD = ea.indicator.IGD(self.NDSet.ObjV,
                                       PF)  # 计算IGD指标
                HV = ea.indicator.HV(self.NDSet.ObjV, PF) #计算HV指标
                Spacing = ea.indicator.Spacing(self.NDSet.ObjV)#计算Spacing指标
                print('GD:%f'%GD)
                print('IGD:%f'%IGD)
                print('HV:%f'%HV)
                print('Spacing:%f'%Spacing)
            """=====================进化过程指标追踪分析========================"""
            if PF is not None:
                metricName = [['IGD'], ['HV']]
                [NDSet_trace, Metrics] = ea.indicator.moea_tracking(
                self.myAlgorithm.pop_trace, PF, metricName,
                self.problem.maxormins)  # 绘制指标追踪分析图
                ea.trcplot(Metrics, labels = metricName, titles = metricName)


