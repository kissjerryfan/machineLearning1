# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from sys import path as paths
from os import path

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


class moea_NSGA2_30p_toZero(ea.MoeaAlgorithm):
    """
moea_NSGA2_toZero : class - 多目标进化NSGA-II算法模板

算法描述:
    采用NSGA-II进行多目标优化，算法详见参考文献[1]。
模板使用注意:
    与模板相比，进行的修改，对参数小于1的，直接设置为0.
参考文献:
    [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective
    genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary
    Computation, 2002, 6(2):0-197.
    """

    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if str(type(population)) != "<class 'Population.Population'>":
            raise RuntimeError('传入的种群对象必须为Population类型')
        self.name = 'NSGA2-30p-toZero'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=1)  # 生成二进制变异算子对象
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1, DisI=20)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')

    def reinsertion(self, population, offspring, NUM):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            注：这里对原版NSGA-II进行等价的修改：先按帕累托分级和拥挤距离来计算出种群个体的适应度，
            然后调用dup选择算子(详见help(ea.dup))来根据适应度从大到小的顺序选择出个体保留到下一代。
            这跟原版NSGA-II的选择方法所得的结果是完全一样的。
        """

        # 父子两代合并
        population = population + offspring
        # 选择个体保留到下一代
        [levels, criLevel] = self.ndSort(self.problem.maxormins * population.ObjV, NUM, None,
                                         population.CV)  # 对NUM个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag]

    def run(self):
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        if population.Chrom is None:
            population.initChrom()  # 初始化种群染色体矩阵（内含解码，详见Population类的源码）
        else:
            population.Phen = population.decoding()  # 染色体解码
        self.problem.aimFunc(population)  # 计算种群的目标函数值
        self.evalsNum = population.sizes  # 记录评价次数
        [levels, criLevel] = self.ndSort(self.problem.maxormins * population.ObjV, NIND, None,
                                         population.CV)  # 对NIND个个体进行非支配分层
        population.FitnV[:, 0] = 1 / levels  # 直接根据levels来计算初代个体的适应度
        # ===========================开始进化============================
        while self.terminated(population) == False:
            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 对选出的个体进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异

            #====================================设置对每条染色体，小于其最大值的参数设置为0======================================
            max_index = offspring.Chrom.argmax(axis = 1)
            min_index = offspring.Chrom.argmin(axis = 1)
            for i in range(offspring.Chrom.shape[0]):
                pos_v = abs(offspring.Chrom[i][max_index[i]] * 0.3)
                neg_v = abs(offspring.Chrom[i][min_index[i]] * 0.3)
                max_v = pos_v
                if neg_v > max_v:
                    max_v = neg_v


                for j in range(offspring.Chrom.shape[1]):
                    if  abs(offspring.Chrom[i][j]) <= max_v:
                        offspring.Chrom[i][j] = 0

            #offspring.Chrom[offspring.Chrom < 1] = 0


            offspring.Phen = offspring.decoding()  # 解码


            self.problem.aimFunc(offspring)  # 求进化后个体的目标函数值
            self.evalsNum += offspring.sizes  # 更新评价次数
            # 重插入生成新一代种群
            population = self.reinsertion(population, offspring, NIND)

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果