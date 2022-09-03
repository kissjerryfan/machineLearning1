import geatpy as ea
import random
import sys
sys.path.append('..')
import algorithms


class CoDE_random10p_toZero(ea.SoeaAlgorithm):

    def __init__(self, problem, population):

        ea.SoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if str(type(population)) != "<class 'Population.Population'>":
            raise RuntimeError('传入的种群对象必须为Population类型')
        self.name = 'CoDE_random10p_toZero'

        self.selFunc = 'urs'  # 基向量的选择方式，采用无约束随机选择

        # 定义参数资源池
        # Mutde是一个用于调用内核中变异函数mutde(差分变异)的变异算子类
        if population.Encoding == 'RI':
            self.mut_parameters_pool = [1.0, 1.0, 0.8]
            self.bin_parameters_pool = [0.1, 0.9, 0.2]
        else:
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self):

        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数

        # ===========================准备进化============================
        if population.Chrom is None:
            population.initChrom(NIND)  # 初始化种群染色体矩阵（内含染色体解码，详见Population类的源码）
        else:
            population.Phen = population.decoding()  # 染色体解码

        # print(population.Chrom)
        #  print(population.Chrom.shape, population.ObjV.shape, population.CV.shape)

        self.problem.aimFunc(population)  # 计算种群的目标函数值
        # print(population.Chrom.shape, population.ObjV.shape, population.CV.shape)
        population.FitnV = ea.scaling(self.problem.maxormins * population.ObjV, population.CV)  # 计算适应度
        self.evalsNum = population.sizes  # 记录评价次数

        # ========================================开始进化===============================================================
        # 在这个判断条件中, terminated中会记录population的迭代记录
        while self.terminated(population) == False:
            # 进行迭代的时候，选择对每一个个体一次差分进化
            u_population = [population.copy(), population.copy(), population.copy()]  # 存储试验个体

            indexes = [t for t in range(NIND)]
            for i in range(NIND):
                # 使用随机抽样来找出不同于当前处理的个体的另外五个不同个体索引，用来生成三个子代后代
                indexes.remove(i)
                r_list = random.sample(indexes, 5)
                indexes.append(i)
                # print(self.population.Field)
                # print(self.population.Field.shape)
                # print(self.population.Field[0])
                # print(self.population.Field[1])

                # 使用 rand/1/bin生成第一个差分算子
                parameter_index = random.randint(0, 2)  # 从参数池中随机选择一个参数
                u_population[0].Chrom[i] = algorithms.mutde_1(population.Chrom[r_list[0]], population.Chrom[r_list[1]],
                                                              population.Chrom[r_list[2]],
                                                              self.mut_parameters_pool[parameter_index])
                u_population[0].Chrom[i] = algorithms.binary_recombination(population.Chrom[i],
                                                                           u_population[0].Chrom[i],
                                                                           self.bin_parameters_pool[parameter_index],
                                                                           self.population.Field[1],
                                                                           self.population.Field[0]
                                                                           )

                # 使用 rand/2/bin生成第二个差分算子
                u_population[1].Chrom[i] = algorithms.mutde_2(population.Chrom[r_list[0]], population.Chrom[r_list[1]],
                                                              population.Chrom[r_list[2]], population.Chrom[r_list[3]],
                                                              population.Chrom[r_list[4]],
                                                              self.mut_parameters_pool[parameter_index])
                u_population[1].Chrom[i] = algorithms.binary_recombination(population.Chrom[i],
                                                                           u_population[1].Chrom[i],
                                                                           self.bin_parameters_pool[parameter_index],
                                                                           self.population.Field[1],
                                                                           self.population.Field[0])

                # 使用 current-to-rand/1生成第三个差分算子
                u_population[2].Chrom[i] = algorithms.mutde_current_to_rand(population.Chrom[i],
                                                                            population.Chrom[r_list[0]],
                                                                            population.Chrom[r_list[1]],
                                                                            population.Chrom[r_list[2]],
                                                                            self.mut_parameters_pool[parameter_index],
                                                                            self.population.Field[1],
                                                                            self.population.Field[0])

            # 计算进化后个体的目标函数值
            for u in u_population:
                algorithms.check_toZero_random(oldChrom=u.Chrom, pz=0.1)
                u.Phen = u.decoding()  # 染色体解码
                self.problem.aimFunc(u)  # 计算目标函数值
                self.evalsNum += u.sizes

            tempPop = population + u_population[0] + u_population[1] + u_population[2]

            tempPop.FitnV = ea.scaling(self.problem.maxormins * tempPop.ObjV, tempPop.CV)

            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]

            # for i in range(NIND):
            #     flag = 0
            #     if u_population[flag].ObjV[i] < u_population[1].ObjV[i]:
            #         flag = 1
            #     if u_population[flag].ObjV[i] < u_population[2].ObjV[i]:
            #         flag = 2
            #     if u_population[flag].ObjV[i] > population.ObjV[i]:
            #         population.Chrom[i] = u_population[flag].Chrom[i]
            #         population.ObjV[i] = u_population[flag].ObjV[i]
            #         population.CV[i] = u_population[flag].CV[i]
            #         population.Phen[i] = u_population[flag].Phen[i]
            #         population.FitnV[i] = u_population[flag].FitnV[i]
            #
            # population.FitnV = ea.scaling(self.problem.maxormins * population.ObjV, population.CV) # 计算适应度

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果