import pandas as pd
import numpy as np
from pathlib import Path
from MSDP import MSDP
from SSDP import SSDP
from BPNN import BPNN
import target_functions as tgf
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import dictionaries
import math


def getfeatures(path, file):
    data = pd.read_csv(path + file + '.csv', header=None)
    columns = data.columns.tolist()
    X = data[columns[:-1]]
    y = data[columns[-1]]
    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    y = y.astype(float)
    return X, y


'''
1. 算法流程：
 使用单目标优化CoDE算法对每一个数据集进行优化吗，并记录最优解和最优解对应的参数，FPA, AAE,和对应的非零参数的个数。
2. 记录信息
a. 文档1：记录每个数据集最优解的【FPA, AAE, 和对应的非零参数解】
b. 文档2：记录每个数据集进化的每一代的【种群平均目标函数值】
c. 文档3：记录每个数据集进化的每一代的【种群最优目标函数值】
d. 文档4：记录每个数据集最优解的参数
e. 文档5+：对每个数据集，记录其迭代的每一代的最优参数
'''


def training_record_for_ssdp(predict_model, l, u, save_folder, soea):
    fileLists = dictionaries.get_filelists()
    path = 'data/'
    save_file = '单目标算法1'
    save_path = '../results/' + save_folder + '/train/'
    doc1 = [['filename', 'FPA', 'AAE', 'numOfNonZero', 'L1', 'MSE']]
    doc2 = []
    doc3 = []
    doc4 = []
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')
            X, y = getfeatures(path, fileLists[i][j])
            model = SSDP(X=X, y=y, model=predict_model, drawing=0, l=l, u=u, soea=soea)
            model.run()


            # 找出单目标的最优的f1, f2, f3
            best_gen = np.argmin(model.problem.maxormins * model.obj_trace[:, 1])
            best_objV = model.obj_trace[best_gen, 1]

            ss_max_parameter = model.var_trace[best_gen]
            if predict_model == 1:
                predvalue = tgf.linear_predict(X, ss_max_parameter)
            elif predict_model == 2:
                predvalue = tgf.bpnn_predict(X, ss_max_parameter)
            elif predict_model == 3:
                predvalue = tgf.nn_predict(X, ss_max_parameter)
            elif predict_model == 4:
                predvalue = tgf.mlp_predict(X, ss_max_parameter)
            elif predict_model == 5:
                predvalue = tgf.mlpn_predict(X, ss_max_parameter, 3)
            elif predict_model == 6:
                predvalue = tgf.mlpn_predict(X, ss_max_parameter, 5)
            else:
                print('error model method number in helpers !!!! ')
            f1 = tgf.FPA(predvalue, y)
            f2 = tgf.AAE(predvalue, y)
            f3 = tgf.numOfnonZero(ss_max_parameter)
            f4 = tgf.l1_values(ss_max_parameter)
            f5 = tgf.MSE(predvalue, y)
            doc1.append([fileLists[i][j], f1, f2, f3, f4, f5])  # FPA, AAE, 和对应的非零参数解

            avg_list = model.obj_trace[:, 0]
            avg_list = list(avg_list)
            avg_list.insert(0, fileLists[i][j])
            doc2.append(avg_list)  # 种群平均目标函数值

            best_list = model.obj_trace[:, 1]
            best_list = list(best_list)
            best_list.insert(0, fileLists[i][j])
            doc3.append(best_list)  # 种群最优目标函数值

            best_param = ss_max_parameter
            best_param = list(best_param)
            best_param.insert(0, fileLists[i][j])
            doc4.append(best_param)  # 种群最优个体的参数

            params = pd.DataFrame(model.var_trace)
            params.to_csv(save_path + fileLists[i][j] + '.csv')

    with open(save_path + 'doc1.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        for row in doc1:
            writer1.writerow(row)

    with open(save_path + 'doc2.csv', 'w', newline='') as file2:
        writer2 = csv.writer(file2)
        for row in doc2:
            writer2.writerow(row)

    with open(save_path + 'doc3.csv', 'w', newline='') as file3:
        writer3 = csv.writer(file3)
        for row in doc3:
            writer3.writerow(row)

    with open(save_path + 'doc4.csv', 'w', newline='') as file4:
        writer4 = csv.writer(file4)
        for row in doc4:
            writer4.writerow(row)


'''
1. 算法流程：
 使用多目标优化算法对每一个数据集进行优化，并记录非支配集对应的【FPA, AAE,和对应的非零参数的个数】(具体自选)。
2. 记录信息
a. 文档1：记录每个数据集非支配集的平均【FPA, AAE, 对应的非零参数解】
b. 文档2：记录每个数据集非支配集的【FPA】
c. 文档3：记录每个数据集非支配集的【AAE】
d. 文档4：记录每个数据集非支配集的【对应的非零参数个数】
e. 文档5：记录每个数据集非支配集的【L1范数值】
f. 文档6：记录每个数据集非支配集的【MSE】
e. 文档(6+):对每个数据集,记录其获取的非支配集的参数
注：具体有几个文档，以及文档具体代表的含义，依据该多目标优化算法所优化的目标决定
target 的值
    0---FPA
    1---AAE
    2---numOfnonZero
    3---L1
    4---MSE
model--'linear', 'BPNN'
l:决策变量的下界
u:决策变量的上界
'''

def training_record_for_ssdp_m(save_folder, predict_model, l, u, soea):
    fileLists = dictionaries.get_filelists()
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    path = 'data/'
  
    doc1 = []
    doc1_names = ['filename']

    doc1_names.append(para_name[0])
    doc2 = []

    doc1_names.append(para_name[1])
    doc3 = []

    doc1_names.append(para_name[2])
    doc4 = []

    doc1_names.append(para_name[3])
    doc5 = []

    doc1_names.append(para_name[4])
    doc6 = []

    save_path = '../results/' + save_folder + '/train/'
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')
            X, y = getfeatures(path, fileLists[i][j])
            model = SSDP(X=X, y=y, model=predict_model, drawing=0, l=l, u=u, soea=soea)
            model.run()

            # 找出单目标的最优的f1, f2, f3
            best_gen = np.argmin(model.problem.maxormins * model.obj_trace[:, 1])
            best_objV = model.obj_trace[best_gen, 1]
            best_Chrom = []
            for tmp_index in range(model.population.ObjV.shape[0]):
                if model.population.ObjV[tmp_index][0] == best_objV:
                    best_Chrom.append(model.population.Chrom[tmp_index])
            best_Chrom = np.array(best_Chrom)

            if predict_model == 1:
                predvalue = tgf.linear_predict(X, best_Chrom)
            elif predict_model == 2:
                predvalue = tgf.bpnn_predict(X, best_Chrom)
            elif predict_model == 3:
                predvalue = tgf.nn_predict(X, best_Chrom)
            elif predict_model == 4:
                predvalue = tgf.mlp_predict(X, best_Chrom)
            elif predict_model == 5:
                predvalue = tgf.mlpn_predict(X,best_Chrom, 3)
            elif predict_model == 6:
                predvalue = tgf.mlpn_predict(X, best_Chrom, 5)
            else:
                print('error model method number in helpers !!!! ')

            f1 = tgf.FPA(predvalue, y)
            f1_set = list(f1)
            f1_set.insert(0, fileLists[i][j])
            doc2.append(f1_set)

            f2 = tgf.AAE(predvalue, y)
            f2_set = list(f2)
            f2_set.insert(0, fileLists[i][j])
            doc3.append(f2_set)

            f3 = tgf.numOfnonZero(best_Chrom)
            f3_set = list(f3)
            f3_set.insert(0, fileLists[i][j])
            doc4.append(f3_set)

            f4 = tgf.l1_values(best_Chrom)
            f4_set = list(f4)
            f4_set.insert(0, fileLists[i][j])
            doc5.append(f4_set)

            f5 = tgf.MSE(predvalue, y)
            f5_set = list(f5)
            f5_set.insert(0, fileLists[i][j])
            doc6.append(f5_set)

            params = pd.DataFrame(best_Chrom)
            params.to_csv(save_path + fileLists[i][j] + '.csv')
            
    with open(save_path + 'doc1.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        for row in doc1:
            writer1.writerow(row)

    with open(save_path + 'doc2.csv', 'w', newline='') as file2:
        writer2 = csv.writer(file2)
        for row in doc2:
            writer2.writerow(row)

    with open(save_path + 'doc3.csv', 'w', newline='') as file3:
        writer3 = csv.writer(file3)
        for row in doc3:
            writer3.writerow(row)

    with open(save_path + 'doc4.csv', 'w', newline='') as file4:
        writer4 = csv.writer(file4)
        for row in doc4:
            writer4.writerow(row)

    with open(save_path + 'doc5.csv', 'w', newline='') as file5:
        writer5 = csv.writer(file5)
        for row in doc5:
            writer5.writerow(row)

    with open(save_path + 'doc6.csv', 'w', newline='') as file6:
        writer6 = csv.writer(file6)
        for row in doc6:
            writer6.writerow(row)
def training_record_for_msdp(save_folder, target, predict_model, l, u, moea, drawing=0, maxgen=100):
    fileLists = dictionaries.get_filelists()
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    path = 'data/'

    doc1 = []
    doc1_names = ['filename']

    doc1_names.append(para_name[0])
    doc2 = []

    doc1_names.append(para_name[1])
    doc3 = []

    doc1_names.append(para_name[2])
    doc4 = []

    doc1_names.append(para_name[3])
    doc5 = []

    doc1_names.append(para_name[4])
    doc6 = []

    save_path = '../results/' + save_folder + '/train/'
   # save_path = '../results/多目标优化__FPA+AAE__NIND=100__MAX_GEN=100__NSGAII_决策变量范围_[-20,20]/train/'
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')
            X, y = getfeatures(path, fileLists[i][j])
            model = MSDP(X=X, y=y, target=target, model=predict_model, l=l, u=u, moea=moea,
                         maxgen=maxgen, drawing=drawing)
            model.run()

            if len(target) > 1:
                avgs = []
                sets = []
                for m in range(len(target)):
                    avg = np.mean(model.NDSet.ObjV[:, m])
                    avgs.append(avg)

                    set_t = model.NDSet.ObjV[:, m]
                    set_t = list(set_t)
                    set_t.insert(0, fileLists[i][j])

                    sets.append(set_t.copy())

                print('各目标值的平均值')
                for vlist in zip(target, avgs):
                    print(vlist[0], vlist[1])
                doc1.append([fileLists[i][j]] + avgs)

                # 目前总共要判断五个指标

                for tar_sets in zip(target, sets):
                    if tar_sets[0] == 0:
                        doc2.append(tar_sets[1])
                    elif tar_sets[0] == 1:
                        doc3.append(tar_sets[1])
                    elif tar_sets[0] == 2:
                        doc4.append(tar_sets[1])
                    elif tar_sets[0] == 3:
                        doc5.append(tar_sets[1])
                    elif tar_sets[0] == 4:
                        doc6.append(tar_sets[1])
                    else:
                        print('target error!!!')
                    print('非支配集的', para_name[tar_sets[0]], tar_sets[1])
                '''
                    0---FPA
                    1---AAE
                    2---numOfnonZero
                    3---L1
                    4---MSE
                '''
                if predict_model == 1:
                    predValue = tgf.linear_predict(X, model.NDSet.Chrom)
                elif predict_model == 2:
                    predValue = tgf.bpnn_predict(X, model.NDSet.Chrom)
                elif predict_model == 3:
                    predValue = tgf.nn_predict(X, model.NDSet.Chrom)
                elif predict_model == 4:
                    predValue = tgf.mlp_predict(X, model.NDSet.Chrom)
                elif predict_model == 5:
                    predValue = tgf.mlpn_predict(X, model.NDSet.Chrom, 3)
                elif predict_model == 6:
                    predValue = tgf.mlpn_predict(X, model.NDSet.Chrom, 5)
                else:
                    print('error predict model number in helpers!!')
                if 0 not in target:
                    f1 = tgf.FPA(predValue, y)
                    f1_set = list(f1)
                    f1_set.insert(0, fileLists[i][j])
                    doc2.append(f1_set)
                if 1 not in target:
                    f2 = tgf.AAE(predValue, y)
                    f2_set = list(f2)
                    f2_set.insert(0, fileLists[i][j])
                    doc3.append(f2_set)
                if 2 not in target:
                    f3 = tgf.numOfnonZero(model.NDSet.Chrom)
                    f3_set = list(f3)
                    f3_set.insert(0, fileLists[i][j])
                    doc4.append(f3_set)
                if 3 not in target:
                    f4 = tgf.l1_values(model.NDSet.Chrom)
                    f4_set = list(f4)
                    f4_set.insert(0, fileLists[i][j])
                    doc5.append(f4_set)
                if 4 not in target:
                    f5 = tgf.MSE(predValue, y)
                    f5_set = list(f5)
                    f5_set.insert(0, fileLists[i][j])
                    doc6.append(f5_set)

            #print('平均FPA, 对应非零参数个数', avg_fpa, avg_aae)

           # print('平均FPA, AAE, 非零参数的个数', avg_fpa, avg_nonp)
            #doc1.append([fileLists[i][j], avg_fpa, avg_nonp])

            params = pd.DataFrame(model.NDSet.Chrom)
            params.to_csv(save_path + fileLists[i][j] + '.csv')

    with open(save_path + 'doc1.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        for row in doc1:
            writer1.writerow(row)

    with open(save_path + 'doc2.csv', 'w', newline='') as file2:
        writer2 = csv.writer(file2)
        for row in doc2:
            writer2.writerow(row)

    with open(save_path + 'doc3.csv', 'w', newline='') as file3:
        writer3 = csv.writer(file3)
        for row in doc3:
            writer3.writerow(row)

    with open(save_path + 'doc4.csv', 'w', newline='') as file4:
        writer4 = csv.writer(file4)
        for row in doc4:
            writer4.writerow(row)

    with open(save_path + 'doc5.csv', 'w', newline='') as file5:
        writer5 = csv.writer(file5)
        for row in doc5:
            writer5.writerow(row)

    with open(save_path + 'doc6.csv', 'w', newline='') as file6:
        writer6 = csv.writer(file6)
        for row in doc6:
            writer6.writerow(row)


# 分别使用单目标优化算法和多目标优化算法运行数据集，并记录单目标的最优解和多目标的非支配集，以及平均FPA和AAE
def idea2_with_plotpic():
    fileLists = dictionaries.get_filelists()
    path = 'data/'
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            X, y = getfeatures(path, fileLists[i][j])
            # 分别运行两个模型——单目标优化和多目标优化
            model_msdp = MSDP(X, y)
            model_msdp.run()
            model_ssdp = SSDP(X, y)
            model_ssdp.run()

            # 求出多目标的值的平均值
            avg_msdp_fpa = np.mean(model_msdp.NDSet.ObjV[:, 0])
            avg_msdp_aae = np.mean(model_msdp.NDSet.ObjV[:, 1])

            print('多目标的fpa, aae值分别为:')
            print(model_msdp.NDSet.ObjV[:, 0])
            print(model_msdp.NDSet.ObjV[:, 1])
            print('多目标的fpa, aae平均值值分别为:', avg_msdp_fpa, avg_msdp_aae)

            # 找出单目标的最优的f1, f2
            best_gen = np.argmin(model_ssdp.problem.maxormins * model_ssdp.obj_trace[:, 1])
            best_objV = model_ssdp.obj_trace[best_gen, 1]
            ss_max_parameter = model_ssdp.var_trace[best_gen]
            predvalue = tgf.linear_predict(X, ss_max_parameter)

            f1 = tgf.FPA(predvalue, y)
            f2 = tgf.AAE(predvalue, y)

            print('单目标的fpa, aae值分别为:', f1, f2)

            # 描绘出多目标优化的f1, f2图
            plt.figure(figsize=(8, 4))
            plt.scatter(model_msdp.NDSet.ObjV[:, 0], model_msdp.NDSet.ObjV[:, 1],
                        color='green', label='msdp')
            # 描绘出单目标的点
            plt.scatter(f1, f2, color='red', label='ssdp')

            plt.title(fileLists[i][j])
            plt.xlabel("FPA")
            plt.ylabel("AAE")
            plt.savefig(fileLists[i][j] + '.csv')
            plt.show()


'''
利用训练集训练得到的，已经存储下来的模型参数，使用测试集进行测试。
1. 对于单目标算法,设置为type1:
    1.a 首先读取每个文件名对应的参数,(e. 文档5+：对每个数据集，记录其迭代的每一代的最优参数),
    然后使用下一个数据集进行测试，此时计算的是每一代的最优模型的性能。
    -------->生成文件<test_file>.csv
    1.b.读取doc4.csv中对应的参数(d. 文档4：记录每个数据集最优解的参数)，此时记录的是每个数据集的单目标优化的最有模型的性能。
    -------->生成文件"doc1.csv"
2. 对于多目标算法，设置为type2：
    2.a首先读取每个文件名对应的参数,(e. 文档(5+):对每个数据集,记录其获取的非支配集的参数),
    然后使用下一个数据集进行测试，此时计算的是非支配集的性能。
    -------->生成文件<test_file>.csv
'''


def test_model(folder_name, predict_model, type):
    fileLists = dictionaries.get_filelists()
    path = 'data/'
    # 1. 修改params_path,来修改获取参数的地址
    # 2. 修改save_path, 来修改存储结果的地址
    folder_name = folder_name
    params_path = '../results/' + folder_name + '/train/'
    save_path = '../results/' + folder_name + '/test/'

    # 对于单目标算法，需要加一个测试最优模型的文档
    if type == 1:
        best_params = pd.read_csv(params_path + 'doc4.csv', index_col=0, header=None)
        doc1 = [['filename', 'FPA', 'AAE', 'numOfnonZero', 'L1', 'MSE']]

    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i]) - 1):
            test_results = [['filename', 'FPA', 'AAE', 'numOfnonZero', 'L1', 'MSE']]
            params = pd.read_csv(params_path + fileLists[i][j] + '.csv', index_col=0, header=0)
            params = params.values
            X, y = getfeatures(path, fileLists[i][j + 1])
            for t in range(params.shape[0]):
                if predict_model == 1:
                    predValue = tgf.linear_predict(X, params[t])
                elif predict_model == 2:
                    predValue = tgf.bpnn_predict(X, params[t])
                elif predict_model == 3:
                    predValue = tgf.nn_predict(X, params[t])
                elif predict_model == 4:
                    predValue = tgf.mlp_predict(X, params[t])
                elif predict_model == 5:
                    predValue = tgf.mlpn_predict(X, params[t], 3)
                elif predict_model == 6:
                    predValue = tgf.mlpn_predict(X, params[t], 5)
                else:
                    print('error predict model number in helpers!!!')
                f1 = tgf.FPA(predValue, y)
                f2 = tgf.AAE(predValue, y)
                f3 = tgf.numOfnonZero(params[t])
                f4 = tgf.l1_values(params[t])
                f5 = tgf.MSE(predValue, y)
                test_results.append([fileLists[i][j + 1], f1, f2, f3, f4, f5])
            test_results = pd.DataFrame(test_results)
            print('save', fileLists[i][j + 1])
            test_results.to_csv(save_path + fileLists[i][j + 1] + '.csv')

            if type == 1:
                best_param = best_params.loc[fileLists[i][j]].values
                if predict_model == 1:

                    predValue = tgf.linear_predict(X, best_param)
                elif predict_model == 2:
                    predValue = tgf.bpnn_predict(X, best_param)
                elif predict_model == 3:
                    predValue = tgf.nn_predict(X, best_param)
                elif predict_model == 4:
                    predValue = tgf.mlp_predict(X, best_param)
                elif predict_model == 5:
                    predValue = tgf.mlpn_predict(X, best_param, 3)
                elif predict_model == 6:
                    predValue = tgf.mlpn_predict(X, best_param, 5)
                else:
                    print('error predict model number')

                f1 = tgf.FPA(predValue, y)
                f2 = tgf.AAE(predValue, y)
                f3 = tgf.numOfnonZero(best_param)
                f4 = tgf.l1_values(best_param)
                f5 = tgf.MSE(predValue, y)
                doc1.append([fileLists[i][j + 1], f1, f2, f3, f4, f5])
    if type == 1:
        with open(save_path + 'doc1.csv', 'w', newline='') as file1:
            writer1 = csv.writer(file1)
            for row in doc1:
                writer1.writerow(row)


'''
1. 算法流程：
 使用多目标优化算法对每一个数据集进行优化，并记录非支配集对应的【FPA, AAE,和对应的非零参数的个数】(具体自选)。
2. 记录信息
【train/文件】
a. 文档1：记录每个数据集非支配集的测试最高的【FPA, AAE, 对应的非零参数解】
b. 文档2：记录每个数据集非支配集的【FPA】
c. 文档3：记录每个数据集非支配集的【AAE】
d. 文档4：记录每个数据集非支配集的【对应的非零参数个数】
e. 文档5：记录每个数据集非支配集的【L1范数值】
f. 文档6：记录每个数据集非支配集的【MSE】
e. 文档(6+):对每个数据集,记录其获取的非支配集的参数
注：具体有几个文档，以及文档具体代表的含义，依据该多目标优化算法所优化的目标决定
【validation/文件】
根据训练集训练出来的非支配集，进行测试。
----------->生成文件<test_file>.csv
target 的值
    0---FPA
    1---AAE
    2---numOfnonZero
    3---L1
    4---MSE
model--'linear', 'BPNN'
l:决策变量的下界
u:决策变量的上界
'''


def train_validation_for_msdp(save_folder, target, predict_model, l, u, moea, validation_size, drawing=0, maxgen=100):
    fileLists = dictionaries.get_filelists()
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    path = 'data/'

    doc1 = []
    doc1_names = ['filename']

    doc1_names.append(para_name[0])
    doc2 = []

    doc1_names.append(para_name[1])
    doc3 = []

    doc1_names.append(para_name[2])
    doc4 = []

    doc1_names.append(para_name[3])
    doc5 = []

    doc1_names.append(para_name[4])
    doc6 = []

    train_save_path = '../results/' + save_folder + '/train/'
    validation_save_path = '../results/' + save_folder + '/validation/'
    # save_path = '../results/多目标优化__FPA+AAE__NIND=100__MAX_GEN=100__NSGAII_决策变量范围_[-20,20]/train/'
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')
            X, y = getfeatures(path, fileLists[i][j])
            X_train, X_validation, y_train, y_validation = train_test_split(
                X, y, test_size=validation_size, random_state=1)
            model = MSDP(X=X_train, y=y_train, target=target, model=predict_model, l=l, u=u, moea=moea,
                         maxgen=maxgen, drawing=drawing)
            model.run()

            if len(target) > 1:
                avgs = []
                sets = []
                for m in range(len(target)):
                    avg = np.mean(model.NDSet.ObjV[:, m])
                    avgs.append(avg)

                    set_t = model.NDSet.ObjV[:, m]
                    set_t = list(set_t)
                    set_t.insert(0, fileLists[i][j])

                    sets.append(set_t.copy())

                print('各目标值的平均值')
                for vlist in zip(target, avgs):
                    print(vlist[0], vlist[1])
                doc1.append([fileLists[i][j]] + avgs)

                # 目前总共要判断五个指标

                for tar_sets in zip(target, sets):
                    if tar_sets[0] == 0:
                        doc2.append(tar_sets[1])
                    elif tar_sets[0] == 1:
                        doc3.append(tar_sets[1])
                    elif tar_sets[0] == 2:
                        doc4.append(tar_sets[1])
                    elif tar_sets[0] == 3:
                        doc5.append(tar_sets[1])
                    elif tar_sets[0] == 4:
                        doc6.append(tar_sets[1])
                    else:
                        print('target error!!!')
                    print('非支配集的', para_name[tar_sets[0]], tar_sets[1])
                '''
                    0---FPA
                    1---AAE
                    2---numOfnonZero
                    3---L1
                    4---MSE
                '''
                if predict_model == 1:
                    predValue = tgf.linear_predict(X_train, model.NDSet.Chrom)
                elif predict_model == 2:
                    predValue = tgf.bpnn_predict(X_train, model.NDSet.Chrom)
                elif predict_model == 3:
                    predValue = tgf.nn_predict(X_train, model.NDSet.Chrom)
                elif predict_model == 4:
                    predValue = tgf.mlp_predict(X_train, model.NDSet.Chrom)
                elif predict_model == 5:
                    predValue = tgf.mlpn_predict(X_train, model.NDSet.Chrom, 3)
                elif predict_model == 6:
                    predValue = tgf.mlpn_predict(X_train, model.NDSet.Chrom, 5)
                else:
                    print('error predict model number in helpers!!')
                if 0 not in target:
                    f1 = tgf.FPA(predValue, y_train)
                    f1_set = list(f1)
                    f1_set.insert(0, fileLists[i][j])
                    doc2.append(f1_set)
                if 1 not in target:
                    f2 = tgf.AAE(predValue, y_train)
                    f2_set = list(f2)
                    f2_set.insert(0, fileLists[i][j])
                    doc3.append(f2_set)
                if 2 not in target:
                    f3 = tgf.numOfnonZero(model.NDSet.Chrom)
                    f3_set = list(f3)
                    f3_set.insert(0, fileLists[i][j])
                    doc4.append(f3_set)
                if 3 not in target:
                    f4 = tgf.l1_values(model.NDSet.Chrom)
                    f4_set = list(f4)
                    f4_set.insert(0, fileLists[i][j])
                    doc5.append(f4_set)
                if 4 not in target:
                    f5 = tgf.MSE(predValue, y_train)
                    f5_set = list(f5)
                    f5_set.insert(0, fileLists[i][j])
                    doc6.append(f5_set)

            # print('平均FPA, 对应非零参数个数', avg_fpa, avg_aae)

            # print('平均FPA, AAE, 非零参数的个数', avg_fpa, avg_nonp)
            # doc1.append([fileLists[i][j], avg_fpa, avg_nonp])

            params = pd.DataFrame(model.NDSet.Chrom)
            params.to_csv(train_save_path + fileLists[i][j] + '.csv')

            validation_results = [['filename', 'FPA', 'AAE', 'numOfnonZero', 'L1', 'MSE']]
            params = params.values

            for t in range(params.shape[0]):
                if predict_model == 1:
                    v_predValue = tgf.linear_predict(X_validation, params[t])
                elif predict_model == 2:
                    v_predValue = tgf.bpnn_predict(X_validation, params[t])
                elif predict_model == 3:
                    v_predValue = tgf.nn_predict(X_validation, params[t])
                elif predict_model == 4:
                    v_predValue = tgf.mlp_predict(X_validation, params[t])
                elif predict_model == 5:
                    v_predValue = tgf.mlpn_predict(X_validation, params[t], 3)
                elif predict_model == 6:
                    v_predValue =tgf.mlpn_predict(X_validation, params[t], 5)
                else:
                    print('error predict model number in helpers!!!')
                v_f1 = tgf.FPA(v_predValue, y_validation)
                v_f2 = tgf.AAE(v_predValue, y_validation)
                v_f3 = tgf.numOfnonZero(params[t])
                v_f4 = tgf.l1_values(params[t])
                v_f5 = tgf.MSE(v_predValue, y_validation)
                validation_results.append([fileLists[i][j], v_f1, v_f2, v_f3, v_f4, v_f5])
            validation_results = pd.DataFrame(validation_results)
            print('save_validation:', fileLists[i][j])
            validation_results.to_csv(validation_save_path + fileLists[i][j] + '.csv')

    with open(train_save_path + 'doc1.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        for row in doc1:
            writer1.writerow(row)

    with open(train_save_path + 'doc2.csv', 'w', newline='') as file2:
        writer2 = csv.writer(file2)
        for row in doc2:
            writer2.writerow(row)

    with open(train_save_path + 'doc3.csv', 'w', newline='') as file3:
        writer3 = csv.writer(file3)
        for row in doc3:
            writer3.writerow(row)

    with open(train_save_path + 'doc4.csv', 'w', newline='') as file4:
        writer4 = csv.writer(file4)
        for row in doc4:
            writer4.writerow(row)

    with open(train_save_path + 'doc5.csv', 'w', newline='') as file5:
        writer5 = csv.writer(file5)
        for row in doc5:
            writer5.writerow(row)

    with open(train_save_path + 'doc6.csv', 'w', newline='') as file6:
        writer6 = csv.writer(file6)
        for row in doc6:
            writer6.writerow(row)


'''
1. 算法流程：
 使用单目标优化CoDE算法对每一个数据集进行优化，并记录最优解和最优解对应的参数，FPA, AAE(因为非零参数没有意义), MSE(优化的目标是MSE)
2. 记录信息
a. 文档1：记录每个数据集最优解的【FPA, AAE， MSE】
b. 文档2：记录每个数据集进化的每一代的【种群平均目标函数值】
c. 文档3：记录每个数据集进化的每一代的【种群最优目标函数值】
d. 文档4：记录每个数据集最优解的参数
e. 文档5+：对每个数据集，记录其迭代的每一代的最优参数
'''
# folder = '单目标优化__BPNN_NIND=100__MAX_GEN=100__CoDE_决策变量范围_[-1,1]'


def train_for_bpnn(save_folder):
    fileLists = dictionaries.get_filelists()
    path = 'data/'
    save_path = '../results/' + save_folder + '/train/'
    doc1 = [['filename', 'FPA', 'AAE', 'MSE']]
    doc2 = []
    doc3 = []
    doc4 = []
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')
            X, y = getfeatures(path, fileLists[i][j])
            model = SSDP(X, y)
            model.run()

            # 找出单目标的最优的f1, f2, f3
            best_gen = np.argmin(model.problem.maxormins * model.obj_trace[:, 1])
            best_objV = model.obj_trace[best_gen, 1]

            ss_max_parameter = model.var_trace[best_gen]

            predvalue = tgf.bpnn_predict(X, y, ss_max_parameter)
            f1 = tgf.FPA(predvalue, y)
            f2 = tgf.AAE(predvalue, y)
            f3 = mean_squared_error(y, predvalue)
            doc1.append([fileLists[i][j], f1, f2, f3])  # FPA, AAE, 和对应的非零参数解

            avg_list = model.obj_trace[:, 0]
            avg_list = list(avg_list)
            avg_list.insert(0, fileLists[i][j])
            doc2.append(avg_list)  # 种群平均目标函数值

            best_list = model.obj_trace[:, 1]
            best_list = list(best_list)
            best_list.insert(0, fileLists[i][j])
            doc3.append(best_list)  # 种群最优目标函数值

            best_param = ss_max_parameter
            best_param = list(best_param)
            best_param.insert(0, fileLists[i][j])
            doc4.append(best_param)  # 种群最优个体的参数

            params = pd.DataFrame(model.var_trace)
            params.to_csv(save_path + fileLists[i][j] + '.csv')

    with open(save_path + 'doc1.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        for row in doc1:
            writer1.writerow(row)

    with open(save_path + 'doc2.csv', 'w', newline='') as file2:
        writer2 = csv.writer(file2)
        for row in doc2:
            writer2.writerow(row)

    with open(save_path + 'doc3.csv', 'w', newline='') as file3:
        writer3 = csv.writer(file3)
        for row in doc3:
            writer3.writerow(row)

    with open(save_path + 'doc4.csv', 'w', newline='') as file4:
        writer4 = csv.writer(file4)
        for row in doc4:
            writer4.writerow(row)


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def comparison_sm_train(single_path, multi_path, parameters, single_name, multi_name, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_file = pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0)
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    multi_files = []
    if 0 in parameters:
        file1 = open(multi_path + 'doc2.csv', 'r')
        multi_files.append(file1)
    if 1 in parameters:
        file2 = open(multi_path + 'doc3.csv', 'r')
        multi_files.append(file2)
    if 2 in parameters:
        file3 = open(multi_path + 'doc4.csv', 'r')
        multi_files.append(file3)
    if len(parameters) == 2:
        reader1 = csv.reader(multi_files[0])
        reader2 = csv.reader(multi_files[1])

        mdata1 = list(reader1)
        mdata2 = list(reader2)

        for i in range(len(mdata1)):
            assert mdata1[i][0] == mdata2[i][0]

            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            single_values = single_file.loc[mdata1[i][0]].values.astype('float64').tolist()

            plt.scatter(single_values[parameters[0]], single_values[parameters[1]],
                        color='red', label=single_name)
            #

            # plt.axvline(x = single_values[-1],
            #             color='red', label=single_name, linestyle = '-')
            m_x = list(map(lambda x: float(x), mdata1[i][1:]))
            m_y = list(map(lambda x: float(x), mdata2[i][1:]))
            plt.scatter(m_x, m_y, color='green', label=multi_name)
            plt.legend()
            plt.savefig(save_path + mdata1[i][0] + '.png')
            plt.close()


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def comparison_sm_test(single_path, multi_path, parameters, single_name, multi_name, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_file = pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0)
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)

            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            plt.scatter(single_file.loc[fileLists[i][j], para_name[parameters[0]]], single_file.loc[fileLists[i][j],
                                                                                                    para_name[parameters[1]]],
                        color='red', label=single_name)
            #

            # plt.axvline(x = single_values[-1],
            #             color='red', label=single_name, linestyle = '-')
            plt.scatter(m_file[para_name[parameters[0]]].values,
                        m_file[para_name[parameters[1]]].values, color='green', label=multi_name)
            plt.legend()
            plt.savefig(save_path + fileLists[i][j] + '.png')
            plt.close()


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集和测试集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def comparison_sm_train_test(single_path, multi_path, parameters, single_name, multi_name, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。

    # 设置单目标，多目标训练集测试集的读取路径
    single_train_path = single_path + 'train/'
    single_test_path = single_path + 'test/'
    multi_train_path = multi_path + 'train/'
    multi_test_path = multi_path + 'test/'

    # 首先读取单目标的训练集和测试集文件
    single_train_file = pd.read_csv(single_train_path + 'doc1.csv', header=0, index_col=0)
    single_test_file = pd.read_csv(single_test_path + 'doc1.csv', header=0, index_col=0)

    # 读取多目标的训练集的文件
    multi_train_files = []
    if 0 in parameters:
        file1 = open(multi_train_path + 'doc2.csv', 'r')
        multi_train_files.append(file1)
    if 1 in parameters:
        file2 = open(multi_train_path + 'doc3.csv', 'r')
        multi_train_files.append(file2)
    if 2 in parameters:
        file3 = open(multi_train_path + 'doc4.csv', 'r')
        multi_train_files.append(file3)
    if len(parameters) == 2:
        multi_train_reader1 = csv.reader(multi_train_files[0])
        multi_train_reader2 = csv.reader(multi_train_files[1])

        train_mdata1 = list(multi_train_reader1)
        train_mdata2 = list(multi_train_reader2)

        for i in range(len(train_mdata1)):
            assert train_mdata1[i][0] == train_mdata2[i][0]

            # 读取多目标的测试集-------首先查看是否存在该测试集，不存在的不需要画图, 则跳过
            if not Path(multi_test_path + train_mdata1[i][0] + '.csv').exists():
                continue

            test_mfile = pd.read_csv(multi_test_path + train_mdata1[i][0] + '.csv', header=1, index_col=1)

            # 绘图基本设置
            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            # 绘制单目标-训练集
            single_train_values = single_train_file.loc[train_mdata1[i][0]].values.astype('float64').tolist()
            plt.scatter(single_train_values[parameters[0]], single_train_values[parameters[1]],
                        color='red', label=single_name + '/train')

            # 绘制单目标-测试集
            plt.scatter(single_test_file.loc[train_mdata1[i][0], para_name[parameters[0]]],
                        single_test_file.loc[train_mdata1[i][0], para_name[parameters[1]]],
                        color='green', label=single_name + '/test')

            # 绘制多目标-训练集
            train_m_x = list(map(lambda x: float(x), train_mdata1[i][1:]))
            train_m_y = list(map(lambda x: float(x), train_mdata2[i][1:]))
            plt.scatter(train_m_x, train_m_y, color='blue', label=multi_name + '/train')

            # 绘制多目标-测试集
            plt.scatter(test_mfile[para_name[parameters[0]]].values, test_mfile[para_name[parameters[1]]].values,
                        color='yellow', label=multi_name + '/test')

            plt.legend()
            plt.savefig(save_path + train_mdata1[i][0] + '.png')
            plt.close()
    elif len(parameters) == 3:
        print('three parameters functions not be completed!!')
    else:
        print('很奇怪！！！！！')


def comparison_mm_train(path1, path2, name1, name2, parameters, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero'}
    m_files1 = []
    m_files2 = []
    if 0 in parameters:
        f11 = open(path1 + 'doc2.csv', 'r')
        m_files1.append(f11)

        f21 = open(path2 + 'doc2.csv', 'r')
        m_files2.append(f21)
    if 1 in parameters:
        f12 = open(path1 + 'doc3.csv', 'r')
        m_files1.append(f12)

        f22 = open(path2 + 'doc3.csv', 'r')
        m_files2.append(f22)
    if 2 in parameters:
        f13 = open(path1 + 'doc4.csv', 'r')
        m_files1.append(f13)

        f23 = open(path2 + 'doc4.csv', 'r')
        m_files2.append(f23)
    if len(parameters) == 2:
        reader11 = csv.reader(m_files1[0])
        reader12 = csv.reader(m_files1[1])

        reader21 = csv.reader(m_files2[0])
        reader22 = csv.reader(m_files2[1])

        mdata11 = list(reader11)
        mdata12 = list(reader12)

        mdata21 = list(reader21)
        mdata22 = list(reader22)

        for i in range(len(mdata11)):
            assert mdata11[i][0] == mdata12[i][0]
            assert mdata21[i][0] == mdata22[i][0]
            assert mdata11[i][0] == mdata21[i][0]

            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            m1_x = list(map(lambda x: float(x), mdata11[i][1:]))
            m1_y = list(map(lambda x: float(x), mdata12[i][1:]))
            plt.scatter(m1_x, m1_y, color='red', label=name1)

            m2_x = list(map(lambda x: float(x), mdata21[i][1:]))
            m2_y = list(map(lambda x: float(x), mdata22[i][1:]))
            plt.scatter(m2_x, m2_y, color='green', label=name2)

            plt.legend()
            plt.savefig(save_path + mdata11[i][0] + '.png')
            plt.close()

    elif len(parameters) == 3:
        print('three parameters functions not be completed!!')
    else:
        print('很奇怪！！！！！')


def comparison_mm_test(path1, path2, name1, name2, parameters, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero'}

    fileLists = dictionaries.get_filelists()

    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            file1 = pd.read_csv(path1 + fileLists[i][j] + '.csv', header=1, index_col=1)
            file2 = pd.read_csv(path2 + fileLists[i][j] + '.csv', header=1, index_col=1)

            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            plt.scatter(file1[para_name[parameters[0]]].values, file1[para_name[parameters[1]]].values, color='red',
                        label=name1)
            plt.scatter(file2[para_name[parameters[0]]].values, file2[para_name[parameters[1]]].values, color='green',
                        label=name2)

            plt.legend()
            plt.savefig(save_path + fileLists[i][j] + '.png')
            plt.close()


def comparison_mm_train_test(path1, path2, name1, name2, parameters, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero'}

    train_path1 = path1 + 'train/'
    test_path1 = path1 + 'test/'

    train_path2 = path2 + 'train/'
    test_path2 = path2 + 'test/'

    files1 = []
    files2 = []
    if 0 in parameters:
        f11 = open(train_path1 + 'doc2.csv', 'r')
        f21 = open(train_path2 + 'doc2.csv', 'r')
        files1.append(f11)
        files2.append(f21)
    if 1 in parameters:
        f12 = open(train_path1 + 'doc3.csv', 'r')
        f22 = open(train_path2 + 'doc3.csv', 'r')
        files1.append(f12)
        files2.append(f22)
    if 2 in parameters:
        f13 = open(train_path1 + 'doc4.csv', 'r')
        f23 = open(train_path2 + 'doc4.csv', 'r')
        files1.append(f13)
        files2.append(f23)

    if len(parameters) == 2:
        reader11 = csv.reader(files1[0])
        reader12 = csv.reader(files1[1])

        reader21 = csv.reader(files2[0])
        reader22 = csv.reader(files2[1])

        train_data11 = list(reader11)
        train_data12 = list(reader12)

        train_data21 = list(reader21)
        train_data22 = list(reader22)

        for i in range(len(train_data11)):
            assert train_data11[i][0] == train_data12[i][0]
            assert train_data21[i][0] == train_data22[i][0]
            assert train_data11[i][0] == train_data21[i][0]

            if not Path(test_path1 + train_data11[i][0] + '.csv').exists():
                continue
            if not Path(test_path2 + train_data11[i][0] + '.csv').exists():
                continue

            test_file1 = pd.read_csv(test_path1 + train_data11[i][0] + '.csv', header=1, index_col=1)
            test_file2 = pd.read_csv(test_path2 + train_data11[i][0] + '.csv', header=1, index_col=1)

            # 绘图基本设置
            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            # 绘制多目标-训练集
            train1_x = list(map(lambda x: float(x), train_data11[i][1:]))
            train1_y = list(map(lambda x: float(x), train_data12[i][1:]))
            plt.scatter(train1_x, train1_y, color='blue', label=name1 + '/train')

            train2_x = list(map(lambda x: float(x), train_data21[i][1:]))
            train2_y = list(map(lambda x: float(x), train_data22[i][1:]))
            plt.scatter(train2_x, train2_y, color='green', label=name2 + '/train')

            # 绘制多目标-测试集
            plt.scatter(test_file1[para_name[parameters[0]]].values, test_file1[para_name[parameters[1]]].values,
                        color='yellow', label=name1 + '/test')

            plt.scatter(test_file2[para_name[parameters[0]]].values, test_file2[para_name[parameters[1]]].values,
                        color='red', label=name2 + '/test')

            plt.legend()
            plt.savefig(save_path + train_data11[i][0] + '.png')
            plt.close()

    elif len(parameters) == 3:
        print('three parameters functions not be completed!!')
    else:
        print('很奇怪！！！！！')


'''
简介：
    1. 该函数用于多个单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def get_m_files(path, parameters):
    files = []
    if 0 in parameters:
        file1 = open(path + 'doc2.csv', 'r')
        files.append(file1)
    if 1 in parameters:
        file2 = open(path + 'doc3.csv', 'r')
        files.append(file2)
    if 2 in parameters:
        file3 = open(path + 'doc4.csv', 'r')
        files.append(file3)
    if 3 in parameters:
        file4 = open(path + 'doc5.csv', 'r')
        files.append(file4)
    if 4 in parameters:
        file5 = open(path + 'doc6.csv', 'r')
        files.append(file5)
    return files


def comparison_difcolor_ssmm_train(single_paths, multi_paths, parameters, single_names, multi_names, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}

    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))

    multi_files = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files.append(get_m_files(multi_path, parameters))

    if len(parameters) == 2:
        if len(multi_files) == 0:

            fileLists = dictionaries.get_filelists()
            for i in range(len(fileLists)):
                for j in range(1, len(fileLists[i])):
                    color = 0
                    plt.figure(figsize=(8, 4))
                    plt.xlabel(para_name[parameters[0]])
                    plt.ylabel(para_name[parameters[1]])
                    for single_file, single_name in zip(single_files, single_names):
                        if single_name == 'CoDE':
                            single_name = 'learning-to-rank'
                        single_values = single_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                        plt.scatter(single_values[parameters[0]], single_values[parameters[1]],
                                    color=color_dict[color], label=single_name)
                        color += 1

                    plt.title(fileLists[i][j] + '_' + fileLists[i][j])
                    plt.legend()
                    plt.savefig(save_path + fileLists[i][j] + '.png')
                    plt.close()

        else:
            readers1 = []
            readers2 = []
            # readers1 记录的是维度1, 算法个数 *文件个数 * 值
            # readers2 记录的是维度2，算法个数 *文件个数 * 值
            for multi_file in multi_files:
                readers1.append(csv.reader(multi_file[0]))
                readers2.append(csv.reader(multi_file[1]))

            # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
            mdatas1 = []
            # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
            mdatas2 = []
            for reader in zip(readers1, readers2):
                mdatas1.append(list(reader[0]))
                mdatas2.append(list(reader[1]))

            for i in range(len(mdatas1[0])):
                color = 0
                for t in range(len(mdatas1)):
                    assert mdatas1[t][i][0] == mdatas2[0][i][0]

                plt.figure(figsize=(8, 4))
                plt.xlabel(para_name[parameters[0]])
                plt.ylabel(para_name[parameters[1]])

                for mdata1, mdata2, multi_name in zip(mdatas1, mdatas2, multi_names):
                    m_x = list(map(lambda x: float(x), mdata1[i][1:]))
                    m_y = list(map(lambda x: float(x), mdata2[i][1:]))
                    plt.scatter(m_x, m_y, color=color_dict[color], label=multi_name)
                    color += 1

                # plt.axvline(x = single_values[-1],
                #             color='red', label=single_name, linestyle = '-')
                for single_file, single_name in zip(single_files, single_names):
                    if single_name == 'CoDE':
                        single_name = 'learning-to-rank'
                    single_values = single_file.loc[mdatas2[0][i][0]].values.astype('float64').tolist()
                    plt.scatter(single_values[parameters[0]], single_values[parameters[1]],
                                color=color_dict[color], label=single_name)
                    color += 1

                plt.title(mdatas1[0][i][0] + '_' + mdatas1[0][i][0])
                plt.legend()
                plt.savefig(save_path + mdatas1[0][i][0] + '.png')
                plt.close()
    elif len(parameters) == 3:
        pass


def comparison_difmarker_ssmm_train(single_paths, multi_paths, parameters, single_names, multi_names, save_path, msize, wsize,figsize, if_show_label=True):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}

    multi_marker_dict = {0: '.', 1: '+', 2: '^', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: 'o', 2: 'x', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))

    multi_files = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files.append(get_m_files(multi_path, parameters))

    if len(parameters) == 2:

        if len(multi_files) == 0:
            fileLists = dictionaries.get_filelists()
            for i in range(len(fileLists)):
                for j in range(1, fileLists[i]):
                    single_marker = 0
                    plt.figure(figsize=(figsize[0], figsize[1]))
                    plt.xlabel(para_name[parameters[0]], fontdict={'size': wsize[0]})
                    plt.ylabel(para_name[parameters[1]], fontdict={'size': wsize[0]})
                    for single_file, single_name in zip(single_files, single_names):
                        if single_name == 'CoDE':
                            single_name = 'learning-to-rank'
                        single_values = single_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                        if if_show_label:
                            plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                     single_marker_dict[single_marker], label=single_name, markersize=msize[0])
                        else:
                            plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                     single_marker_dict[single_marker], markersize=msize[0])
                            print(single_marker_dict[single_marker], single_name)
                        single_marker += 1
                    plt.title(fileLists[i][j] + '_' + fileLists[i][j], fontdict={'size': wsize[1]})
                    if if_show_label:
                        plt.legend(prop={'size': wsize[2]})
                    plt.tick_params(labelsize=wsize[3])
                    plt.savefig(save_path + fileLists[i][j] + '.png')
                    plt.close()

        else:
            readers1 = []
            readers2 = []
            # readers1 记录的是维度1, 算法个数 *文件个数 * 值
            # readers2 记录的是维度2，算法个数 *文件个数 * 值
            for multi_file in multi_files:
                readers1.append(csv.reader(multi_file[0]))
                readers2.append(csv.reader(multi_file[1]))

            # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
            mdatas1 = []
            # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
            mdatas2 = []
            for reader in zip(readers1, readers2):
                mdatas1.append(list(reader[0]))
                mdatas2.append(list(reader[1]))

            for i in range(len(mdatas1[0])):
                multi_marker = 0
                single_marker = 0
                for t in range(len(mdatas1)):
                    assert mdatas1[t][i][0] == mdatas2[0][i][0]

                plt.figure(figsize=(figsize[0], figsize[1]))
                plt.xlabel(para_name[parameters[0]], fontdict={'size': wsize[0]})
                plt.ylabel(para_name[parameters[1]], fontdict={'size': wsize[0]})

                for mdata1, mdata2, multi_name in zip(mdatas1, mdatas2, multi_names):
                    if multi_name == 'CoDE':
                        multi_name = 'learning-to-rank'
                    m_x = list(map(lambda x: float(x), mdata1[i][1:]))
                    m_y = list(map(lambda x: float(x), mdata2[i][1:]))
                    if if_show_label:
                        plt.plot(m_x, m_y, multi_marker_dict[multi_marker], label=multi_name, markersize=msize[1])
                    else:
                        plt.plot(m_x, m_y, multi_marker_dict[multi_marker], markersize=msize[1])
                        print(multi_marker_dict[multi_marker], multi_name)
                    multi_marker += 1

                # plt.axvline(x = single_values[-1],
                #             color='red', label=single_name, linestyle = '-')
                for single_file, single_name in zip(single_files, single_names):
                    if single_name == 'CoDE':
                        single_name = 'learning-to-rank'
                    single_values = single_file.loc[mdatas2[0][i][0]].values.astype('float64').tolist()
                    if if_show_label:
                        plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                 single_marker_dict[single_marker], label=single_name, markersize=msize[0])
                    else:
                        plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                 single_marker_dict[single_marker], markersize=msize[0])
                        print(single_marker_dict[single_marker], single_name)
                    single_marker += 1
                plt.title(mdatas1[0][i][0] + '_' + mdatas1[0][i][0], fontdict={'size': wsize[1]})
                if if_show_label:
                    plt.legend(prop={'size': wsize[2]})
                plt.tick_params(labelsize=wsize[3])
                plt.savefig(save_path + mdatas1[0][i][0] + '.png')
                plt.close()
    elif len(parameters) == 3:
        pass


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def comparison_difmarker_ssmm_test(single_paths, multi_paths, parameters, single_names, multi_names, save_path, wsize, msize,figsize, if_show_label=True):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}
    multi_marker_dict = {0: '.', 1: '+', 2: '^', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: 'o', 2: 'x', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            plt.figure(figsize=(figsize[0], figsize[1]))
            plt.xlabel(para_name[parameters[0]], fontdict={'size': wsize[0]})
            plt.ylabel(para_name[parameters[1]], fontdict={'size': wsize[0]})

            single_marker = 0
            multi_marker = 0

            for multi_path, multi_name in zip(multi_paths, multi_names):
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                if multi_name == 'CoDE':
                        multi_name = 'learning-to-rank'
                write_name = multi_name.replace('nonz', 'NNZ')
                if if_show_label:
                    plt.plot(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values,
                    multi_marker_dict[multi_marker], label=write_name, ms=msize[1])
                else:
                    plt.plot(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values,
                             multi_marker_dict[multi_marker], ms=msize[1])
                    print(multi_marker_dict[multi_marker], write_name)
                multi_marker += 1

            for single_file, single_name in zip(single_files, single_names):
                if single_name == 'CoDE':
                    single_name = 'learning-to-rank'
                if if_show_label:
                    plt.plot(single_file.loc[fileLists[i][j], para_name[parameters[0]]], single_file.loc[fileLists[i][j],
                                                                                                         para_name[parameters[1]]],
                             single_marker_dict[single_marker], label=single_name, ms=msize[0])
                else:
                    plt.plot(single_file.loc[fileLists[i][j], para_name[parameters[0]]], single_file.loc[fileLists[i][j],
                                                                                                         para_name[parameters[1]]],
                             single_marker_dict[single_marker], ms=msize[0])
                    print(single_marker_dict[single_marker], single_name)
                single_marker += 1

            plt.title(fileLists[i][j - 1] + '_' + fileLists[i][j], fontdict={'size': wsize[1]})
            if if_show_label:
                plt.legend(prop={'size': wsize[2]})
            plt.tick_params(labelsize=wsize[3])
            plt.savefig(save_path + fileLists[i][j - 1] + '_' + fileLists[i][j] + '.png')
            plt.close()


def comparison_difcolor_ssmm_test(single_paths, multi_paths, parameters, single_names, multi_names, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            color = 0

            for multi_path, multi_name in zip(multi_paths, multi_names):
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                plt.scatter(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values,
                            color=color_dict[color], label=multi_name)
                color += 1

            for single_file, single_name in zip(single_files, single_names):
                if single_name == 'CoDE':
                    single_name = 'learning-to-rank'

                plt.scatter(single_file.loc[fileLists[i][j], para_name[parameters[0]]], single_file.loc[fileLists[i][j],
                                                                                                        para_name[parameters[1]]],
                            color=color_dict[color], label=single_name)
                color += 1

            plt.title(fileLists[i][j - 1] + '_' + fileLists[i][j])
            plt.legend()
            plt.savefig(save_path + fileLists[i][j - 1] + '_' + fileLists[i][j] + '.png')
            plt.close()


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集和测试集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def comparison_ssmm_train_test(single_paths, multi_paths, parameters, single_names, multi_names, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。

    # 设置单目标，多目标训练集测试集的读取路径
    single_train_paths = []
    single_test_paths = []

    for single_path in single_paths:
        single_train_paths.append(single_path + 'train/')
        single_test_paths.append(single_path + 'test/')

    multi_train_paths = []
    multi_test_paths = []

    for multi_path in multi_paths:
        multi_train_paths.append(multi_path + 'train/')
        multi_test_paths.append(multi_path + 'test/')

    # 首先读取单目标的训练集和测试集文件
    single_train_files = []
    for single_train_path in single_train_paths:
        single_train_files.append(pd.read_csv(single_train_path + 'doc1.csv', header=0, index_col=0))
    single_test_files = []
    for single_test_path in single_test_paths:
        single_test_files.append(pd.read_csv(single_test_path + 'doc1.csv', header=0, index_col=0))

    # 读取多目标的训练集的文件
    multi_train_files = []
    for multi_train_path in multi_train_paths:
        multi_train_files.append(get_m_files(multi_train_path, parameters))

    if len(parameters) == 2:
        multi_train_readers1 = []
        multi_train_readers2 = []

        # readers1 记录的是维度1, 算法个数 *文件个数 * 值
        # readers2 记录的是维度2，算法个数 *文件个数 * 值
        for multi_train_file in multi_train_files:
            multi_train_readers1.append(csv.reader(multi_train_file[0]))
            multi_train_readers2.append(csv.reader(multi_train_file[1]))

        # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
        train_mdatas1 = []
        # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
        train_mdatas2 = []
        for reader in zip(multi_train_readers1, multi_train_readers2):
            train_mdatas1.append(list(reader[0]))
            train_mdatas2.append(list(reader[1]))

        for i in range(len(train_mdatas1[0])):
            for t in range(len(train_mdatas1)):
                assert train_mdatas1[t][i][0] == train_mdatas2[0][i][0]
            # 读取多目标的测试集-------首先查看是否存在该测试集，不存在的不需要画图, 则跳过
            if not Path(multi_test_paths[0] + train_mdatas1[0][i][0] + '.csv').exists():
                continue

            test_mfiles = []
            for multi_test_path in multi_test_paths:
                test_mfiles.append(pd.read_csv(multi_test_path +
                                               train_mdatas1[0][i][0] + '.csv', header=1, index_col=1))

            # 绘图基本设置
            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])
            color = 0

            # 绘制多目标-训练集

            for mdata1, mdata2, multi_name in zip(train_mdatas1, train_mdatas2, multi_names):
                m_x = list(map(lambda x: float(x), mdata1[i][1:]))
                m_y = list(map(lambda x: float(x), mdata2[i][1:]))
                plt.scatter(m_x, m_y, color=color_dict[color], label=multi_name + '/train')
                color += 1

            # 绘制多目标-测试集
            for test_mfile, multi_name in zip(test_mfiles, multi_names):
                plt.scatter(test_mfile[para_name[parameters[0]]].values, test_mfile[para_name[parameters[1]]].values,
                            color=color_dict[color], label=multi_name + '/test')
                color += 1

            # 绘制单目标-训练集

            for single_train_file, single_name in zip(single_train_files, single_names):
                single_train_values = single_train_file.loc[train_mdatas1[0][i][0]].values.astype('float64').tolist()
                plt.scatter(single_train_values[parameters[0]], single_train_values[parameters[1]],
                            color=color_dict[color], label=single_name + '/train')
                color += 1

            # 绘制单目标-测试集
            for single_test_file, single_name in zip(single_test_files, single_names):
                if single_name == 'CoDE':
                    single_name = 'learning-to-rank'
                plt.scatter(single_test_file.loc[train_mdatas1[0][i][0], para_name[parameters[0]]],
                            single_test_file.loc[train_mdatas1[0][i][0], para_name[parameters[1]]],
                            color=color_dict[color], label=single_name + '/test')

            plt.legend()
            plt.savefig(save_path + train_mdatas1[0][i][0] + '.png')
            plt.close()
    elif len(parameters) == 3:
        print('three parameters functions not be completed!!')
    else:
        print('很奇怪！！！！！')


def comparison_difmarker_line_train(single_paths, multi_paths, line_paths, parameters, single_names, multi_names,
                                    line_names, save_path, if_show_label=True, if_show_rtborder = True):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}

    multi_marker_dict = {0: '.', 1: '+', 2: 'x', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: '^', 2: 'o', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    line_files = []
    for line_path in line_paths:
        line_files.append(pd.read_csv(line_path + 'doc1.csv', header=0, index_col=0))

    multi_files = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files.append(get_m_files(multi_path, parameters))

    if len(parameters) == 2:

        if len(multi_files) == 0:
            fileLists = dictionaries.get_filelists()
            for i in range(len(fileLists)):
                for j in range(1, fileLists[i]):
                    single_marker = 0
                    plt.figure(figsize=(8, 4))
                    plt.xlabel(para_name[parameters[0]], fontdict={'size': 14})
                    plt.ylabel(para_name[parameters[1]], fontdict={'size': 14})
                    y_max = 0
                    for single_file, single_name in zip(single_files, single_names):
                        if single_name == 'CoDE':
                            single_name = 'learning-to-rank'
                        single_values = single_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                        if if_show_label:
                            plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                     single_marker_dict[single_marker], label=single_name, markersize=16)
                        else:
                            plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                     single_marker_dict[single_marker], markersize=16)
                            print(single_marker_dict[single_marker], single_name)
                        single_marker += 1
                        max_y = np.max(y_max, single_values[parameters[1]])
                    for line_file, line_name in zip(line_files, line_names):
                        if line_name == 'CoDE':
                            line_name = 'learning-to-rank'
                        line_values = line_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                        plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax=y_max, colors='r', linestyles='--', label=line_name)
                        y_tmp = ('%.4f' % line_values[parameters[1]])
                        ss_label = para_name[parameters[1]] + ' = ' + str(y_tmp)
                        plt.text(line_values[parameters[0]], max_y, ss_label, ha='right',va='top',fontdict={'size': 14, 'color':  'black'})

                    plt.title(fileLists[i][j] + '_' + fileLists[i][j], fontdict={'size': 14})
                    if not if_show_rtborder:
                        ax = plt.axes()
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    if if_show_label:
                        plt.legend(prop={'size': 14})
                    plt.savefig(save_path + fileLists[i][j] + '.png')
                    plt.close()

        else:
            readers1 = []
            readers2 = []
            # readers1 记录的是维度1, 算法个数 *文件个数 * 值
            # readers2 记录的是维度2，算法个数 *文件个数 * 值
            for multi_file in multi_files:
                readers1.append(csv.reader(multi_file[0]))
                readers2.append(csv.reader(multi_file[1]))

            # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
            mdatas1 = []
            # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
            mdatas2 = []
            for reader in zip(readers1, readers2):
                mdatas1.append(list(reader[0]))
                mdatas2.append(list(reader[1]))

            for i in range(len(mdatas1[0])):
                multi_marker = 0
                single_marker = 0
                y_max = 0
                for t in range(len(mdatas1)):
                    assert mdatas1[t][i][0] == mdatas2[0][i][0]
                plt.figure(figsize=(8, 4))
                plt.xlabel(para_name[parameters[0]], fontdict={'size': 14})
                plt.ylabel(para_name[parameters[1]], fontdict={'size': 14})

                for mdata1, mdata2, multi_name in zip(mdatas1, mdatas2, multi_names):
                    m_x = list(map(lambda x: float(x), mdata1[i][1:]))
                    m_y = list(map(lambda x: float(x), mdata2[i][1:]))
                    y_max = max(y_max, max(m_y))
                    if if_show_label:
                        plt.plot(m_x, m_y, multi_marker_dict[multi_marker], label=multi_name, markersize=8)
                    else:
                        plt.plot(m_x, m_y, multi_marker_dict[multi_marker], markersize=8)
                        print(multi_marker_dict[multi_marker], multi_name)
                    multi_marker += 1

                # plt.axvline(x = single_values[-1],
                #             color='red', label=single_name, linestyle = '-')
                for single_file, single_name in zip(single_files, single_names):
                    if single_name == 'CoDE':
                        single_name = 'learning-to-rank'
                    single_values = single_file.loc[mdatas2[0][i][0]].values.astype('float64').tolist()
                    if if_show_label:
                        plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                 single_marker_dict[single_marker], label=single_name, markersize=16)
                    else:
                        plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                 single_marker_dict[single_marker], markersize=16)
                        print(single_marker_dict[single_marker], single_name)
                    single_marker += 1
                    y_max = max(y_max, single_values[parameters[1]])
                for line_file, line_name in zip(line_files, line_names):
                    if line_name == 'CoDE':
                        line_name = 'learning-to-rank'
                    line_values = line_file.loc[mdatas2[0][i][0]].values.astype('float64').tolist()
                    plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax=y_max, colors='r', linestyles='--', label=line_name)
                    y_tmp = ('%.4f' % line_values[parameters[1]])
                    ss_label = para_name[parameters[1]] + ' = ' + str(y_tmp)
                    plt.text(line_values[parameters[0]], y_max, ss_label, ha='right',va='top',fontdict={'size': 14, 'color':  'black'})
                plt.title(mdatas1[0][i][0] + '_' + mdatas1[0][i][0], fontdict={'size': 14})
                if if_show_label:
                    plt.legend(prop={'size': 14})
                if not if_show_rtborder:
                    ax = plt.axes()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                plt.savefig(save_path + mdatas1[0][i][0] + '.png')
                plt.close()
    elif len(parameters) == 3:
        pass


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def comparison_difmarker_line_test(single_paths, multi_paths,line_paths, parameters, single_names, multi_names, line_names, save_path,
                                   if_show_label=True, if_show_rtborder = True):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}
    multi_marker_dict = {0: '.', 1: '+', 2: 'x', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: '^', 2: 'o', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    line_files = []
    for line_path in line_paths:
        line_files.append(pd.read_csv(line_path + 'doc1.csv', header=0, index_col=0))

    # single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]], fontdict={'size': 14})
            plt.ylabel(para_name[parameters[1]], fontdict={'size': 14})

            single_marker = 0
            multi_marker = 0
            y_max = 0
            for multi_path, multi_name in zip(multi_paths, multi_names):
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                write_name = multi_name.replace('nonz', 'NNZ')
                y_max = max(y_max, max(m_file[para_name[parameters[1]]].values.tolist()))
                if if_show_label:
                    plt.plot(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values,
                             multi_marker_dict[multi_marker], label=write_name, ms=8)
                else:
                    plt.plot(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values,
                             multi_marker_dict[multi_marker], ms=8)
                    print(multi_marker_dict[multi_marker], write_name)
                multi_marker += 1

            for single_file, single_name in zip(single_files, single_names):
                if single_name == 'CoDE':
                    single_name = 'learning-to-rank'
                if if_show_label:
                    plt.plot(single_file.loc[fileLists[i][j], para_name[parameters[0]]],
                             single_file.loc[fileLists[i][j],
                                             para_name[parameters[1]]],
                             single_marker_dict[single_marker], label=single_name, ms=16)
                else:
                    plt.plot(single_file.loc[fileLists[i][j], para_name[parameters[0]]],
                             single_file.loc[fileLists[i][j],
                                             para_name[parameters[1]]],
                             single_marker_dict[single_marker], ms=16)
                    print(single_marker_dict[single_marker], single_name)
                y_max = max(y_max, single_file.loc[fileLists[i][j], para_name[parameters[1]]])
                single_marker += 1
            for line_file, line_name in zip(line_files, line_names):
                if line_name == 'CoDE':
                    line_name = 'learning-to-rank'
                line_values = line_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                print(fileLists[i][j], line_values[parameters[0]])
                plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax= y_max, colors='r', linestyles='--', label=line_name)
                y_tmp = ('%.4f' % line_values[parameters[1]])
                ss_label = para_name[parameters[1]] + ' = ' + str(y_tmp)
                plt.text(line_values[parameters[0]], y_max, ss_label, ha='right',va='top',fontdict={'size': 14, 'color':  'black'})
            if not if_show_rtborder:
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.title(fileLists[i][j - 1] + '_' + fileLists[i][j], fontdict={'size': 14})
            if if_show_label:
                plt.legend(prop={'size': 14})
            plt.savefig(save_path + fileLists[i][j - 1] + '_' + fileLists[i][j] + '.png')
            plt.close()

def combine_difmarker_line_train(single_paths, multi_paths, line_paths, parameters, single_names, multi_names,rows, columns,
                                    line_names, save_path,if_show_label, if_show_rtborder = True):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}

    multi_marker_dict = {0: '.', 1: '+', 2: 'x', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: '^', 2: 'o', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    line_files = []
    for line_path in line_paths:
        line_files.append(pd.read_csv(line_path + 'doc1.csv', header=0, index_col=0))

    multi_files = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files.append(get_m_files(multi_path, parameters))

    if len(parameters) == 2:

        if len(multi_files) == 0:
            fileLists = dictionaries.get_filelists()
            plt.figure(figsize=(8 * columns, 4 * rows))
            fig_number = 0

            for i in range(len(fileLists)):
                for j in range(1, fileLists[i]):
                    single_marker = 0
                    fig_number += 1
                    plt.subplot(rows, columns, fig_number)
                    plt.xlabel(para_name[parameters[0]], fontdict={'size': 14})
                    plt.ylabel(para_name[parameters[1]], fontdict={'size': 14})
                    y_max = 0
                    for single_file, single_name in zip(single_files, single_names):
                        if single_name == 'CoDE':
                            single_name = 'learning-to-rank'
                        single_values = single_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                        if if_show_label:
                            plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                     single_marker_dict[single_marker], label=single_name, markersize=16)
                        else:
                            plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                     single_marker_dict[single_marker], markersize=16)
                            print(single_marker_dict[single_marker], single_name)
                        single_marker += 1
                        max_y = np.max(y_max, single_values[parameters[1]])
                    for line_file, line_name in zip(line_files, line_names):
                        if line_name == 'CoDE':
                            line_name = 'learning-to-rank'
                        
                        line_values = line_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                        if if_show_label:
                            plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax=y_max, colors='r', linestyles='--', label=line_name)
                        else:
                            plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax=y_max, colors='r', linestyles='--')
                        
                        y_tmp = ('%.4f' % line_values[parameters[1]])
                        ss_label = para_name[parameters[1]] + ' = ' + str(y_tmp)
                        plt.text(line_values[parameters[0]], max_y, ss_label, ha='right',va='top',fontdict={'size': 14, 'color':  'black'})

                    plt.title(fileLists[i][j] + '_' + fileLists[i][j], fontdict={'size': 14})
            if not if_show_rtborder:
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            if if_show_label:
                plt.legend(prop={'size': 14})
            plt.savefig(save_path + 'train.png')
            plt.close()

        else:
            readers1 = []
            readers2 = []
            # readers1 记录的是维度1, 算法个数 *文件个数 * 值
            # readers2 记录的是维度2，算法个数 *文件个数 * 值
            for multi_file in multi_files:
                readers1.append(csv.reader(multi_file[0]))
                readers2.append(csv.reader(multi_file[1]))

            # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
            mdatas1 = []
            # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
            mdatas2 = []
            for reader in zip(readers1, readers2):
                mdatas1.append(list(reader[0]))
                mdatas2.append(list(reader[1]))
            plt.figure(figsize=(8 * columns, 4 * rows))
            fig_number = 0
            for i in range(len(mdatas1[0])):
                multi_marker = 0
                single_marker = 0
                y_max = 0
                for t in range(len(mdatas1)):
                    assert mdatas1[t][i][0] == mdatas2[0][i][0]
                fig_number += 1
                plt.subplot(rows, columns, fig_number)
                plt.xlabel(para_name[parameters[0]], fontdict={'size': 14})
                plt.ylabel(para_name[parameters[1]], fontdict={'size': 14})

                for mdata1, mdata2, multi_name in zip(mdatas1, mdatas2, multi_names):
                    if multi_name == 'CoDE':
                        multi_name = 'learning-to-rank'
                    m_x = list(map(lambda x: float(x), mdata1[i][1:]))
                    m_y = list(map(lambda x: float(x), mdata2[i][1:]))
                    y_max = max(y_max, max(m_y))
                    if if_show_label:
                        plt.plot(m_x, m_y, multi_marker_dict[multi_marker], label=multi_name, markersize=8)
                    else:
                        plt.plot(m_x, m_y, multi_marker_dict[multi_marker], markersize=8)
                        print(multi_marker_dict[multi_marker], multi_name)
                    multi_marker += 1

                # plt.axvline(x = single_values[-1],
                #             color='red', label=single_name, linestyle = '-')
                for single_file, single_name in zip(single_files, single_names):
                    if single_name == 'CoDE':
                        single_name = 'learning-to-rank'
                    single_values = single_file.loc[mdatas2[0][i][0]].values.astype('float64').tolist()
                    if if_show_label:
                        plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                 single_marker_dict[single_marker], label=single_name, markersize=16)
                    else:
                        plt.plot(single_values[parameters[0]], single_values[parameters[1]],
                                 single_marker_dict[single_marker], markersize=16)
                        print(single_marker_dict[single_marker], single_name)
                    single_marker += 1
                    y_max = max(y_max, single_values[parameters[1]])
                for line_file, line_name in zip(line_files, line_names):
                    if line_name == 'CoDE':
                        line_name = 'learning-to-rank'
                    line_values = line_file.loc[mdatas2[0][i][0]].values.astype('float64').tolist()
                    if if_show_label:
                        plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax=y_max, colors='r', linestyles='--', label=line_name)
                    else:
                        plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax=y_max, colors='r', linestyles='--') 
                    y_tmp = ('%.4f' % line_values[parameters[1]])
                    ss_label = para_name[parameters[1]] + ' = ' + str(y_tmp)
                    plt.text(line_values[parameters[0]], y_max, ss_label, ha='right',va='top',fontdict={'size': 14, 'color':  'black'})
                plt.title(mdatas1[0][i][0] + '_' + mdatas1[0][i][0], fontdict={'size': 14})
            if if_show_label:
                plt.legend(prop={'size': 14})
            if not if_show_rtborder:
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.savefig(save_path + 'train.png')
            plt.close()
    elif len(parameters) == 3:
        pass


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def combine_difmarker_line_test(single_paths, multi_paths,line_paths, parameters, single_names, multi_names, line_names, save_path, rows, columns,
                                   if_show_label, if_show_rtborder = True):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}
    multi_marker_dict = {0: '.', 1: '+', 2: 'x', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: '^', 2: 'o', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    line_files = []
    for line_path in line_paths:
        line_files.append(pd.read_csv(line_path + 'doc1.csv', header=0, index_col=0))

    # single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    plt.figure(figsize=(8 * columns, 4 * rows))
    fig_number = 0
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            fig_number += 1
            plt.subplot(rows, columns, fig_number)
            plt.xlabel(para_name[parameters[0]], fontdict={'size': 14})
            plt.ylabel(para_name[parameters[1]], fontdict={'size': 14})

            single_marker = 0
            multi_marker = 0
            y_max = 0
            for multi_path, multi_name in zip(multi_paths, multi_names):
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                write_name = multi_name.replace('nonz', 'NNZ')
                y_max = max(y_max, max(m_file[para_name[parameters[1]]].values.tolist()))
                if if_show_label:
                    plt.plot(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values,
                             multi_marker_dict[multi_marker], label=write_name, ms=8)
                else:
                    plt.plot(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values,
                             multi_marker_dict[multi_marker], ms=8)
                    print(multi_marker_dict[multi_marker], write_name)
                multi_marker += 1

            for single_file, single_name in zip(single_files, single_names):
                if single_name == 'CoDE':
                    single_name = 'learning-to-rank'
                if if_show_label:
                    plt.plot(single_file.loc[fileLists[i][j], para_name[parameters[0]]],
                             single_file.loc[fileLists[i][j],
                                             para_name[parameters[1]]],
                             single_marker_dict[single_marker], label=single_name, ms=16)
                else:
                    plt.plot(single_file.loc[fileLists[i][j], para_name[parameters[0]]],
                             single_file.loc[fileLists[i][j],
                                             para_name[parameters[1]]],
                             single_marker_dict[single_marker], ms=16)
                    print(single_marker_dict[single_marker], single_name)
                y_max = max(y_max, single_file.loc[fileLists[i][j], para_name[parameters[1]]])
                single_marker += 1
            for line_file, line_name in zip(line_files, line_names):
                if line_name == 'CoDE':
                    line_name = 'learning-to-rank'
                line_values = line_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                print(fileLists[i][j], line_values[parameters[0]])
                if if_show_label:
                    plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax= y_max, colors='r', linestyles='--', label=line_name)
                else:
                    plt.vlines(x=line_values[parameters[0]], ymin= 0, ymax= y_max, colors='r', linestyles='--')
                y_tmp = ('%.4f' % line_values[parameters[1]])
                ss_label = para_name[parameters[1]] + ' = ' + str(y_tmp)
                plt.text(line_values[parameters[0]], y_max, ss_label, ha='right',va='top',fontdict={'size': 14, 'color':  'black'})
            plt.title(fileLists[i][j - 1] + '_' + fileLists[i][j], fontdict={'size': 14})
    if not if_show_rtborder:
            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    if if_show_label:
        plt.legend(prop={'size': 14})
    plt.savefig(save_path + 'test.png')
    plt.close()


    
def comparison_difmarker_log04_train(single_paths, multi_paths,single_names, multi_names, save_path, msize, wsize,figsize, if_show_label=True):
    parameters = [0, 4]
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}

    multi_marker_dict = {0: '.', 1: '+', 2: '^', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: 'o', 2: 'x', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))

    multi_files = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files.append(get_m_files(multi_path, parameters))

    if len(multi_files) == 0:
        fileLists = dictionaries.get_filelists()
        for i in range(len(fileLists)):
            for j in range(1, fileLists[i]):
                single_marker = 0
                plt.figure(figsize=(figsize[0], figsize[1]))
                plt.xlabel(para_name[parameters[0]], fontdict={'size': wsize[0]})
                plt.ylabel('ln('+para_name[parameters[1]]+')', fontdict={'size': wsize[0]})
                for single_file, single_name in zip(single_files, single_names):
                    if single_name == 'CoDE':
                        single_name = 'learning-to-rank'
                    single_values = single_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                    fpas = single_values[parameters[0]]
                    mses = single_values[parameters[1]]
                    log_mses = math.log(mses)
                    if if_show_label:
                        plt.plot(fpas, log_mses,
                                    single_marker_dict[single_marker], label=single_name, markersize=msize[0])
                    else:
                        plt.plot(fpas, log_mses,
                                    single_marker_dict[single_marker], markersize=msize[0])
                        print(single_marker_dict[single_marker], single_name)
                    single_marker += 1
                plt.title(fileLists[i][j] + '_' + fileLists[i][j], fontdict={'size': wsize[1]})
                if if_show_label:
                    plt.legend(prop={'size': wsize[2]})
                plt.tick_params(labelsize=wsize[3])
                plt.savefig(save_path + fileLists[i][j] + '.png')
                plt.close()

    else:
        readers1 = []
        readers2 = []
        # readers1 记录的是维度1, 算法个数 *文件个数 * 值
        # readers2 记录的是维度2，算法个数 *文件个数 * 值
        for multi_file in multi_files:
            readers1.append(csv.reader(multi_file[0]))
            readers2.append(csv.reader(multi_file[1]))

        # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
        mdatas1 = []
        # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
        mdatas2 = []
        for reader in zip(readers1, readers2):
            mdatas1.append(list(reader[0]))
            mdatas2.append(list(reader[1]))

        for i in range(len(mdatas1[0])):
            multi_marker = 0
            single_marker = 0
            for t in range(len(mdatas1)):
                assert mdatas1[t][i][0] == mdatas2[0][i][0]

            plt.figure(figsize=(figsize[0], figsize[1]))
            plt.xlabel(para_name[parameters[0]], fontdict={'size': wsize[0]})
            plt.ylabel('ln('+para_name[parameters[1]]+')', fontdict={'size': wsize[0]})

            for mdata1, mdata2, multi_name in zip(mdatas1, mdatas2, multi_names):
                m_x = list(map(lambda x: float(x), mdata1[i][1:]))
                m_y = list(map(lambda x: float(x), mdata2[i][1:]))
                log_m_y = [math.log(tmp) for tmp in m_y]
                if multi_name == 'CoDE':
                        multi_name = 'learning-to-rank'
                if if_show_label:
                    plt.plot(m_x, log_m_y, multi_marker_dict[multi_marker], label=multi_name, markersize=msize[1])
                else:
                    plt.plot(m_x, log_m_y, multi_marker_dict[multi_marker], markersize=msize[1])
                    print(multi_marker_dict[multi_marker], multi_name)
                multi_marker += 1

            # plt.axvline(x = single_values[-1],
            #             color='red', label=single_name, linestyle = '-')
            for single_file, single_name in zip(single_files, single_names):
                if single_name == 'CoDE':
                    single_name = 'learning-to-rank'
                single_values = single_file.loc[mdatas2[0][i][0]].values.astype('float64').tolist()
                fpas = single_values[parameters[0]]
                mses = single_values[parameters[1]]
                log_mses = math.log(mses)
                if if_show_label:
                    plt.plot(fpas, log_mses, single_marker_dict[single_marker], label=single_name, markersize=msize[0])
                else:
                    plt.plot(fpas, log_mses, single_marker_dict[single_marker], markersize=msize[0])
                    print(single_marker_dict[single_marker], single_name)
                single_marker += 1
            plt.title(mdatas1[0][i][0] + '_' + mdatas1[0][i][0], fontdict={'size': wsize[1]})
            if if_show_label:
                plt.legend(prop={'size': wsize[2]})
            plt.tick_params(labelsize=wsize[3])
            plt.savefig(save_path + mdatas1[0][i][0] + '.png')
            plt.close()


'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''


def comparison_difmarker_log04_test(single_paths, multi_paths, single_names, multi_names, save_path, wsize, msize, figsize, if_show_label=True):
    parameters = [0, 4]
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    color_dict = {0: '#FF0000', 1: '#008000', 2: '#0000FF', 3: '#FFFF00', 4: '#FFA500', 5: '#800080', 6: '#EE82EE',
                  7: '#000000', 8: '#FF1493', 9: '#CD853F', 10: '#00FF00', 11: '#00008B', 12: '#FF6347'}
    multi_marker_dict = {0: '.', 1: '+', 2: '^', 3: '1', 4: '2', 5: '|', 6: '3', 7: 'd'}
    single_marker_dict = {0: 's', 1: 'o', 2: 'x', 3: '*'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            plt.figure(figsize=(figsize[0], figsize[1]))
            plt.xlabel(para_name[parameters[0]], fontdict={'size': wsize[0]})
            plt.ylabel('ln('+para_name[parameters[1]]+')', fontdict={'size': wsize[0]})

            single_marker = 0
            multi_marker = 0

            for multi_path, multi_name in zip(multi_paths, multi_names):
                if multi_name == 'CoDE':
                        multi_name = 'learning-to-rank'
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                write_name = multi_name.replace('nonz', 'NNZ')
                fpas = m_file[para_name[parameters[0]]].values
                mses =  m_file[para_name[parameters[1]]].values
                log_mses = [math.log(tmp) for tmp in mses]
                if if_show_label:
                    plt.plot(fpas, log_mses,
                             multi_marker_dict[multi_marker], label=write_name, ms=msize[1])
                else:
                    plt.plot(fpas, log_mses,
                             multi_marker_dict[multi_marker], ms=msize[1])
                    print(multi_marker_dict[multi_marker], write_name)
                multi_marker += 1

            for single_file, single_name in zip(single_files, single_names):
                if single_name == 'CoDE':
                    single_name = 'learning-to-rank'
                fpas = single_file.loc[fileLists[i][j], para_name[parameters[0]]]
                mses = single_file.loc[fileLists[i][j], para_name[parameters[1]]]
                log_mses = math.log(mses)
                if if_show_label:
                    plt.plot(fpas,log_mses ,
                             single_marker_dict[single_marker], label=single_name, ms=msize[0])
                else:
                    plt.plot(fpas,log_mses, single_marker_dict[single_marker], ms=msize[0])
                    print(single_marker_dict[single_marker], single_name)
                single_marker += 1

            plt.title(fileLists[i][j - 1] + '_' + fileLists[i][j], fontdict={'size': wsize[1]})
            if if_show_label:
                plt.legend(prop={'size': wsize[2]})
            plt.tick_params(labelsize=wsize[3])
            plt.savefig(save_path + fileLists[i][j - 1] + '_' + fileLists[i][j] + '.png')
            plt.close()