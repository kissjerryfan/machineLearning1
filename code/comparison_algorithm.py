from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import helpers
import target_functions as tgf
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import dictionaries
import os

def training_test_with_sklearnmodel(save_folder, model):
    fileLists = dictionaries.get_filelists()
    path = 'data/'
    save_train_path = '../results/compared_algorithms/' + save_folder+'/train/'
    save_test_path = '../results/compared_algorithms/' + save_folder+'/test/'

    train_doc1 = [['filename', 'FPA', 'AAE', 'numOfnonZero', 'L1', 'MSE']]

    test_doc1 = [['filename', 'FPA', 'AAE', 'numOfnonZero', 'L1', 'MSE']]

    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')

            # 模型训练
            train_X, train_y = helpers.getfeatures(path, fileLists[i][j])
            model.fit(train_X, train_y)

            # 训练集的结果保存
            predValue = model.predict(train_X)
            if isinstance(model, MLPRegressor):
                parameters = np.concatenate((model.coefs_[0].flatten(), model.coefs_[1].flatten()))
            elif isinstance(model, RandomForestRegressor):
                parameters = np.array([0])
            else:
                parameters = np.concatenate((model.coef_ ,[model.intercept_]))

            f1 = tgf.FPA(predValue, train_y)
            f2 = tgf.AAE(predValue, train_y)

            f3 = tgf.numOfnonZero(parameters)
            f4 = tgf.l1_values(parameters)

            f5 = tgf.MSE(predValue, train_y)

            train_doc1.append([fileLists[i][j], f1, f2, f3, f4, f5])

            #测试集的结果保存
            # 如果是最后一个文件，说明不需要
            if j == len(fileLists[i])-1:
                continue

            test_X, test_y = helpers.getfeatures(path, fileLists[i][j+1])
            pred_test = model.predict(test_X)
            test_f1 = tgf.FPA(pred_test, test_y)
            test_f2 = tgf.AAE(pred_test, test_y)

            test_f5 = tgf.MSE(pred_test, test_y)


            test_doc1.append([fileLists[i][j+1], test_f1, test_f2, f3, f4, test_f5])

    with open(save_train_path + 'doc1.csv', 'w', newline='') as train_file:
        train_writer = csv.writer(train_file)
        for row in train_doc1:
            train_writer.writerow(row)
    with open(save_test_path + 'doc1.csv', 'w', newline='') as test_file:
        test_writer = csv.writer(test_file)
        for row in test_doc1:
            test_writer.writerow(row)


'''
a. 文档1：记录每个数据集非支配集的平均【FPA, AAE, 对应的非零参数解】
b. 文档2：记录每个数据集非支配集的【FPA】
c. 文档3：记录每个数据集非支配集的【AAE】
d. 文档4：记录每个数据集非支配集的【对应的非零参数个数】
e. 文档5：记录每个数据集非支配集的【L1范数值】
f. 文档6：记录每个数据集非支配集的【MSE】
'''
def training_test_10times_sklearnmodel(save_folder, model):
    fileLists = dictionaries.get_filelists()
    path = 'data/'
    save_train_path = '../results/compared_algorithms/' + save_folder+'/train/'
    save_test_path = '../results/compared_algorithms/' + save_folder+'/test/'

    train_doc1 = [['filename', 'FPA', 'AAE', 'numOfnonZero', 'L1', 'MSE']]
    train_doc2 = []
    train_doc3 = []
    train_doc4 = []
    train_doc5 = []
    train_doc6 = []

    test_doc1 = [['filename', 'FPA', 'AAE', 'numOfnonZero', 'L1', 'MSE']]
    test_doc2 = []
    test_doc3 = []
    test_doc4 = []
    test_doc5 = []
    test_doc6 = []


    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            print('\n\n\n' + fileLists[i][j] + '\n\n\n')

            # 模型训练
            train_X, train_y = helpers.getfeatures(path, fileLists[i][j])
            train_f1s = []
            train_f2s = []
            train_f3s = []
            train_f4s = []
            train_f5s = []
            if j < len(fileLists[i]) - 1:
                test_X, test_y = helpers.getfeatures(path, fileLists[i][j + 1])
                test_f1s = []
                test_f2s = []
                test_f3s = []
                test_f4s = []
                test_f5s = []


            for times in range(10):
                model.fit(train_X, train_y)
                # 训练集的结果保存
                predValue = model.predict(train_X)
                parameters = np.concatenate((model.coef_ ,[model.intercept_]))

                f1 = tgf.FPA(predValue, train_y)
                f2 = tgf.AAE(predValue, train_y)

                f3 = tgf.numOfnonZero(parameters)
                f4 = tgf.l1_values(parameters)

                f5 = tgf.MSE(predValue, train_y)

                train_f1s.append(f1)
                train_f2s.append(f2)
                train_f3s.append(f3)
                train_f4s.append(f4)
                train_f5s.append(f5)



                #测试集的结果保存
                # 如果是最后一个文件，说明不需要
                if j == len(fileLists[i])-1:
                    continue


                pred_test = model.predict(test_X)
                test_f1 = tgf.FPA(pred_test, test_y)
                test_f2 = tgf.AAE(pred_test, test_y)

                test_f5 = tgf.MSE(pred_test, test_y)

                test_f1s.append(test_f1)
                test_f2s.append(test_f2)
                test_f5s.append(test_f5)

            avg_train_f1 = np.mean(train_f1s)
            avg_train_f2 = np.mean(train_f2s)
            avg_train_f3 = np.mean(train_f3s)
            avg_train_f4 = np.mean(train_f4s)
            avg_train_f5 = np.mean(train_f5s)

            train_doc1.append([fileLists[i][j], avg_train_f1, avg_train_f2, avg_train_f3, avg_train_f4, avg_train_f5])

            train_f1s.insert(0, fileLists[i][j])
            train_f2s.insert(0, fileLists[i][j])
            train_f3s.insert(0, fileLists[i][j])
            train_f4s.insert(0, fileLists[i][j])
            train_f5s.insert(0, fileLists[i][j])

            train_doc2.append(train_f1s.copy())
            train_doc3.append(train_f2s.copy())
            train_doc4.append(train_f3s.copy())
            train_doc5.append(train_f4s.copy())
            train_doc6.append(train_f5s.copy())

            if j == len(fileLists[i]) - 1:
                continue

            avg_test_f1 = np.mean(test_f1s)
            avg_test_f2 = np.mean(test_f2s)
            avg_test_f5 = np.mean(test_f5s)

            test_f1s.insert(0, fileLists[i][j+1])
            test_f2s.insert(0, fileLists[i][j+1])

            test_f3s = train_f3s.copy()
            test_f3s[0] = fileLists[i][j+1]
            test_f4s = train_f4s.copy()
            test_f4s[0] = fileLists[i][j+1]

            test_f5s.insert(0, fileLists[i][j+1])


            test_doc1.append([fileLists[i][j+1], avg_test_f1, avg_test_f2, avg_train_f3, avg_train_f4, avg_test_f5])
            test_doc2.append(test_f1s.copy())
            test_doc3.append(test_f2s.copy())
            test_doc4.append(test_f3s.copy())
            test_doc5.append(test_f5s.copy())
            test_doc6.append(test_f5s.copy())

    with open(save_train_path + 'doc1.csv', 'w', newline='') as train_file1:
        train_writer = csv.writer(train_file1)
        for row in train_doc1:
            train_writer.writerow(row)
    with open(save_test_path + 'doc1.csv', 'w', newline='') as test_file1:
        test_writer = csv.writer(test_file1)
        for row in test_doc1:
            test_writer.writerow(row)

    with open(save_train_path + 'doc2.csv', 'w', newline='') as train_file2:
        train_writer = csv.writer(train_file2)
        for row in train_doc2:
            train_writer.writerow(row)
    with open(save_test_path + 'doc2.csv', 'w', newline='') as test_file2:
        test_writer = csv.writer(test_file2)
        for row in test_doc2:
            test_writer.writerow(row)

    with open(save_train_path + 'doc3.csv', 'w', newline='') as train_file3:
        train_writer = csv.writer(train_file3)
        for row in train_doc3:
            train_writer.writerow(row)
    with open(save_test_path + 'doc3.csv', 'w', newline='') as test_file3:
        test_writer = csv.writer(test_file3)
        for row in test_doc3:
            test_writer.writerow(row)

    with open(save_train_path + 'doc4.csv', 'w', newline='') as train_file4:
        train_writer = csv.writer(train_file4)
        for row in train_doc4:
            train_writer.writerow(row)
    with open(save_test_path + 'doc4.csv', 'w', newline='') as test_file4:
        test_writer = csv.writer(test_file4)
        for row in test_doc4:
            test_writer.writerow(row)

    with open(save_train_path + 'doc5.csv', 'w', newline='') as train_file5:
        train_writer = csv.writer(train_file5)
        for row in train_doc5:
            train_writer.writerow(row)
    with open(save_test_path + 'doc5.csv', 'w', newline='') as test_file5:
        test_writer = csv.writer(test_file5)
        for row in test_doc5:
            test_writer.writerow(row)
    with open(save_train_path + 'doc6.csv', 'w', newline='') as train_file6:
        train_writer = csv.writer(train_file6)
        for row in train_doc6:
            train_writer.writerow(row)
    with open(save_test_path + 'doc6.csv', 'w', newline='') as test_file6:
        test_writer = csv.writer(test_file6)
        for row in test_doc6:
            test_writer.writerow(row)

def comparison_ssm_train(single_path1, single_path2, multi_path, parameters, single_name1, single_name2, multi_name, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2:'numOfnonZero'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_file1 = pd.read_csv(single_path1+'doc1.csv', header = 0, index_col = 0)
    single_file2 = pd.read_csv(single_path2 + 'doc1.csv', header=0, index_col=0)
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    multi_files = []
    if 0 in parameters:
        file1 = open(multi_path+'doc2.csv', 'r')
        multi_files.append(file1)
    if 1 in parameters:
        file2 = open(multi_path+'doc3.csv', 'r')
        multi_files.append(file2)
    if 2 in parameters:
        file3 = open(multi_path+'doc4.csv', 'r')
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

            # plt.axvline(x = single_values[-1],
            #             color='red', label=single_name, linestyle = '-')
            m_x = list(map(lambda x:float(x), mdata1[i][1:]))
            m_y = list(map(lambda x:float(x), mdata2[i][1:]))
            plt.scatter(m_x, m_y, color='green', label=multi_name)

            single_values1 = single_file1.loc[mdata1[i][0]].values.astype('float64').tolist()
            plt.scatter(single_values1[parameters[0]], single_values1[parameters[1]],
                        color='red', label=single_name1)

            single_values2 = single_file2.loc[mdata1[i][0]].values.astype('float64').tolist()
            plt.scatter(single_values2[parameters[0]], single_values2[parameters[1]],
                        color='blue', label=single_name2)
            #

            plt.legend()
            plt.savefig(save_path + mdata1[i][0] + '.png')
            plt.close()

    # fileLists = get_filelists()
    # for i in range(len(fileLists)):
    #     for j in range(1, len(fileLists[i])):
    #         if len(parameters) == 2:
    #             plt.figure(figsize=(8, 4))
    #             plt.xlabel(para_name[parameters[0]])
    #             plt.ylabel(para_name[parameters[1]])
    #             single_values = single_file.loc[fileLists[i][j]].values.astype('float64').tolist()
    #             plt.scatter(single_values[parameters[0]], single_values[parameters[1]],
    #                         color = 'red', label = single_name)
    #             m_x = multi_files[0].loc[fileLists[i][j]].values.astype('float64')
    #             m_y = multi_files[1].loc[fileLists[i][j]].values.astype('float64')
    #             plt.scatter(m_x, m_y, color = 'green', label = multi_name)
    #             plt.savefig(save_path+fileLists[i][j]+'.png')
    #




'''
简介：
    1. 该函数用于单目标算法和多目标算法的训练集的对比
    2. parameters用于指定，所对比的参数是什么
        0---FPA
        1---AAE
        2---numOfnonZero
'''
def comparison_ssm_test(single_path1, single_path2, multi_path, parameters, single_name1, single_name2, multi_name, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2:'numOfnonZero'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。
    single_file1 = pd.read_csv(single_path1+'doc1.csv', header = 0, index_col = 0)
    single_file2 = pd.read_csv(single_path2 + 'doc1.csv', header=0, index_col=0)
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = helpers.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            m_file = pd.read_csv(multi_path+fileLists[i][j]+'.csv', header=1, index_col=1)

            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])


            plt.scatter(single_file1.loc[fileLists[i][j],para_name[parameters[0]]], single_file1.loc[fileLists[i][j],
                                                                                                   para_name[parameters[1]]],
                        color='red', label=single_name1)

            plt.scatter(single_file2.loc[fileLists[i][j], para_name[parameters[0]]], single_file2.loc[fileLists[i][j],
                                                                                                    para_name[
                                                                                                        parameters[1]]],
                        color='blue', label=single_name2)
            #

            # plt.axvline(x = single_values[-1],
            #             color='red', label=single_name, linestyle = '-')
            plt.scatter(m_file[para_name[parameters[0]]].values, m_file[para_name[parameters[1]]].values, color='green', label=multi_name)
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
def comparison_ssm_train_test(single_path1, single_path2, multi_path, parameters, single_name1, single_name2, multi_name, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero'}
    # 需要注意的是，恰好再测试集和数据集时，记录最优的模型对应的性能都是doc1文件。

    # 设置单目标，多目标训练集测试集的读取路径
    single_train_path1 = single_path1 + 'train/'
    single_test_path1 = single_path1 + 'test/'
    single_train_path2 = single_path2 + 'train/'
    single_test_path2 = single_path2 + 'test/'
    multi_train_path = multi_path + 'train/'
    multi_test_path = multi_path + 'test/'

    #首先读取单目标的训练集和测试集文件
    single_train_file1 = pd.read_csv(single_train_path1 + 'doc1.csv', header=0, index_col=0)
    single_test_file1 = pd.read_csv(single_test_path1 + 'doc1.csv', header = 0, index_col=0)

    single_train_file2 = pd.read_csv(single_train_path2 + 'doc1.csv', header=0, index_col=0)
    single_test_file2 = pd.read_csv(single_test_path2 + 'doc1.csv', header=0, index_col=0)

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
            if not Path(multi_test_path+train_mdata1[i][0]+'.csv').exists():
                continue

            test_mfile = pd.read_csv(multi_test_path+train_mdata1[i][0]+'.csv', header = 1, index_col=1)


            # 绘图基本设置
            plt.figure(figsize=(8, 4))
            plt.xlabel(para_name[parameters[0]])
            plt.ylabel(para_name[parameters[1]])

            # 绘制单目标-训练集1
            single_train_values = single_train_file1.loc[train_mdata1[i][0]].values.astype('float64').tolist()
            plt.scatter(single_train_values[parameters[0]], single_train_values[parameters[1]],
                        color = 'red', label = single_name1+'/train')

            # 绘制单目标-测试集1
            plt.scatter(single_test_file1.loc[train_mdata1[i][0], para_name[parameters[0]]],
                        single_test_file1.loc[train_mdata1[i][0], para_name[parameters[1]]],
                        color='green', label=single_name1+'/test')

            # 绘制单目标-训练集2
            single_train_values = single_train_file2.loc[train_mdata1[i][0]].values.astype('float64').tolist()
            plt.scatter(single_train_values[parameters[0]], single_train_values[parameters[1]],
                        color='red', label=single_name2 + '/train')

            # 绘制单目标-测试集2
            plt.scatter(single_test_file2.loc[train_mdata1[i][0], para_name[parameters[0]]],
                        single_test_file2.loc[train_mdata1[i][0], para_name[parameters[1]]],
                        color='green', label=single_name2 + '/test')

            #绘制多目标-训练集
            train_m_x = list(map(lambda x: float(x), train_mdata1[i][1:]))
            train_m_y = list(map(lambda x: float(x), train_mdata2[i][1:]))
            plt.scatter(train_m_x, train_m_y, color = 'blue', label = multi_name+'/train')

            #绘制多目标-测试集
            plt.scatter(test_mfile[para_name[parameters[0]]].values, test_mfile[para_name[parameters[1]]].values,
                        color = 'yellow', label = multi_name+'/test')



            plt.legend()
            plt.savefig(save_path + train_mdata1[i][0] + '.png')
            plt.close()
    elif len(parameters) == 3:
        print('three parameters functions not be completed!!')
    else:
        print('很奇怪！！！！！')
