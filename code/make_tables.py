import pandas as pd
import csv
import helpers
import numpy as np
import dictionaries
import target_functions as tgf
from algorithms import minmaxScaler
import random

def table_for_train(single_paths, multi_paths, parameters, single_names, multi_names, save_path, if_max):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']

    for multi_name in multi_names:
        doc_name.append(multi_name)
    for single_name in single_names:
        doc_name.append(single_name)
    doc = [doc_name]

    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))

    multi_files = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files.append(helpers.get_m_files(multi_path, [parameters]))

    readers = []
    # readers1 记录的是维度1, 算法个数 *文件个数 * 值
    # readers2 记录的是维度2，算法个数 *文件个数 * 值
    if len(multi_files) == 0:
        fileLists = dictionaries.get_filelists()
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i])):
                lines = [fileLists[i][j]]
                for single_file, single_name in zip(single_files, single_names):
                    single_values = single_file.loc[fileLists[i][j]].values.astype('float64').tolist()
                    t = single_values[parameters]
                    lines.append(t)
                doc.append(lines.copy())
        with open(save_path + para_name[parameters] + '_train' + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in doc:
                writer.writerow(row)

    else:
        for multi_file in multi_files:
            readers.append(csv.reader(multi_file[0]))



        # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
        mdatas = []
        # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
        for reader in readers:
            mdatas.append(list(reader))

        for i in range(len(mdatas[0])):
            multi_marker = 0
            single_marker = 0
            lines = [mdatas[0][i][0]]

            for mdata, multi_name in zip(mdatas, multi_names):
                m_values = list(map(lambda x: float(x), mdata[i][1:]))

                if if_max:
                    best_value = max(m_values)
                else:
                    best_value = min(m_values)
                lines.append(best_value)

            # plt.axvline(x = single_values[-1],
            #             color='red', label=single_name, linestyle = '-')
            for single_file, single_name in zip(single_files, single_names):

                single_values = single_file.loc[mdatas[0][i][0]].values.astype('float64').tolist()
                t = single_values[parameters]
                lines.append(t)
            doc.append(lines.copy())
        with open(save_path + para_name[parameters]+'_train'+'.csv', 'w', newline = '') as file:
            writer = csv.writer(file)
            for row in doc:
                writer.writerow(row)

def table_for_test(single_paths, multi_paths, parameters, single_names, multi_names, save_path, if_max):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    for single_name in single_names:
        doc_name.append(single_name)
    doc = [doc_name]

    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            lines = [fileLists[i][j]]

            for multi_path, multi_name in zip(multi_paths, multi_names):
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                m_values = m_file[para_name[parameters]].values
                if if_max:
                    best_index = np.argmax(m_values)
                else:
                    best_index = np.argmin(m_values)
                lines.append(m_values[best_index])

            for single_file, single_name in zip(single_files, single_names):
                lines.append(single_file.loc[fileLists[i][j], para_name[parameters]])
            doc.append(lines.copy())
            with open(save_path + para_name[parameters]+'_test' + '.csv', 'w', newline='') as file2:
                writer = csv.writer(file2)
                for row in doc:
                    writer.writerow(row)


'''
---------------------------------------------------------------------------------------------------------
'''

def table_range_train(single_paths, multi_paths, parameters, single_names, multi_names, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']

    for multi_name in multi_names:
        doc_name.append(multi_name)
    for single_name in single_names:
        doc_name.append(single_name)
    doc = [doc_name]

    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))

    multi_files = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files.append(helpers.get_m_files(multi_path, [parameters]))

    readers = []
    # readers1 记录的是维度1, 算法个数 *文件个数 * 值
    # readers2 记录的是维度2，算法个数 *文件个数 * 值
    for multi_file in multi_files:
        readers.append(csv.reader(multi_file[0]))

    # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
    mdatas = []
    # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
    for reader in readers:
        mdatas.append(list(reader))

    for i in range(len(mdatas[0])):
        multi_marker = 0
        single_marker = 0
        lines = [mdatas[0][i][0]]

        for mdata, multi_name in zip(mdatas, multi_names):
            m_values = list(map(lambda x: float(x), mdata[i][1:]))

            t_range = '[' + str(min(m_values)) + ', ' + str(max(m_values)) + ']'
            lines.append(t_range)

        # plt.axvline(x = single_values[-1],
        #             color='red', label=single_name, linestyle = '-')
        for single_file, single_name in zip(single_files, single_names):

            single_values = single_file.loc[mdatas[0][i][0]].values.astype('float64').tolist()
            t = single_values[parameters]
            lines.append(t)
        doc.append(lines.copy())
    with open(save_path + para_name[parameters]+'_range_train'+'.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)
def table_range_test(single_paths, multi_paths, parameters, single_names, multi_names, save_path):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    for single_name in single_names:
        doc_name.append(single_name)
    doc = [doc_name]

    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            lines = [fileLists[i][j]]

            for multi_path, multi_name in zip(multi_paths, multi_names):
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                m_values = m_file[para_name[parameters]].values
                t_range = '['+str(m_values[np.argmin(m_values)]) + ', ' + str(m_values[np.argmax(m_values)]) + ']'
                lines.append(t_range)

            for single_file, single_name in zip(single_files, single_names):
                lines.append(single_file.loc[fileLists[i][j], para_name[parameters]])
            doc.append(lines.copy())
    with open(save_path + para_name[parameters]+'_range_test' + '.csv', 'w', newline='') as file2:
        writer = csv.writer(file2)
        for row in doc:
            writer.writerow(row)
'''
----------------------------------------------------------------------------------------------------------------
'''
def find_m_corresponding_value_train(single_paths, multi_paths, max_parameter, target_parameter, single_names, multi_names, save_path, if_max):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']

    for multi_name in multi_names:
        doc_name.append(multi_name)
    for single_name in single_names:
        doc_name.append(single_name)
    doc = [doc_name]

    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))

    multi_files1 = []
    multi_files2 = []
    # 每一个文档后面都是之前的m——file
    for multi_path in multi_paths:
        multi_files1.append(helpers.get_m_files(multi_path, [max_parameter]))
        multi_files2.append(helpers.get_m_files(multi_path, [target_parameter]))

    readers1 = []
    readers2 = []
    # readers1 记录的是维度1, 算法个数 *文件个数 * 值
    # readers2 记录的是维度2，算法个数 *文件个数 * 值
    for multi_file1 in multi_files1:
        readers1.append(csv.reader(multi_file1[0]))
    for multi_file2 in multi_files2:
        readers2.append(csv.reader(multi_file2[0]))


    # mdatas1 记录的是维度1 算法个数 *文件个数 * 值
    mdatas1 = []
    mdatas2 = []
    # ndatas2 记录的是维度2 算法个数 *文件个数 * 值
    for reader1 in readers1:
        mdatas1.append(list(reader1))
    for reader2 in readers2:
        mdatas2.append(list(reader2))

    for i in range(len(mdatas1[0])):
        multi_marker = 0
        single_marker = 0
        lines = [mdatas2[0][i][0]]

        for mdata1,mdata2, multi_name in zip(mdatas1, mdatas2, multi_names):
            m_values1 = list(map(lambda x: float(x), mdata1[i][1:]))
            m_values2 = list(map(lambda x: float(x), mdata2[i][1:]))
            if if_max:
                m_index = m_values1.index(max(m_values1))
            else:
                m_index = m_values1.index(min(m_values1))
            lines.append(m_values2[m_index])

        # plt.axvline(x = single_values[-1],
        #             color='red', label=single_name, linestyle = '-')
        for single_file, single_name in zip(single_files, single_names):

            single_values = single_file.loc[mdatas1[0][i][0]].values.astype('float64').tolist()
            t = single_values[target_parameter]
            lines.append(t)
        doc.append(lines.copy())
    with open(save_path + para_name[target_parameter] + '(bybest'+ para_name[max_parameter]+')_train'+'.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)

def find_m_corresponding_value_test(single_paths, multi_paths, max_parameter, target_parameter, single_names, multi_names, save_path, if_max):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    for single_name in single_names:
        doc_name.append(single_name)
    doc = [doc_name]

    single_files = []
    for single_path in single_paths:
        single_files.append(pd.read_csv(single_path + 'doc1.csv', header=0, index_col=0))
    #single_file = pd.read_csv(single_path + 'doc5.csv', header=0, index_col=0)
    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(2, len(fileLists[i])):
            lines = [fileLists[i][j]]

            for multi_path, multi_name in zip(multi_paths, multi_names):
                m_file = pd.read_csv(multi_path + fileLists[i][j] + '.csv', header=1, index_col=1)
                m_values1 = m_file[para_name[max_parameter]].values
                m_values2 = m_file[para_name[target_parameter]].values
                if if_max:
                    m_index = np.argmax(m_values1)
                else:
                    m_index = np.argmin(m_values1)
                lines.append(m_values2[m_index])

            for single_file, single_name in zip(single_files, single_names):
                lines.append(single_file.loc[fileLists[i][j], para_name[target_parameter]])
            doc.append(lines.copy())
    with open(save_path + para_name[target_parameter] + '(bybest'+ para_name[max_parameter]+')_test' + '.csv', 'w', newline='') as file2:
        writer = csv.writer(file2)
        for row in doc:
            writer.writerow(row)


def find_train_best_test_value(multi_paths, multi_names, parameter, save_path, if_max, save_file):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    doc = [doc_name]

    train_files = []
    for multi_path in multi_paths:
        train_files.append(helpers.get_m_files(multi_path+'train/', [parameter]))
    readers = []
    for train_file in train_files:
        readers.append(csv.reader(train_file[0]))

    train_datas = []
    for reader in readers:
        train_datas.append(list(reader))

    fileLists = dictionaries.get_filelists()
    t = 0
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])-1):
            lines = [fileLists[i][j]+'_'+fileLists[i][j+1]]
            while train_datas[0][t][0] != fileLists[i][j]:
                t += 1



            for train_data, multi_name, multi_path in zip(train_datas, multi_names, multi_paths):
                train_values = list(map(lambda x:float(x), train_data[t][1:]))
                if if_max:
                    b_index = train_values.index(max(train_values))
                else:
                    b_index = train_values.index(min(train_values))

                test_file = pd.read_csv(multi_path+'test/'+fileLists[i][j+1] + '.csv', header = 1, index_col=1)
                test_values = test_file[para_name[parameter]].values
                lines.append(test_values[b_index])
            doc.append(lines.copy())
    save_file = save_file+para_name[parameter]
    with open(save_path + save_file + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)


def find_validation_best_test(multi_paths, multi_names, parameter, save_path, if_max):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    doc = [doc_name]

    fileLists = dictionaries.get_filelists()
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])-1):
            lines = [fileLists[i][j]+'_'+fileLists[i][j+1]]
            for multi_path, multi_name in zip(multi_paths, multi_names):
                validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j]+'.csv', header=1,index_col=1)
                test_file = pd.read_csv(multi_path+'test/'+fileLists[i][j+1]+'.csv', header=1, index_col=1)

                validation_values = validation_file[para_name[parameter]].values
                test_values = test_file[para_name[parameter]].values
                if if_max:
                    b_index = np.argmax(validation_values)
                else:
                    b_index = np.argmin(validation_values)

                lines.append(test_values[b_index])
            doc.append(lines.copy())
    save_file = 'bvtest_'+para_name[parameter]
    with open(save_path + save_file + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)

def findbest_tvratio_test(multi_paths, multi_names, parameter, save_path, if_max, pt):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    doc = [doc_name]

    train_files = []

    for multi_path in multi_paths:
        train_files.append(helpers.get_m_files(multi_path + 'train/', [parameter]))
    readers = []
    for train_file in train_files:
        readers.append(csv.reader(train_file[0]))

    train_datas = []
    for reader in readers:
        train_datas.append(list(reader))

    fileLists = dictionaries.get_filelists()
    t = 0
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i]) - 1):
            lines = [fileLists[i][j] + '_' + fileLists[i][j + 1]]

            while train_datas[0][t][0] != fileLists[i][j]:
                t += 1

            for train_data, multi_path, multi_name in zip(train_datas, multi_paths, multi_names):
                train_values = list(map(lambda x: float(x), train_data[t][1:]))
                validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j] + '.csv', header=1,
                                              index_col=1)
                test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                validation_values = validation_file[para_name[parameter]].values
                train_values = np.array(train_values)

                balance = pt * train_values + (1-pt) * validation_values
                test_values = test_file[para_name[parameter]].values

                if if_max:
                    b_index = np.argmax(balance)
                else:
                    b_index = np.argmin(balance)

                lines.append(test_values[b_index])
            doc.append(lines.copy())
    save_file = 'tvratio_test_' + para_name[parameter]
    with open(save_path + save_file + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)

def find_tv_balancebest_test(multi_paths, multi_names,parameters, target,save_path, if_max,pt, pratios,random_size = 0, best_size = 0):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    doc = [doc_name]

    train_files = []
    for multi_path in multi_paths:
        train_files.append(helpers.get_m_files(multi_path + 'train/', parameters))

    if len(parameters) == 2:
        readers1 = []
        readers2 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
        train_datas1 = []
        train_datas2 = []
        for reader in zip(readers1,readers2):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))

        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = [fileLists[i][j] + '_' + fileLists[i][j + 1]]

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1,train_data2, multi_path, multi_name in zip(train_datas1, train_datas2, multi_paths, multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j] + '.csv', header=1,
                                                  index_col=1)
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    validation_values1 = validation_file[para_name[parameters[0]]].values
                    validation_values2 = validation_file[para_name[parameters[1]]].values
                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)

                    standard_validation_values1 = minmaxScaler(validation_values1)
                    standard_validation_values2 = minmaxScaler(validation_values2)

                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)

                    if parameters[0] == 0:
                        validation_values = pratios[0] * standard_validation_values1 - pratios[1] * standard_validation_values2
                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2
                    else:
                        print('error parameters!!!')

                    balance = pt * train_values + (1 - pt) * validation_values
                    test_values = test_file[para_name[target]].values
                    if random_size > 1 and best_size == 0:
                        size = min(random_size, len(balance))

                    elif best_size == 0 and random_size > 0:
                        size = int(random_size * len(balance))
                    elif best_size > 1 and random_size == 0:
                        size = min(best_size, len(balance))
                    elif best_size > 0 and random_size == 0:
                        size = int(best_size * len(balance))
                    else:
                        print('error random best size!')

                    if best_size == 0:
                        index_population = [i_tmp for i_tmp in range(0, len(balance))]
                        random_index = random.sample(index_population, size)
                        balance = balance[random_index]
                        test_values = test_values[random_index]

                        if if_max:
                            b_index = np.argmax(balance)
                        else:
                            b_index = np.argmin(balance)
                        lines.append(test_values[b_index])
                    else:
                        new_balance = []
                        new_test  = []
                        for choose in range(size):
                            if if_max:
                                choose_index = np.argmax(balance)
                            else:
                                choose_index = np.argmin(balance)


                            new_balance.append(balance[choose_index])
                            balance = np.delete(balance, choose_index)
                            new_test.append(test_values[choose_index])
                            test_values = np.delete(test_values, choose_index)
                        lines.append(random.choice(new_test))
                doc.append(lines.copy())
    elif len(parameters) == 3:
        readers1 = []
        readers2 = []
        readers3 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
            readers3.append(csv.reader(train_file[2]))
        train_datas1 = []
        train_datas2 = []
        train_datas3 = []
        for reader in zip(readers1,readers2,readers3):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))
            train_datas3.append(list(reader[2]))
        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = [fileLists[i][j] + '_' + fileLists[i][j + 1]]

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1, train_data2,train_data3, multi_path, multi_name in zip(train_datas1, train_datas2, train_datas3, multi_paths,
                                                                            multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    train_values3 = list(map(lambda x: float(x), train_data3[t][1:]))
                    validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j] + '.csv', header=1,
                                                  index_col=1)
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    validation_values1 = validation_file[para_name[parameters[0]]].values
                    validation_values2 = validation_file[para_name[parameters[1]]].values
                    validation_values3 = validation_file[para_name[parameters[2]]].values
                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)
                    train_values3 = np.array(train_values3)

                    standard_validation_values1 = minmaxScaler(validation_values1)
                    standard_validation_values2 = minmaxScaler(validation_values2)
                    standard_validation_values3 = minmaxScaler(validation_values3)
                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)
                    standard_train_values3 = minmaxScaler(train_values3)
                    if parameters[0] == 0:
                        validation_values = pratios[0] * standard_validation_values1 - pratios[1] * standard_validation_values2 - pratios[2] * standard_validation_values3

                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2 - pratios[2] * standard_train_values3[2]
                    else:
                        print('error parameters!!!!')
                    balance = pt * train_values + (1 - pt) * validation_values
                    test_values = test_file[para_name[target]].values

                    if random_size > 1 and best_size == 0:
                        size = min(random_size, len(balance))

                    elif best_size == 0 and random_size > 0:
                        size = int(random_size * len(balance))
                    elif best_size > 1 and random_size == 0:
                        size = min(best_size, len(balance))
                    elif best_size > 0 and random_size == 0:
                        size = int(best_size * len(balance))
                    else:
                        print('error random best size!')

                    if best_size == 0:
                        index_population = [i_tmp for i_tmp in range(0, len(balance))]
                        random_index = random.sample(index_population, size)
                        balance = balance[random_index]
                        test_values = test_values[random_index]

                        if if_max:
                            b_index = np.argmax(balance)
                        else:
                            b_index = np.argmin(balance)
                        lines.append(test_values[b_index])
                    else:
                        new_balance = []
                        new_test = []
                        for choose in range(size):
                            if if_max:
                                choose_index = np.argmax(balance)
                            else:
                                choose_index = np.argmin(balance)

                            new_balance.append(balance[choose_index])
                            balance = np.delete(balance, choose_index)
                            new_test.append(test_values[choose_index])
                            test_values = np.delete(test_values, choose_index)
                        lines.append(random.choice(new_test))
                doc.append(lines.copy())
    else:
        print('error target num', len(parameters))


    save_file = 'balance_tvratio_test_' + para_name[target]
    with open(save_path + save_file + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)

def find_corresponding_model(multi_paths, multi_names,parameters, target,save_path, if_max,pt, pratios,random_size = 0, best_size = 0):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    target_num = len(target)
    for multi_name in multi_names:
        doc_name.append(multi_name)
    docs = []
    for i_docs in range(target_num):
        doc = [doc_name]
        docs.append(doc)

    train_files = []
    for multi_path in multi_paths:
        train_files.append(helpers.get_m_files(multi_path + 'train/', parameters))

    if len(parameters) == 2:
        readers1 = []
        readers2 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
        train_datas1 = []
        train_datas2 = []
        for reader in zip(readers1,readers2):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))

        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = []
                for i_lines in range(target_num):
                    lines.append([fileLists[i][j] + '_' + fileLists[i][j + 1]])

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1,train_data2, multi_path, multi_name in zip(train_datas1, train_datas2, multi_paths, multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j] + '.csv', header=1,
                                                  index_col=1)
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    validation_values1 = validation_file[para_name[parameters[0]]].values
                    validation_values2 = validation_file[para_name[parameters[1]]].values
                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)

                    standard_validation_values1 = minmaxScaler(validation_values1)
                    standard_validation_values2 = minmaxScaler(validation_values2)

                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)

                    if parameters[0] == 0:
                        validation_values = pratios[0] * standard_validation_values1 - pratios[1] * standard_validation_values2
                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2
                    else:
                        print('error parameters!!!')

                    balance = pt * train_values + (1 - pt) * validation_values
                    test_values = []
                    for i_target in target:
                        test_values.append(test_file[para_name[i_target]].values)
                    if random_size > 1 and best_size == 0:
                        size = min(random_size, len(balance))

                    elif best_size == 0 and random_size > 0:
                        size = int(random_size * len(balance))
                    elif best_size > 1 and random_size == 0:
                        size = min(best_size, len(balance))
                    elif best_size > 0 and random_size == 0:
                        size = int(best_size * len(balance))
                    else:
                        print('error random best size!')

                    if best_size == 0:
                        index_population = [i_tmp for i_tmp in range(0, len(balance))]
                        random_index = random.sample(index_population, size)
                        balance = balance[random_index]
                        new_test_values = []
                        for test_value in test_values:
                            new_test_values.append(test_value[random_index])

                        if if_max:
                            b_index = np.argmax(balance)
                        else:
                            b_index = np.argmin(balance)
                        for line, new_test_value in zip(lines, new_test_values):
                            line.append(new_test_value[b_index])
                    else:
                        new_balance = []
                        new_tests  = []
                        for i_ntests in range(target_num):
                            new_tests.append([])
                        for choose in range(size):
                            if if_max:
                                choose_index = np.argmax(balance)
                            else:
                                choose_index = np.argmin(balance)


                            new_balance.append(balance[choose_index])
                            balance = np.delete(balance, choose_index)
                            for test_value, new_test in zip(test_values, new_tests):
                                new_test.append(test_value[choose_index])
                                test_value = np.delete(test_value, choose_index)
                        for line, new_test in zip(lines, new_tests):
                            line.append(random.choice(new_test))
                for doc, line in zip(docs, lines):
                    doc.append(line.copy())
    elif len(parameters) == 3:
        readers1 = []
        readers2 = []
        readers3 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
            readers3.append(csv.reader(train_file[2]))
        train_datas1 = []
        train_datas2 = []
        train_datas3 = []
        for reader in zip(readers1,readers2,readers3):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))
            train_datas3.append(list(reader[2]))
        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = []
                for i_lines in range(target_num):
                    lines.append([fileLists[i][j] + '_' + fileLists[i][j + 1]])

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1, train_data2,train_data3, multi_path, multi_name in zip(train_datas1, train_datas2, train_datas3, multi_paths,
                                                                            multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    train_values3 = list(map(lambda x: float(x), train_data3[t][1:]))
                    validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j] + '.csv', header=1,
                                                  index_col=1)
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    validation_values1 = validation_file[para_name[parameters[0]]].values
                    validation_values2 = validation_file[para_name[parameters[1]]].values
                    validation_values3 = validation_file[para_name[parameters[2]]].values
                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)
                    train_values3 = np.array(train_values3)

                    standard_validation_values1 = minmaxScaler(validation_values1)
                    standard_validation_values2 = minmaxScaler(validation_values2)
                    standard_validation_values3 = minmaxScaler(validation_values3)
                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)
                    standard_train_values3 = minmaxScaler(train_values3)
                    if parameters[0] == 0:
                        validation_values = pratios[0] * standard_validation_values1 - pratios[1] * standard_validation_values2 - pratios[2] * standard_validation_values3

                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2 - pratios[2] * standard_train_values3[2]
                    else:
                        print('error parameters!!!!')
                    balance = pt * train_values + (1 - pt) * validation_values
                    test_values = []
                    for i_target in target:
                        test_values.append(test_file[para_name[i_target]].values)

                    if random_size > 1 and best_size == 0:
                        size = min(random_size, len(balance))

                    elif best_size == 0 and random_size > 0:
                        size = int(random_size * len(balance))
                    elif best_size > 1 and random_size == 0:
                        size = min(best_size, len(balance))
                    elif best_size > 0 and random_size == 0:
                        size = int(best_size * len(balance))
                    else:
                        print('error random best size!')

                    if best_size == 0:
                        index_population = [i_tmp for i_tmp in range(0, len(balance))]
                        random_index = random.sample(index_population, size)
                        print(balance, random_index)
                        balance = balance[random_index]
                        print('attention!!!!!!')
                        print(test_values, random_index)
                        new_test_values = []
                        for test_value in test_values:
                            print('\n\n', test_value, random_index)
                            new_test_values.append(test_value[random_index])

                        if if_max:
                            b_index = np.argmax(balance)
                        else:
                            b_index = np.argmin(balance)
                        for line, new_test_value in zip(lines, new_test_values):
                            line.append(new_test_value[b_index])
                    else:
                        new_balance = []
                        new_tests = []
                        for i_ntests in range(target_num):
                            new_tests.append([])
                        for choose in range(size):
                            if if_max:
                                choose_index = np.argmax(balance)
                            else:
                                choose_index = np.argmin(balance)

                            new_balance.append(balance[choose_index])
                            balance = np.delete(balance, choose_index)
                            for test_value, new_test in zip(test_values, new_tests):
                                new_test.extend(test_value[choose_index])
                                test_value = np.delete(test_value, choose_index)
                        for line, new_test in zip(lines, new_tests):
                            line.append(random.choice(new_test))
                for doc, line in zip(docs, lines):
                    doc.append(line.copy())
    else:
        print('error target num', len(parameters))

    for doc, i_target in zip(docs, target):
        save_file = 'choosemodel_' + para_name[i_target]
        with open(save_path + save_file + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in doc:
                writer.writerow(row)

def find_corresponding_model_withoutsplit(multi_paths, multi_names,parameters, target,save_path, if_max,pratios,random_size = 0, best_size = 0):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    target_num = len(target)
    for multi_name in multi_names:
        doc_name.append(multi_name)
    docs = []
    for i_docs in range(target_num):
        doc = [doc_name]
        docs.append(doc)

    train_files = []
    for multi_path in multi_paths:
        train_files.append(helpers.get_m_files(multi_path + 'train/', parameters))

    if len(parameters) == 2:
        readers1 = []
        readers2 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
        train_datas1 = []
        train_datas2 = []
        for reader in zip(readers1,readers2):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))

        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = []
                for i_lines in range(target_num):
                    lines.append([fileLists[i][j] + '_' + fileLists[i][j + 1]])

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1,train_data2, multi_path, multi_name in zip(train_datas1, train_datas2, multi_paths, multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)
                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)

                    if parameters[0] == 0:
                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2
                    else:
                        print('error parameters!!!')

                    balance = train_values
                    test_values = []
                    for i_target in target:
                        test_values.append(test_file[para_name[i_target]].values)
                    if random_size > 1 and best_size == 0:
                        size = min(random_size, len(balance))

                    elif best_size == 0 and random_size > 0:
                        size = int(random_size * len(balance))
                    elif best_size > 1 and random_size == 0:
                        size = min(best_size, len(balance))
                    elif best_size > 0 and random_size == 0:
                        size = int(best_size * len(balance))
                    else:
                        print('error random best size!')

                    if best_size == 0:
                        index_population = [i_tmp for i_tmp in range(0, len(balance))]
                        random_index = random.sample(index_population, size)
                        balance = balance[random_index]
                        new_test_values = []
                        for test_value in test_values:
                            new_test_values.append(test_value[random_index])

                        if if_max:
                            b_index = np.argmax(balance)
                        else:
                            b_index = np.argmin(balance)
                        for line, new_test_value in zip(lines, new_test_values):
                            line.append(new_test_value[b_index])
                    else:
                        new_balance = []
                        new_tests  = []
                        for i_ntests in range(target_num):
                            new_tests.append([])
                        for choose in range(size):
                            if if_max:
                                choose_index = np.argmax(balance)
                            else:
                                choose_index = np.argmin(balance)


                            new_balance.append(balance[choose_index])
                            balance = np.delete(balance, choose_index)
                            for test_value, new_test in zip(test_values, new_tests):
                                new_test.append(test_value[choose_index])
                                test_value = np.delete(test_value, choose_index)
                        for line, new_test in zip(lines, new_tests):
                            line.append(random.choice(new_test))
                for doc, line in zip(docs, lines):
                    doc.append(line.copy())
    elif len(parameters) == 3:
        readers1 = []
        readers2 = []
        readers3 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
            readers3.append(csv.reader(train_file[2]))
        train_datas1 = []
        train_datas2 = []
        train_datas3 = []
        for reader in zip(readers1,readers2,readers3):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))
            train_datas3.append(list(reader[2]))
        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = []
                for i_lines in range(target_num):
                    lines.append([fileLists[i][j] + '_' + fileLists[i][j + 1]])

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1, train_data2,train_data3, multi_path, multi_name in zip(train_datas1, train_datas2, train_datas3, multi_paths,
                                                                            multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    train_values3 = list(map(lambda x: float(x), train_data3[t][1:]))
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)
                    train_values3 = np.array(train_values3)

                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)
                    standard_train_values3 = minmaxScaler(train_values3)
                    if parameters[0] == 0:
                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2 - pratios[2] * standard_train_values3
                    else:
                        print('error parameters!!!!')
                    balance = train_values
                    test_values = []
                    for i_target in target:
                        test_values.append(test_file[para_name[i_target]].values)

                    if random_size > 1 and best_size == 0:
                        size = min(random_size, len(balance))

                    elif best_size == 0 and random_size > 0:
                        size = int(random_size * len(balance))
                    elif best_size > 1 and random_size == 0:
                        size = min(best_size, len(balance))
                    elif best_size > 0 and random_size == 0:
                        size = int(best_size * len(balance))
                    else:
                        print('error random best size!')

                    if best_size == 0:
                        index_population = [i_tmp for i_tmp in range(0, len(balance))]
                        random_index = random.sample(index_population, size)
                        print(balance, random_index)
                        balance = balance[random_index]
                        print('attention!!!!!!')
                        print(test_values, random_index)
                        new_test_values = []
                        for test_value in test_values:
                            print('\n\n', test_value, random_index)
                            new_test_values.append(test_value[random_index])

                        if if_max:
                            b_index = np.argmax(balance)
                        else:
                            b_index = np.argmin(balance)
                        for line, new_test_value in zip(lines, new_test_values):
                            line.append(new_test_value[b_index])
                    else:
                        new_balance = []
                        new_tests = []
                        for i_ntests in range(target_num):
                            new_tests.append([])
                        for choose in range(size):
                            if if_max:
                                choose_index = np.argmax(balance)
                            else:
                                choose_index = np.argmin(balance)

                            new_balance.append(balance[choose_index])
                            balance = np.delete(balance, choose_index)
                            for test_value, new_test in zip(test_values, new_tests):
                                new_test.extend(test_value[choose_index])
                                test_value = np.delete(test_value, choose_index)
                        for line, new_test in zip(lines, new_tests):
                            line.append(random.choice(new_test))
                for doc, line in zip(docs, lines):
                    doc.append(line.copy())
    else:
        print('error target num', len(parameters))

    for doc, i_target in zip(docs, target):
        save_file = 'choosemodel_' + para_name[i_target]
        with open(save_path + save_file + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in doc:
                writer.writerow(row)

def find_validation_trainbest_test(multi_paths, multi_names,parameters, target,save_path, if_max, pratios, best_size = 0):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    doc_name = ['filename']
    for multi_name in multi_names:
        doc_name.append(multi_name)
    doc = [doc_name]

    train_files = []
    for multi_path in multi_paths:
        train_files.append(helpers.get_m_files(multi_path + 'train/', parameters))

    if len(parameters) == 2:
        readers1 = []
        readers2 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
        train_datas1 = []
        train_datas2 = []
        for reader in zip(readers1,readers2):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))

        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = [fileLists[i][j] + '_' + fileLists[i][j + 1]]

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1,train_data2, multi_path, multi_name in zip(train_datas1, train_datas2, multi_paths, multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j] + '.csv', header=1,
                                                  index_col=1)
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    validation_values1 = validation_file[para_name[parameters[0]]].values
                    validation_values2 = validation_file[para_name[parameters[1]]].values
                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)

                    standard_validation_values1 = minmaxScaler(validation_values1)
                    standard_validation_values2 = minmaxScaler(validation_values2)

                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)

                    if parameters[0] == 0:

                        validation_values = pratios[0] * standard_validation_values1 - pratios[1] * standard_validation_values2

                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2
                    else:
                        print('error parameters!!')


                    test_values = test_file[para_name[target]].values


                    if best_size > 1:
                        size = min(best_size, len(train_values))
                    else:
                        size = int(best_size * len(train_values))
                    new_train_values = []
                    new_validation_values = []
                    new_test_values  = []
                    for choose in range(size):
                        if if_max:
                            choose_index = np.argmax(train_values)
                        else:
                            choose_index = np.argmin(train_values)

                        new_train_values.append(train_values[choose_index])
                        train_values = np.delete(train_values, choose_index)
                        new_validation_values.append(validation_values)
                        validation_values = np.delete(validation_values, choose_index)
                        new_test_values.append(test_values[choose_index])
                        test_values = np.delete(test_values, choose_index)
                    if if_max:
                        b_index = np.argmax(new_validation_values)
                    else:
                        b_index = np.argmin(new_validation_values)
                    lines.append(new_test_values[b_index])
                doc.append(lines.copy())
    elif len(parameters) == 3:
        readers1 = []
        readers2 = []
        readers3 = []
        for train_file in train_files:
            readers1.append(csv.reader(train_file[0]))
            readers2.append(csv.reader(train_file[1]))
            readers3.append(csv.reader(train_file[2]))
        train_datas1 = []
        train_datas2 = []
        train_datas3 = []
        for reader in zip(readers1,readers2,readers3):
            train_datas1.append(list(reader[0]))
            train_datas2.append(list(reader[1]))
            train_datas3.append(list(reader[2]))
        fileLists = dictionaries.get_filelists()
        t = 0
        for i in range(len(fileLists)):
            for j in range(1, len(fileLists[i]) - 1):
                lines = [fileLists[i][j] + '_' + fileLists[i][j + 1]]

                while train_datas1[0][t][0] != fileLists[i][j]:
                    t += 1

                for train_data1, train_data2,train_data3, multi_path, multi_name in zip(train_datas1, train_datas2, train_datas3, multi_paths,
                                                                            multi_names):
                    train_values1 = list(map(lambda x: float(x), train_data1[t][1:]))
                    train_values2 = list(map(lambda x: float(x), train_data2[t][1:]))
                    train_values3 = list(map(lambda x: float(x), train_data3[t][1:]))
                    validation_file = pd.read_csv(multi_path + 'validation/' + fileLists[i][j] + '.csv', header=1,
                                                  index_col=1)
                    test_file = pd.read_csv(multi_path + 'test/' + fileLists[i][j + 1] + '.csv', header=1, index_col=1)

                    validation_values1 = validation_file[para_name[parameters[0]]].values
                    validation_values2 = validation_file[para_name[parameters[1]]].values
                    validation_values3 = validation_file[para_name[parameters[2]]].values
                    train_values1 = np.array(train_values1)
                    train_values2 = np.array(train_values2)
                    train_values3 = np.array(train_values3)

                    standard_validation_values1 = minmaxScaler(validation_values1)
                    standard_validation_values2 = minmaxScaler(validation_values2)
                    standard_validation_values3 = minmaxScaler(validation_values3)
                    standard_train_values1 = minmaxScaler(train_values1)
                    standard_train_values2 = minmaxScaler(train_values2)
                    standard_train_values3 = minmaxScaler(train_values3)

                    if parameters[0] == 0:
                        validation_values = pratios[0] * standard_validation_values1 - pratios[1] * standard_validation_values2 - pratios[2] * standard_validation_values3

                        train_values = pratios[0] * standard_train_values1 - pratios[1] * standard_train_values2 - pratios[2] * standard_train_values3[2]
                    else:
                        print('error parameters!!')

                    test_values = test_file[para_name[target]].values

                    if best_size > 1:
                        size = min(best_size, len(train_values))
                    else:
                        size = int(best_size * len(train_values))
                    new_train_values = []
                    new_validation_values = []
                    new_test_values = []
                    for choose in range(size):
                        if if_max:
                            choose_index = np.argmax(train_values)
                        else:
                            choose_index = np.argmin(train_values)

                        new_train_values.append(train_values[choose_index])
                        train_values = np.delete(train_values, choose_index)
                        new_validation_values.append(validation_values)
                        validation_values = np.delete(validation_values, choose_index)
                        new_test_values.append(test_values[choose_index])
                        test_values = np.delete(test_values, choose_index)

                    new_test_values = np.array(new_test_values)
                    new_validation_values = np.array(new_train_values)
                    new_train_values = np.array(new_train_values)
                    if if_max:
                        b_index = np.argmax(new_validation_values)
                    else:
                        b_index = np.argmin(new_validation_values)
                    lines.append(new_test_values[b_index])
                doc.append(lines.copy())
    else:
        print('error target num', len(parameters))


    save_file = 'balance_tvratio_test_' + para_name[target]
    with open(save_path + save_file + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)



def find_train_test_rank(moea, op_target, target):
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    moea_name = dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1])
    op_target_name = dictionaries.get_target_composition(op_target)
    train_path = '../results/multi-objective/' + moea_name + '/' + op_target_name + '/train/'
    test_path = '../results/multi-objective/' + moea_name + '/' + op_target_name + '/test/'
    save_path =  '../results/tables/'
    save_file = para_name[target]+'_ttrank'

    doc = [['filename/(train/test)']]

    train_file = helpers.get_m_files(train_path, [target])
    train_reader = csv.reader(train_file[0])
    train_data = list(train_reader)


    fileLists = dictionaries.get_filelists()
    t = 0
    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])-1):
            while t < len(train_data):
                if train_data[t][0] == fileLists[i][j]:
                    break
                else:
                    t += 1

            train_values = list(map(lambda x : float(x), train_data[t][1:]))
            train_lines = tgf.bubbleSort(train_values.copy())
            train_lines.insert(0, fileLists[i][j]+'/train')

            test_file = pd.read_csv(test_path+fileLists[i][j+1]+'.csv', header=1, index_col=1)
            test_values = test_file[para_name[target]].values
            test_lines = tgf.bubbleSort(test_values.copy())
            test_lines.insert(0, fileLists[i][j+1]+'/test')

            doc.append(train_lines.copy())
            doc.append(test_lines.copy())
    with open(save_path + save_file+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in doc:
            writer.writerow(row)




'''
==============================================================================================================
============================================具体调用===========================================================
==============================================================================================================
'''
def ssmm_table_train(moea_list, sklearn_list, soea_list, op_targets , target, if_max = True, if_split = False):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0])+'/'+dictionaries.get_moea_name(moea[1]))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    soea_names = []
    for soea in soea_list:
        soea_names.append(dictionaries.get_model_method_name(soea[0])+'/'+dictionaries.get_soea_name(soea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            if if_split:
                multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/train/')
            else:
                multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/train/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/train/')
    for soea_name in soea_names:
        single_paths.append('../results/single-objective/' + soea_name  + '/train/')
    single_names = sklearn_names +soea_names
    save_path = '../results/tables/'
    table_for_train(single_paths=single_paths, single_names=single_names,
                    multi_paths=multi_paths,multi_names=multi_names,
                    parameters=target, save_path=save_path, if_max = if_max)

def ssmm_table_test(moea_list, sklearn_list, soea_list, op_targets, target, if_max = True, if_split = False):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0])+'/'+dictionaries.get_moea_name(moea[1]))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    soea_names = []
    for soea in soea_list:
        soea_names.append(dictionaries.get_model_method_name(soea[0])+'/'+dictionaries.get_soea_name(soea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            if if_split:
                multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/test/')
            else:
                multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/test/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/test/')
    for soea_name in soea_names:
        single_paths.append('../results/single-objective/' + soea_name + '/test/')
    single_names = sklearn_names + soea_names
    save_path = '../results/tables/'
    table_for_test(single_paths=single_paths, single_names=single_names,
                    multi_paths=multi_paths,multi_names=multi_names,
                    parameters=target, save_path=save_path, if_max = if_max)

'''
----------------------------------------------------------------------------------------------------------------
'''
def ssmm_table_range_train(moea_list, sklearn_list, soea_list, op_targets , target, if_split = False):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    soea_names = []
    for soea in soea_list:
        soea_names.append(dictionaries.get_model_method_name(soea[0]) + '/' + dictionaries.get_soea_name(soea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            if if_split:
                multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/train/')
            else:
                multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/train/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/train/')
    for soea_name in soea_names:
        single_paths.append('../results/single-objective/' + soea_name  + '/train/')
    single_names = sklearn_names +soea_names
    save_path = '../results/tables/'
    table_range_train(single_paths=single_paths, single_names=single_names,
                    multi_paths=multi_paths,multi_names=multi_names,
                    parameters=target, save_path=save_path)

def ssmm_table_range_test(moea_list, sklearn_list, soea_list, op_targets, target, if_split = False):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    soea_names = []
    for soea in soea_list:
        soea_names.append(dictionaries.get_model_method_name(soea[0]) + '/' + dictionaries.get_soea_name(soea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            if if_split:
                multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/test/')
            else:
                multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/test/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/test/')
    for soea_name in soea_names:
        single_paths.append('../results/single-objective/' + soea_name + '/test/')
    single_names = sklearn_names + soea_names
    save_path = '../results/tables/'
    table_range_test(single_paths=single_paths, single_names=single_names,
                    multi_paths=multi_paths,multi_names=multi_names,
                    parameters=target, save_path=save_path)


'''
---------------------------------------------------------------------------------------------------------------------
'''
def ssmm_find_best_other_train(moea_list, sklearn_list, soea_list, op_targets , best_target, target, if_max):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    soea_names = []
    for soea in soea_list:
        soea_names.append(dictionaries.get_model_method_name(soea[0]) + '/' + dictionaries.get_soea_name(soea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/train/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/train/')
    for soea_name in soea_names:
        single_paths.append('../results/single-objective/' + soea_name  + '/train/')
    single_names = sklearn_names +soea_names
    save_path = '../results/tables/'
    find_m_corresponding_value_train(single_paths = single_paths, multi_paths = multi_paths,
                                     max_parameter = best_target, target_parameter = target,
                                     single_names = single_names, multi_names = multi_names, save_path = save_path,
                                     if_max = if_max)

def ssmm_find_best_other_test(moea_list, sklearn_list, soea_list, op_targets,best_target, target, if_max):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    soea_names = []
    for soea in soea_list:
        soea_names.append(dictionaries.get_model_method_name(soea[0]) + '/' + dictionaries.get_soea_name(soea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/test/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/test/')
    for soea_name in soea_names:
        single_paths.append('../results/single-objective/' + soea_name + '/test/')
    single_names = sklearn_names + soea_names
    save_path = '../results/tables/'
    find_m_corresponding_value_test(single_paths=single_paths, multi_paths=multi_paths,
                                     max_parameter=best_target, target_parameter=target,
                                     single_names=single_names, multi_names=multi_names, save_path=save_path,
                                     if_max=if_max)

# path是放置多目标算法的位置


# 寻找非分割的训练最优的对应的测试
def make_btrain_test(moea_list, op_targets, target, if_max):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    save_path = '../results/tables/'
    save_file = 'best_train_test_'
    find_train_best_test_value(multi_paths =multi_paths, multi_names=multi_names, parameter = target, save_path=save_path, if_max=if_max,save_file = save_file)

# 寻找分割后的validation最优的
def make_bv_test(moea_list, op_targets, target, if_max):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    save_path = '../results/tables/'
    find_validation_best_test(multi_paths =multi_paths, multi_names=multi_names, parameter = target, save_path=save_path, if_max=if_max)
#寻找按比例的train和validation最优的
def make_best_tvratio_test(moea_list, op_targets, target, if_max, train_ratio):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    save_path = '../results/tables/'
    findbest_tvratio_test(multi_paths =multi_paths, multi_names=multi_names, parameter = target, save_path=save_path, if_max=if_max, pt = train_ratio)


# 寻找参数加权，train/validation加权后，最优的，同时提供先随机后最有和先最优后随机的选项
def make_balance_tvratio(moea_list, op_targets,parameters, target, if_max, train_ratio, pratios, random_size = 0, best_size = 0):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    save_path = '../results/tables/'
    find_tv_balancebest_test(multi_paths=multi_paths, multi_names=multi_names, parameters=parameters,target=target, save_path=save_path,
                          if_max=if_max, pt=train_ratio, pratios=pratios, random_size=random_size, best_size = best_size)

# 从最优的几个train中，选出最优的validation的值
def make_validation_trainbest_test(moea_list, op_targets,parameters, target, if_max, pratios,best_size = 0):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    save_path = '../results/tables/'
    find_validation_trainbest_test(multi_paths=multi_paths, multi_names=multi_names, parameters=parameters,target=target, save_path=save_path,
                          if_max=if_max, pratios=pratios, best_size = best_size)
def choose_model(moea_list,op_targets,parameters, target, if_max, train_ratio, pratios, if_split,random_size = 0, best_size = 0,soea_list = []):
    moea_names = []
    for moea in moea_list:
        moea_names.append(dictionaries.get_model_method_name(moea[0]) + '/' + dictionaries.get_moea_name(moea[1]))
    op_target_names = []
    for op_target in op_targets:
        op_target_names.append(dictionaries.get_target_composition(op_target))
    multi_paths = []
    multi_names = []
    for moea_name in moea_names:
        for op_target_name in op_target_names:
            if if_split:
                multi_paths.append('../results/split-train/' + moea_name + '/' + op_target_name + '/')
            else:
                multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
            else:
                write_algorithm = moea_name
            write_target = op_target_name.replace('nonz', 'NNZ')
            multi_names.append(write_algorithm + '/' + write_target)
    save_path = '../results/tables/'

    if len(soea_list) > 0:
        for soea in soea_list:
            soea_name = dictionaries.get_model_method_name(soea[0]) + '/' + dictionaries.get_soea_name(soea[1])
            multi_paths.append('../results/multi-objective/' + soea_name+'/')
            multi_names.append(soea_name)
        

    if if_split:
        find_corresponding_model(multi_paths=multi_paths, multi_names=multi_names, parameters=parameters,target=target, save_path=save_path,
                          if_max=if_max, pt = train_ratio, pratios=pratios, random_size=random_size, best_size = best_size)
    else:
        find_corresponding_model_withoutsplit(multi_paths=multi_paths, multi_names=multi_names, parameters=parameters,target=target, save_path=save_path,
                          if_max=if_max, pratios=pratios, random_size=random_size, best_size = best_size)   
'''
=========================================================================================================================================
'''

# algoritm指的是算法的名字，即列名；
# type =1 是大于；type=2是大于等于。
def counting(path, filename, algorithm, type = 1):
    df = pd.read_csv(path + filename + '.csv', header= 0, index_col= 0)
    compare = df[algorithm].values.copy()
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            print(i, j, compare[i])
            if type == 1 and df.iloc[i,j] > compare[i]:
                df.iloc[i,j] = 1
            elif type == 2 and df.iloc[i,j] >= compare[i]:
                df.iloc[i,j] = 1
            else:
                df.iloc[i,j] = 0
            print(compare[i])
    df.to_csv(path+filename+'_counts'+str(type)+'.csv')



