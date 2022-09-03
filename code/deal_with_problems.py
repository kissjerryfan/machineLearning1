import helpers
import pandas as pd
import target_functions as tgf
import csv

def regenerate_msdp_train_target(path):
    fileLists = helpers.get_filelists()
    data_path = 'data/'

    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}

    doc1_names = []
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

    for i in range(len(fileLists)):
        for j in range(1, len(fileLists[i])):
            params = pd.read_csv(path+fileLists[i][j]+'.csv', index_col = 0, header = 0)
            params = params.values
            X, y = helpers.getfeatures(data_path, fileLists[i][j])
            f1_values = [fileLists[i][j]]
            f2_values = [fileLists[i][j]]
            f3_values = [fileLists[i][j]]
            f4_values = [fileLists[i][j]]
            f5_values = [fileLists[i][j]]
            for t in range(params.shape[0]):
                predvalue = tgf.linear_predict(X, params[t])
                f1 = tgf.FPA(predvalue, y)
                f2 = tgf.AAE(predvalue, y)
                f3 = tgf.numOfnonZero(params[t])
                f4 = tgf.l1_values(params[t])
                f5 = tgf.MSE(predvalue, y)
                f1_values += [f1]
                f2_values += [f2]
                f3_values += [f3]
                f4_values += [f4]
                f5_values += [f5]
            doc2.append(f1_values.copy())
            doc3.append(f2_values.copy())
            doc4.append(f3_values.copy())
            doc5.append(f4_values.copy())
            doc6.append(f5_values.copy())

    with open(path + 'doc2.csv', 'w', newline='') as file2:
        writer2 = csv.writer(file2)
        for row in doc2:
            writer2.writerow(row)

    with open(path + 'doc3.csv', 'w', newline='') as file3:
        writer3 = csv.writer(file3)
        for row in doc3:
            writer3.writerow(row)

    with open(path + 'doc4.csv', 'w', newline='') as file4:
        writer4 = csv.writer(file4)
        for row in doc4:
            writer4.writerow(row)

    with open(path + 'doc5.csv', 'w', newline='') as file5:
        writer5 = csv.writer(file5)
        for row in doc5:
            writer5.writerow(row)

    with open(path + 'doc6.csv', 'w', newline='') as file6:
        writer6 = csv.writer(file6)
        for row in doc6:
            writer6.writerow(row)

def deal_all(algorithms):
    targets = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2], [0, 1, 3], [0, 2, 4], [0, 3, 4]]
    for algorithm in algorithms:
        algorithm_name = helpers.get_moea_name(algorithm)
        for target in targets:
            target_name = helpers.get_target_composition(target)
            path = '../results/linear/' + algorithm_name + '/' + target_name + '/train/'
            regenerate_msdp_train_target(path=path)
            print(algorithm_name, target_name, 'successful!!')



