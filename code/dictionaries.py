


def get_moea_name(moea):
    algorithm_dict = {1: 'nsga2', 2: 'nsga2_DE', 3: 'nsga2_toZero', 4: 'nsga2_DE_toZero',
                      5: 'nsga3', 6: 'nsga3_DE', 7: 'awGA', 8: 'RVEA', 9: 'nsga2_10p_toZero',
                      10: 'nsga2_20p_toZero', 11: 'nsga2_30p_toZero', 12:'nsga2_random10p_toZero',
                      13:'nsga2_random20p_toZero', 14:'nsga2_random30p_toZero'}
    return algorithm_dict[moea]


def get_sklearn_name(num):
    algorithm_dict = {1: 'LinearRegression', 2: 'RidgeCV', 3: 'RidgeRegression', 4:'LassoLarsCV', 5:'MLPRegressor',
                      6:'RandomForestRegressor'}
    return algorithm_dict[num]

def get_soea_name(num):
    algorithm_dict = {1: 'CoDE', 2: 'DE_rand_1_bin', 3:'CoDE_toZero', 4: 'CoDE_10p_toZero', 5:'CoDE_20p_toZero',
                      6:'CoDE_10p_lr_toZero', 7:'CoDE_20p_lr_toZero', 8:'CoDE_random10p_toZero', 9:'CoDE_random20p_toZero',
                      10:'CoDE_random30p_toZero'}
    return algorithm_dict[num]

def get_model_method_name(num):
    model_dict = {1: 'linear', 2: 'BPNN', 3:'NN', 4:'MLP', 5:'mlp3', 6: 'mlp5', 7:'logistic'}
    return model_dict[num]

def get_targets_name(num):
    targets_dict = {0: 'FPA', 1:'AAE', 2:'numOfnonZero', 3:'l1', 4:'MSE'}
    return targets_dict[num]


'''
0---FPA
1---AAE
2---numOfnonZero
3---l1
4---MSE
'''
# 注意，顺序很重要！！！！


def get_target_composition(target):
    composition_dict = {0: 'FPA', 1: 'AAE', 2: 'nonz', 3: 'L1', 4: 'MSE'}
    target_name = ''
    for t in target:
        if target_name == '':
            target_name += composition_dict[t]
        else:
            target_name += '_' + composition_dict[t]
    return target_name
# 从指定的文件读入特征X, y

def get_filelists():
    fileLists = []
    fileLists.append(['ant', 'ant-1.3', 'ant-1.4', 'ant-1.5', 'ant-1.6', 'ant-1.7'])
    fileLists.append(['camel', 'camel-1.0', 'camel-1.2', 'camel-1.4', 'camel-1.6'])
    fileLists.append(['ivy', 'ivy-1.1', 'ivy-1.4', 'ivy-2.0'])
    fileLists.append(['jedit', 'jedit-3.2', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'])
    fileLists.append(['log4j', 'log4j-1.0', 'log4j-1.1', 'log4j-1.2'])
    fileLists.append(['lucene', 'lucene-2.0', 'lucene-2.2', 'lucene-2.4'])
    fileLists.append(['poi', 'poi-1.5', 'poi-2.0', 'poi-2.5', 'poi-3.0'])
    fileLists.append(['synapse', 'synapse-1.0', 'synapse-1.1', 'synapse-1.2'])
    fileLists.append(['velocity', 'velocity-1.4', 'velocity-1.5', 'velocity-1.6'])
    fileLists.append(['xalan', 'xalan-2.4', 'xalan-2.5', 'xalan-2.6', 'xalan-2.7'])
    fileLists.append(['xerces', 'xerces-init', 'xerces-1.2', 'xerces-1.3', 'xerces-1.4'])
    return fileLists

