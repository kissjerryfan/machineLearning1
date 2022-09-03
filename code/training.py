import comparison_algorithm
import helpers
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLarsCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import dictionaries

'''
=========================================训练模型=============================================================
'''

'''
=================================================通用函数================================================
'''

def run_ssdp_train(soea, predict_model):
    soea_name = dictionaries.get_soea_name(soea)
    predict_model_name = dictionaries.get_model_method_name(predict_model)
    folder_name = 'single-objective/'+ predict_model_name +'/'+ soea_name
    l = -20
    u = 20
    helpers.training_record_for_ssdp(predict_model=predict_model, l = l, u=u, save_folder=folder_name, soea = soea)


def run_msdp_train(moea, targets, predict_model):
    moea_name = dictionaries.get_moea_name(moea)
    predict_model_name = dictionaries.get_model_method_name(predict_model)
    targets_name = dictionaries.get_target_composition(targets)
    folder_name = 'multi-objective/'+ predict_model_name +'/'+ moea_name + '/' + targets_name
    l = -20
    u = 20
    helpers.training_record_for_msdp(save_folder=folder_name, target= targets, predict_model=predict_model, l = l, u = u,
                                     moea=moea)
def train_ssdp_m(soea, predict_model):
    soea_name = dictionaries.get_soea_name(soea)
    predict_model_name = dictionaries.get_model_method_name(predict_model)
    folder_name = 'multi-objective/'+ predict_model_name +'/'+ soea_name
    l = -20
    u = 20
    helpers.training_record_for_ssdp_m(predict_model=predict_model, l = l, u=u, save_folder=folder_name, soea = soea)


def split_train_test_msdp(moea, targets, predict_model, validation_size = 0.2):
    moea_name = dictionaries.get_moea_name(moea)
    predict_model_name = dictionaries.get_model_method_name(predict_model)
    targets_name = dictionaries.get_target_composition(targets)
    folder_name = 'split-train/' + predict_model_name + '/' + moea_name + '/' + targets_name
    l = -20
    u = 20
    helpers.train_validation_for_msdp(save_folder=folder_name, target=targets, predict_model=predict_model, l=l, u=u,
                                     moea=moea, validation_size=validation_size)















'''
----------------------------------------------单目标-------------------------------------------------
'''


def run_ssdp_bpnn():
    save_folder = '单目标优化__BPNN_NIND=100__MAX_GEN=100__CoDE_决策变量范围_[-1,1]'
    helpers.train_for_bpnn(save_folder)


def run_ssdp_linear_CoDE():
    save_folder = 'linear/single-objective/CoDE_new'
    method = 'linear'
    l = -20
    u = 20
    helpers.training_record_for_ssdp(method=method, l=l, u=u, save_folder=save_folder)


'''
-------------------------------------------多目标----------------------------------------------------
'''
#---------------------------------------------------------------------------------------
#----------------------------------------只谈目标不谈算法-----------------------------
#--------------------------------------------------------------------------
# FPA, nonz, MSE


def run_msdp_FPA_nonz_MSE_train(moea):
    target_name = 'FPA_nonz_MSE'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 2, 4]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)
# FPA, nonz, AAE


def run_msdp_FPA_AAE_nonz_train(moea):
    target_name = 'FPA_AAE_nonz'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 1, 2]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)
# FPA, nonz


def run_msdp_FPA_nonz_train(moea):

    target_name = 'FPA_nonz'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 2]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)
# FPA, MSE


def run_msdp_FPA_MSE_train(moea):

    target_name = 'FPA_MSE'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 4]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)

# FPA, L1, MSE


def run_msdp_FPA_L1_MSE_train(moea):
    target_name = 'FPA_L1_MSE'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 3, 4]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)
# FPA, L1, AAE


def run_msdp_FPA_AAE_L1_train(moea):

    target_name = 'FPA_AAE_L1'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 1, 3]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)

# FPA, L1


def run_msdp_FPA_L1_train(moea):

    target_name = 'FPA_L1'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 3]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)

# FPA, AAE


def run_msdp_FPA_AAE_train(moea):

    target_name = 'FPA_AAE'
    save_file = dictionaries.get_moea_name(moea) + '/' + target_name
    target = [0, 1]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = moea
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)


#----------------------------------------算法+目标
def run_msdp_fpa_aae_nonz_nsga2_toZero_train():
    save_file = '多目标优化_FPA+AAE+numofnonzero_nsga2_toZero'
    target = [0, 1, 2]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = 3
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)


def run_msdp_fpa_nonz_nsga2_DE_toZero_train():
    save_file = '多目标优化_FPA+numofnonzero_nsga2_DE_toZero'
    target = [0, 2]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = 4
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)


def run_msdp_fpa_aae_nonz_nsga2_DE_toZero_train():
    save_file = '多目标优化_FPA+AAE+numofnonzero_nsga2_DE_toZero'
    target = [0, 1, 2]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = 4
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing, maxgen=maxgen)


def run_msdp_fpa_l1_nsga2_train():
    save_file = '多目标优化_FPA+L1_nsga2'
    target = [0, 3]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = 1
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing,
                                     maxgen=maxgen)


def run_msdp_fpa_l1_nsga2_toZero_train():
    save_file = '多目标优化_FPA+L1_nsga2_toZero'
    target = [0, 3]
    predict_model = 'linear'
    l = -20
    u = 20
    maxgen = 100
    drawing = 0
    moea = 3
    helpers.training_record_for_msdp(save_file=save_file, target=target,
                                     predict_model=predict_model, l=l, u=u, moea=moea, drawing=drawing,
                                     maxgen=maxgen)


'''
---------------------------------------训练sklearn中的模型------------------------------------------------------
'''


def run_linearRegresion_model():
    save_folder = 'LinearRegression'
    model = LinearRegression()
    comparison_algorithm.training_test_with_sklearnmodel(save_folder=save_folder, model=model)



def run_RRgcv_model(single_time):
    save_folder = 'RidgeGCV'
    alpha = list(np.linspace(start=0.1, stop=1000, num=10000))
    # alpha = [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 25, 30, 50, 100, 300, 500, 1000]
    model = RidgeCV(alphas=alpha)
    if single_time:
        comparison_algorithm.training_test_with_sklearnmodel(save_folder=save_folder, model=model)
    else:
        save_folder += '10t'
        comparison_algorithm.training_test_10times_sklearnmodel(save_folder=save_folder, model=model)



def run_lassoLarsCV(single_time):
    save_folder = 'LassoLarsCV'
    model = LassoLarsCV(cv = 10)
    if single_time:
        comparison_algorithm.training_test_with_sklearnmodel(save_folder=save_folder, model=model)
    else:
        save_folder += '10t'
        comparison_algorithm.training_test_10times_sklearnmodel(save_folder=save_folder, model=model)
        
def run_Lars(single_time):
    save_folder = 'Lars'
    model = Lars()
    if single_time:
        comparison_algorithm.training_test_with_sklearnmodel(save_folder=save_folder, model=model)
    else:
        save_folder += '10t'
        comparison_algorithm.training_test_10times_sklearnmodel(save_folder=save_folder, model=model)

def run_LassoLars(single_time):
    save_folder = 'LassoLars'
    model = LassoLars(alpha=1.0)
    if single_time:
        comparison_algorithm.training_test_with_sklearnmodel(save_folder=save_folder, model=model)
    else:
        save_folder += '10t'
        comparison_algorithm.training_test_10times_sklearnmodel(save_folder=save_folder, model=model)




def run_MLPRegressor(single_time):
    save_folder = 'MLPRegressor'
    model = MLPRegressor(hidden_layer_sizes=3, activation='logistic', max_iter=1000)
    if single_time:
        comparison_algorithm.training_test_with_sklearnmodel(save_folder=save_folder, model=model)
    else:
        save_folder += '10t'
        comparison_algorithm.training_test_10times_sklearnmodel(save_folder=save_folder, model=model)

def run_randomForestRegressor(single_time):
    save_folder = 'RandomForestRegressor'
    model = RandomForestRegressor(n_estimators=100)
    if single_time:
        comparison_algorithm.training_test_with_sklearnmodel(save_folder=save_folder, model=model)
    else:
        save_folder += '10t'
        comparison_algorithm.training_test_10times_sklearnmodel(save_folder=save_folder, model=model)