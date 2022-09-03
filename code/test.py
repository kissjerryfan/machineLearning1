import helpers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import comparison_algorithm
import dictionaries


'''
==============================================测试模型===================================================
'''

'''
-------------------------------------------多目标测试模型,填算法和目标----------------------------------
'''
## 注意target的顺序很重要
def run_msdp_test(method, moea, target):
    moea_name = dictionaries.get_moea_name(moea)
    target_name = dictionaries.get_target_composition(target)
    method_name = dictionaries.get_model_method_name(method)
    folder_name = 'multi-objective/' + method_name +'/'+ moea_name + '/' + target_name
    type = 2
    helpers.test_model(folder_name=folder_name, type = type, predict_model=method)


def run_ssdp_test(method,soea):
    type = 1
    soea_name = dictionaries.get_soea_name(soea)
    method_name = dictionaries.get_model_method_name(method)
    folder_name = 'single-objective/' + method_name + '/' + soea_name
    helpers.test_model(folder_name=folder_name, type = type, predict_model=method)

def run_train_split_test(method, moea, target):
    moea_name = dictionaries.get_moea_name(moea)
    target_name = dictionaries.get_target_composition(target)
    method_name = dictionaries.get_model_method_name(method)
    folder_name = 'split-train/' + method_name +'/'+ moea_name + '/' + target_name
    type = 2
    helpers.test_model(folder_name=folder_name, type = type, predict_model=method)



def run_ssdpM_test(method,soea):
    type = 2
    soea_name = dictionaries.get_soea_name(soea)
    method_name = dictionaries.get_model_method_name(method)
    folder_name = 'multi-objective/' + method_name + '/' + soea_name
    helpers.test_model(folder_name=folder_name, type = type, predict_model=method)
'''
--------------------------------------------------------------------------------------------------------
'''


# 测试单目标-线性模型-CoDE算法
def run_ssdp_linear_CoDE_test():
    folder_name = 'linear/CoDE'
    type = 1
    helpers.test_model(folder_name=folder_name, type=type)

# 测试多目标-FPA+nonz-线性模型-NSGA2-toZero
def run_msdp_fpa_nonz_linear_nsga2_toZeor_test():
    folder_name = 'linear/多目标优化_FPA+numofnonzero_nsga2_toZero'
    type = 2
    helpers.test_model(folder_name=folder_name, type=type)

# 测试多目标-FPA+nonz-线性模型-NSGA2-DE-toZero
def run_msdp_fpa_nonz_linear_nsga2_DE_toZero_test():
    folder_name = 'linear/多目标优化_FPA+numofnonzero_nsga2_DE_toZero'
    type = 2
    helpers.test_model(folder_name=folder_name, type=type)


# 测试多目标-FPA+AAE+nonz-线性模型-NSGA2-toZero
def run_msdp_fpa_aae_nonz_linear_nsga2_toZero_test():
    folder_name = 'linear/多目标优化_FPA+AAE+numofnonzero_nsga2_toZero'
    type = 2
    helpers.test_model(folder_name=folder_name, type=type)


# 测试多目标-FPA+AAE+nonz-线性模型-NSGA2-DE-toZero
def run_msdp_fpa_aae_nonz_linear_nsga2_DE_toZero_test():
    folder_name = 'linear/多目标优化_FPA+AAE+numofnonzero_nsga2_DE_toZero'
    type = 2
    helpers.test_model(folder_name=folder_name, type=type)

def run_msdp_fpa_l1_linear_nsga2_test():
    folder_name = 'linear/多目标优化_FPA+L1_nsga2'
    type = 2
    helpers.test_model(folder_name=folder_name, type=type)


def run_msdp_fpa_l1_linear_nsga2_toZero_test():
    folder_name = 'linear/多目标优化_FPA+L1_nsga2_toZero'
    type = 2
    helpers.test_model(folder_name=folder_name, type=type)


