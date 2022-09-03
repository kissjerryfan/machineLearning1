import helpers
import dictionaries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import comparison_algorithm




'''
==============================================结果对比绘图=================================================

'''

'''
-----------------------------------------------------------------------------------------------------------
-------------------------------测试同样优化目标的算法，单目标算法，和多目标算法--------------------------------------\
输入:多目标算法数字列表，单目标算法文档列表, 目标
-----------------------------------------------------------------------------------------------------------
'''
def plotting_ssmm_train(moea_list, sklearn_list, soea_list, target):
    moea_names = []
    for i in moea_list:
        moea_names.append(dictionaries.get_moea_name(i))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    soea_names = []
    for i in soea_list:
        soea_names.append(dictionaries.get_soea_name(i))
    target_name = dictionaries.get_target_composition(target)
    multi_paths = []
    for moea_name in moea_names:
        multi_paths.append('../results/linear/' + moea_name + '/' + target_name + '/train/')
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/train/')
    for soea_name in soea_names:
        single_paths.append('../results/linear/' + soea_name + '/' + '/train/')
    single_names = sklearn_names +soea_names
    save_path = '../results/plotting/all/' + target_name + '/train/'
    helpers.comparison_ssmm_train(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                  single_names=single_names, multi_names=moea_names, save_path=save_path)

def plotting_ssmm_test(moea_list, sklearn_list, soea_list, target):
    moea_names = []
    for i in moea_list:
        moea_names.append(helpers.get_moea_name(i))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(helpers.get_sklearn_name(i))
    soea_names = []
    for i in soea_list:
        soea_names.append(helpers.get_soea_name(i))
    target_name = helpers.get_target_composition(target)
    multi_paths = []
    for moea_name in moea_names:
        multi_paths.append('../results/linear/' + moea_name + '/' + target_name + '/test/')
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/test/')
    for soea_name in soea_names:
        single_paths.append('../results/linear/' + soea_name + '/' + '/test/')
    single_names = sklearn_names +soea_names
    save_path = '../results/plotting/all/' + target_name + '/test/'
    helpers.comparison_ssmm_test(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                  single_names=single_names, multi_names=moea_names, save_path=save_path)

def plotting_ssmm_train_test(moea_list, sklearn_list, soea_list, target):
    moea_names = []
    for i in moea_list:
        moea_names.append(helpers.get_moea_name(i))
    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(helpers.get_sklearn_name(i))
    soea_names = []
    for i in soea_list:
        soea_names.append(helpers.get_soea_name(i))
    target_name = helpers.get_target_composition(target)
    multi_paths = []
    for moea_name in moea_names:
        multi_paths.append('../results/linear/' + moea_name + '/' + target_name + '/')
    single_paths = []
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/')
    for soea_name in soea_names:
        single_paths.append('../results/linear/' + soea_name + '/')
    single_names = sklearn_names +soea_names
    save_path = '../results/plotting/all/' + target_name + '/train-test/'
    helpers.comparison_ssmm_train_test(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                  single_names=single_names, multi_names=moea_names, save_path=save_path)


'''
-----------------------------------------------------------------------------------------------------------
-------------------------------测试同样优化目标的算法，单目标算法，和多目标算法--------------------------------------\
输入:多目标算法数字列表，单目标算法文档列表, 目标

1224:新添加功能，
对于moea_list,soea_list,里面每一个元素都是一个二元list, t, t[0]指的是所使用的模型比如linear或者mlp, t[1]指的是所使用的算法
-----------------------------------------------------------------------------------------------------------
'''
'''
plotting_mix_ssmm_train(moea_list = [1, 3, 10], sklearn_list = [1, 2], soea_list = [1], op_targets = [[0, 1], [0, 2], [0, 1, 2], [0, 3], [0, 2, 4], [0, 4]] , target = [0, 1])
'''
'''
1:
plotting_mix_ssmm_train(moea_list = [1, 3], sklearn_list = [],soea_list = [1], op_targets = [[0, 2, 4], [0, 4]], target = [0, 4], color = False, type =1) 
2:
plotting_mix_ssmm_train(moea_list = [1, 3], sklearn_list = [1, 2],soea_list = [], op_targets = [[0, 2, 4], [0, 4]], target = [0, 4], color = False, type =2)
3:
plotting_mix_ssmm_train(moea_list = [1, 3], sklearn_list = [1, 2],soea_list = [1], op_targets = [[0, 2, 4], [0, 4]], target = [0, 2], color = False, type =2) 

'''
def plotting_mix_ssmm_train(moea_list, sklearn_list, soea_list, op_targets , target, color = True, type = 2, if_show_label = True, save_folder = 'mix'):
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
            multi_paths.append('../results/multi-objective/' + moea_name + '/' + op_target_name + '/train/')
            if type == 1:
                write_algorithm = moea_name.replace('nsga2_toZero', 'multi-objective-revised')
                write_algorithm = write_algorithm.replace('nsga2', 'multi-objective')
                # if moea_name == 'nsga2':
                #     write_algorithm = 'multi-objective'
                # elif moea_name == 'nsga2_toZero':
                #     write_algorithm = 'multi-objective-revised'
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
    target_name = dictionaries.get_target_composition(target)
    save_path = '../results/plotting/' + save_folder+'/' + target_name + '/train/'
    if color:
        helpers.comparison_difcolor_ssmm_train(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                  single_names=single_names, multi_names=multi_names, save_path=save_path)
    else:
        helpers.comparison_difmarker_ssmm_train(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                  single_names=single_names, multi_names=multi_names, save_path=save_path, if_show_label=if_show_label)


def plotting_mix_ssmm_test(moea_list, sklearn_list, soea_list, op_targets, target, color = True, type = 2, if_show_label = True, save_folder = 'mix'):
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
                # if moea_name == 'nsga2':
                #     write_algorithm = 'multi-objective'
                # elif moea_name == 'nsga2_toZero':
                #     write_algorithm = 'multi-objective-revised'
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
    target_name = dictionaries.get_target_composition(target)
    save_path = '../results/plotting/' + save_folder +'/'+ target_name + '/test/'
    if color:
        helpers.comparison_difcolor_ssmm_test(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                  single_names=single_names, multi_names=multi_names, save_path=save_path)
    else:
        helpers.comparison_difmarker_ssmm_test(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                  single_names=single_names, multi_names=multi_names, save_path=save_path, if_show_label=if_show_label)



# msize[0] 单目标的marker的大小，msize[1] 多目标的marker大小
# wsize------label的fontsize，title的fontsize， legend的fontsize
# 多目标优化算法直接指定
# moea_list = [[a, b, [c, d]]]--->a是模型，b是优化算法，[c,d]是优化目标
def plotting_universal_train(moea_list, sklearn_list, soea_list, target, line_list = [], color = True, type = 2,if_show_label = True, save_folder = 'universal', if_line = False, if_combine = False, if_show_rtborder = True, rows = 14, columns = 3, if_sM = False, msize = [16, 8], wsize = [14, 14, 14, 14], figsize = [8,4]):
    multi_names = []
    multi_paths = []

    moea_methods = []

    for moea in moea_list:
        moea_method = dictionaries.get_model_method_name(moea[0])
        moea_name = dictionaries.get_moea_name(moea[1])
        op_target_name = dictionaries.get_target_composition(moea[2])
        multi_paths.append('../results/multi-objective/' +moea_method + '/'+ moea_name + '/' + op_target_name + '/train/')
        if type == 1:
            write_algorithm = moea_name.replace('nsga2_random20p_toZero', 'revised_NSGA_II')
            write_algorithm = write_algorithm.replace('nsga2', 'NSGA_II')
            # if moea_name == 'nsga2':
            #     write_algorithm = 'multi-objective'
            # elif moea_name == 'nsga2_toZero':
            #     write_algorithm = 'multi-objective-revised'
        else:
            write_algorithm = moea_name
        write_target = op_target_name.replace('nonz', 'NNZ')
        moea_methods.append(moea_method)
        if type == 1:
            multi_names.append(write_algorithm)
        else:
            multi_names.append(write_algorithm + '/' + write_target)
    
    single_paths = []

    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/train/')
    
    soea_names = []
    soea_methods = []
    for soea in soea_list:
        soea_methods.append(dictionaries.get_model_method_name(soea[0]))
        soea_names.append(dictionaries.get_soea_name(soea[1]))
    if if_sM:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            multi_paths.append('../results/multi-objective/' + soea_method + '/'+soea_name  + '/train/')
        multi_names = multi_names + soea_names
        single_names = sklearn_names
    else:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            single_paths.append('../results/single-objective/' + soea_method + '/'+ soea_name  + '/train/')
        single_names = sklearn_names +soea_names
    
    line_names = []
    for line in line_list:
        line_names.append(dictionaries.get_model_method_name(line[0]) + '/' + dictionaries.get_soea_name(line[1]))
    line_paths = []
    for line_name in line_names:
        line_paths.append('../results/single-objective/' + line_name + '/train/')

    target_name = dictionaries.get_target_composition(target)
    save_path = '../results/plotting/' + save_folder+'/' + target_name + '/train/'
    print(multi_names, multi_paths)
    print(single_names, single_paths)
    if if_combine:
        helpers.combine_difmarker_line_train(single_paths=single_paths, multi_paths=multi_paths, line_paths = line_paths, parameters=target,
                                      single_names=single_names, multi_names=multi_names, line_names = line_names, save_path=save_path,
                                      if_show_label = if_show_label,
                                      if_show_rtborder = if_show_rtborder, rows = rows, columns = columns)
    
    if if_line:
        helpers.comparison_difmarker_line_train(single_paths=single_paths, multi_paths=multi_paths, line_paths = line_paths, parameters=target,
                                      single_names=single_names, multi_names=multi_names, line_names = line_names, save_path=save_path,
                                      if_show_rtborder = if_show_rtborder)
    else :
        if color:
            helpers.comparison_difcolor_ssmm_train(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                      single_names=single_names, multi_names=multi_names, save_path=save_path)
        else:
            helpers.comparison_difmarker_ssmm_train(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                      single_names=single_names, multi_names=multi_names, save_path=save_path, if_show_label=if_show_label, 
                                      wsize = wsize, msize = msize, figsize = figsize)


def plotting_universal_test(moea_list, sklearn_list, soea_list, target, line_list = [], color = True, type = 2, if_show_label = True, save_folder = 'universal', if_line = False, if_combine = False, if_show_rtborder = True, rows =10, columns = 3, if_sM = False, msize = [16, 8], wsize = [14, 14, 14, 14], figsize =[8, 4]):
    multi_names = []
    multi_paths = []

    moea_methods = []
    for moea in moea_list:
        moea_method = dictionaries.get_model_method_name(moea[0])
        moea_name = dictionaries.get_moea_name(moea[1])
        op_target_name = dictionaries.get_target_composition(moea[2])
        multi_paths.append('../results/multi-objective/' + moea_method + '/'+ moea_name + '/' + op_target_name + '/test/')
        if type == 1:
            write_algorithm = moea_name.replace('nsga2_random20p_toZero', 'revised_NSGA_II')
            write_algorithm = write_algorithm.replace('nsga2', 'NSGA_II')
            # if moea_name == 'nsga2':
            #     write_algorithm = 'multi-objective'
            # elif moea_name == 'nsga2_toZero':
            #     write_algorithm = 'multi-objective-revised'
        else:
            write_algorithm = moea_name
        write_target = op_target_name.replace('nonz', 'NNZ')
        moea_methods.append(moea_method)
        if type == 1:
            multi_names.append(write_algorithm)
        else:
            multi_names.append(write_algorithm + '/' + write_target)
        
    single_paths = []

    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/test/')
    
    soea_names = []
    soea_methods = []
    for soea in soea_list:
        soea_methods.append(dictionaries.get_model_method_name(soea[0]))
        soea_names.append(dictionaries.get_soea_name(soea[1]))
    if if_sM:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            multi_paths.append('../results/multi-objective/' + soea_method + '/'+soea_name  + '/test/')
        multi_names = multi_names + soea_names
        single_names = sklearn_names
    else:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            single_paths.append('../results/single-objective/' + soea_method + '/'+ soea_name  + '/test/')
        single_names = sklearn_names +soea_names

    line_names = []
    for line in line_list:
        line_names.append(dictionaries.get_model_method_name(line[0]) + '/' + dictionaries.get_soea_name(line[1]))
    line_paths = []
    for line_name in line_names:
        line_paths.append('../results/single-objective/' + line_name + '/test/')
    single_names = sklearn_names + soea_names
    target_name = dictionaries.get_target_composition(target)
    save_path = '../results/plotting/' + save_folder +'/'+ target_name + '/test/'
    if if_combine:
         helpers.combine_difmarker_line_test(single_paths=single_paths, multi_paths=multi_paths,
                                               line_paths=line_paths, parameters=target,
                                               single_names=single_names, multi_names=multi_names,
                                               line_names=line_names, save_path=save_path,
                                            if_show_rtborder = if_show_rtborder,
                                            if_show_label = if_show_label, rows = rows, columns = columns)
    if if_line:
        helpers.comparison_difmarker_line_test(single_paths=single_paths, multi_paths=multi_paths,
                                               line_paths=line_paths, parameters=target,
                                               single_names=single_names, multi_names=multi_names,
                                               if_show_label = if_show_label,
                                               line_names=line_names, save_path=save_path, if_show_rtborder = if_show_rtborder)
    else:
        if color:
            helpers.comparison_difcolor_ssmm_test(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                      single_names=single_names, multi_names=multi_names, save_path=save_path)
        else:
            helpers.comparison_difmarker_ssmm_test(single_paths=single_paths, multi_paths=multi_paths, parameters=target,
                                      single_names=single_names, multi_names=multi_names, save_path=save_path, 
                                      if_show_label=if_show_label, wsize = wsize, msize = msize, figsize = figsize)



def plotting_universal_log04_train(moea_list, sklearn_list, soea_list, type = 2, if_show_label = True, save_folder = 'log04', if_sM = False, msize = [16, 8], wsize = [14, 14, 14, 14], figsize = [8,4]):
    multi_names = []
    multi_paths = []

    for moea in moea_list:
        model_method =  dictionaries.get_model_method_name(moea[0])
        moea_name = dictionaries.get_moea_name(moea[1])
        op_target_name = dictionaries.get_target_composition(moea[2])
        multi_paths.append('../results/multi-objective/' +model_method+'/'+moea_name + '/' + op_target_name + '/train/')
        if type == 1:
            write_algorithm = moea_name.replace('nsga2_random20p_toZero', 'revised_NSGA_II')
            write_algorithm = write_algorithm.replace('nsga2', 'NSGA_II')
            # if moea_name == 'nsga2':
            #     write_algorithm = 'multi-objective'
            # elif moea_name == 'nsga2_toZero':
            #     write_algorithm = 'multi-objective-revised'
        else:
            write_algorithm = moea_name
        write_target = op_target_name.replace('nonz', 'NNZ')
        if type == 1:
            multi_names.append(write_algorithm)
        else:
            multi_names.append(write_algorithm + '/' + write_target)
    
    single_paths = []

    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/train/')
    
    soea_names = []
    soea_methods = []
    for soea in soea_list:
        soea_methods.append(dictionaries.get_model_method_name(soea[0]))
        soea_names.append(dictionaries.get_soea_name(soea[1]))
    if if_sM:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            multi_paths.append('../results/multi-objective/' + soea_method + '/'+soea_name  + '/train/')
        multi_names = multi_names + soea_names
        single_names = sklearn_names
    else:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            single_paths.append('../results/single-objective/' + soea_method + '/'+ soea_name  + '/train/')
        single_names = sklearn_names +soea_names

    save_path = '../results/plotting/' + save_folder+'/train/'
    print(multi_names, multi_paths)
    print(single_names, single_paths)

    helpers.comparison_difmarker_log04_train(single_paths=single_paths, multi_paths=multi_paths,
                                single_names=single_names, multi_names=multi_names, save_path=save_path,
                                 if_show_label=if_show_label, wsize = wsize, msize = msize, figsize = figsize)

def plotting_universal_log04_test(moea_list, sklearn_list, soea_list, type = 2, if_show_label = True, save_folder = 'log04', if_sM = False, msize = [16, 8], wsize = [14, 14, 14, 14], figsize = [8, 4]):
    multi_names = []
    multi_paths = []

    for moea in moea_list:
        model_method =  dictionaries.get_model_method_name(moea[0])
        moea_name = dictionaries.get_moea_name(moea[1])
        op_target_name = dictionaries.get_target_composition(moea[2])
        multi_paths.append('../results/multi-objective/' +model_method+'/'+moea_name + '/' + op_target_name + '/test/')
        if type == 1:
            write_algorithm = moea_name.replace('nsga2_random20p_toZero', 'revised_NSGA_II')
            write_algorithm = write_algorithm.replace('nsga2', 'NSGA_II')
            # if moea_name == 'nsga2':
            #     write_algorithm = 'multi-objective'
            # elif moea_name == 'nsga2_toZero':
            #     write_algorithm = 'multi-objective-revised'
        else:
            write_algorithm = moea_name
        write_target = op_target_name.replace('nonz', 'NNZ')
        if type == 1:
            multi_names.append(write_algorithm)
        else:
            multi_names.append(write_algorithm + '/' + write_target)
    single_paths = []

    sklearn_names = []
    for i in sklearn_list:
        sklearn_names.append(dictionaries.get_sklearn_name(i))
    for sklearn_name in sklearn_names:
        single_paths.append('../results/compared_algorithms/' + sklearn_name + '/test/')
    
    soea_names = []
    soea_methods = []
    for soea in soea_list:
        soea_methods.append(dictionaries.get_model_method_name(soea[0]))
        soea_names.append(dictionaries.get_soea_name(soea[1]))
    if if_sM:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            multi_paths.append('../results/multi-objective/' + soea_method + '/'+soea_name  + '/test/')
        multi_names = multi_names + soea_names
        single_names = sklearn_names
    else:
        for soea_name, soea_method in zip(soea_names, soea_methods):
            single_paths.append('../results/single-objective/' + soea_method + '/'+ soea_name  + '/test/')
        single_names = sklearn_names +soea_names

    save_path = '../results/plotting/' + save_folder+'/test/'
    print(multi_names, multi_paths)
    print(single_names, single_paths)

    helpers.comparison_difmarker_log04_test(single_paths=single_paths, multi_paths=multi_paths,
                                single_names=single_names, multi_names=multi_names, save_path=save_path, 
                                if_show_label=if_show_label, wsize = wsize, msize = msize, figsize = figsize)













# 【linearRegression】和【linear-多目标-FPA-AAE-NSGA2】的对比
def run_sm_LR_fpa_aae_NSGA2_train():
    single_path = '../results/compared_algorithms/' + 'LinearRegression' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/train/'
    parameters = [0, 1]
    single_name = 'LinearRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'LinearRegressionvs多目标_FPA_AAE_NSGA2' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_LR_fpa_aae_NSGA2_test():
    single_path = '../results/compared_algorithms/' + 'LinearRegression' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/test/'
    parameters = [0, 1]
    single_name = 'LinearRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'LinearRegressionvs多目标_FPA_AAE_NSGA2' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_LR_fpa_aae_NSGA2_train_test():
    single_path = '../results/compared_algorithms/' + 'LinearRegression/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2/'
    parameters = [0, 1]
    single_name = 'LinearRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'LinearRegressionvs多目标_FPA_AAE_NSGA2' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)


# 【RidgeRegression】和【linear-多目标-FPA-AAE-NSGA2】的对比
def run_sm_RR_fpa_aae_NSGA2_train():
    single_path = '../results/compared_algorithms/' + 'RidgeRegression' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/train/'
    parameters = [0, 1]
    single_name = 'RidgeRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'RidgeRegressionvs多目标_FPA_AAE_NSGA2' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_RR_fpa_aae_NSGA2_test():
    single_path = '../results/compared_algorithms/' + 'RidgeRegression' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/test/'
    parameters = [0, 1]
    single_name = 'RidgeRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'RidgeRegressionvs多目标_FPA_AAE_NSGA2' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_RR_fpa_aae_NSGA2_train_test():
    single_path = '../results/compared_algorithms/' + 'RidgeRegression/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2/'
    parameters = [0, 1]
    single_name = 'RidgeRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'RidgeRegressionvs多目标_FPA_AAE_NSGA2' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)



# 【linear-单目标-CoDE】和【linear-多目标-FPA-AAE-NSGA2】的对比
def run_sm_fpa_aae_CoDE_NSGA2_train():
    single_path = '../results/linear/' + '单目标优化_CoDE' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/train/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_AAE_NSGA2' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_aae_CoDE_NSGA2_test():
    single_path = '../results/linear/' + '单目标优化_CoDE' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/test/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_AAE_NSGA2' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_aae_CoDE_NSGA2_train_test():
    single_path = '../results/linear/' + '单目标优化_CoDE/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_AAE_NSGA2' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)



# 【linear-单目标-CoDE】和【linear-多目标-FPA-nonz-NSGA2-toZero】的对比
def run_sm_fpa_nonz_CoDE_NSGA2_toZero_train():
    single_path = '../results/linear/' + '单目标优化_CoDE' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero' + '/train/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_nonz_NSGA2_toZero' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_nonz_CoDE_NSGA2_toZero_test():
    single_path = '../results/linear/' + '单目标优化_CoDE' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero' + '/test/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_nonz_NSGA2_toZero' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_nonz_CoDE_NSGA2_toZero_train_test():
    single_path = '../results/linear/' + '单目标优化_CoDE/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_nonz_NSGA2_toZero' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)



# 【linear-单目标-CoDE】和【linear-多目标-FPA-nonz-NSGA2-DE-toZero】的对比
def run_sm_fpa_nonz_CoDE_NSGA2_DE_toZero_train():
    single_path = '../results/linear/' + '单目标优化_CoDE' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero' + '/train/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_nonz_NSGA2_DE_toZero' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)


def run_sm_fpa_nonz_CoDE_NSGA2_DE_toZero_test():
    single_path = '../results/linear/' + '单目标优化_CoDE' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero' + '/test/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_nonz_NSGA2_DE_toZero' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)


def run_sm_fpa_nonz_CoDE_NSGA2_DE_toZero_train_test():
    single_path = '../results/linear/' + '单目标优化_CoDE/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_CoDEvs多目标_FPA_nonz_NSGA2_DE_toZero' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)




#---------------------------------------------rand_1_bin--------------------------------------------

# 【linear-单目标-DE_rand_1_bin】和【linear-多目标-FPA-AAE-NSGA2】的对比
def run_sm_fpa_aae_rand_1_bin_NSGA2_train():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/train/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_AAE_NSGA2' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_aae_rand_1_bin_NSGA2_test():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/test/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_AAE_NSGA2' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_aae_rand_1_bin_NSGA2_train_test():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_AAE_NSGA2' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)



# 【linear-单目标-CoDE】和【linear-多目标-FPA-nonz-NSGA2-toZero】的对比
def run_sm_fpa_nonz_rand_1_bin_NSGA2_toZero_train():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero' + '/train/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_nonz_NSGA2_toZero' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_nonz_rand_1_bin_NSGA2_toZero_test():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero' + '/test/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_nonz_NSGA2_toZero' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_nonz_rand_1_bin_NSGA2_toZero_train_test():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_nonz_NSGA2_toZero' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)



# 【linear-单目标-CoDE】和【linear-多目标-FPA-nonz-NSGA2-DE-toZero】的对比
def run_sm_fpa_nonz_rand_1_bin_NSGA2_DE_toZero_train():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero' + '/train/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_nonz_NSGA2_DE_toZero' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)


def run_sm_fpa_nonz_rand_1_bin_NSGA2_DE_toZero_test():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero' + '/test/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/CoDE/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_nonz_NSGA2_DE_toZero' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)


def run_sm_fpa_nonz_rand_1_bin_NSGA2_DE_toZero_train_test():
    single_path = '../results/linear/' + '单目标优化_rand_1_bin/'
    multi_path = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero/'
    parameters = [0, 2]
    single_name = 'single-objective/linear/rand_1_bin/FPA'
    multi_name = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '单目标_rand_1_binvs多目标_FPA_nonz_NSGA2_DE_toZero' + '/train-test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)








'''
----------------------------------------------------------------------------------------------
'''

def run_sm_fpa_aae_rand1_train():
    single_path = '../results/' + '单目标优化算法1' + '/train/'
    multi_path = '../results/' + '多目标优化__FPA+AAE__NIND=100__MAX_GEN=100__NSGAII_决策变量范围_[-20,20]' + '/train/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/rand/1/bin'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE/'
    save_path = '../results/plotting/' + '单目标vs多目标_FPA_AAE' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)

def run_sm_fpa_aae_rand1_test():
    single_path = '../results/' + '单目标优化算法1' + '/test/'
    multi_path = '../results/' + '多目标优化__FPA+AAE__NIND=100__MAX_GEN=100__NSGAII_决策变量范围_[-20,20]' + '/test/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/rand/1/bin'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE/'
    save_path = '../results/plotting/' + '单目标vs多目标_FPA_AAE_rand_1_bin' + '/test/'
    helpers.comparison_sm_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)




def run_sm_fpa_aae_CoDE_train():
    single_path = '../results/' + '单目标优化__NIND=100__MAX_GEN=100__CoDE_决策变量范围_[-20,20]' + '/train/'
    multi_path = '../results/' + '多目标优化__FPA+AAE__NIND=100__MAX_GEN=100__NSGAII_决策变量范围_[-20,20]' + '/train/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/CoDE/FPA+AAE'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE/'
    save_path = '../results/plotting/' + '单目标vs多目标_FPA_AAE_CoDE' + '/train/'
    helpers.comparison_sm_train(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)


def run_sm_fpa_aae_CoDE_train_test():
    single_path = '../results/' + '单目标优化__NIND=100__MAX_GEN=100__CoDE_决策变量范围_[-20,20]/'
    multi_path = '../results/' + '多目标优化__FPA+AAE__NIND=100__MAX_GEN=100__NSGAII_决策变量范围_[-20,20]/'
    parameters = [0, 1]
    single_name = 'single-objective/linear/CoDE/FPA+AAE'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE/'
    save_path = '../results/plotting/' + '单目标vs多目标_FPA_AAE_CoDE' + '/train_test/'
    helpers.comparison_sm_train_test(single_path=single_path, multi_path=multi_path,
                                parameters=parameters, single_name=single_name, multi_name=multi_name, save_path=save_path)


'''
===============================================两个多目标算法的对比==================================================
'''
#【linear-多目标-FPA-nonZero-NSGA2-DE-toZero】和【linear-多目标-FPA-nonZero-NSGA2-toZero】的对比
def run_mm_fpa_nonz_NSGA2_NSGA2_DE_toZero_train():
    path1 = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero' + '/train/'
    path2 = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero' + '/train/'

    parameters = [0, 2]
    name1 = 'multi-objective/linear//NSGA-II-toZero/FPA+nonz'
    name2 = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '多目标_NSGA2_toZerovs多目标_NSGA2_DE_toZero_FPA_nonz' + '/train/'
    helpers.comparison_mm_train(path1=path1, path2=path2, name1 = name1, name2 = name2,
                                parameters=parameters, save_path=save_path)

def run_mm_fpa_nonz_NSGA2_NSGA2_DE_toZero_test():
    path1 = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero' + '/test/'
    path2 = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero' + '/test/'

    parameters = [0, 2]
    name1 = 'multi-objective/linear//NSGA-II-toZero/FPA+nonz'
    name2 = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '多目标_NSGA2_toZerovs多目标_NSGA2_DE_toZero_FPA_nonz' + '/test/'
    helpers.comparison_mm_test(path1=path1, path2=path2, name1 = name1, name2 = name2,
                                parameters=parameters, save_path=save_path)

def run_mm_fpa_nonz_NSGA2_NSGA2_DE_toZero_train_test():
    path1 = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_toZero/'
    path2 = '../results/linear/' + '多目标优化_FPA+nonz_nsga2_DE_toZero/'

    parameters = [0, 2]
    name1 = 'multi-objective/linear//NSGA-II-toZero/FPA+nonz'
    name2 = 'multi-objective/linear/NSGA-II-DE-toZero/FPA+nonz'
    save_path = '../results/plotting/linear/' + '多目标_NSGA2_toZerovs多目标_NSGA2_DE_toZero_FPA_nonz' + '/train-test/'
    helpers.comparison_mm_train_test(path1=path1, path2=path2, name1 = name1, name2 = name2,
                                parameters=[0,2], save_path=save_path)
'''
======================================================单单多对比========================================================
'''
# 【linearRegression】【RidgeRegression】和【linear-多目标-FPA-AAE-NSGA2】的对比
def run_ssm_LR_RR_fpa_aae_NSGA2_train():
    single_path1 = '../results/compared_algorithms/' + 'LinearRegression' + '/train/'
    single_path2 = '../results/compared_algorithms/' + 'RidgeRegression' + '/train/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/train/'
    parameters = [0, 1]
    single_name1 = 'LinearRegression'
    single_name2 = 'RidgeRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'LR_RR_多目标_FPA_AAE_NSGA2' + '/train/'
    comparison_algorithm.comparison_ssm_train(single_path1=single_path1, single_path2 = single_path2, multi_path=multi_path,
                                parameters=parameters, single_name1=single_name1, single_name2 = single_name2,
                                 multi_name=multi_name, save_path=save_path)

def run_ssm_LR_RR_fpa_aae_NSGA2_test():
    single_path1 = '../results/compared_algorithms/' + 'LinearRegression' + '/test/'
    single_path2 = '../results/compared_algorithms/' + 'RidgeRegression' + '/test/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2' + '/test/'
    parameters = [0, 1]
    single_name1 = 'LinearRegression'
    single_name2 = 'RidgeRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'LR_RR_多目标_FPA_AAE_NSGA2' + '/test/'
    comparison_algorithm.comparison_ssm_test(single_path1=single_path1, single_path2 = single_path2, multi_path=multi_path,
                                parameters=parameters, single_name1=single_name1, single_name2 = single_name2,
                                multi_name=multi_name, save_path=save_path)

def run_ssm_LR_RR_fpa_aae_NSGA2_train_test():
    single_path1 = '../results/compared_algorithms/' + 'LinearRegression/'
    single_path2 = '../results/compared_algorithms/' + 'RidgeRegression/'
    multi_path = '../results/linear/' + '多目标优化_FPA+AAE_NSGA2/'
    parameters = [0, 1]
    single_name1 = 'LinearRegression'
    single_name2 = 'RidgeRegression'
    multi_name = 'multi-objective/linear/NSGA-II/FPA+AAE'
    save_path = '../results/plotting/sklearn/' + 'LR_RR_多目标_FPA_AAE_NSGA2' + '/train-test/'
    comparison_algorithm.comparison_ssm_train_test(single_path1=single_path1, single_path2=single_path2, multi_path=multi_path,
                                parameters=parameters, single_name1=single_name1, single_name2=single_name2,
                                      multi_name=multi_name, save_path=save_path)
