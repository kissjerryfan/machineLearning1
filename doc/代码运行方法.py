#=================================================================================
#=================================================================================
#===============================一、训练模型=======================================
# 调用函数见[code/training.py]



# 1. 训练及测试sklearn模型------------
# (1) 调用函数见“行298~364”
# (2) 参数“single_time=True”表示仅运行一次，“single_time=False”表示运行10次。
# (3) 结果保存在“\results\compared_algorithms”中；
# (4) 运行前需先创建好对应的三个文件夹——如：“\results\compared_algorithms\LassoLarsCV”，及在该文件夹下创建”train“”test“文件夹
training.run_linearRegresion_model()
training.run_RRgcv_model(single_time)
training.run_lassoLarsCV(single_time)
training.run_Lars(single_time)
training.run_LassoLars(single_time)
training.run_MLPRegressor(single_time)
training.run_randomForestRegressor(single_time)

# 2. 训练单目标优化模型-仅记录一个结果----------
# (1) 调用函数见“行20~26”
# (2) 参数“soea”表示运行所使用的优化算法，参数“predict_model”表示运行所使用的模型（如linear）
# (3) 结果保存在“\results\single-objective\运行所使用的模型(如linear)\”中；
# (4) 运行前需先创建好对应的文件夹——如：“\results\single-objective\linear\CoDE\train”

# soea ：{1: 'CoDE', 2: 'DE_rand_1_bin', 3:'CoDE_toZero', 4: 'CoDE_10p_toZero', 5:'CoDE_20p_toZero',
#                      6:'CoDE_10p_lr_toZero', 7:'CoDE_20p_lr_toZero', 8:'CoDE_random10p_toZero', 9:'CoDE_random20p_toZero',
#                     10:'CoDE_random30p_toZero'}
# predict_model ：{1: 'linear', 2: 'BPNN', 3:'NN', 4:'MLP', 5:'mlp3', 6: 'mlp5'}
training.run_ssdp_train(soea, predict_model)
training.run_ssdp_train(soea = 1, predict_model = 1) # 运行linear/CoDE算法

# 3. 训练单目标优化模型-记录多个结果----------
# (1) 其他与第2点类似，但有以下不同：
# (2) 结果保存在“\results\multi-objective\运行所使用的模型(如linear)\”中；
# (3) 运行前在(2)文件夹内创建如第2点所述的文件夹，如“\results\multi-objective\linear\CoDE\train”
training.train_ssdp_m(soea, predict_model)


# 3. 训练多目标优化模型----------
# (1) 调用函数见“行29~37”
# (2) 参数“moea”表示运行所使用的优化算法，参数“predict_model”表示运行所使用的模型（如linear）,参数“targets”表示优化的目标
# (3) 结果保存在“\results\multi-objective\运行所使用的模型(如linear)\优化目标(如FPA_MSE)”中；
# (4) 运行前需先创建好对应的个文件夹——如：“\results\multi-objective\linear\nsga2\FPA_MSE\train”

# predict_model:{1: 'linear', 2: 'BPNN', 3:'NN', 4:'MLP', 5:'mlp3', 6: 'mlp5'}
# targets可选：{0: 'FPA', 1:'AAE', 2:'numOfnonZero', 3:'l1', 4:'MSE'}
# 注：targets需从小到大排列，即升序排列。如FPA+MSE+L1---->[0,3,4]
# moea：{1: 'nsga2', 2: 'nsga2_DE', 3: 'nsga2_toZero', 4: 'nsga2_DE_toZero',
#                      5: 'nsga3', 6: 'nsga3_DE', 7: 'awGA', 8: 'RVEA', 9: 'nsga2_10p_toZero',
#                      10: 'nsga2_20p_toZero', 11: 'nsga2_30p_toZero', 12:'nsga2_random10p_toZero',
#                      13:'nsga2_random20p_toZero', 14:'nsga2_random30p_toZero'}
training.run_msdp_train(moea, targets, predict_model)
training.run_msdp_train(moea = 13, targets = [0, 2, 4], predict_model = 1) # 运行linear/nsga2_random20p_toZero/FPA_NNZ_MSE




#=================================================================================
#=================================================================================
#===============================二、测试模型=======================================
#=================================================================================
# 调用函数见 [code/test.py]


# 1. 测试单目标优化模型-仅记录一个结果----------
# (1) 调用函数见“行25-30”
# (2) 参数“soea”表示运行所使用的优化算法，参数“method”表示运行所使用的模型（如linear）
# (3) 结果保存在“\results\single-objective\运行所使用的模型(如linear)\”中；
# (4) 运行前需先创建好对应的文件夹——如：“\results\single-objective\linear\CoDE\test”

# soea ：{1: 'CoDE', 2: 'DE_rand_1_bin', 3:'CoDE_toZero', 4: 'CoDE_10p_toZero', 5:'CoDE_20p_toZero',
#                      6:'CoDE_10p_lr_toZero', 7:'CoDE_20p_lr_toZero', 8:'CoDE_random10p_toZero', 9:'CoDE_random20p_toZero',
#                     10:'CoDE_random30p_toZero'}
# method ：{1: 'linear', 2: 'BPNN', 3:'NN', 4:'MLP', 5:'mlp3', 6: 'mlp5'}
test.run_ssdp_test(soea, method)
test.run_ssdp_test(soea = 1, method = 1) # 根据linear/CoDE算法训练结果进行测试

# 2. 测试单目标优化模型-记录多个结果----------
# (1) 其他与第1点类似，但有以下不同：
# (2) 结果保存在“\results\multi-objective\运行所使用的模型(如linear)\”中；
# (3) 运行前在(2)文件夹内创建如第2点所述的文件夹，如“\results\multi-objective\linear\CoDE\trst”
test.run_ssdpM_test(moea, soea)


# 3. 测试多目标优化模型----------
# (1) 调用函数见“行16~22”
# (2) 参数“moea”表示运行所使用的优化算法，参数“method”表示运行所使用的模型（如linear）,参数“targets”表示优化的目标
# (3) 结果保存在“\results\multi-objective\运行所使用的模型(如linear)\优化目标(如FPA_MSE)”中；
# (4) 运行前需先创建好对应的个文件夹——如：“\results\multi-objective\linear\nsga2\FPA_MSE\test”

# method:{1: 'linear', 2: 'BPNN', 3:'NN', 4:'MLP', 5:'mlp3', 6: 'mlp5'}
# targets可选：{0: 'FPA', 1:'AAE', 2:'numOfnonZero', 3:'l1', 4:'MSE'}
# 注：targets需从小到大排列，即升序排列。如FPA+MSE+L1---->[0,3,4]
# moea：{1: 'nsga2', 2: 'nsga2_DE', 3: 'nsga2_toZero', 4: 'nsga2_DE_toZero',
#                      5: 'nsga3', 6: 'nsga3_DE', 7: 'awGA', 8: 'RVEA', 9: 'nsga2_10p_toZero',
#                      10: 'nsga2_20p_toZero', 11: 'nsga2_30p_toZero', 12:'nsga2_random10p_toZero',
#                      13:'nsga2_random20p_toZero', 14:'nsga2_random30p_toZero'}
test.run_msdp_test(moea, targets, method)
test.run_msdp_test(moea = 13, targets = [0, 2, 4], method = 1) # 根据linear/nsga2_random20p_toZero/FPA_NNZ_MSE训练结果进行测试







