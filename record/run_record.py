

# -----------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------#
# ================================ 0302 号总结中的绘图======================================#
# -----------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------#

plotting.plotting_mix_ssmm_train(moea_list, sklearn_list, soea_list, op_targets, target, color = True, type = 2, if_show_label = True, save_folder = 'mix')

# ==============================linear-single-obj

#---------------------------plot
plotting.plotting_mix_ssmm_train(moea_list = [], sklearn_list = [4, 6], soea_list = [[1, 1], [1,3], [1,4],[1,5], [1,6], [1,7], [1,8], [1,9], [1,10]], op_targets=[], target=[0,4])
plotting.plotting_mix_ssmm_test(moea_list = [], sklearn_list = [4, 6], soea_list = [[1, 1], [1,2], [1,3], [1,4],[1,5], [1,6], [1,7], [1,8], [1,9], [1,10]], op_targets=[], target=[0,4])
#----------------------------table
make_tables.ssmm_table_train(moea_list = [], sklearn_list= [4,6], soea_list = [[1, 1], [1,3], [1,4],[1,5], [1,6], [1,7], [1,8], [1,9], [1,10]], op_targets = [], target=0)
make_tables.ssmm_table_test(moea_list = [], sklearn_list= [4,6], soea_list = [[1, 1], [1,3], [1,4],[1,5], [1,6], [1,7], [1,8], [1,9], [1,10]], op_targets = [], target=0)



# ================================mlp3-single

#-------------------------------plot
plotting.plotting_mix_ssmm_train(moea_list = [], sklearn_list = [4, 6], soea_list = [[5, 1], [5,3], [5,4],[5,5], [5,6], [5,7], [5,8], [5,9], [5,10]], op_targets=[], target=[0,4])
plotting.plotting_mix_ssmm_test(moea_list = [], sklearn_list = [4, 6], soea_list = [[5, 1], [5,3], [5,4],[5,5], [5,6], [5,7], [5,8], [5,9], [5,10]], op_targets=[], target=[0,4])

#------------------------------table
make_tables.ssmm_table_train(moea_list = [], sklearn_list= [4,6], soea_list =  [[5, 1], [5,3], [5,4],[5,5], [5,6], [5,7], [5,8], [5,9], [5,10]], op_targets = [], target=0)
make_tables.ssmm_table_test(moea_list = [], sklearn_list= [4,6], soea_list =  [[5, 1], [5,3], [5,4],[5,5], [5,6], [5,7], [5,8], [5,9], [5,10]], op_targets = [], target=0)


#==============================linear-multi-obj
#--------------------------plot
plotting.plotting_mix_ssmm_train(moea_list = [[1, 3], [1, 12], [1, 14]], sklearn_list = [4, 6], soea_list = [], op_targets=[[0, 3, 4]], target=[0,4])
plotting.plotting_mix_ssmm_test(moea_list = [[1, 3], [1, 12], [1, 14]], sklearn_list = [4, 6], soea_list = [], op_targets=[[0, 3, 4]], target=[0,4])

#--------------------------table
# 1
make_tables.ssmm_table_train(moea_list = [[1, 1], [1, 9], [1, 10], [1, 11]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0, 1], [0,2], [0,1,2], [0,4], [0,2,4],[0,3], [0,1,3], [0, 3, 4]], target=0)
make_tables.ssmm_table_test(moea_list = [[1, 1], [1, 9], [1, 10], [1, 11]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0, 1], [0,2], [0,1,2], [0,4], [0,2,4],[0,3], [0,1,3], [0, 3, 4]], target=0)
# 2
make_tables.ssmm_table_train(moea_list = [[1, 3]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0,2], [0,4], [0,2,4],[0,3],[0, 3, 4]], target=0)
make_tables.ssmm_table_test(moea_list = [[1, 3]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0,2], [0,4], [0,2,4],[0,3],[0, 3, 4]], target=0)

# 3
make_tables.ssmm_table_train(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0,4], [0,2,4],[0, 3, 4]], target=0)
make_tables.ssmm_table_test(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0,4], [0,2,4],[0, 3, 4]], target=0)
#==============================mlp3-multi-obj
#--------------------------plot
plotting.plotting_mix_ssmm_train(moea_list = [[5, 12], [5, 13], [5, 14]], sklearn_list = [4, 6], soea_list = [], op_targets=[[0,3,4]], target=[0,4])
plotting.plotting_mix_ssmm_test(moea_list = [[5, 12], [5, 13], [5, 14]], sklearn_list = [4, 6], soea_list = [], op_targets=[[0,3,4]], target=[0,4])

#--------------------------table
make_tables.ssmm_table_train(moea_list = [[5, 12], [5, 13], [5, 14]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0, 3, 4]], target=0)
make_tables.ssmm_table_test(moea_list = [[5, 12], [5, 13], [5, 14]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0, 3, 4]], target=0)
#==============================mlp5-multi-obj
#--------------------------plot
plotting.plotting_mix_ssmm_train(moea_list = [[6, 12], [6, 13], [6, 14]], sklearn_list = [4, 6], soea_list = [], op_targets=[[0,3,4]], target=[0,4])
plotting.plotting_mix_ssmm_test(moea_list = [[6, 12], [6, 13], [6, 14]], sklearn_list = [4, 6], soea_list = [], op_targets=[[0,3,4]], target=[0,4])

#--------------------------table
make_tables.ssmm_table_train(moea_list = [[6, 12], [6, 13], [6, 14]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0, 3, 4]], target=0)
make_tables.ssmm_table_test(moea_list = [[6, 12], [6, 13], [6, 14]], sklearn_list= [4, 6], soea_list =  [], op_targets = [[0, 3, 4]], target=0)

# -----------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------#
# ================================ 0314 号总结中的绘图======================================#
# -----------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------#


#============================单目标优化FPA的对应NONZ值--------------------------------------#
# ==============================linear-single-obj
#----------------------------table
make_tables.ssmm_table_train(moea_list = [], sklearn_list= [4,6], soea_list = [[1, 1], [1,3], [1,4],[1,5], [1,6], [1,7], [1,8], [1,9], [1,10]], op_targets = [], target=2)
make_tables.ssmm_table_test(moea_list = [], sklearn_list= [4,6], soea_list = [[1, 1], [1,3], [1,4],[1,5], [1,6], [1,7], [1,8], [1,9], [1,10]], op_targets = [], target=2)

# ================================mlp3-single

#------------------------------table
make_tables.ssmm_table_train(moea_list = [], sklearn_list= [4,6], soea_list =  [[5, 1], [5,3], [5,4],[5,5], [5,6], [5,7], [5,8], [5,9], [5,10]], op_targets = [], target=2)
make_tables.ssmm_table_test(moea_list = [], sklearn_list= [4,6], soea_list =  [[5, 1], [5,3], [5,4],[5,5], [5,6], [5,7], [5,8], [5,9], [5,10]], op_targets = [], target=2)


#============================nsga2结果和lasso的对比--------------------------------------#
plotting.plotting_universal_test(moea_list = [[1, 3, [0, 2]], [1, 3, [0, 4]], [1, 3, [0, 2, 4]]], sklearn_list = [4], soea_list = [], target=[0,4])
plotting.plotting_universal_train(moea_list = [[1, 3, [0, 2]], [1, 3, [0, 4]], [1, 3, [0, 2, 4]]], sklearn_list = [4], soea_list = [], target=[0,4])

plotting.plotting_universal_test(moea_list = [[1, 3, [0, 2]], [1, 3, [0, 4]], [1, 3, [0, 2, 4]]], sklearn_list = [4], soea_list = [], target=[0,2])
plotting.plotting_universal_train(moea_list = [[1, 3, [0, 2]], [1, 3, [0, 4]], [1, 3, [0, 2, 4]]], sklearn_list = [4], soea_list = [], target=[0,2])

#mlp和rf的对比
plotting.plotting_universal_test(moea_list = [[5, 12, [0, 3, 4]], [5, 13, [0, 3, 4]], [5, 14, [0, 3, 4]]], sklearn_list = [6], soea_list = [], target=[0,4])
plotting.plotting_universal_train(moea_list = [[5, 12, [0, 3, 4]], [5, 13, [0, 3, 4]], [5, 14, [0, 3, 4]]], sklearn_list = [6], soea_list = [], target=[0,4])



#============================ mlp3选取出来最优的值对应的mse的结果与rf的对比====================
train_ratio = 0.5,pratios=[1,0,0], random_size = 1

train_ratio = 0.5,pratios=[1,0,0], random_size = 0.5
train_ratio = 0.8,pratios=[8,4,1], random_size = 1
train_ratio = 0.8,pratios=[1,0,0], random_size = 1
 # save_file = 'balance_tvratio_test_' + para_name[target]
make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 0, if_max = True, train_ratio = 0.5,pratios=[1,0,0], random_size = 1)
make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 0, if_max = True, train_ratio = 0.5,pratios=[1,0,0], random_size = 0.5)
make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 0, if_max = True, train_ratio = 0.8,pratios=[8,4,1], random_size = 1)
make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 0, if_max = True, train_ratio = 0.8,pratios=[1,0,0], random_size = 1)

make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 4, if_max = True, train_ratio = 0.5,pratios=[1,0,0], random_size = 1)
make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 4, if_max = True, train_ratio = 0.5,pratios=[1,0,0], random_size = 0.5)
make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 4, if_max = True, train_ratio = 0.8,pratios=[8,4,1], random_size = 1)
make_tables.make_balance_tvratio(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = 4, if_max = True, train_ratio = 0.8,pratios=[1,0,0], random_size = 1)

# 同时获取相应的fpa及mse值
make_tables.choose_model(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = [0, 4], if_max = True, train_ratio = 0.5,pratios=[1,0,0], random_size = 1)
make_tables.choose_model(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = [0, 4], if_max = True, train_ratio = 0.5,pratios=[1,0,0], random_size = 0.5)
make_tables.choose_model(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = [0, 4], if_max = True, train_ratio = 0.8,pratios=[8,4,1], random_size = 1)
make_tables.choose_model(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = [0, 4], if_max = True, train_ratio = 0.8,pratios=[1,0,0], random_size = 1)
make_tables.choose_model(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = [0, 4], if_max = True, train_ratio = 0.8,pratios=[1,1,1], random_size = 0.3)



make_tables.choose_model(moea_list = [[6, 13], [5, 14]], op_targets =[[0, 3, 4]], parameters = [0, 3, 4], target = [0, 4], if_max = True, train_ratio = 0.8,pratios=[1,1,1], random_size = 0.2)


#====================================================0321=================================================
#=============================================================================================================
#============================================================================================================
#===================绘制0321不同算法对fpa+mse/fpa+nnz对比/


#------------------------------------linear/fpa+mse+l1/三种random的对比
#----------------fpa/mse对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 3, 4]], target=[0,4],color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 3, 4]], target=[0,4], color = False)
#fpa/nnz对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 3, 4]], target=[0,2], color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 3, 4]], target=[0,2], color = False)


#------------------------------------linear/fpa+mse+nnz/三种random的对比
#----------------fpa/mse对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2, 4]], target=[0,4],color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2, 4]], target=[0,4], color = False)
#fpa/nnz对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2, 4]], target=[0,2], color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2, 4]], target=[0,2], color = False)

#------------------------------------linear/fpa+mse/三种random的对比
#----------------fpa/mse对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 4]], target=[0,4],color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 4]], target=[0,4], color = False)
#fpa/nnz对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 4]], target=[0,2], color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 12], [1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 4]], target=[0,2], color = False)


#-------linear/fpa+mse+l1/random20p,linear/fpa+mse+l1/random30p,
#----------------fpa/mse对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2,4], [0, 3, 4]], target=[0,4],color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2,4], [0, 3, 4]], target=[0,4], color = False)
#fpa/nnz对比
plotting.plotting_mix_ssmm_train(moea_list = [[1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2,4], [0, 3, 4]], target=[0,2], color = False)
plotting.plotting_mix_ssmm_test(moea_list = [[1, 13], [1, 14]], sklearn_list = [2, 4], soea_list = [], op_targets=[[0, 2,4], [0, 3, 4]], target=[0,2], color = False)

#-----------------------------从非支配集中选取train最优的值
make_tables.make_btrain_test(moea_list = [[1, 13], [1, 14]], op_targets = [[0, 3, 4], [0, 2, 4], [0, 4]], target = 0, if_max = True)
make_tables.make_btrain_test(moea_list = [[1, 9], [1, 10], [1, 11]], op_targets = [[0, 3, 4],[0, 2], [0, 3], [0, 4], [0, 2, 4], [0, 4], [0, 1], [0, 1, 2]], target = 0, if_max = True)


#-----------------------------运行一下FPA+NNZ的三个random，运行一下FPA+L1的三个random
training.run_msdp_train(moea = 12, targets = [0, 2], predict_model = 1)
training.run_msdp_train(moea = 13, targets = [0, 2], predict_model = 1)
training.run_msdp_train(moea = 14, targets = [0, 2], predict_model = 1)

training.run_msdp_train(moea = 12, targets = [0, 3], predict_model = 1)
training.run_msdp_train(moea = 13, targets = [0, 3], predict_model = 1)
training.run_msdp_train(moea = 14, targets = [0, 3], predict_model = 1)


test.run_msdp_test(method = 1, moea = 12, target = [0, 2])
test.run_msdp_test(method = 1, moea = 13, target = [0, 2])
test.run_msdp_test(method = 1, moea = 14, target = [0, 2])

test.run_msdp_test(method = 1, moea = 12, target = [0, 3])
test.run_msdp_test(method = 1, moea = 13, target = [0, 3])
test.run_msdp_test(method = 1, moea = 14, target = [0, 3])
#---------------------------根据之前选取train最优的结果，选取几个模型进行对比
# linear/nsga2_30p_toZero/FPA_NNZ; linear/nsga2_30p_toZero/FPA_NNZ_MSE; linear/nsga2_10p_toZero/FPA_NNZ_MSE; linear/fpa+mse+nnz/random30p.
plotting.plotting_universal_test(moea_list = [[1, 11, [0, 2]], [1, 11, [0,3, 4]], [1, 9, [0, 2, 4]], [1, 14, [0, 3, 4]]], sklearn_list = [2,4], soea_list = [], target=[0,2], color = False)
plotting.plotting_universal_train(moea_list =  [[1, 11, [0, 2]], [1, 11, [0,3, 4]], [1, 9, [0, 2, 4]], [1, 14, [0, 3, 4]]], sklearn_list = [2, 4], soea_list = [], target=[0,2], color = False)

plotting.plotting_universal_test(moea_list = [[1, 11, [0, 2]], [1, 11, [0,3, 4]], [1, 9, [0, 2, 4]], [1, 14, [0, 3, 4]]], sklearn_list = [2,4], soea_list = [], target=[0,4], color = False)
plotting.plotting_universal_train(moea_list =  [[1, 11, [0, 2]], [1, 11, [0,3, 4]], [1, 9, [0, 2, 4]], [1, 14, [0, 3, 4]]], sklearn_list = [2, 4], soea_list = [], target=[0,4], color = False)


#---------------------------按照比例，以及引入随机来选取非支配解
#-------------------------首先选fpa最好的
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,0,0], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 3, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,0,0], random_size = 1, if_split = False)

#-------------------------最优的测试fpa是多少
make_tables.ssmm_table_test(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], sklearn_list =[], soea_list = [], op_targets=[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 3, 4]] , target = 0, if_max = True, if_split = False)
make_tables.ssmm_table_test(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], sklearn_list =[2, 4], soea_list = [], op_targets=[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]] , target = 0, if_max = True, if_split = False)

#---------------------------1:1:1-034
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,1], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 3, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,1], random_size = 1, if_split = False)

#---------------------------1:1:1-024
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,1], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,1], random_size = 1, if_split = False)

#--------------------------2:2:1-024
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[2,2,1], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[2,2,1], random_size = 1, if_split = False)

#--------------------------2:2:1-034

make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[2,2,1], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 2, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[2,2,1], random_size = 1, if_split = False)

#-------------------------4:2:1-024
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[4,2,1], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[4,2,1], random_size = 1, if_split = False)

#-------------------------4:2:1-034
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[4,2,1], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 2, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[4,2,1], random_size = 1, if_split = False)

#-------------------------1:1:0-024
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,0], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,0], random_size = 1, if_split = False)

#-------------------------1:1:0-034
make_tables.choose_model(moea_list = [[1, 3], [1, 12],[1, 13], [1, 14]], op_targets =[[0, 2], [0, 3], [0, 4], [0, 3, 4], [0, 2, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,0], random_size = 1, if_split = False)
make_tables.choose_model(moea_list = [[1, 1], [1, 9],[1, 10], [1, 11]], op_targets =[[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2],[0, 1, 3], [0, 2, 4], [0, 3, 4]], parameters = [0, 3, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[1,1,0], random_size = 1, if_split = False)


#-----------------------------绘图：绘制linear/nsga2_random20p_toZero/FPA_NNZ_MSE；linear/nsga2_random30p_toZero/FPA_L1_MSE
#ridge；lasso对比
#--------------------fpa+mse
plotting.plotting_universal_test(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,4], color = False)
plotting.plotting_universal_train(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,4], color = False)

#-------------------fpa+nnz
plotting.plotting_universal_test(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,2], color = False)
plotting.plotting_universal_train(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,2], color = False)



#======================================================================
#-=====================================================================
#=========================0329绘图=====================================

#=============绘制nsga2-20p-toZero/FPA+MSE+NNZ/linear与nsga2-30p-toZero/FPA+MSE+NNZ与linear/CoDE对比
plotting.plotting_universal_test(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,4],if_line = True, line_list = [[1, 1]], color = False)
plotting.plotting_universal_train(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,4], if_line = True, line_list = [[1, 1]], color = False)

#=============组合到一张图绘制nsga2-20p-toZero/FPA+MSE+NNZ/linear与nsga2-30p-toZero/FPA+MSE+NNZ与linear/CoDE对比
plotting.plotting_universal_test(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,4],if_line = True, line_list = [[1, 1]], color = False, if_combine = True, if_show_label = False)
plotting.plotting_universal_train(moea_list = [[1, 13, [0, 2, 4]], [1, 14, [0, 3, 4]],], sklearn_list = [2, 4], soea_list = [], target=[0,4], if_line = True, line_list = [[1, 1]], color = False, if_combine = True, if_show_label = False)


#=================================================================\
#=================================================================
#======================0406绘图====================================

plotting.plotting_universal_log04_train(moea_list=[[1, 13,[0,2,4]]], sklearn_list=[1,2,4], soea_list=[[1,1]])
plotting.plotting_universal_log04_test(moea_list=[[1, 13,[0,2,4]]], sklearn_list=[1,2,4], soea_list=[[1,1]])



#=================================================================\
#=================================================================
#======================0407重新运行====================================
training.train_ssdp_m(soea=1, predict_model=1)
test.run_ssdpM_test(method = 1, soea = 1)



#=======================================绘图nsga2/nsga2-random20ptozero/code/04=====
plotting.plotting_universal_log04_train(moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]])
plotting.plotting_universal_log04_test(moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]])
#=======================================绘图nsga2/nsga2-random20ptozero/code /04无标签=====

plotting.plotting_universal_log04_train(if_show_label = False, moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]])
plotting.plotting_universal_log04_test(if_show_label = False,moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]])


#=======================================绘图nsga2/nsga2-random20ptozero/code/02=====
plotting.plotting_universal_test(moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False)
plotting.plotting_universal_train(moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False)

#=======================================绘图nsga2/nsga2-random20ptozero/code 无标签/02=====

plotting.plotting_universal_test(moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False, if_show_label = False)
plotting.plotting_universal_train(moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False, if_show_label = False)


#=============================================================================
#===============================0408==========================================
#=====================================绘图记录多个值的CoDE算法

#===============================确定fontsize 和markersize
plotting.plotting_universal_log04_train(wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5], moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]], if_sM = True)
plotting.plotting_universal_log04_test(wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5],moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]], if_sM = True)

#=======================================绘图nsga2/nsga2-random20ptozero/code/04=====
plotting.plotting_universal_log04_train(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5], moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]], if_sM = True)
plotting.plotting_universal_log04_test(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5], moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]], if_sM = True)
#=======================================绘图nsga2/nsga2-random20ptozero/code /04无标签=====

plotting.plotting_universal_log04_train(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5],if_show_label = False, moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]], if_sM = True)
plotting.plotting_universal_log04_test(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5],if_show_label = False,moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]], if_sM = True)


#=======================================绘图nsga2/nsga2-random20ptozero/code/02=====
plotting.plotting_universal_test(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5],moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False, if_sM = True)
plotting.plotting_universal_train(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5],moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False, if_sM = True)

#=======================================绘图nsga2/nsga2-random20ptozero/code 无标签/02=====

plotting.plotting_universal_test(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5],moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False, if_show_label = False, if_sM = True)
plotting.plotting_universal_train(type = 1,wsize = [20, 20, 10, 14], msize = [16, 16], figsize = [8, 5],moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False, if_show_label = False, if_sM = True)



#================================================================================
################################################################################
#===============================表格结果=========================================
make_tables.choose_model(moea_list = [[1, 1], [1, 13]], soea_list = [[1,1]],op_targets =[[0, 2, 4]], parameters = [0, 2, 4],train_ratio = 1, target = [0,2, 4], if_max = True,pratios=[4,2,1], random_size = 1, if_split = False)


#================================================================================
#=========================绘图之毕业论文用图==================================
#================================绘图========================================
#=======================================绘图nsga2/nsga2-random20ptozero/code/04=====
plotting.plotting_universal_log04_test(type = 1,wsize = [20, 30, 10, 20], msize = [16, 16], figsize = [10, 6],if_show_label = False,moea_list=[[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list=[], soea_list=[[1,1]], if_sM = True)

#=======================================绘图nsga2/nsga2-random20ptozero/code 无标签/02=====

plotting.plotting_universal_test(type = 1,wsize = [20, 30, 10, 20], msize = [16, 16], figsize = [8, 5],moea_list = [[1, 13,[0,2,4]], [1, 1,[0,2,4]]], sklearn_list = [], soea_list = [[1, 1]], target=[0,2], color = False, if_show_label = False, if_sM = True)

