import helpers
def training_record_for_mma(save_folder, target, predict_model, l, u, moea, drawing=0, maxgen=100):
    fileLists =
    para_name = {0: 'FPA', 1: 'AAE', 2: 'numOfnonZero', 3: 'L1', 4: 'MSE'}
    path = 'mma_data/'
    fileLists = ['results_blood', 'results_blood_without', 'results_urine','results_urine_without','results_urinewithblood','results_urinewithblood_without']

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
            print('\n\n\n' + fileLists[i] + '\n\n\n')
            X, y = helpers.getfeatures(path, fileLists[i][j])
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