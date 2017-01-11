import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
import sys

from LCTM import models
from LCTM import metrics

sc_vals = [1, 10, 20, 30, 40, 50, 100, 150]
users = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
supertrials = ['1', '2', '3', '4', '5']

for sc_length in sc_vals:
    final_avg_acc = []
    final_avg_overlap = []
    final_avg_edit = []

    #for st in supertrials:
    for user in users:

        #print('Test:', 'supertrial=', st, 'sc=', sc_length)
        print('Test:', 'user=', user, 'sc=', sc_length)
        train = sc.loadmat('/cis/home/divya/SDSDL/sparse-z/LOUO/Suturing/LOUO-'+user+'.mat')
        test = sc.loadmat('/cis/home/divya/SDSDL/sparse-z/LOUO/Suturing/LOUO-'+user+'-test.mat')


        #print(train.keys())
        #print(test.keys())
        X_train = train['X_train']
        y_train = train['y_train']
        X_test = test['X_test']
        y_test = test['y_test']

        X_train_lst = []
        y_train_lst = []
        index = 0

#        all_training_data = np.empty((0, 400), float)

        m,n = y_train.shape
        for i in range(m):
            for j in range(n):
                trial = y_train[i][j]
                if trial.shape != (0, 0):
                    #print(trial[0].shape)
                    #print(trial[0])
                    y_train_lst.append(np.int64(trial[0]))
                    x = X_train[:, index:index+trial.shape[1]]
                    #print(x.shape)
                    X_train_lst.append(x)
                    #all_training_data = np.r_[all_training_data, x.T]
                    #print(np.linalg.norm(x))
                    index += trial.shape[1]

        print("Training on", index-1, "samples...")

        #print('all training data size', all_training_data.shape)
        #norms = [np.linalg.norm(all_training_data[:, x]) for x in range(all_training_data.shape[1])]
        #norms = np.array(norms)[:,None]
        #print(norms.shape)
        #print(np.amin(norms), np.amax(norms))

        #norms = sum(norms) / len(norms)
        #print(norms)
        maxnorms = 0;

        for i in range(len(X_train_lst)):
            #divide by the L2 norm for each feature vector
            #print(X_train_lst[i].shape)
            norms = [np.linalg.norm(X_train_lst[i][:, x])for x in range(X_train_lst[i].shape[1] )]
            #X_train_lst[i] = X_train_lst[i] / norms
            maxnorms = max(norms)
            X_train_lst[i] = X_train_lst[i] / norms
            

        #X_std = np.hstack(X_train_lst).std(1)[:,None]
        #X_train_lst = [(x-x.mean(1)[:,None])/X_std for x in X_train_lst]

        norms = 0
        model = models.ChainModel(skip=int(sc_length))
        #model.inference_type = "filtered"
        #model.filter_len = max(int(sys.argv[1])//2, 1)
        model.fit(X_train_lst, y_train_lst, n_iter=200)

        m,n = y_test.shape

        avg_acc = []
        avg_overlap = []
        avg_edit = []

        index = 0
        for i in range(m):
            for j in range(n):
                gt = y_test[i][j]
                if gt.shape != (0,0):
                    #print(gt[0].shape)
                    #print(gt[0])
                    test_trial = X_test[:, index:index+gt.shape[1]]
                    #print(test_trial.shape)
                    norms = [np.linalg.norm(test_trial[:, x])for x in range(test_trial.shape[1] )]
                    test_trial = test_trial / norms
                    #test_trial = test_trial 
                    #test_trial = (test_trial - test_trial.mean(1)[:,None])/X_std 
                    pred = model.predict(test_trial)
                    #print (pred)
                    
                    acc = sum(pred == gt[0]) / len(pred)
                    avg_acc.append(acc)
                    avg_overlap.append(metrics.overlap_score(pred, gt[0]))
                    avg_edit.append(metrics.edit_score(pred, gt[0]))
                    
                    #print(acc)
                    index += test_trial.shape[1]

        print("Tested", index-1, "samples..")
        final_avg_acc.append(sum(avg_acc)/len(avg_acc)*100.0 )
        final_avg_overlap.append(sum(avg_overlap)/len(avg_overlap)*1.0) 
        final_avg_edit.append(sum(avg_edit)/len(avg_edit)*1.0)
        
        #print(sum(avg_acc)/len(avg_acc)*100.0 )
        #print(sum(avg_overlap)/len(avg_overlap)*1.0) 
        #print(sum(avg_edit)/len(avg_edit)*1.0)

        #to visualize the trainied pairwise weights
        #x = model.get_weights('pw')
        #np.savetxt('sparse_pw', np.exp(x))

    print('SC', sc_length)
    print(sum(final_avg_acc)/len(final_avg_acc))
    print(sum(final_avg_overlap)/len(final_avg_overlap))
    print(sum(final_avg_edit)/len(final_avg_edit))

