import scipy.io as sc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import crf

from LCTM import models
from LCTM import metrics

import matlab.engine

#Written to perform joint learning on Suturing LOUO B
#To change the training and testing data, update the following variables to the desired path: train, test, initialization_data_filename

LOUO = 'B'

###### CHANGE THESE PATHS TO DIRECT TO LOCAL sparse-z DIRECTORY
train = sc.loadmat('/cis/home/divya/SDSDL/sparse-z/LOUO/Suturing/LOUO-'+LOUO+'.mat')
test = sc.loadmat('/cis/home/divya/SDSDL/sparse-z/LOUO/Suturing/LOUO-'+LOUO+'-test.mat')
initialization_data_filename = '/cis/home/divya/SDSDL/initialization/LOUO/Suturing-LOUO-'+LOUO+'-n-35-forsupervised.mat'
######

X_train = train['X_train']
y_train = train['y_train']
X_test = test['X_test']
y_test = test['y_test']

n_psi_update = 2
n_wp_iter = 100
skip_chain = 100
c = 1

gradient_in = 'joint_learning/input-iter'
gradient_out = 'joint_learning/output-iter'
prev_psi_update = initialization_data_filename

metrics, y_star, y_gt, ws, hamming, obj = crf.crf_optimize(X_train, y_train, X_test, y_test, None, n_wp_iter, skip_chain)

#print and plot the objective function and testing accuracy
plt.plot(list(obj.keys()), list(obj.values()))
accuracies = {}
accuracies[n_wp_iter] = metrics['accuracy']

sc.savemat(gradient_in + '1', {'y_star': y_star, 'y_gt': y_gt })

for cur_psi in [x+1 for x in range(n_psi_update)]:
    
    eng = matlab.engine.start_matlab()
    eng.sgd_dict_par(initialization_data_filename, gradient_in + str(cur_psi), prev_psi_update, cur_psi)

    update = sc.loadmat(gradient_out + str(cur_psi))
    X_train = update['X_train']
    y_train = update['y_train']
    X_test = update['X_test']
    y_test = update['y_test']

    prev_psi_update = gradient_out + str(cur_psi)

    metrics, y_star, y_gt, ws, hamming, obj = crf.crf_optimize(X_train, y_train, X_test, y_test, ws, n_wp_iter, skip_chain)

    #plot the new objective function and testing accuracy
    plt.plot([cur_psi * n_wp_iter + x for x in list(obj.keys())], list(obj.values()))
    accuracies[(cur_psi+1) * n_wp_iter] = metrics['accuracy']

    sc.savemat(gradient_in + str(cur_psi+1), {'y_star': y_star, 'y_gt': y_gt })

#plot and save
plt.ylabel('Objective Function')
plt.xlabel('Number of w,P iterations')
plt.savefig('joint_learning/objective_plot')
plt.gcf().clear()
plt.ylabel('Testing Accuracy')
plt.xlabel('Number of w,P iterations')
plt.scatter(list(accuracies.keys()), list(accuracies.values()))
plt.savefig('joint_learning/test_accuracy_plot')

