# CROSS VALIDATION for Suturing B
# Test to determine optimal value for C hyperparameter

import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
import crf

from LCTM import models
from LCTM import metrics

import matlab.engine

# Extract training and testing data
# These are the sparse codes

###### CHANGE THESE PATHS TO DIRECT TO LOCAL sparse-z DIRECTORY
train = sc.loadmat('/cis/home/divya/SDSDL/sparse-z/LOUO/Suturing/LOUO-B.mat')
test = sc.loadmat('/cis/home/divya/SDSDL/sparse-z/LOUO/Suturing/LOUO-B-test.mat')
######

X_train = train['X_train']
y_train = train['y_train']
X_test = test['X_test']
y_test = test['y_test']

# Split X_train into training data and validation set
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

y_train_cv_valset = y_train[0,:]
y_train_cv = y_train[1:-1, :]
split_valset_index = sum([x.shape[1] for x in y_train_cv_valset])

y_train_cv_valset = np.reshape(y_train_cv_valset, (1, 5))

num_training = X_train.shape[1]
print(split_valset_index)

X_train_cv_valset = X_train[:, 0:split_valset_index]
X_train_cv = X_train[:, split_valset_index:num_training]

print(X_train_cv.shape, y_train_cv.shape)
print(X_train_cv_valset.shape, y_train_cv_valset.shape)

# n_psi_update = 0
# no updates to psi in these tests
n_wp_iter = 300
skip_chain = 1

acc = {}

for p in range(10):
    c = np.power(10, p)
    print('Cross val test for c = ', c)
    # gradient_in = 'joint_learning/input-iter'
    # gradient_out = 'joint_learning/output-iter'
    # initialization_data_filename = '/cis/home/divya/SDSDL/initialization/LOUO/Suturing-LOUO-B-n-35-forsupervised.mat'
    # prev_psi_update = initialization_data_filename

    metrics, y_star, y_gt, ws, hamming, obj = crf.crf_optimize(X_train_cv, y_train_cv, X_train_cv_valset, y_train_cv_valset, None, n_wp_iter, skip_chain, c)
    #print(hamming)
    #print(obj)

    # np.savetxt('c_crossval/obj_psi-update-'+str(n_psi_update)+'_wp-iter-'+str(n_wp_iter)+'_c-'+str(c)+'_objective', list(obj.items()))
    # np.savetxt('c_crossval/obj_psi-update-'+str(n_psi_update)+'_wp-iter-'+str(n_wp_iter)+'_c-'+str(c)+'_hamming', list(hamming.items()))
    
    print('c =', c, ':', metrics['accuracy'])

    acc[c] = metrics['accuracy']


print('Accuracies', acc)
best_c = max(acc.keys(), key=lambda k: acc[k])
print('Best c = ', best_c)

# Use cross-validated C to train original train and test data
metrics, y_star, y_gt, ws, hamming, obj = crf.crf_optimize(X_train, y_train, X_test, y_test, None, n_wp_iter, skip_chain, best_c)
print(metrics)



# np.savetxt('c_crossval/test_accuracy', list(acc.items()))
