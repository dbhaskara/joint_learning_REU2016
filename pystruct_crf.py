"""
Input SDSDL sparse codes to pystruct ChainCRF() Model

"""
import scipy.io as sc
import numpy as np
import matplotlib
# Trying to enable graphics with X11 Forwarding..
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from pystruct.plot_learning import plot_learning


from LCTM import models
from LCTM import metrics
from LCTM import utils

# Script command line input
skip = sys.argv[1]
user = sys.argv[2]


# Import Matlab files from sparse-louo
train = sc.loadmat('/cis/home/divya/SDSDL/sparse-louo/louo-'+user+'.mat')
test = sc.loadmat('/cis/home/divya/SDSDL/sparse-louo/louo-'+user+'-test.mat')

# Extract sparse codes: Training and Testing
X_train = train['X_train']
y_train = train['y_train']
X_test = test['X_test']
y_test = test['y_test']

# Reformat the training data
X_train_lst = []
y_train_lst = []
index = 0

all_training_data = np.empty((0, 400), float)
all_testing_data = np.empty((0, 400), float)

# Extract the individual sequences
m,n = y_train.shape
for i in range(m):
    for j in range(n):
        trial = y_train[i][j]
        if trial.shape != (0, 0):
            print(trial[0].shape)
            #print(trial[0])
            y_train_lst.append(np.int64(trial[0]))
            x = X_train[:, index:index+trial.shape[1]]
            print(x.shape)
            X_train_lst.append(x)
            all_training_data = np.r_[all_training_data, x.T]
            #print(np.linalg.norm(x))
            index += trial.shape[1]

print("Training on", index-1, "samples...")

print('all training data size', all_training_data.shape)
norms = [np.linalg.norm(all_training_data[:, x]) for x in range(all_training_data.shape[1])]
norms = np.array(norms)[:,None]
#print(norms.shape)

#norms = sum(norms) / len(norms)
#print(norms)

#for i in range(len(X_train_lst)):
#    X_train_lst[i] = X_train_lst[i] / norms

#X_std = np.hstack(X_train_lst).std(1)[:,None]
#X_train_lst = [(x-x.mean(1)[:,None])/X_std for x in X_train_lst]


m,n = y_test.shape

#avg_acc = []
#avg_overlap = []
#avg_edit = []

X_test_lst = []
y_test_lst = []
index = 0

for i in range(m):
    for j in range(n):
        gt = y_test[i][j]
        if gt.shape != (0,0):
            #print(gt[0].shape)
            #print(gt[0])
            test_trial = X_test[:, index:index+gt.shape[1]]
            #print(test_trial.shape)
            #test_trial = test_trial / norms
            #test_trial = (test_trial - test_trial.mean(1)[:,None])/X_std 
            #pred = model.predict(test_trial)
            #print (pred)
            all_testing_data = np.r_[all_testing_data, test_trial.T]
            #acc = sum(pred == gt[0]) / len(pred)
            #avg_acc.append(acc)
            #avg_overlap.append(metrics.overlap_score(pred, gt[0]))
            #avg_edit.append(metrics.edit_score(pred, gt[0]))
            
            X_test_lst.append(test_trial)
            y_test_lst.append(np.int64(gt[0]))
            index += test_trial.shape[1]

#print("Tested", index-1, "samples..")
#print("accuracy:", sum(avg_acc)/len(avg_acc)*100.0 )
#print("overlap score:", sum(avg_overlap)/len(avg_overlap)*1.0 )
#print("edit score:", sum(avg_edit)/len(avg_edit)*1.0 )
ts_norms = [np.linalg.norm(all_testing_data[:, x]) for x in range(all_testing_data.shape[1])]
ts_norms = np.array(ts_norms)[:,None]

n_train = len(X_train_lst)
n_test = len(X_test_lst)

y_all = utils.remap_labels(np.hstack([y_train_lst, y_test_lst]))
y_train_lst, y_test_lst = y_all[:n_train], y_all[-n_test:]

#X_std = np.hstack(X_train_lst).std(1)[:,None]
#X_train_lst = [(x-x.mean(1)[:,None])/X_std for x in X_train_lst]
#X_test_lst = [(x-x.mean(1)[:,None])/X_std for x in X_test_lst]

for i in range(len(X_train_lst)):
    X_train_lst[i] = (X_train_lst[i]).T

for i in range(len(X_test_lst)):
    X_test_lst[i] = (X_test_lst[i]).T


model = ChainCRF()
ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
ssvm.fit(X_train_lst, y_train_lst)

plot_learning(ssvm, time=False)

print(ssvm.score(X_test_lst, y_test_lst))


