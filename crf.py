import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
import sys

from LCTM import models
from LCTM import metrics
from LCTM import utils
from LCTM import ssvm

"""
Function to train LCTM Chain Model on JIGSAWS data
Optimization of the CRF (unary and pairwise terms)

INPUTS
X_train, y_train: training data
X_test, y_test: testing data
init_ws: initializations for CRF weights of type Weights
n_iter: number of iterations of CRF optimization
skip: skip-chain value for pairwise term calculation

OUTPUTS
objective: objective function values for iterations [0, n_iter]
y_star: y* from loss augmented inference
ws: instance of Weights class containing optimized unary and pairwise terms
"""
def crf_optimize(X_train, y_train, X_test, y_test, init_ws, n_iter, skip, c = 1.): 
    
    # ---------------- FORMAT DATA -------------------

    
    # Format the data into a list of sequences
    X_train_lst = [] # Each sequence is of size features x timesteps
    y_train_lst = [] # Each sequence's ground truth is of size 1 x timesteps
    index = 0

    # X_train_lst
    m,n = y_train.shape
    for i in range(m):
        for j in range(n):
            trial = y_train[i][j]
            if trial.shape != (0, 0):
                y_train_lst.append(np.int64(trial[0]))
                x = X_train[:, index:index+trial.shape[1]]
                X_train_lst.append(x)
                index += trial.shape[1]
    n_train_frames = index;

    # Normalize X_train_lst
    for i in range(len(X_train_lst)):
        # Divide by the L2 norm for each feature vector
        norms = [np.linalg.norm(X_train_lst[i][:, x])for x in range(X_train_lst[i].shape[1] )]
        X_train_lst[i] = X_train_lst[i]/ norms
    norms = 0
    
    X_test_lst = []
    y_test_lst = []
    index = 0

    # X_test_lst
    m,n = y_test.shape
    for i in range(m):
        for j in range(n):
            test_trial = y_test[i][j]
            if test_trial.shape != (0, 0):
                y_test_lst.append(np.int64(test_trial[0]))
                x = X_test[:, index:index+test_trial.shape[1]]
                X_test_lst.append(x)
                index += test_trial.shape[1]
    n_test_frames = index;

    # Normalize X_test_lst
    for i in range(len(X_test_lst)):
        # Divide by the L2 norm for each feature vector
        norms = [np.linalg.norm(X_test_lst[i][:, x])for x in range(X_test_lst[i].shape[1] )]
        X_test_lst[i] = X_test_lst[i]/ norms
   
    #Re-map the labels
    n_train = len(X_train_lst)
    n_test = len(X_test_lst)
    y_all, y_map = utils.remap_labels(np.hstack([y_train_lst, y_test_lst]))
    y_train_lst, y_test_lst = y_all[:n_train], y_all[-n_test:]

    y_map = y_map.tolist()
    #print(y_map)

    # ---------------- TRAIN -------------------

    print("Training on", n_train_frames, "frames...")
    
    # Instantiate and train the ChainModel
    model = models.ChainModel(debug=True, skip=skip)
    if init_ws != None:
        model.n_features = X_train_lst[0].shape[0]
        model.n_classes = np.max(list(map(np.max, y_train_lst)))+1
        model.max_segs = utils.max_seg_count(y_train_lst)
        model.ws = init_ws
    hamming_dist, objective = model.fit(X_train_lst, y_train_lst, n_iter=n_iter, C=c)
    
    ws = model.ws

    # Calculate the most violated constraint y* using loss augmented inference
    y_star = []
    y_gt = y_train_lst
    for i in range(len(X_train_lst)):
        pred, score = model.predict(Xi = X_train_lst[i], Yi = y_train_lst[i], is_training = True)
        y_star.append(pred)

   
    #Map the labels back to the original ones
    for i in range(n_train):
        for t in range(len(y_star[i])):
            y_star[i][t] = y_map.index(y_star[i][t])
        for t in range(len(y_gt[i])):
            y_gt[i][t] = y_map.index(y_gt[i][t])
    y_star = np.array(y_star)
    y_gt = np.array(y_gt)
   
    # ---------------- TEST -------------------

    print("Testing on", n_test_frames, "frames...")

    # Accumulate metrics: accuracy, overlap score, and edit score
    avg_acc = []
    avg_overlap = []
    avg_edit = []

    for i in range(len(X_test_lst)):
        pred, score = model.predict(Xi = X_test_lst[i])
        acc = sum(pred == y_test_lst[i]) / len(pred)
        avg_acc.append(acc)
        avg_overlap.append(metrics.overlap_score(pred, y_test_lst[i]))
        avg_edit.append(metrics.edit_score(pred, y_test_lst[i]))

    hamming_test = np.mean([model.objective(model.predict(X_test_lst[i])[0], y_test_lst[i]) for i in range(len(X_test_lst))])
    objective_test = ssvm.objective_mm(model, X=X_test_lst, Y=y_test_lst)
    #print(hamming_test)
    #print(objective_test)

    
    # Metrics for test set
    accuracy = sum(avg_acc)/len(avg_acc)*100.0 
    overlap_score = sum(avg_overlap)/len(avg_overlap)*1.0 
    edit_score = sum(avg_edit)/len(avg_edit)*1.0
    final_metrics = {'accuracy': accuracy, 'overlap_score': overlap_score, 'edit_score': edit_score}            

    print("Tested", index, "samples..")
    print("accuracy:", accuracy)
    print("overlap score:", overlap_score)
    print("edit score:", edit_score)

    return final_metrics, y_star, y_gt, ws, hamming_dist, objective


