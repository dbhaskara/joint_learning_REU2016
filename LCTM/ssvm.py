import numpy as np
from numba import jit, float64, int64
from copy import deepcopy

from LCTM import weights

def hamming_loss(Yi, Y_truth):
    return np.sum(Yi!=Y_truth).astype(np.float)

def objective_01(model, Yi, Y_truth):
    return np.sum(Yi!=Y_truth).astype(np.float)

# Loss defined by max-margin formulation
def objective_mm(model, X, Y):
    n_samples = len(X) #number of sequences in the training set 
    max_margins = []

    for i in range(n_samples):
        predict, score = model.predict(Xi=X[i], Yi=Y[i], is_training=True)
        
        # Compute hamming loss
        #hamming = hamming_loss(predict, Y[i])

        # Compute max-margin loss
        # ..by first computing the scores
        #_, n_timesteps = X[i].shape
        #n_nodes = model.n_nodes
        
        #score = np.zeros([n_nodes, n_timesteps], np.float64)

        #for key in model.potentials:
        #    score = model.potentials[key].compute(model, X[i], score)
        
        # ..then computing the score for each sequence, summed over time
        score_predict = np.sum(np.choose(predict, score))
        score_groundtruth = np.sum(np.choose(Y[i], score))
        
        mm = score_predict - score_groundtruth  
        max_margins.append(mm)

    # Calculate j, the loss for the training set
    j = np.mean(max_margins)
    
    # Add regularizer(s)
    w_regularizer =  pow(np.linalg.norm(model.ws['unary']), 2) 
    pw_regularizer = pow(np.linalg.norm(model.ws['pw']), 2)

#    j += w_regularizer + pw_regularizer 

    return j 


def compute_costs(model, Xi, Yi):
    # costs = weights.Weights()
    # costs.init_weights(model)
    costs = deepcopy(model.ws) * 0
    for key in model.potentials:
        # print(key)
        costs[key] += model.potentials[key].cost_fcn(model, Xi, Yi)
    return costs

def compute_ssvm_gradient(model, Xi, Yi, cost_truth=None, C=1.):
    
    # Predict states
    if model.is_latent:
        Zi = predict_best_latent(model, Xi, Yi)
        cost_truth = compute_costs(model, Xi, Zi)
        predict, score = model.predict(Xi, Yi=Yi, is_training=True, output_latent=True)
    else:
        predict, score = model.predict(Xi, Yi=Yi, is_training=True)
    
    # Get costs for the predicted labels
    cost_predict = compute_costs(model, Xi, predict)

    # Compute gradient between expected and predicted labelings
    w_diff = (cost_predict-cost_truth)*C
    #print('model', np.mean(np.mean(model.ws['unary'])))
    #print('wdiff', np.mean(np.mean(w_diff['unary'])))
    
    # Regularizer
#    w_diff += model.ws


    return w_diff

def reduce_latent_states(score, n_latent, n_classes):
    score = score.reshape([n_classes, n_latent, -1])
    return score.max(1)

@jit("int64[:](float64[:,:], int64[:], int64)")
def predict_best_latent_(score, Yi, n_latent):
    n_timesteps = score.shape[1]
    path = np.empty(n_timesteps, np.int64)
    
    for t in range(n_timesteps):
        start = Yi[t]*n_latent
        stop = (Yi[t]+1)*n_latent
        best_latent = score[start:stop, t].argmax()
        path[t] = Yi[t]*n_latent + best_latent

    return path

def predict_best_latent(model, Xi, Yi):
    n_timesteps = Xi.shape[1]

    # Initialize score
    n_nodes = model.n_nodes
    score = np.zeros([n_nodes, n_timesteps], np.float64)

    # Add potentials to score
    for key in model.potentials:
        score = model.potentials[key].compute(model, Xi, score)
    
    return predict_best_latent_(score, Yi, model.n_latent)


@jit("float64[:,:](float64[:,:], int64[:])")
def loss_augmented_unaries(score, Yi):
    n_classes, T = score.shape
    score += 1
    for t in range(T):
        score[Yi[t],t] -= 1

    return score

@jit("float64[:,:](float64[:,:], int64[:], int64)")
def latent_loss_augmented_unaries(score, Yi, n_latent):
    n_classes, T = score.shape
    score += 1
    for t in range(T):
        start = Yi[t]*n_latent
        stop = (Yi[t]+1)*n_latent
        score[start:stop,t] -= 1

    return score
