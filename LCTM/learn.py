import numpy as np
import scipy.io as sc
from functools import reduce
from copy import deepcopy
from LCTM import ssvm
from LCTM import utils

def pretrain_weights(model, X, Y):
    # Take mean of all potentials
    n_samples = len(X)

    # Compute potential costs for each (correct) labeling
    costs = [ssvm.compute_costs(model, X[i], Y[i]) for i in range(n_samples)]
    costs = reduce(lambda x,y: x+y, costs)
    for key in costs:
        norms = np.linalg.norm(costs[key])
        # norms[norms==0] = 1
        costs[key] /= norms

    model.ws = costs

def subgradient_descent(model, X, Y, n_iter=100, C=1., pretrain=True, verbose=True, 
                        gradient_method="adagrad", learning_rate=0.1, decay_rate=.99, 
                        batch_size=5):
                        

    hamming_dist = {}
    objective = {}
    
    if model.debug:
        np.random.seed(1234)

    n_samples = len(X)

    # Check that Xi is of size FxT
    if X[0].shape[0] > X[0].shape[0]:
        X = [x.T for x in X]
    
    # if weights haven't been set yet then initialize
    if model.n_classes is None:
        model.n_features = X[0].shape[0]
        model.n_classes = np.max(list(map(np.max, Y)))+1
        model.max_segs = utils.max_seg_count(Y)
        model.ws.init_weights(model)

        if pretrain:
            if model.is_latent:
                Z = [utils.partition_latent_labels(Y[i], model.n_latent) for i in range(n_samples)]
                pretrain_weights(model, X, Z)
            else:
                pretrain_weights(model, X, Y)

    
    costs_truth = [ssvm.compute_costs(model, X[i], Y[i]) for i in range(n_samples)]
    # print("Unaries costs", [c['unary'].sum() for c in costs_truth])
    cache = deepcopy(costs_truth[0]) * 0.

    ########## objective before training
    #w_p_init = sc.loadmat('/cis/home/divya/LCTM/w_p_init.mat')
    #init_w = w_p_init['w']
    #init_pw = w_p_init['pw']
    sample_set = list(range(len(X)))
    #print('before iter', model.ws['unary'])
    #print('before iter', model.ws['pw'])
    #print('unary == u', model.ws['unary'] == init_w)  
    #print('pairwise == pw', model.ws['pw'] == init_pw)  
    objective_hamming = np.mean([model.objective(model.predict(X[i])[0], Y[i]) for i in sample_set])
    objective_mm = ssvm.objective_mm(model, X=X, Y=Y)
    #print("Iter {}, obj={}".format(0, objective_new))
    #print("Objective: ", objective_mm)
    hamming_dist[0] = objective_hamming
    objective[0] = objective_mm



    for t in range(n_iter):
        batch_samples = np.random.randint(0, n_samples, batch_size)

        # Compute gradient
        w_diff = [ssvm.compute_ssvm_gradient(model, X[j], Y[j], costs_truth[j], C) for j in batch_samples]
        w_diff = reduce(lambda x,y: x+y, w_diff)
        w_diff /= batch_size

        # ===Weight Update===
        # Vanilla SGD
        if gradient_method == "sgd":
            eta = learning_rate * (1 - t/n_iter)
            w_diff = w_diff*eta
        # Adagrad
        elif gradient_method == "adagrad":
            cache += w_diff*w_diff
            w_diff = w_diff / (cache + 1e-8).sqrt() * learning_rate
        # RMSProp
        elif gradient_method == "rmsprop":
            # cache = decay_rate*cache + (1-decay_rate)*w_diff.^2
            if t == 0:
                cache += w_diff*w_diff
            else:
                cache *= decay_rate
                cache += w_diff*w_diff*(1-decay_rate)
            w_diff = w_diff / sqrt(cache + 1e-8) * learning_rate
        
        model.ws -= w_diff


        # Print and compute objective
        if verbose and ((t+1)%50==0) :
            objective_hamming = np.mean([model.objective(model.predict(X[i])[0], Y[i]) for i in sample_set])
            objective_mm = ssvm.objective_mm(model, X=X, Y=Y)
            hamming_dist[t+1] = objective_hamming
            objective[t+1] = objective_mm
            print("Iter {}, obj_h={}".format(t+1, objective_hamming))
            print("Objective: ", objective_mm)
            model.logger.objectives[t+1] = objective_hamming
   
    if n_iter not in objective.keys():
            objective_hamming = np.mean([model.objective(model.predict(X[i])[0], Y[i]) for i in sample_set])
            objective_mm = ssvm.objective_mm(model, X=X, Y=Y)
            hamming_dist[t+1] = objective_hamming
            objective[t+1] = objective_mm
        
    #sc.savemat('w_p_FROMSGD.mat', dict(unary=model.ws['unary'], pairwise=model.ws['pw']))
    #np.savetxt('obj_collect', obj_collect)
    return hamming_dist, objective
