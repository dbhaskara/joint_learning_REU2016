import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import os
import sys

from LCTM import utils
from LCTM import models
from LCTM import datasets
from LCTM import metrics
from LCTM.utils import imshow_


#DIMENSIONS
dim = 76

#PATH TO SUTURING DATA
suturing = "/cis/home/divya/jigsaw/Suturing/kinematics/AllGestures"
transcriptions = "/cis/home/divya/jigsaw/Suturing/transcriptions"

#LEAVE ONE USER OUT
louo = sys.argv[2]

#LEAVE ONE SUPERTRIAL OUT
loso = '1'

#EXTRACT TRAINING DATA
training_data = [] 
labels_supervised = []

testing_data = []
ground_truth = []

#to calculate norms
all_training_data = np.empty((0, 14), float)

for name in os.listdir(suturing):
    path_to_file = os.path.join(suturing, name);
    infile = open(path_to_file, 'r')
    features = np.array(infile.read().split(), dtype='float');
    infile.close()
    features = np.reshape(features, (-1, dim))
    print(path_to_file, features.shape)
    
    path_to_trans = os.path.join(transcriptions, name)
    try:
        infile = open(path_to_trans, 'r')
    except FileNotFoundError:
        print (path_to_trans, 'not found')
        continue
    segments = infile.readlines()
    start_task = int(segments[0].split()[0]) - 1
    labels = []
    for period in segments:
        start, finish, gest = period.split()
        start = int(start)
        finish = int(finish)
        gest = int(gest[1:])
        labels += [gest for x in range(finish - start + 1)]

    print(np.array(labels).shape)

    end_task = int(segments[-1].split()[1]) 


    features = features[start_task : end_task]
    print(features.shape)

    header = ["x"]*38
    header += ["l_pos_x", "l_pos_y", "l_pos_z"]+["x"]*9+["l_vel_x", "l_vel_y", "l_vel_z"]+["x"]*3+["l_gripper"]
    header += ["r_pos_x", "r_pos_y", "r_pos_z"]+["x"]*9+["r_vel_x", "r_vel_y", "r_vel_z"]+["x"]*3+["r_gripper"]

    use = ["l_pos_x", "l_pos_y", "l_pos_z", "r_pos_x", "r_pos_y", "r_pos_z"]
    use += ["l_vel_x", "l_vel_y", "l_vel_z", "r_vel_x", "r_vel_y", "r_vel_z"]
    use += ["l_gripper", "r_gripper"]
    feature_idxs = [np.nonzero(np.array(header)==u)[0][0] for u in use]

#   use only 14 relevant features
    features = features[:, feature_idxs]
    print(features.shape)


    if name[9] == louo:
        testing_data.append(features.T)
        ground_truth.append(np.array(labels))
    else:
        training_data.append(features.T)
        labels_supervised.append(np.array(labels))
        all_training_data = np.r_[all_training_data, features]
    
print ("Collected", len(training_data), "training data videos")

print('testing size:', len(testing_data))
X_std = np.hstack(testing_data).std(1)[:,None]
training_data = [(x-x.mean(1)[:,None])/X_std for x in training_data]
testing_data = [(x-x.mean(1)[:,None])/X_std for x in testing_data]

print('all training data size', all_training_data.shape)
norms = [np.linalg.norm(all_training_data[:, x]) for x in range(all_training_data.shape[1])]
print(norms)
print(np.amin(norms), np.amax(norms))#print('norm', np.linalg.norm())

model = models.ChainModel(skip=int(sys.argv[1]))
model.fit(training_data, labels_supervised, n_iter=200)

model.inference_type = "framewise"

fla_acc = []
overlap = []
edit = []

for x in range(len(testing_data)):
    a = model.predict(testing_data[x])
    b = ground_truth[x]
    acc = np.sum(a == b) / a.shape[0]
    fla_acc.append(acc)
    overlap.append(metrics.overlap_score(a, b))
    edit.append(metrics.edit_score(a, b))

#    if x == 4:
#        np.savetxt('lctm_I4_gt', b)
#        np.savetxt('lctm_I4_pred', a)
#        print('****** acc', acc*100.0)
#        print('****** over', overlap[-1])
#        print('****** edit', edit[-1])




print ('accuracy', sum(fla_acc) / len (fla_acc) * 100.0)
print ('overlap', sum(overlap) / len (overlap) * 1.0)
print ('edit', sum(edit) / len (edit) * 1.0)

np.savetxt('sccrf_pw', np.exp(model.get_weights('pw')))
