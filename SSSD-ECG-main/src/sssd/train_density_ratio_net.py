
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from density_ratio_guidance import Bandit_Critic_Guide

import json
import numpy as np
import os
import ipdb

import sys
sys.path.append('Diffusion_RL/SSSD-ECG-main/src/sssd/utils')
from util_generation import bandit_get_args, calc_diffusion_hyperparams, std_normal
from load_ptbxl import load_ptbxl_data
from load_icbeb import load_icbeb_data

import math
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import ipdb

args = bandit_get_args()


# guidance_net = Bandit_MlpScoreNet(1000, 1000, args)
# guidance_net = Bandit_ScoreBase(1000, 1000, args)
guidance_net = Bandit_Critic_Guide(3904, 0, args)

X_train_PTBXL, X_val_PTBXL, X_test_PTBXL, y_train_PTBXL, y_val_PTBXL, y_test_PTBXL = load_ptbxl_data()
X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()
# ipdb.set_trace()

X_train_ICBEB = X_test_ICBEB
y_train_ICBEB = y_test_ICBEB

label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
# first filter out the corresponding labeled data
indices = (y_train_PTBXL[:, label_map] == 1).any(axis=1)

# Filter data and labels based on indices
X_train_PTBXL = X_train_PTBXL[indices]
y_train_PTBXL = y_train_PTBXL[indices]

# Standardize the data of icbeb
indices = [i for i, d in enumerate(X_train_ICBEB) if d.shape[0] >= 1000]
X_train_ICBEB = [d[:1000,:] for d in X_train_ICBEB if d.shape[0] >= 1000]
y_train_ICBEB = y_train_ICBEB[indices]


# Standardize label of icbeb 
new_labels = np.zeros((y_train_ICBEB.shape[0], 71))

# Iterate through each example and update new_labels
for i in range(y_train_ICBEB.shape[0]):
    for j in range(y_train_ICBEB.shape[1]):
        new_labels[i, label_map[j]] = y_train_ICBEB[i, j]

y_train_ICBEB = new_labels
data_ptbxl = np.transpose(X_train_PTBXL,(0,2,1))


''' class balance processing'''
train_data = []
for i in range(len(data_ptbxl)):
    train_data.append([data_ptbxl[i], y_train_PTBXL[i]])

for i in range(len(X_train_ICBEB)):
    train_data.append([np.transpose(X_train_ICBEB[i], (1,0)), y_train_ICBEB[i]])

trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=128, drop_last=True)


'''
# Class Balance Batch Sampler
class CustomBatchSampler(Sampler):
    def __init__(self, dataset_a, dataset_b, batch_size):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.batch_size = batch_size
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        self.max_len = max(self.len_a, self.len_b)

    def __iter__(self):
        count = 0
        while count < self.max_len:
            # Get indices for dataset A
            idx_a = [(i % self.len_a) for i in range(count, count + self.batch_size // 2)]
            # Get indices for dataset B
            idx_b = [(i % self.len_b) for i in range(count, count + self.batch_size // 2)]
            yield idx_a + idx_b
            count += self.batch_size // 2

    def __len__(self):
        return math.ceil(self.max_len / (self.batch_size // 2)) * 2


custom_batch_sampler = CustomBatchSampler(dataset_a, dataset_b, batch_size)

# Creating DataLoader with the custom batch sampler
dataloader = DataLoader([dataset_a, dataset_b], batch_sampler=custom_batch_sampler)

# Iterate over the DataLoader
for batch_indices in dataloader:
    print(batch_indices)
'''

    
# index_8 = torch.tensor([0,2,3,4,5,6,7,11])
# index_4 = torch.tensor([1,8,9,10]) 
    
# training
n_iter = 1

while n_iter < 1000 + 1:
    for audio, label in trainloader:
        # we left the selection for diffusion model, we need to use 12 leads for classifier
        # audio = torch.index_select(audio, 1, index_8).float().cuda()
        label = label.float().cuda()
        
        loss = guidance_net.update_qt(audio, label)
        
        
        if n_iter % 100 == 0:
            print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

        # save checkpoint
        if n_iter > 0 and n_iter % 1000 == 0:
            checkpoint_name = 'guidance_{}.pkl'.format(n_iter)
            torch.save({'model_state_dict': guidance_net.qt.state_dict(),
                        'optimizer_state_dict': guidance_net.qt_optimizer.state_dict()},
                        os.path.join("sssd_label_cond_ptbxl_for_icbeb_guided_", checkpoint_name))
                        # os.path.join("sssd_label_cond_ptbxl_for_icbeb_guided", checkpoint_name))
            print('model at iteration %s is saved' % n_iter)

        n_iter += 1

