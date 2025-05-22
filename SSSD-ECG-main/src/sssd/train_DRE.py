import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/models")
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/")

# from util_utility.utils import apply_standardizer

from density_ratio_guidance import load_pretrained_model


from experiments.scp_experiment import SCP_Experiment
from util_utility import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *

import ipdb

from load_ptbxl import load_ptbxl_data
from load_icbeb import load_icbeb_data

pre_trained = load_pretrained_model()



'''
# standard_scaler = pickle.load(open('Diffusion_RL/ecg_ptbxl_benchmarking/output/exp_ICBEB_from_scatch/data/standard_scaler.pkl', "rb"))
# standard_scaler = pickle.load(open('Diffusion_RL/ecg_ptbxl_benchmarking/output/exp0/data/standard_scaler.pkl', "rb"))

datafolder = 'physionet.org/files/ptb-xl/1.0.3/'
# datafolder_icbeb = 'Diffusion_RL/ecg_ptbxl_benchmarking/data/ICBEB/' #normal ICBEB
datafolder_icbeb = 'Diffusion_RL/ecg_ptbxl_benchmarking/data/ICBEB/'   #vanilla diffusion 
outputfolder = 'Diffusion_RL/ecg_ptbxl_benchmarking/output/exp_ICBEB_reward_guided_by_ptbxl'

models = [
    conf_fastai_xresnet1d50,
    # conf_fastai_resnet1d_wang,
    # conf_fastai_lstm,
    # conf_fastai_lstm_bidir,
    # conf_fastai_fcn_wang,
    # conf_fastai_inception1d,
    # conf_wavelet_standard_nn,
    ]

##########################################
# STANDARD SCP EXPERIMENTS ON PTBXL
##########################################

experiments = [
    ('exp0', 'all'),
    # ('exp1', 'diagnostic'),
    # ('exp1.1', 'subdiagnostic'),
    # ('exp1.1.1', 'superdiagnostic'),
    # ('exp2', 'form'),
    # ('exp3', 'rhythm')
    ]


for name, task in experiments:
    e = SCP_Experiment(name, task, datafolder, outputfolder, models)
    X_train_PTBXL, X_val_PTBXL, X_test_PTBXL, y_train_PTBXL, y_val_PTBXL, y_test_PTBXL = e.get_ptbxl_unnormalized()

e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models) #_vanilla_diffusion
X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB= e.get_ICBEB_unnormalized()
'''

'''

# original ptbxl data
X_train_PTBXL = np.load('physionet.org/files/Real/train_ptbxl_1000.npy')
y_train_PTBXL = np.load('physionet.org/files/Real/1000_train_labels.npy')   

# original ICBEB data
X_train_ICBEB = np.load('Diffusion_RL/ecg_ptbxl_benchmarking/output/x_train.npy',allow_pickle=True)
y_train_ICBEB = np.load('Diffusion_RL/ecg_ptbxl_benchmarking/output/exp_ICBEB_finetune_discriminator/data/y_train.npy',allow_pickle=True)    
'''

X_train_PTBXL, X_val_PTBXL, X_test_PTBXL, y_train_PTBXL, y_val_PTBXL, y_test_PTBXL = load_ptbxl_data()
X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()

####### only use 690 samples for training DRE ------------------------
X_train_ICBEB = X_test_ICBEB
y_train_ICBEB = y_test_ICBEB



# ipdb.set_trace()

label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
# first filter out the corresponding labeled data
indices = (y_train_PTBXL[:, label_map] == 1).any(axis=1)

# Filter data and labels based on indices
X_train_PTBXL = X_train_PTBXL[indices]
y_train_PTBXL = y_train_PTBXL[indices]


# Standardize the data of icbeb
indices = [i for i, d in enumerate(X_train_ICBEB) if d.shape[0] >= 1000]
X_train_ICBEB = [d[:1000,:] for d in X_train_ICBEB if d.shape[0] >= 1000]
X_train_ICBEB = np.array(X_train_ICBEB)
y_train_ICBEB = y_train_ICBEB[indices]


# Standardize label of icbeb 
new_labels = np.zeros((y_train_ICBEB.shape[0], 71))

# Iterate through each example and update new_labels
for i in range(y_train_ICBEB.shape[0]):
    for j in range(y_train_ICBEB.shape[1]):
        new_labels[i, label_map[j]] = y_train_ICBEB[i, j]

y_train_ICBEB = new_labels

X_train_PTBXL = X_train_PTBXL[:690]
y_train_PTBXL = y_train_PTBXL[:690]

# to estimate the joint distribution of q(x,y)/p(x,y), we need to cancat x and y

'''
# embedding = nn.Embedding(71, 12)
# embedding_matrix = embedding.weight.data
# torch.save(embedding_matrix, 'embedding_matrix.pth')
loaded_embedding_matrix = torch.load('embedding_matrix.pth').numpy()
# embedding.weight = nn.Parameter(loaded_embedding_matrix)

y_train_PTBXL = np.matmul(y_train_PTBXL,loaded_embedding_matrix)
y_train_ICBEB = np.matmul(y_train_ICBEB,loaded_embedding_matrix)

y_train_PTBXL = np.expand_dims(y_train_PTBXL, axis=1)
y_train_ICBEB = np.expand_dims(y_train_ICBEB, axis=1)

X_train_PTBXL = np.concatenate((X_train_PTBXL, y_train_PTBXL), axis=1)
X_train_PTBXL = np.concatenate((X_train_ICBEB, y_train_ICBEB), axis=1)
'''

def cancat_data(X_train, syn_data):
    concatenated_data = []
    for element in X_train:
        concatenated_data.append(element)

    for element in syn_data:
        concatenated_data.append(element)
    return np.array(concatenated_data)

def perturb(X_train, y_train):
    # ipdb.set_trace()
    one_hot_label = np.zeros((y_train.size, 2))
    one_hot_label[np.arange(y_train.size), y_train.astype(int)] = 1
    idx = np.random.permutation(len(y_train))
    X_train,one_hot_label = X_train[idx], one_hot_label[idx]
    return X_train,one_hot_label

X_train = cancat_data(X_train_PTBXL,X_train_ICBEB)
y_train = np.concatenate((np.zeros(X_train_PTBXL.shape[0]),np.ones(X_train_ICBEB.shape[0])),0)

# X_val = cancat_data(X_val_PTBXL,X_val_ICBEB)
# y_val = np.concatenate((np.zeros(X_val_PTBXL.shape[0]),np.ones(X_val_ICBEB.shape[0])),0)

# X_test = cancat_data(X_test_PTBXL,X_test_ICBEB) 
# y_test = np.concatenate((np.zeros(X_test_PTBXL.shape[0]),np.ones(X_test_ICBEB.shape[0])),0)

X_train, y_train = perturb(X_train, y_train)
# X_val, y_val = perturb(X_val, y_val)
# X_test, y_test = perturb(X_test, y_test)



# pre_trained.fit(X_train, y_train, X_val, y_val)
pre_trained.fit(X_train, y_train, X_train, y_train)

# y_val_pred = pre_trained.predict(X_val)
# print(utils.evaluate_experiment(y_val, y_val_pred))

y_train_pred = pre_trained.predict(X_train)
print(utils.evaluate_experiment(y_train, y_train_pred))


