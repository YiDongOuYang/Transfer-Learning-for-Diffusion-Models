import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from utils.util_generation import find_max_epoch, print_size, training_loss_label, calc_diffusion_hyperparams
from models.SSSD_ECG import SSSD_ECG


import sys
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/models")
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/")
from load_icbeb import load_icbeb_data

import ipdb


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
         batch_size):
  
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
            
    # predefine model
    net = SSSD_ECG(**model_config).cuda()

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')
        
        
    # load synthetic data
    # data_ptbxl = np.load('physionet.org/files/Dataset/data/ptbxl_train_data.npy')
    # labels_ptbxl = np.load('physionet.org/files/Dataset/labels/ptbxl_train_labels.npy')    

    # original ptbxl data
    # data_ptbxl = np.load('physionet.org/files/Real/train_ptbxl_1000.npy')
    # labels_ptbxl = np.load('physionet.org/files/Real/1000_train_labels.npy')   

    # ipdb.set_trace()



    ''' 
    # original ICBEB data
    data_ptbxl = np.load('Diffusion_RL/ecg_ptbxl_benchmarking/output/x_train.npy',allow_pickle=True)
    labels_ptbxl = np.load('Diffusion_RL/ecg_ptbxl_benchmarking/output/exp_ICBEB_finetune/data/y_train.npy',allow_pickle=True)    
    '''
    # Since I am going to load 690 the training samples, therefore I use this (normalized data)
    X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()
    data_ptbxl = X_test_ICBEB
    labels_ptbxl = y_test_ICBEB



    label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
    # Standardize label of icbeb 
    new_labels = np.zeros((labels_ptbxl.shape[0], 71))

    # Iterate through each example and update new_labels
    for i in range(labels_ptbxl.shape[0]):
        for j in range(labels_ptbxl.shape[1]):
            new_labels[i, label_map[j]] = labels_ptbxl[i, j]

    labels_ptbxl = new_labels  

    ckpt_iter = 0


    data_ptbxl_all = []
    for i in data_ptbxl:
        if i.shape[0]>=1000:
            data_ptbxl_all.append(i[:1000])
        else:
            exit(1)
    data_ptbxl_all = np.array(data_ptbxl_all)
    data_ptbxl = np.transpose(data_ptbxl_all,(0,2,1))
    
    
    train_data = []
    for i in range(len(data_ptbxl)):
        train_data.append([data_ptbxl[i], labels_ptbxl[i]])
        
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=8, drop_last=True)
       
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    
    
    # training
    n_iter = ckpt_iter + 1
    
    while n_iter < n_iters + 1:
        for audio, label in trainloader:
            
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
           
            
            # back-propagation
            optimizer.zero_grad()
            X = audio, label
            
            loss = training_loss_label(net, nn.MSELoss(), X, diffusion_hyperparams)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG.json',
    #                     help='JSON file for configuration')
    parser.add_argument('-c', '--config', type=str, default='Diffusion_RL/SSSD-ECG-main/src/sssd_0514/config/config_SSSD_ECG.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)
    
    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']


    train(**train_config)

