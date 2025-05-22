

import numpy as np
import torch
from models.SSSD_ECG import SSSD_ECG

import sys
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/models")
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/")

from util_utility.utils import apply_standardizer

from fastai_model import fastai_model
import pickle
import torch.nn.functional as F
import json
import numpy as np
import os
import torch.nn as nn

import ipdb
from utils.util_generation import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams, std_normal
from load_icbeb import load_icbeb_data


def load_pretrained_model():
    modelname = 'fastai_xresnet1d50'
    pretrainedfolder = 'Diffusion_RL/ecg_ptbxl_benchmarking/output/exp0/models/fastai_xresnet1d50/'
    mpath='./for_finetune/' # <=== path where the finetuned model will be stored
    num_classes = 2
    n_classes_pretrained = 71 # <=== because we load the model from exp0, this should be fixed because this depends the experiment
    sampling_frequency = 100
    input_shape = [1000, 12]

    model = fastai_model(
        modelname, 
        num_classes, 
        sampling_frequency, 
        mpath, 
        input_shape=input_shape, 
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained, 
        pretrained=True,
        epochs_finetuning=100,
    )
    return model

def load_density_ratio_model():
    modelname = 'fastai_xresnet1d50'
    pretrainedfolder = 'Diffusion_RL/SSSD-ECG-main/src/sssd/for_finetune/models/fastai_xresnet1d50/'
    mpath='./for_finetune/' # <=== path where the finetuned model will be stored
    num_classes = 2
    n_classes_pretrained = 2 # <=== because we load the model from exp0, this should be fixed because this depends the experiment
    sampling_frequency = 100
    input_shape = [1000, 12]

    model = fastai_model(
        modelname, 
        num_classes, 
        sampling_frequency, 
        mpath, 
        input_shape=input_shape, 
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained, 
        pretrained=True
    )
    return model

def load_diffusion_model():
    model_config = {
        "in_channels": 8,
        "out_channels":8,
        "num_res_layers": 36,
        "res_channels": 256,
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 1000,
        "s4_d_state":64,
        "s4_dropout":0.0,
        "s4_bidirectional":1,
        "s4_layernorm":1,
        "label_embed_dim":128,
        "label_embed_classes":71
    }
    
    net = SSSD_ECG(**model_config).cuda()
    model_path = 'sssd_label_cond_ptbxl_for_icbeb_finetune/ch256_T200_betaT0.02/100000.pkl'
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
    except:
        raise Exception('No valid model found')
    return net


def density_ratio_estimation(model, X, Y):

    X = X.numpy()

    y_pred = model.predict_guidance(X) 
    
    y_pred = torch.tensor(y_pred)
    y_pred = F.softmax(y_pred)
    # ipdb.set_trace()
    density_ratio = y_pred[:,1]/y_pred[:,0]
    return torch.log(torch.tensor(density_ratio)).cuda() #torch.tensor(density_ratio).cuda()  #torch.log(torch.tensor(density_ratio))#this is the  version 

def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GuidanceQt(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [action_dim+32+state_dim, 512, 512, 512, 512, 1]
        self.qt = mlp(dims, activation=SiLU)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=10, stride=2)
        
    def forward(self, action, t, condition=None):

        action = self.conv1(action)
        action = self.conv2(action)

        # Flatten for input to linear layers
        action = action.view(action.size(0), -1)
    
        ''' '''
        #在training guidance network的时候需要加上这个 
        # ipdb.set_trace()
        t = torch.squeeze(t) 

        embed = self.embed(t)
        

        '''
        #for sampling
        
        embed = self.embed(t)
        '''

        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.qt(ats)


class Bandit_Critic_Guide(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        self.qt = GuidanceQt(adim, sdim).to(args.device)
        self.qt_optimizer = torch.optim.Adam(self.qt.parameters(), lr=3e-4)
        self.guidance_net = load_density_ratio_model()# 将density ratio estimator load进来
        
        self.args = args
        self.guidance_scale = 1.0
        self.alpha = 1.0


        with open('Diffusion_RL/SSSD-ECG-main/src/sssd/config/config_SSSD_ECG.json') as f:
            data = f.read()

        config = json.loads(data)

        diffusion_config = config["diffusion_config"]
        diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config) 
        _dh = diffusion_hyperparams
        self.T, self.Alpha_bar = _dh["T"], _dh["Alpha_bar"].cuda()

        self.model = load_diffusion_model()
        X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()
        X_train_ICBEB = X_test_ICBEB
        y_train_ICBEB = y_test_ICBEB
        indices = [i for i, d in enumerate(X_train_ICBEB) if d.shape[0] >= 1000]
        X_train_ICBEB = [d[:1000,:] for d in X_train_ICBEB if d.shape[0] >= 1000]
        y_train_ICBEB = y_train_ICBEB[indices]

        new_labels = np.zeros((y_train_ICBEB.shape[0], 71))

        label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
        # Iterate through each example and update new_labels
        for i in range(y_train_ICBEB.shape[0]):
            for j in range(y_train_ICBEB.shape[1]):
                new_labels[i, label_map[j]] = y_train_ICBEB[i, j]

        self.target_labels = torch.tensor(new_labels).float().cuda()
        self.target_sample = torch.tensor(X_train_ICBEB)
        self.target_sample = torch.permute(self.target_sample, (0,2,1))
        index_8 = torch.tensor([0,2,3,4,5,6,7,11])
        index_4 = torch.tensor([1,8,9,10])
        self.target_sample = torch.index_select(self.target_sample, 1, index_8).float().cuda()

    def forward(self, a, condition=None):
        return self.qt(a, condition)

    def calculate_guidance(self, a, t, condition=None):
        with torch.enable_grad():
            a.requires_grad_(True)
            Q_t = self.qt(a, t, condition)
            # Q_t = torch.log(Q_t) #!!! 这个是un的version
            guidance =  self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
        return guidance.detach()

    def calculated_consistence_regularization(self):
        random_indices = torch.randint(0, self.target_sample.size(0), (8,))
        sample = self.target_sample[random_indices]
        label = self.target_labels[random_indices]

        B, C, L = sample.shape  # B is batchsize, C=1, L is audio length
        diffusion_steps = torch.randint(self.T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
        z = std_normal(sample.shape)
        transformed_X = torch.sqrt(self.Alpha_bar[diffusion_steps]) * sample + torch.sqrt(1-self.Alpha_bar[diffusion_steps]) * z
        score = self.model((transformed_X, label, diffusion_steps.view(B,1),))  

        transformed_X.requires_grad = True
        Q_t = self.qt(transformed_X, diffusion_steps)
        # guidance = self.guidance_scale * Q_t
        guidance =  self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), transformed_X, retain_graph=True)[0]
        # ipdb.set_trace()
        loss = torch.mean(torch.sum((score  +  guidance - z)**2, dim=(1,)))
        return loss

    def update_qt(self, audio, label):

        index_8 = torch.tensor([0,2,3,4,5,6,7,11])
        index_4 = torch.tensor([1,8,9,10]) 
        reward = density_ratio_estimation(self.guidance_net,audio, label)
        audio = torch.index_select(audio, 1, index_8).float().cuda()
        
        
        if self.args.method == "mse":

            
            B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
            # diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
            diffusion_steps = torch.randint(self.T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
            # ！！！ it is very important to restrict the T near 0
            # ipdb.set_trace()
            z = std_normal(audio.shape)
            transformed_X = torch.sqrt(self.Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-self.Alpha_bar[diffusion_steps]) * z
            loss = torch.mean((self.qt(transformed_X, diffusion_steps, None)- reward * self.alpha)**2)
            # loss = torch.mean((self.qt(transformed_X, diffusion_steps, label)- reward * self.alpha)**2)


            loss_consistence_regularization = self.calculated_consistence_regularization()
            # ipdb.set_trace()
            loss += loss_consistence_regularization

        # elif self.args.method == "emse":
        #     pass

        # elif self.args.method == "CEP":
        #     pass

        else:
            raise NotImplementedError

        self.qt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qt_optimizer.step()

        return loss.detach().cpu().numpy()