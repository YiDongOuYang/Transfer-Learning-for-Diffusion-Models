from SNIP_for_reward import load_pretrained_model
import pickle
import torch
import numpy as np
import os


import sys
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/models")
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/")

from util_utility import utils
from util_utility.utils import apply_standardizer

import ipdb




def infer_label(syn_x):
    pretrained_model = load_pretrained_model()
    # standard_scaler = pickle.load(open('Diffusion_RL/ecg_ptbxl_benchmarking/output/exp_ICBEB_from_scatch/data/standard_scaler.pkl', "rb"))
    # X = apply_standardizer(syn_x, standard_scaler) 
    with torch.no_grad():
        y_val_pred = pretrained_model.predict(syn_x)
    max_indices = np.argmax(y_val_pred, axis=1)

    # Create a one-hot embedding tensor
    one_hot = np.zeros_like(y_val_pred)
    one_hot[np.arange(len(max_indices)), max_indices] = 1
    return one_hot
    