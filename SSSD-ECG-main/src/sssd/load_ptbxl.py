from experiments.scp_experiment import SCP_Experiment
from util_utility import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *

def load_ptbxl_data():


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
        X_train_PTBXL, X_val_PTBXL, X_test_PTBXL, y_train_PTBXL, y_val_PTBXL, y_test_PTBXL = e.get_ptbxl()

    return X_train_PTBXL, X_val_PTBXL, X_test_PTBXL, y_train_PTBXL, y_val_PTBXL, y_test_PTBXL