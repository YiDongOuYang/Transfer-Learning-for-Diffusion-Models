from experiments.scp_experiment import SCP_Experiment
from util_utility import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *

def load_icbeb_data():


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

    e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models) #_vanilla_diffusion
    X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB= e.get_ICBEB()


    return X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB