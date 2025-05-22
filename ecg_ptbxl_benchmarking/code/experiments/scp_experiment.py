from util_utility import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat

import sys
sys.path.append("Diffusion_RL/SSSD-ECG-main/src/sssd/")
from assign_pesudo_label_for_syn_data import infer_label

import ipdb

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat'):
        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.outputfolder = outputfolder
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

        # create folder structure if needed
        if not os.path.exists(self.outputfolder+self.experiment_name):
            os.makedirs(self.outputfolder+self.experiment_name)
            if not os.path.exists(self.outputfolder+self.experiment_name+'/results/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/results/')
            if not os.path.exists(outputfolder+self.experiment_name+'/models/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/models/')
            if not os.path.exists(outputfolder+self.experiment_name+'/data/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/data/')

    def get_ptbxl(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_ICBEB(self):
        # Load ICBEB data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # ipdb.set_trace()
        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        # ipdb.set_trace()

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def get_ptbxl_unnormalized(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        # self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_ICBEB_unnormalized(self):
        # Load ICBEB data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # ipdb.set_trace()
        # Preprocess signal data
        # self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        # ipdb.set_trace()

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def load_data_for_embedding(self, data_path):

        self.data = np.load(data_path+'/all_generated_data.npy')
        self.labels = np.load(data_path+'/all_generated_label.npy')

        # ipdb.set_trace()
        self.data = self.data.reshape(17200,12,1000)
        # self.data = self.data.reshape(16500,12,1000)
        # self.data = self.data.reshape(17152,12,1000)
        self.data = np.transpose(self.data, (0, 2, 1))

        # self.input_shape = self.data[0].shape
        # # 10th fold for testing (9th for now)
        # self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        # self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # # 9th fold for validation (8th for now)
        # self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        # self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # # rest for training
        # self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        # self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]


        # Preprocess signal data
        # self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.data, self.data[-3:-1], self.data[-3:-1], self.outputfolder+self.experiment_name+'/data/')
        # ipdb.set_trace()
        return self.X_train, self.X_val, self.X_test

    def prepare(self):
        
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        aaa = self.X_train
        self.X_train = self.X_test
        self.X_test = aaa


        bbb = self.y_train
        self.y_train = self.y_test
        self.y_test = bbb

        # selected_indices = np.random.choice(len(self.X_test), size=690, replace=False)
        # ipdb.set_trace()
        # self.X_test = self.X_test[selected_indices]
        # self.y_test = self.y_test[selected_indices]

        # ipdb.set_trace()
        '''
        # ADDed!! This is for using synthetic data for PTB-XL
        synth_path = 'physionet.org/files/Dataset'

        self.y_train = np.load(os.path.join(synth_path, 'labels', 'ptbxl_train_labels.npy'))
        self.y_val = np.load(os.path.join(synth_path, 'labels', 'ptbxl_validation_labels.npy'))
        self.y_test = np.load(os.path.join(synth_path, 'labels', 'ptbxl_test_labels.npy'))
        self.X_train = np.load(os.path.join(synth_path, 'data', 'ptbxl_train_data.npy'))
        self.X_val = np.load(os.path.join(synth_path, 'data', 'ptbxl_validation_data.npy'))
        self.X_test = np.load(os.path.join(synth_path, 'data', 'ptbxl_test_data.npy'))

        self.X_train = np.transpose(self.X_train, (0, 2, 1))
        self.X_val = np.transpose(self.X_val, (0, 2, 1))
        self.X_test = np.transpose(self.X_test, (0, 2, 1))

        self.input_shape = self.X_train[0].shape
        '''

        # ipdb.set_trace()
        # for vanilla diffusion model 
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02_True/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02_True/all_generated_label.npy')

        # for reward guided diffusion model 
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_label.npy')
        
        # for direct transfer diffusion model from ptbxl
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb/ch256_T200_betaT0.02/all_generated_label.npy')

        # for density ratio guided diffusion model from ptbxl
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided/ch256_T200_betaT0.02/all_generated_label.npy')

        # for density ratio guided diffusion model from ptbxl (improved )
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided_/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided_/ch256_T200_betaT0.02/all_generated_label.npy')
          
        # for vanilla diffusion model 690 samples
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_label.npy')

        # for direct transfer diffusion model from ptbxl 690 samples
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ptbxl_for_icbeb_finetune/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ptbxl_for_icbeb_finetune/ch256_T200_betaT0.02/all_generated_label.npy')
        
        # for density ratio guided diffusion model from ptbxl (improved ) 690sample
        syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ptbxl_for_icbeb_guided_/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514åå/sssd_label_cond_ptbxl_for_icbeb_guided_/ch256_T200_betaT0.02/all_generated_label.npy')
                  
        # syn_data = syn_data.reshape(16500,12,1000)
        syn_data = syn_data.reshape(17152,12,1000)
        # syn_data = syn_data.reshape(17200,12,1000)
        syn_data = np.transpose(syn_data, (0, 2, 1))
        syn_label = infer_label(syn_data)

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')

        

        
        
        '''
        syn_label = syn_label.reshape(17200,9)
        '''
        '''
        # syn_label = syn_label.reshape(17200,71)
        # syn_label = syn_label.reshape(16500,71)
        syn_label = syn_label.reshape(17152,71)

        label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]

        # output_tensor = np.zeros((17200, 9))
        # output_tensor = np.zeros((16500, 9))
        output_tensor = np.zeros((17152, 9))

        # Iterate over each batch in the input tensor
        for batch_idx in range(syn_label.shape[0]):
            # Iterate over each element in the one-hot encoded label (71 elements)
            for label_idx in label_map:
                # Find the corresponding index in the new label mapping
                new_label_idx = label_map.index(label_idx)
                # Update the corresponding entry in the output tensor
                output_tensor[batch_idx, new_label_idx] += syn_label[batch_idx, label_idx]

        syn_label = output_tensor
        '''

        concatenated_data = []

        for element in self.X_train:
            concatenated_data.append(element)

        for element in syn_data:
            concatenated_data.append(element)
        
        self.X_train = np.array(concatenated_data)
        self.y_train = np.append(self.y_train, syn_label,axis=0)

        # ipdb.set_trace()
        idx = np.random.permutation(len(self.y_train))
        self.X_train,self.y_train = self.X_train[idx], self.y_train[idx]

        # ipdb.set_trace()


        #only use 690sample for trianing 
        # aaa = self.X_test 
        # self.X_test = self.X_train
        # self.X_train = aaa

        # aaa = self.y_test
        # self.y_test = self.y_train
        # self.y_train = aaa


        #self.X_train.dump('Diffusion_RL/ecg_ptbxl_benchmarking/output/x_train.npy')
        #self.X_val.dump('Diffusion_RL/ecg_ptbxl_benchmarking/output/X_val.npy')
        #self.X_test.dump('Diffusion_RL/ecg_ptbxl_benchmarking/output/X_test.npy')


        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test.npy')

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        mean_y = np.mean(self.y_train, axis=0)
        np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred.npy')
        np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred.npy')
        np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred.npy')

    def perform(self):
        for model_description in self.models:
            modelname = model_description['modelname']
            modeltype = model_description['modeltype']
            modelparams = model_description['parameters']

            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            # create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')

            n_classes = self.Y.shape[1]
            # load respective model
            if modeltype == 'WAVELET':
                from models.wavelet import WaveletModel
                model = WaveletModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from models.fastai_model import fastai_model
                model = fastai_model(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "YOUR_MODEL_TYPE":
                # YOUR MODEL GOES HERE!
                from models.your_model import YourModel
                model = YourModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            else:
                assert(True)
                break

            # fit model
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            # predict and dump
            model.predict(self.X_train).dump(mpath+'y_train_pred.npy')
            model.predict(self.X_val).dump(mpath+'y_val_pred.npy')
            model.predict(self.X_test).dump(mpath+'y_test_pred.npy')

        modelname = 'ensemble'
        # create ensemble predictions via simple mean across model predictions (except naive predictions)
        ensemblepath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(ensemblepath):
            os.makedirs(ensemblepath)
        if not os.path.exists(ensemblepath+'results/'):
            os.makedirs(ensemblepath+'results/')
        # load all predictions
        ensemble_train, ensemble_val, ensemble_test = [],[],[]
        for model_description in os.listdir(self.outputfolder+self.experiment_name+'/models/'):
            if not model_description in ['ensemble', 'naive']:
                mpath = self.outputfolder+self.experiment_name+'/models/'+model_description+'/'
                ensemble_train.append(np.load(mpath+'y_train_pred.npy', allow_pickle=True))
                ensemble_val.append(np.load(mpath+'y_val_pred.npy', allow_pickle=True))
                ensemble_test.append(np.load(mpath+'y_test_pred.npy', allow_pickle=True))
        # dump mean predictions
        np.array(ensemble_train).mean(axis=0).dump(ensemblepath + 'y_train_pred.npy')
        np.array(ensemble_test).mean(axis=0).dump(ensemblepath + 'y_test_pred.npy')
        np.array(ensemble_val).mean(axis=0).dump(ensemblepath + 'y_val_pred.npy')

    # def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True):
    def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=True, dumped_bootstraps=False):
        # get labels
        y_train = np.load(self.outputfolder+self.experiment_name+'/data/y_train.npy', allow_pickle=True)
        #y_val = np.load(self.outputfolder+self.experiment_name+'/data/y_val.npy', allow_pickle=True)
        y_test = np.load(self.outputfolder+self.experiment_name+'/data/y_test.npy', allow_pickle=True)

        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                #train_samples = np.array(utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples))
                test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstraping_samples))
                #val_samples = np.array(utils.get_appropriate_bootstrap_samples(y_val, n_bootstraping_samples))
            else:
                test_samples = np.load(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
        else:
            #train_samples = np.array([range(len(y_train))])
            test_samples = np.array([range(len(y_test))])
            #val_samples = np.array([range(len(y_val))])

        # store samples for future evaluations
        #train_samples.dump(self.outputfolder+self.experiment_name+'/train_bootstrap_ids.npy')
        test_samples.dump(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy')
        #val_samples.dump(self.outputfolder+self.experiment_name+'/val_bootstrap_ids.npy')

        # iterate over all models fitted so far
        for m in sorted(os.listdir(self.outputfolder+self.experiment_name+'/models')):
            print(m)
            mpath = self.outputfolder+self.experiment_name+'/models/'+m+'/'
            rpath = self.outputfolder+self.experiment_name+'/models/'+m+'/results/'

            # load predictions
            y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            #y_val_pred = np.load(mpath+'y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

            if self.experiment_name == 'exp_ICBEB':
                # compute classwise thresholds such that recall-focused Gbeta is optimized
                thresholds = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_train, y_train_pred)
            else:
                thresholds = None

            pool = multiprocessing.Pool(n_jobs)

            # tr_df = pd.concat(pool.starmap(utils.generate_results, zip(train_samples, repeat(y_train), repeat(y_train_pred), repeat(thresholds))))
            # tr_df_point = utils.generate_results(range(len(y_train)), y_train, y_train_pred, thresholds)
            # tr_df_result = pd.DataFrame(
            #     np.array([
            #         tr_df_point.mean().values, 
            #         tr_df.mean().values,
            #         tr_df.quantile(0.05).values,
            #         tr_df.quantile(0.95).values]), 
            #     columns=tr_df.columns,
            #     index=['point', 'mean', 'lower', 'upper'])

            te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))
            te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred, thresholds)
            te_df_result = pd.DataFrame(
                np.array([
                    te_df_point.mean().values, 
                    te_df.mean().values,
                    te_df.quantile(0.05).values,
                    te_df.quantile(0.95).values]), 
                columns=te_df.columns, 
                index=['point', 'mean', 'lower', 'upper'])

            # val_df = pd.concat(pool.starmap(utils.generate_results, zip(val_samples, repeat(y_val), repeat(y_val_pred), repeat(thresholds))))
            # val_df_point = utils.generate_results(range(len(y_val)), y_val, y_val_pred, thresholds)
            # val_df_result = pd.DataFrame(
            #     np.array([
            #         val_df_point.mean().values, 
            #         val_df.mean().values,
            #         val_df.quantile(0.05).values,
            #         val_df.quantile(0.95).values]), 
            #     columns=val_df.columns, 
            #     index=['point', 'mean', 'lower', 'upper'])

            pool.close()

            # dump results
            #tr_df_result.to_csv(rpath+'tr_results.csv')
            #val_df_result.to_csv(rpath+'val_results.csv')
            te_df_result.to_csv(rpath+'te_results.csv')
