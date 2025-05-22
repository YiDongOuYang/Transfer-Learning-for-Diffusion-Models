import os
import argparse
import json
import numpy as np
import torch

from utils.util_generation import find_max_epoch, print_size, sampling_label_guided, calc_diffusion_hyperparams, bandit_get_args
from models.SSSD_ECG import SSSD_ECG

from density_ratio_guidance import Bandit_Critic_Guide
import ipdb


def generate_four_leads(tensor):
    leadI = tensor[:,0,:].unsqueeze(1)
    leadschest = tensor[:,1:7,:]
    leadavf = tensor[:,7,:].unsqueeze(1)

    leadII = (0.5*leadI) + leadavf

    leadIII = -(0.5*leadI) + leadavf
    leadavr = -(0.75*leadI) -(0.5*leadavf)
    leadavl = (0.75*leadI) - (0.5*leadavf)

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)

    return leads12


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter):
    
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
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

    
    args = bandit_get_args()
    guidance_net = Bandit_Critic_Guide(3904, 0, args)

    guidance_net.qt.load_state_dict(torch.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ptbxl_for_icbeb_guided_/guidance_1000.pkl')['model_state_dict'])


    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

    # label for ptbxl
    # labels = np.load('physionet.org/files/Real/1000_train_labels.npy') #17441 samples to generate; batch size=
    
    # label for ICBEB
    # labels = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/syn_all_labels.npy') #17441 samples to generate; batch size=
    
    # label ptbxl_for_icbeb
    labels = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb/ch256_T200_betaT0.02/syn_all_labels.npy') #17441 samples to generate; batch size=

    all_generated_data = []
    all_label = []
    batch_for_sampling = 256
    for i in range(num_samples//batch_for_sampling-1):
        label = labels[i*batch_for_sampling:(i+1)*batch_for_sampling]
        cond = torch.from_numpy(label).cuda().float()

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
        generated_audio = sampling_label_guided(net, guidance_net, (batch_for_sampling,8,1000), 
                            diffusion_hyperparams,
                            cond=cond)

        # ipdb.set_trace()

        generated_audio12 = generate_four_leads(generated_audio)

        end.record()
        torch.cuda.synchronize()
        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(batch_for_sampling,
                                                                            ckpt_iter, 
                                                                            int(start.elapsed_time(end)/1000)))

        all_generated_data.append(generated_audio12.detach().cpu().numpy())
        all_label.append(cond.detach().cpu().numpy())

        if i==0:
            # ipdb.set_trace()
            new_data = os.path.join(ckpt_path, 'all_generated_data_500.npy')
            new_label = os.path.join(ckpt_path, 'all_generated_label_500.npy')

            np.save(new_data, np.array(all_generated_data))
            np.save(new_label, np.array(all_label))
        

    new_data = os.path.join(ckpt_path, 'all_generated_data.npy')
    new_label = os.path.join(ckpt_path, 'all_generated_label.npy')

    np.save(new_data, np.array(all_generated_data))
    np.save(new_label, np.array(all_label))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSD_ECG.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default=100000,
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=17442,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
            #  use_model=train_config["use_model"],
             data_path=trainset_config["data_path"],
            #  masking=train_config["masking"],
            #  missing_k=train_config["missing_k"]
             )

