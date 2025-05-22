# Transfer-Learning-for-Diffusion-Models

This repo contains the pytorch code for experiments in the paper [Transfer Learning for Diffusion Models](https://arxiv.org/abs/2405.16876)

by Yidong Ouyang, Liyan Xie, Hongyuan Zha, Guang Cheng.

We propose a novel approach to transfer knowledge from existing pre-trained models to specific target domains with limited data We prove that the optimal diffusion model for the target domain integrates pre-trained diffusion
models on the source domain with additional guidance from a domain classifier.
We further extend TGDP to a conditional version for modeling the joint distribution
of data and its corresponding labels, together with two additional regularization
terms to enhance the model performance. 

### Usage

#### ECG benchmark

Training domain classifier

```
cd SSSD-ECG-main/src/sssd
python train_DRE.py
```

Training guidance network

```
python train_density_ratio_net.py
```

Sampling

```
python inference_guided_ptbxl.py
```

Utility evaluation

```
cd ecg_ptbxl_benchmarking/code
python reproduce_results.py

```

### References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{
  Ouyang2024trans,
  title={Transfer Learning for Diffusion Models},
  author={Yidong Ouyang, Liyan Xie, Hongyuan Zha, Guang Cheng},
  booktitle={Conference on Neural Information Processing Systems},
  year={2024},
}
```

This implementation is heavily based on 
* [Diffusion Model for ECG](https://github.com/AI4HealthUOL/SSSD-ECG) 
* [ECG benchmark](https://github.com/helme/ecg_ptbxl_benchmarking) 
