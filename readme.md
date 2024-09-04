# Description
This repository consists of all the necessary code to reproduce the experimental results presented in the paper entitled with *Learning nonvacuous generalization bounds from optimization*.

## Requirements
Ubuntu 20.04、GeForce RTX 3090、CUDA 11.4、Pytorch 1.12、Python 3.8

Important packages include: timm、wandb、colorama、pytorchcv、nolds、cyminiball、prefetch_generator

For more details, please refer to requirements.txt.

## Usage
The train.py defines the full pipline from training a neural network to estimating the generalization bound $\varrho_{\mathrm{bound}}$, you may execute
`
sh run.sh
`
for a simple test on CIFAR-10/100 that may take a few hours and 
`sh run_imagenet.sh
`
for a large-scale test on ImageNet-1K that may take a few days.

## Other details
* hurst_exponent.ipynb elaborates on how to estimate the hurst exponents for different coordinates of a pretrained neural network

* num_coordinate.ipynb investigates the influence of the number of used coordinates on estimating the Hausdorff dimension according to Equation (3)

* ph_dim.ipynb and blumenthal_getoor.ipynb estimate the Hausdorff dimension using methods from Simsekli et al., 2020 and Dupuis et al., 2023

* supp folder provides the complete code to probe the effects of different sources of stochasticity, such as random initlization points, random training subsets created from CIFAR-10, and random order of mini-batches