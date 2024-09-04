import matplotlib.pyplot as plt
import torch
import os
from os.path import join
import numpy as np
import json
import argparse
import glob
import re
from prefetch_generator import BackgroundGenerator

import os
from datetime import datetime
# hurst exponent for some coordinates are not available (i.e. outliers)
import warnings
warnings.filterwarnings("ignore")

from pytorchcv.model_provider import get_model as ptcv_get_model
from timm import create_model

from data import cifar_dataset, imagenet_dataset

from utils import DictToClass
from utils import cycle_loader, get_grads, fractal_dimension
from utils import calc_hurst_exponent, error_func

def main():
    parser = argparse.ArgumentParser(description='Monitor the bound during the course of training')
    parser.add_argument('--data_name', default='imagenet-1k', choices=['cifar10', 'cifar100', 'imagenet-1k'])
    parser.add_argument('--save_dir', default='runs/vit_small_patch32_224_adamw/cosine_lr=3.00e-03_bs=1024_wd=1.00e-01_corr-1.0_-1_cat[]_seed=3', type=str)
    parser.add_argument('--num_components', default=500000, type=int)
    parser.add_argument('--len_of_sequence', default=3000, type=int)

    args = parser.parse_args()

    with open(os.path.join(args.save_dir, 'config.json'), 'r') as f:
        config = f.read()
    config = json.loads(config)
    args = config.update(args.__dict__)
    args = DictToClass(config)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    if args.data_name == 'cifar10' or args.data_name == 'cifar100':
        train_set, _ = cifar_dataset(data_name=args.data_name, root=args.data_dir, label_corruption=args.label_corruption, example_per_class=args.example_per_class, categories=args.categories)
    elif args.data_name == 'imagenet-1k':
        train_set, _ = imagenet_dataset(args)

    train_loader_no_shuffle = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory, drop_last=True)
    train_loader_cycle = cycle_loader(train_loader)

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    if args.data_name == 'cifar10' or args.data_name == 'cifar100':
        net = ptcv_get_model(args.arch, pretrained=False).to(device)
    elif args.data_name == 'imagenet-1k':
        net = create_model(args.arch, pretrained=False).to(device)

    checkpoints = glob.glob(os.path.join(args.save_dir, 'net_*.pt'))
    regex = re.compile(r'\d+')
    checkpoints = sorted(checkpoints, key=lambda s: int(regex.findall(s.split('/')[-1])[0]))

    predictions = []
    dimensions = []
    true_gradient_norms = []
    stochastic_gradient_norms = []
    stochastic_noise_norms = []
    for cnt, checkpoint in enumerate(checkpoints):
        net.load_state_dict(torch.load(checkpoint, map_location=device))
    
        # estimate true gradient
        true_gradient = 0.0
        prediction = []
        net.train()
        for x, y in train_loader_no_shuffle:
            x, y = x.to(device), y.to(device)
            net.zero_grad()
            yhat = net(x)
            loss = loss_func(yhat, y)
            loss.mean().backward()
            curr_gradient = get_grads(net)
            true_gradient = true_gradient + curr_gradient * len(x)
            prediction.append(loss.detach().cpu())
        true_gradient = true_gradient / len(train_loader_no_shuffle.dataset)
        true_gradient_norms.append(true_gradient.norm(p=2).item())
        prediction = torch.hstack(prediction)
        predictions.append(prediction)

        # estimated gradient
        start_t = datetime.now()
        num_components = args.num_components # number of parameters used to estimate Hurst parameter 
        len_of_sequence = args.len_of_sequence # number of mini-batches to generate a stochastic sequence
        tot_param = len(true_gradient)        
        if tot_param < num_components:
            num_components = tot_param
        fixed_dims = torch.randperm(tot_param)[:num_components]

        stochastic_grads = []
        stochastic_grads_norm = []
        stochastic_noise_norm = []

        net.train()
        for j, (x, y) in enumerate(train_loader_cycle):
            if j == len_of_sequence:
                break
            x, y = x.to(device), y.to(device)
            net.zero_grad()
            yhat = net(x)
            loss = loss_func(yhat, y)
            loss.mean().backward()
            curr_gradient = get_grads(net)
            stochastic_noise_norm.append((curr_gradient-true_gradient).norm(p=2).item())
            stochastic_grads_norm.append(curr_gradient.norm(p=2).item())
            stochastic_grads.append(curr_gradient[fixed_dims])
            
        stochastic_grads = torch.vstack(stochastic_grads)
        stochastic_grads = stochastic_grads - true_gradient[fixed_dims]
        stochastic_grads = stochastic_grads.double().cpu().numpy()
        stochastic_grads = stochastic_grads[:, ~np.isnan(stochastic_grads).any(axis=0)] # delete unvalid elements in case of nan
        
        stochastic_gradient_norms.append(np.mean(stochastic_grads_norm))
        stochastic_noise_norms.append(np.mean(stochastic_noise_norm))
        
        # here we exploit a parallel trick to accelerate the process
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(os.cpu_count(), 36)) as pool:
            exponents = [pool.apply_async(calc_hurst_exponent, args=(stochastic_grads[:, j],), error_callback=error_func) for j in range(stochastic_grads.shape[1])]
            exponents = [p.get() for p in exponents]
        
        # remove the outliers
        exponents = np.array(exponents)
        exponents = exponents[exponents >= 0.01]
        exponents = exponents[exponents <= 0.99]
        end_t = datetime.now()
        elapsed_sec = (end_t - start_t).total_seconds()

        dimension = fractal_dimension(exponents)
        dimensions.append(dimension)

        # print("Hurst parameter: {:.4f}, time elapsed: {:.2f} seconds".format(hurst_index, elapsed_sec))
        print(f'{cnt} out of {len(checkpoints)} checkpoints (true_gradient_norm: {true_gradient_norms[-1]:.2f}, stochastic_gradient_norm: {stochastic_gradient_norms[-1]:.2f}, stochastic_noise_norm: {stochastic_noise_norms[-1]:.2f}, dimension: {dimensions[-1]:.2f}, average hurst exponent: {np.median(exponents):.2f}, time elapsed: {elapsed_sec:.2f})')
        
        torch.save([predictions, dimensions, true_gradient_norms, stochastic_gradient_norms, stochastic_noise_norms], os.path.join(args.save_dir, 'evolution_info.pt'))
    
if __name__ == '__main__':
    main()