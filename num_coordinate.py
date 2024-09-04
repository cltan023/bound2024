import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import torch
import os
import numpy as np

import json
from utils import DictToClass

def main():
    model_dir = 'runs/wrn28_10_cifar10_sgd/cosine_lr=5.00e-02_bs=128_wd=5.00e-04_corr-1.0_5000_cat[]_seed=1'
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        args = f.read()
    args = json.loads(args)
    args = DictToClass(args)

    # device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')

    from data import cifar_dataset
    from utils import cycle_loader

    train_set, test_set = cifar_dataset(data_name=args.data_name, root=args.data_dir, label_corruption=args.label_corruption, example_per_class=args.example_per_class, categories=args.categories)

    train_loader_no_shuffle = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory, drop_last=True)
    train_loader_cycle = cycle_loader(train_loader)

    from pytorchcv.model_provider import get_model as ptcv_get_model

    net =  ptcv_get_model(args.arch, pretrained=False).to(device)
    net.load_state_dict(torch.load(os.path.join(model_dir, 'state_dict.pt'), map_location=device))

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    from utils import get_grads

    true_gradient = 0.0
    net.train()
    for x, y in train_loader_no_shuffle:
        x, y = x.to(device), y.to(device)
        net.zero_grad()
        yhat = net(x)
        loss = loss_func(yhat, y)
        loss.mean().backward()
        curr_gradient = get_grads(net)
        true_gradient = true_gradient + curr_gradient * len(x)
    true_gradient = true_gradient / len(train_loader_no_shuffle.dataset)

    for num_components in [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]:
        for seed in [1, 2, 3]:
            # num_components = 300000 # number of coordinates used to estimate Hurst parameter 
            len_of_sequence = 3000 # number of mini-batches to generate a stochastic sequence
            tot_param = len(true_gradient)        
            if tot_param < num_components:
                num_components = tot_param
            fixed_dims = torch.randperm(tot_param)[:num_components]

            stochastic_grads = []

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
                stochastic_grads.append(curr_gradient[fixed_dims])
            stochastic_grads = torch.vstack(stochastic_grads)

            stochastic_grads = stochastic_grads - true_gradient[fixed_dims]

            stochastic_grads = stochastic_grads.double().cpu().numpy()
            stochastic_grads = stochastic_grads[:, ~np.isnan(stochastic_grads).any(axis=0)] # delete unvalid elements in case of nan

            print('valid number of stochastic gradient noise sequence: ', stochastic_grads.shape[-1])

            # here we exploit a parallel trick to accelerate the process
            import nolds
            from datetime import datetime
            from multiprocessing import get_context
            from utils import error_func, calc_hurst_exponent
            from utils import fractal_dimension

            start_t = datetime.now()
            with get_context("spawn").Pool(min(os.cpu_count(), 32)) as pool:
                exponents = [pool.apply_async(calc_hurst_exponent, args=(stochastic_grads[:, j],), error_callback=error_func) for j in range(stochastic_grads.shape[1])]
                exponents = [p.get() for p in exponents]
                
            # remove the failures
            exponents = np.array(exponents)
            exponents = exponents[exponents >= 0.01]
            exponents = exponents[exponents <= 0.99]
            dim_h = fractal_dimension(exponents)
            end_t = datetime.now()
            elapsed_sec = (end_t - start_t).total_seconds()

            print(f'{num_components}-{seed}: hausdorff dimension: {dim_h:.2f}, average Hurst exponent: {np.mean(exponents):.2f}')
            print(f'elapsed time: {elapsed_sec:.2f} seconds')
            
if __name__ == '__main__':
    main()
