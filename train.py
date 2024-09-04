import numpy as np
import torch
import os
from os.path import join
import argparse
from tqdm import tqdm
from colorama import Fore, Style
import wandb
import json
from prefetch_generator import BackgroundGenerator
# hurst exponent for some coordinates are not available (i.e. outliers)
import warnings
warnings.filterwarnings("ignore")

from data import cifar_dataset, imagenet_dataset

from pytorchcv.model_provider import get_model as ptcv_get_model
from timm import create_model

from datetime import datetime
from utils import cycle_loader, get_grads, fractal_dimension
from utils import calc_hurst_exponent, error_func
from utils import init_random_state, validate

# avaiable neural architectures 
model_zoo = [
'resnet18', 'resnet50',
'vit_small_patch32_224', 'vit_small_patch16_224',
'resnet20_cifar10', 'resnet20_cifar100', 
'resnet56_cifar10', 'resnet56_cifar100',
'resnext29_32x4d_cifar10', 'resnext29_32x4d_cifar100', 
'wrn28_10_cifar10', 'wrn28_10_cifar100',
'pyramidnet110_a270_cifar10', 'pyramidnet110_a270_cifar100'
]
        
def main():
    parser = argparse.ArgumentParser(description='non-vacuous generalization bounds for deep neural networks')
    
    # data configuration
    parser.add_argument('--data_dir', default='../public_dataset', type=str)
    
    parser.add_argument('--data_name', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet-1k'])
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--label_corruption', default=-1.0, type=float)
    parser.add_argument('--example_per_class', default=-1, type=int)
    parser.add_argument('--categories', default=[], type=int, nargs='+')
    
    parser.add_argument('--batch_size_train', default=512, type=int)
    parser.add_argument('--batch_size_eval', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    
    # optimizer configuration
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5.0e-4, type=float)
    parser.add_argument('--scheduler', default='multistep', choices=['cosine', 'const', 'multistep'])

    # architecture configuration
    parser.add_argument('--arch', default='resnet20_cifar10')
    # some other choices
    
    # train configuration
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--tot_epochs', default=50, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--save_dir', default='runs', type=str)
    parser.add_argument('--project', default='hurst2024', type=str)
    parser.add_argument('--wandb', default=1, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--miniball', default=0, type=int)
    
    # hurst exponent estimation
    parser.add_argument('--num_components', default=500000, type=int)
    parser.add_argument('--len_of_sequence', default=3000, type=int)
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        
    args.instance = f'{args.scheduler}_lr={args.learning_rate:.2e}_bs={args.batch_size_train}_wd={args.weight_decay:.2e}_corr{args.label_corruption}_{args.example_per_class}_cat{args.categories}_seed={args.seed}'

    # timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # save_dir = join(args.save_dir, args.project, args.instance, timestamp)
    save_dir = join(args.save_dir, f'{args.arch}_{args.optimizer}', args.instance)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(join(save_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.wandb:
        # wandb_run = wandb.init(config=args, project=args.project, name=args.instance, settings=wandb.Settings(code_dir="."))
        wandb_run = wandb.init(config=args, project=args.project, name=args.instance)
        
    init_random_state(args.seed)
    if args.data_name == 'cifar10' or args.data_name == 'cifar100':
        net = ptcv_get_model(args.arch, pretrained=False).to(device)
    elif args.data_name == 'imagenet-1k':
        net = create_model(args.arch, pretrained=False).to(device)
    else:
        raise NotImplementedError
    
    if args.data_name == 'cifar10' or args.data_name == 'cifar100':
        train_set, test_set = cifar_dataset(data_name=args.data_name, root=args.data_dir, label_corruption=args.label_corruption, example_per_class=args.example_per_class, categories=args.categories)
    elif args.data_name == 'imagenet-1k':
        train_set, test_set = imagenet_dataset(args)
    else:
        raise NotImplementedError
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory, drop_last=True)
    train_loader_cycle = cycle_loader(train_loader)
    init_random_state(args.seed) # make sure the mini-batch order is the same
    train_loader_no_shuffle = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_memory, drop_last=False)
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=False)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'{args.optimizer} is currently not implemented!')
        
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tot_epochs)
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.3*args.tot_epochs), int(0.6*args.tot_epochs)], gamma=0.1)
    elif args.scheduler == 'const':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.tot_epochs)
        print('use constant learning rate, be careful!')

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    
    train_loss_vec_hist = []
    stochastic_grad_norm_hist = []
    true_generalization_bound = []
    
    train_loss, train_acc, train_loss_vec = validate(net, train_loader_no_shuffle, loss_func, device, train=True)
    test_loss, test_acc, _ = validate(net, test_loader, loss_func, device, train=False)
    train_loss_vec_hist.append(train_loss_vec)
    true_generalization_bound.append(test_loss-train_loss)
    
    with tqdm(total=args.tot_epochs, colour='MAGENTA', ascii=True) as pbar:
        for epoch in range(args.tot_epochs):
            net.train()
            grad_norm_per_epoch = []
            for x, y in BackgroundGenerator(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                yhat = net(x)
                loss = loss_func(yhat, y)
                loss.mean().backward()
                curr_grad = get_grads(net)
                grad_norm_per_epoch.append(curr_grad.norm(p=2).item())
                optimizer.step()
                    
            scheduler.step()
            
            if epoch % args.log_interval == 0:
                train_loss, train_acc, train_loss_vec = validate(net, train_loader_no_shuffle, loss_func, device, train=False)
                test_loss, test_acc, _ = validate(net, test_loader, loss_func, device, train=False)
                
                train_loss_vec_hist.append(train_loss_vec)
                true_generalization_bound.append(test_loss-train_loss)
                stochastic_grad_norm_hist.append(np.mean(grad_norm_per_epoch))
                if args.wandb:
                    wandb_run.log({'train_acc': train_acc*100, 'test_acc': test_acc*100, 'train_loss': train_loss, 'test_loss': test_loss, 'lr': scheduler.get_last_lr()[0], 'true_bound': true_generalization_bound[-1]})
                                
                message = f'epoch: {epoch} '
                message += f'lr: {scheduler.get_last_lr()[0]:.6f} '
                message += f'train_loss: {Fore.RED}{train_loss:.4f}{Style.RESET_ALL} '
                message += f'train_acc: {Fore.RED}{train_acc*100:.2f}%{Style.RESET_ALL} '
                message += f'test_loss: {Fore.GREEN}{test_loss:.4f}{Style.RESET_ALL} '
                message += f'test_acc: {Fore.GREEN}{test_acc*100:.2f}%{Style.RESET_ALL} '
                
                if args.verbose:
                    torch.save(net.state_dict(), join(save_dir, f'net_{epoch}.pt'))
                pbar.set_description(message)
                pbar.update()
                    
            if train_acc >= 0.995 or train_loss <= 0.01:
                torch.save(net.state_dict(), join(save_dir, f'state_dict.pt'))
                break
            
    # once training is over, we are now ready to compute the upper bound of generalization error
    # first calculate the easy terms
    train_loss_vec_hist = torch.vstack(train_loss_vec_hist)
    if args.miniball:
        import cyminiball as miniball
        _, radius = miniball.compute(train_loss_vec_hist.numpy().astype(np.float64))
        diam_a = 2.0 * np.sqrt(radius)
    else:
        diam_a = (train_loss_vec_hist[0] - train_loss_vec_hist[-1]).norm(p=2).item()
        
    lip = max(stochastic_grad_norm_hist) * args.batch_size_train ** 0.5
    alpha = np.log2(1.0 / diam_a * len(train_loader_no_shuffle.dataset) ** 0.5 * lip)
    alpha = alpha ** 0.5
    beta = diam_a * 12 / len(train_loader_no_shuffle.dataset)
    
    # compute the remaining Hausdorff dimension
    net.train()
    # estimate true gradient
    true_gradient = 0.0
    for x, y in train_loader_no_shuffle:
        x, y = x.to(device), y.to(device)
        net.zero_grad()
        loss = loss_func(net(x), y)
        loss.mean().backward()
        curr_gradient = get_grads(net)
        true_gradient = true_gradient + curr_gradient * len(x)
    true_gradient = true_gradient / len(train_loader_no_shuffle.dataset)
    
    # estimated stochastic gradient noise
    start_t = datetime.now()
    tot_param = len(true_gradient)
    # number of parameters used to estimate Hurst parameter 
    num_components = args.num_components
    # number of mini-batches to generate a stochastic sequence 
    len_of_sequence = args.len_of_sequence         
    if tot_param < num_components:
        num_components = tot_param
    fixed_dims = torch.randperm(tot_param)[:num_components]
    
    net.train()
    stochastic_grads = []
    for j, (x, y) in enumerate(train_loader_cycle):
        if j == len_of_sequence:
            break
        x, y = x.to(device), y.to(device)
        net.zero_grad()
        loss = loss_func(net(x), y)
        loss.mean().backward()
        curr_gradient = get_grads(net)
        stochastic_grads.append(curr_gradient[fixed_dims])
    stochastic_grads = torch.vstack(stochastic_grads)
    stochastic_grads = stochastic_grads - true_gradient[fixed_dims]
    stochastic_grads = stochastic_grads.double().cpu().numpy()
    # delete unvalid elements in case of nan
    stochastic_grads = stochastic_grads[:, ~np.isnan(stochastic_grads).any(axis=0)]
    
    # delete unnecessary variables for multiprocessing
    del train_loader_no_shuffle, train_loader_cycle, train_loader, test_loader
    
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
    
    # hausdorff dimension
    dim_w = fractal_dimension(exponents)
    # final upper bound
    upper_bound = (2.0 * dim_w) ** 0.5 * beta * (alpha + 1.0 / alpha)
    
    print(f'dim_w: {dim_w:.2f}, diam_a: {diam_a:.2f}, lip: {lip:.2f}, average Hurst exponent: {np.mean(exponents):.2f}')
    print(f'time elapsed: {elapsed_sec:.2f} seconds, upper bound is {upper_bound:.4f} and true bound is {max(true_generalization_bound):.4f}')
    
    torch.save([train_loss_vec_hist, stochastic_grad_norm_hist, true_generalization_bound, exponents], join(save_dir, 'summary.pt'))
            
if __name__ == '__main__':
    main()