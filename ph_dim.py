import warnings
warnings.filterwarnings("ignore")

import torch
import os
import numpy as np

import json
from utils import DictToClass

num_examples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
seeds = [1, 2, 3]
for num_example in num_examples:
    for seed in seeds:
        model_dir = f'runs/resnet56_cifar10_sgd/cosine_lr=5.00e-02_bs=128_wd=5.00e-04_corr-1.0_{num_example}_cat[]_seed={seed}'
        # model_dir ='runs/resnet56_cifar10_sgd/cosine_lr=4.00e-02_bs=64_wd=5.00e-04_corr-1.0_-1_cat[]_seed=1'

        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            args = f.read()
        args = json.loads(args)
        args = DictToClass(args)

        # device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:0')

        from data import cifar_dataset
        from utils import cycle_loader

        train_set, test_set = cifar_dataset(data_name=args.data_name, root=args.data_dir, label_corruption=args.label_corruption, example_per_class=args.example_per_class, categories=args.categories)

        train_loader_no_shuffle = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers,
            pin_memory=args.pin_memory, drop_last=True)
        train_loader_cycle = cycle_loader(train_loader)

        from pytorchcv.model_provider import get_model as ptcv_get_model

        net =  ptcv_get_model(args.arch, pretrained=False).to(device)
        net.load_state_dict(torch.load(os.path.join(model_dir, 'state_dict.pt'), map_location=device))

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        # note: using momentum will cause ph_dim always approximately equals 1
        # note: the authors use vaniall SGD with constant learning rate
        optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.0, weight_decay=args.weight_decay)

        from utils import get_params
        from utils import validate

        max_points = 3000 # too large value sometimes will cause training failure
        train_loss_hist = []
        weights_hist = []
        for j, (x, y) in enumerate(train_loader_cycle):
            if j == max_points:
                break
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            yhat = net(x)
            loss = loss_func(yhat, y)
            loss.mean().backward()
            optimizer.step()
            
            curr_params = get_params(net)
            weights_hist.append(curr_params)
            
            # train_loss, train_acc, train_loss_vec = validate(net, train_loader_no_shuffle, loss_func, device, train=False)
            # train_loss_hist.append(train_loss_vec)
            if j % 500 == 0:
                train_loss, train_acc, train_loss_vec = validate(net, train_loader_no_shuffle, loss_func, device, train=False)
                print(f'iteration={j}, train_loss={train_loss:.4f}, train_acc={train_acc*100:.4f}%')
        weights_hist = torch.stack(weights_hist, dim=0).cpu().numpy()

        from indicator import fast_ripser
        ph_dim_euclidean = fast_ripser(weights_hist[1:], max_points=max_points, min_points=200, point_jump=20)

        # print('PH dimension', ph_dim_euclidean)
        print(f'num_example={num_example}, seed={seed}, ph_dim_euclidean={ph_dim_euclidean}')
