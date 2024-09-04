import warnings
warnings.filterwarnings("ignore")

import torch
import os
import numpy as np

import json
from utils import DictToClass

num_examples = [100, 150, 200, 250, 300, 350, 400, 450, 500]
seeds = [1, 2, 3]
for num_example in num_examples:
    for seed in seeds:
        model_dir = f'runs/resnet56_cifar100_sgd/cosine_lr=5.00e-02_bs=128_wd=5.00e-04_corr-1.0_{num_example}_cat[]_seed={seed}'
        # model_dir ='runs/resnet56_cifar10_sgd/cosine_lr=4.00e-02_bs=1024_wd=5.00e-04_corr-1.0_-1_cat[]_seed=1'

        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            args = f.read()
        args = json.loads(args)
        args = DictToClass(args)

        # device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:0')

        from data import cifar_dataset
        from utils import cycle_loader

        train_set, test_set = cifar_dataset(data_name=args.data_name, root=args.data_dir, label_corruption=args.label_corruption, example_per_class=args.example_per_class, categories=args.categories)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers,
            pin_memory=args.pin_memory, drop_last=True)

        train_loader_cycle = cycle_loader(train_loader)

        from pytorchcv.model_provider import get_model as ptcv_get_model

        net =  ptcv_get_model(args.arch, pretrained=False).to(device)
        net.load_state_dict(torch.load(os.path.join(model_dir, 'state_dict.pt'), map_location=device))

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        # note: the authors of Sim¸sekli, 2020 use vaniall SGD with constant learning rate
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.0, weight_decay=args.weight_decay)

        from indicator import peek_model_size
        from utils import validate

        ms = peek_model_size(net)
        iter_num = 391 # number of mini-batches in one epoch
        parameter_arrays = []
        for mod_size in ms:
            # print("ozan", iter_num, mod_size)
            parameter_arrays.append(torch.zeros(iter_num, mod_size))

        for j, (x, y) in enumerate(train_loader_cycle):
            if j == iter_num:
                break
            net.train()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = net(x)
            loss = loss_func(yhat, y).mean()
            loss.backward()
            optimizer.step()
            
            read_mem_cnt = 0
            for p in net.parameters():
                cpu_data = p.data.cpu().view(1,-1)
                if len(p.shape) < 2:
                    continue
                parameter_arrays[read_mem_cnt][j, 0:np.prod(p.shape)] = cpu_data
                read_mem_cnt +=1
                
            if j % 50 == 0:
                # train_loss, train_acc, train_loss_vec = validate(net, train_loader_no_shuffle, loss_func, device, train=False)
                # print(f'iteration={j}, train_loss={train_loss:.4f}, train_acc={train_acc*100:.4f}%')
                print(f'iteration={j}, train_loss={loss.item():.4f}')

        from indicator import estimator_scalar, estimator_vector_full, estimator_vector_mean, estimator_vector_projected

        # All models are stored in the memory, so we need to call estimators
        # Following Sim¸sekli, 2020, we only use estimator_vector_projected function, others are included for future research
        # alpha_full_est = []
        alpha_proj_med_est = []
        alpha_proj_max_est = []
        # alpha_mean_est = []
        # alpha_scalar_est = []
        for param in parameter_arrays:
            # alpha_full = estimator_vector_full(param)
            # alpha_full_est.append(alpha_full)

            alpha_proj_med, alpha_proj_max = estimator_vector_projected(param)
            alpha_proj_med_est.append(alpha_proj_med)
            alpha_proj_max_est.append(alpha_proj_max)

            # alpha_mean = estimator_vector_mean(param)
            # alpha_mean_est.append(alpha_mean)

            # alpha_scalar = estimator_scalar(param)
            # alpha_scalar_est.append(alpha_scalar) 

        print(f'num_example={num_example}, seed={seed}, mean value: {np.mean(alpha_proj_med_est):.4f}, max value: {np.mean(alpha_proj_max_est):.4f}')


