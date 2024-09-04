import wandb

project = 'hurst2024'
config = {
    'program': 'train.py',
    'name': 'learning_rate_and_batch_size_resnet56_cifar100',
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'best_acc'},
    'parameters': 
    {
        'seed': {'value': 1},
        'scheduler': {'value': 'cosine'},
        'data_name': {'value': 'cifar100'},
        'num_classes': {'value': 100},
        'arch': {'value': 'resnet56_cifar100'},
        'optimizer': {'value': 'sgd'},
        'scheduler': {'value': 'cosine'},
        'learning_rate': {'values': [2.0e-2, 4.0e-2, 6.0e-2, 8.0e-2, 1.0e-1]},
        'batch_size_train': {'values': [1024, 512, 256, 128, 64]},
        'weight_decay': {'value': 5.0e-4},
        'tot_epochs': {'value': 200}
     }
}

sweep_id = wandb.sweep(sweep=config, project=project)
print(sweep_id)
