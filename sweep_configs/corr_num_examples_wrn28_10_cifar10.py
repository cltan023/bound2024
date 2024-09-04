import wandb
import numpy as np

project = 'hurst2024'
config = {
    'program': 'train.py',
    'name': 'corr_num_examples_wrn28_10_cifar10',
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'best_acc'},
    'parameters': 
    {
        'seed': {'values': [1, 2, 3]},
        'scheduler': {'value': 'cosine'},
        'data_name': {'value': 'cifar10'},
        'num_classes': {'value': 10},
        'arch': {'value': 'wrn28_10_cifar10'},
        'optimizer': {'value': 'sgd'},
        'learning_rate': {'value': 0.05},
        'batch_size_train': {'value': 128},
        'weight_decay': {'value': 5.0e-4},
        'tot_epochs': {'value': 50},
        'example_per_class': {'values': list(range(500, 5000+1, 500))},
     }
}

sweep_id = wandb.sweep(sweep=config, project=project)
print(sweep_id)
