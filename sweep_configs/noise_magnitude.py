import wandb

project = 'hurst2024'
config = {
    'program': 'train.py',
    'name': 'noise_magnitude',
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'best_acc'},
    'parameters': 
    {
        'seed': {'value': 1},
        'scheduler': {'value': 'cosine'},
        'data_name': {'value': 'cifar10'},
        'num_classes': {'value': 10},
        'arch': {'value': 'resnet20_cifar10'},
        'optimizer': {'value': 'sgd'},
        'scheduler': {'value': 'cosine'},
        'learning_rate': {'values': [2.0e-2, 1.0e-1]},
        'batch_size_train': {'values': [1024, 128]},
        'weight_decay': {'value': 5.0e-4},
        'tot_epochs': {'value': 50},
        'verbose': {'value': 1}, # save model weights for each epoch and use as inputs for evolution_history.py
     }
}

sweep_id = wandb.sweep(sweep=config, project=project)
print(sweep_id)
