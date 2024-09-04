import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

from .cifar10 import CIFAR10Partial, CIFAR10RandomLabels
from .cifar100 import CIFAR100Partial, CIFAR100RandomLabels

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

# transform_train = transforms.Compose([transforms.ToTensor(),
#                                      transforms.Normalize(
#                                          mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

# transform_test = transforms.Compose([transforms.ToTensor(),
#                                      transforms.Normalize(
#                                          mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

transform_train = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

def cifar_dataset(data_name='cifar10', root='../public_dataset', label_corruption=-1.0, example_per_class=-1, categories=None):
    if data_name == 'cifar10':
        if label_corruption > 0:
            train_set = CIFAR10RandomLabels(root=root, train=True, download=True, transform=transform_train, corrupt_prob=label_corruption)
        elif example_per_class > 0:
            train_set = CIFAR10Partial(root=root, train=True, download=True, transform=transform_train, example_per_class=example_per_class, categories=categories)
        else:
            train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif data_name == 'cifar100':
        if label_corruption > 0:
            train_set = CIFAR100RandomLabels(root=root, train=True, download=True, transform=transform_train, corrupt_prob=label_corruption)
        elif example_per_class > 0:
            train_set = CIFAR100Partial(root=root, train=True, download=True, transform=transform_train, example_per_class=example_per_class, categories=categories)
        else:
            train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    else:
        raise NotImplementedError(f'data set {data_name} not provided!')
    
    return train_set, test_set
            