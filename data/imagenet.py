import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

def imagenet_dataset(args):
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    basic_transform = [transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip()]
        
    basic_transform += [transforms.ToTensor(), normalize]
        
    train_transform = transforms.Compose(basic_transform)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_set = datasets.ImageFolder(train_dir, train_transform)
    test_set = datasets.ImageFolder(test_dir, test_transform)
    
    return train_set, test_set