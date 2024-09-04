#!/bin/bash

arch=resnet56_cifar100

python train.py --data_name cifar100 --num_classes 100 --arch $arch --optimizer sgd --momentum 0.9 --learning_rate 0.05 --batch_size_train 128 --weight_decay 5.0e-4 --tot_epochs 200 --scheduler cosine --seed 1 --wandb 1



