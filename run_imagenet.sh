#!/bin/bash

arch=resnet18

python train.py --data_name imagenet-1k --num_classes 1000 --arch $arch --optimizer sgd --learning_rate 0.1 --batch_size_train 256 --weight_decay 5.0e-4 --tot_epochs 100 --scheduler cosine --seed 2 --data_dir /mnt/data0/public_dataset/imagenet-1k --verbose 1 --wandb 1

arch=vit_small_patch32_224

python train.py --data_name imagenet-1k --num_classes 1000 --arch $arch --optimizer adamw --learning_rate 3.0e-3 --batch_size_train 1024 --weight_decay 0.1 --tot_epochs 300 --scheduler cosine --seed 2 --data_dir /mnt/data0/public_dataset/imagenet-1k --verbose 1 --wandb 1