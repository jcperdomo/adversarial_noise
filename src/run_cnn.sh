#!/bin/bash

EXP_NAME="facescrub_d128"
EXP_DIR="/n/regal/rush_lab/awang/data/$EXP_NAME"
DATE="$(date +%m-%d)"
mkdir -p logs/$DATE
mkdir -p out/$DATE
mkdir -p checkpoints/$DATE

#python codebase/main.py --data_path data/lfw_test --log_file logs/06-20/lfw_test.log --optimizer adagrad --n_epochs 10 --init_scale .1 --n_kernels 128 --learning_rate 1
#python codebase/main.py --data_path data/mnist --log_file logs/06-21/mnist_test.log --optimizer adagrad --n_epochs 10 --init_scale .1 --n_kernels 32 --learning_rate 1 --n_modules 4
#python codebase/main.py --data_path data/cifar10 --log_file logs/06-21/cifar10_test.log --optimizer adagrad --n_epochs 15 --init_scale .1 --n_kernels 64 --learning_rate 1. --n_modules 5
#python codebase/main.py --data_path data/cifar100coarse --log_file logs/06-21/cifar100coarse.log --optimizer adagrad --n_epochs 20 --init_scale .1 --n_kernels 64 --learning_rate 1. --n_modules 5
#python codebase/main.py --data_path data/cifar100fine --log_file logs/06-21/cifar100fine.log --optimizer adagrad --n_epochs 20 --init_scale .1 --n_kernels 364 --learning_rate 1. --n_modules 5

# Save a model
#python codebase/main.py --data_path data/cifar10 --log_file logs/$DATE/cifar10.log --im_file data/cifar10/te.hdf5 --out_file out/$DATE/cifar10.hdf5 --save_model_to checkpoints/$DATE/cifar10.ckpt --optimizer adagrad --n_epochs 5 --init_scale .1 --n_kernels 128 --learning_rate 1. --n_modules 5 --generator fast_gradient --eps .3 --alpha 0.1

# Load a model
#python codebase/main.py --data_path data/cifar10 --log_file logs/$DATE/cifar10.log --im_file data/cifar10/te.hdf5 --out_file out/$DATE/cifar10.hdf5 --save_model_to checkpoints/$DATE/cifar10 --load_model_from checkpoints/$DATE/cifar10.ckpt --optimizer adagrad --n_epochs 15 --init_scale .1 --n_kernels 128 --learning_rate 1. --n_modules 5 --generator fast_gradient --eps .3 --alpha 0.1

# Fast, no model saving, few epochs
python -m src/codebase/main --data_path $EXP_DIR --log_file logs/$DATE/$EXP_NAME.log --im_file $EXP_DIR/te.hdf5 --out_file out/$DATE/$EXP_NAME.hdf5 --optimizer adagrad --n_epochs 2 --init_scale .1 --n_kernels 64 --learning_rate .1 --n_modules 7 --generator carlini --eps .3 --alpha 0.1
