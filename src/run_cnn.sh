#!/bin/bash

#python codebase/main.py --data_path data/lfw_test --log_file logs/06-20/lfw_test.log --optimizer adagrad --n_epochs 10 --init_scale .1 --n_kernels 128 --learning_rate 1
#python codebase/main.py --data_path data/mnist --log_file logs/06-21/mnist_test.log --optimizer adagrad --n_epochs 10 --init_scale .1 --n_kernels 32 --learning_rate 1 --n_modules 4
#python codebase/main.py --data_path data/cifar10 --log_file logs/06-21/cifar10_test.log --optimizer adagrad --n_epochs 15 --init_scale .1 --n_kernels 64 --learning_rate 1. --n_modules 5
#python codebase/main.py --data_path data/cifar100coarse --log_file logs/06-21/cifar100coarse.log --optimizer adagrad --n_epochs 20 --init_scale .1 --n_kernels 64 --learning_rate 1. --n_modules 5
#python codebase/main.py --data_path data/cifar100fine --log_file logs/06-21/cifar100fine.log --optimizer adagrad --n_epochs 20 --init_scale .1 --n_kernels 364 --learning_rate 1. --n_modules 5

python codebase/main.py --data_path data/cifar10 --log_file logs/06-21/cifar10.log --im_file data/cifar10/val.hdf5 --out_file out/06-21/cifar10.log --optimizer adagrad --n_epochs 15 --init_scale .1 --n_kernels 64 --learning_rate 1. --n_modules 5 --generator fast_gradient
