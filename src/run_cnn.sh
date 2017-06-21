#!/bin/bash

python codebase/main.py --data_path data/lfw_test --log_file logs/06-20/lfw_test.log --optimizer adagrad --n_epochs 10 --init_scale .1 --n_kernels 128 --learning_rate 1
