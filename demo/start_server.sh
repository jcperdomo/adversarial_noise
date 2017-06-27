#!/bin/bash

python -m demo/main --data_path src/data/cifar10 --load_model_from src/checkpoints/06-26/cifar10.ckpt --n_modules 5
