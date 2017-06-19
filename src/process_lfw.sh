#!/bin/bash

python codebase/utils/process_lfw.py --data_path data/raw/lfw/lfw-deepfunneled --save_data_to data/raw/lfw/lfw-deepfunneled.hdf5 --out_path data/lfw_test --resize 64
#python codebase/utils/process_lfw.py --load_data_from data/raw/lfw/lfw-deepfunneled.hdf5 --out_path data/lfw_test --resize 64
