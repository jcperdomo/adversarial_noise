#!/bin/bash
#
#SBATCH -t 1-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=35000
#SBATCH -o logs/sbatch.out
#SBATCH -e logs/sbatch.err
#SBATCH --mail-type=end
#SBATCH --mail-user=alexwang@college.harvard.edu

#python codebase/utils/process_lfw.py --data_path data/raw/lfw/lfw-deepfunneled --save_data_to data/raw/lfw/lfw-deepfunneled.hdf5 --out_path data/lfw_test --resize 64
#python codebase/utils/process_lfw.py --load_data_from data/raw/lfw/lfw-deepfunneled.hdf5 --out_path data/lfw_test --resize 64

python codebase/utils/process_facescrub.py --data_path data/raw/facescrub/dim128 --out_path data/facescrub_test
