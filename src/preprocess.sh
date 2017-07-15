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

#python codebase/utils/process_facescrub.py --data_path /n/regal/rush_lab/awang/facescrub/dim128/ --out_path /n/regal/rush_lab/awang/data/facescrub/ --save_data_to /n/regal/rush_lab/awang/facescrub/dim128_n50.hdf5 --n_classes 50 --n_te_exs 100
python codebase/utils/process_facescrub.py --out_path /n/regal/rush_lab/awang/data/facescrub/ --load_data_from /n/regal/rush_lab/awang/facescrub/dim128_n50.hdf5 --n_te_exs 100
