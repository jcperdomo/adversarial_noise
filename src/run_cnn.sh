#!/bin/bash
#
#SBATCH -t 1-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -o src/logs/sbatch.out
#SBATCH -e src/logs/sbatch.err
#SBATCH --mail-type=end
#SBATCH --mail-user=alexwang@college.harvard.edu

DATE="$(date +%m-%d)"
mkdir -p src/logs/$DATE
mkdir -p src/out/$DATE
mkdir -p src/checkpoints/$DATE

# Train a good model and save it
EXP_NAME="facescrub_d128"
EXP_DIR="/n/regal/rush_lab/awang/data/facescrub_d128"
#python -m src/codebase/main --data_path $EXP_DIR --log_file src/logs/$DATE/$EXP_NAME.log --out_file src/out/$DATE/$EXP_NAME.hdf5 --save_model_to src/checkpoints/$DATE/$EXP_NAME.ckpt --optimizer adagrad --n_epochs 25 --init_scale .1 --n_kernels 128 --learning_rate .1 --n_modules 7 --batch_size 50

# Load a saved model
MODEL_PATH="src/checkpoints/07-10/$EXP_NAME.ckpt"
#python -m codebase/main --data_path $EXP_DIR --log_file src/logs/$DATE/$EXP_NAME.log --im_file $EXP_DIR/te.hdf5 --out_file src/out/$DATE/$EXP_NAME.hdf5 --load_model_from $MODEL_PATH --optimizer adagrad --n_epochs 25 --init_scale .1 --n_kernels 128 --learning_rate .1 --n_modules 7 --batch_size 50

# Fast, no model saving, few epochs
EXP_NAME="facescrub_d128_fast"
EXP_DIR="/n/regal/rush_lab/awang/data/facescrub_d128"
#python -m src/codebase/main --data_path $EXP_DIR --log_file src/logs/$DATE/$EXP_NAME.log --im_file $EXP_DIR/te.hdf5 --out_file src/out/$DATE/$EXP_NAME.hdf5 --out_path src/out/$DATE/$EXP_NAME --optimizer adagrad --n_epochs 10 --init_scale .1 --n_kernels 32 --learning_rate .1 --n_modules 7 --generator fast_gradient --eps .1 --alpha 0.1 --batch_size 200
python -m src/codebase/main --data_path $EXP_DIR --log_file src/logs/$DATE/$EXP_NAME.log --im_file $EXP_DIR/te.hdf5 --out_file src/out/$DATE/$EXP_NAME.hdf5 --out_path src/out/$DATE/$EXP_NAME --load_model_from $MODEL_PATH --optimizer adagrad --n_epochs 0 --init_scale .1 --n_kernels 128 --learning_rate .1 --n_modules 7 --generator carlini --eps .1 --alpha 0.0 --batch_size 200 --n_generator_steps 10000 --generator_learning_rate .01 --generator_opt_const 10.
