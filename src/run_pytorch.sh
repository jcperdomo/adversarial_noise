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

EXP_NAME="facescrub"
EXP_DIR="/n/regal/rush_lab/awang/data/facescrub_pytorch"
MODEL="load_model_from"
MODEL_PATH="src/checkpoints/07-18/$EXP_NAME.ckpt"

DATE="$(date +%m-%d)"
LOG_PATH="src/logs/$DATE"
OUT_PATH="src/outs/$DATE/$EXP_NAME"
CKPT_PATH="src/checkpoints/$DATE"
mkdir -p $LOG_PATH
mkdir -p $OUT_PATH
mkdir -p $CKPT_PATH
TRAIN_NEW=${1:-"0"}

OPTIMIZER=sgd
N_EPOCHS=10
N_MODULES=7
N_KERNELS=64
LEARNING_RATE=.01
BATCH_SIZE=25

GENERATOR=random
TARGET='none'
GEN_EPS=.1
GEN_ALPHA=0.0

if [ ! -f "$MODEL_PATH.meta" ] || [ $TRAIN_NEW -eq "1" ]; then
    # Train a good model and save it
    echo "Training a model from scratch"
    MODEL="save_model_to"
    MODEL_PATH="$CKPT_PATH/$EXP_NAME.ckpt"
else
    # Load a saved model
    echo "Loading a model"
    N_EPOCHS=0
fi
CMD="python -m src/codebase_pytorch/main --data_path $EXP_DIR --log_file $LOG_PATH/$EXP_NAME.log --im_file $EXP_DIR/te.hdf5 --out_file $OUT_PATH/$EXP_NAME.hdf5 --out_path $OUT_PATH --$MODEL $MODEL_PATH --optimizer $OPTIMIZER --n_epochs $N_EPOCHS --init_scale .1 --n_kerns $N_KERNELS --lr $LEARNING_RATE --n_modules $N_MODULES --batch_size $BATCH_SIZE --generator $GENERATOR --alpha $GEN_ALPHA --eps $GEN_EPS --n_generator_steps 1 --target $TARGET"
eval $CMD
