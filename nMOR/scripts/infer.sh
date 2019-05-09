#!/bin/bash

# nMOR Package directory and base directory
nMOR_DIR=path/to/nMOR
export PYTHONPATH=${nMOR_DIR}
BASE_DIR=`pwd`

# Model and data directories
OUTPUT_DIR=${BASE_DIR}
DATA_DIR=/path/to/data/directory
NUM_UNITS=$1
NUM_INFER_STEPS=$2
CKPT=$3

# Checkpoint directory
CKPT_DIR=/path/to/checkpint/nMOR_model.ckpt-${CKPT}

# Dataset to perform inference on and output directory
INFER_DATA_DIR=${DATA_DIR}/path/to/inference/data.h5
INFER_OUTPUT=infer_ckpt-${CKPT}.h5

python3 -m nMOR.nMOR \
    --num_units=${NUM_UNITS} \
    --num_infer_steps=${NUM_INFER_STEPS} \
    --time_major=True \
    --optimizer=adam \
    --out_dir=${OUTPUT_DIR} \
    --infer_data_file=${INFER_DATA_DIR} \
    --infer_output_file=${INFER_OUTPUT} \
    --data_size 64 64 \
    --batch_size=1 \
    --random_seed=1 \
    --num_gpus=0 \
    --ckpt=${CKPT_DIR} \
    --override_loaded_hparams=True \
    --num_intra_threads=16 \
    --num_inter_threads=0
