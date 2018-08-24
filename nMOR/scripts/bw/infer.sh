#!/bin/bash

# nMOR package directory and base directory
nMOR_DIR=${HOME}/nMOR
BASE_DIR=`pwd`

# Model and data directories from args
OUTPUT_DIR=${BASE_DIR}/$1
DATA_DIR=${BASE_DIR}/$2

# Checkpoint directory
CKPT_DIR=${OUTPUT_DIR}/ckpt/nMOR_model.ckpt-20000

# Inference hparams
INFER_HPARAMS=${OUTPUT_DIR}/infer_hparams.json

# Dataset to perform inference on and output directory
INFER_DATA_DIR=${DATA_DIR}/ad.200.testset.h5
INFER_OUTPUT=${OUTPUT_DIR}/infer/ks.infer-ckpt-20000.h5


python3 -m nMOR.nMOR \
    --num_units=${NUM_UNITS} \
    --num_proj_units 128 256 512 \
    --time_major=True \
    --out_dir=${OUTPUT_DIR} \
    --infer_data_file=${INFER_DATA_DIR} \
    --infer_output_file=${INFER_OUTPUT} \
    --data_size=1024 \
    --random_seed=1 \
    --num_gpus=1 \
    --ckpt=${CKPT_DIR} \
    --hparams_path=${INFER_HPARAMS} \
    --override_loaded_hparams=True
