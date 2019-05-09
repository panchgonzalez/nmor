#!/bin/bash

# nMOR package directory and base directory
nMOR_DIR=/path/to/nMOR
BASE_DIR=`pwd`

# Model and data directories from args
OUTPUT_DIR=${BASE_DIR}
DATA_DIR=${HOME}/path/to/data/directory
NUM_UNITS=8
NUM_STEPS=$1

# Train and test datasets to use
TRAIN_DATA_DIR=${DATA_DIR}/path/to/train/data.h5
TEST_DATA_DIR=${DATA_DIR}/path/to/test/data.h5
INFER_DATA_DIR=${DATA_DIR}/path/to/sample/inference/data.h5  # Optional

python3 -m nMOR.nMOR \
    --num_units=${NUM_UNITS} \
    --time_major=True \
    --optimizer=adam \
    --learning_rate=0.0005 \
    --decay_scheme=luong234 \
    --num_steps=${NUM_STEPS} \
    --out_dir=${OUTPUT_DIR} \
    --data_file=${TRAIN_DATA_DIR} \
    --dev_data_file=${TEST_DATA_DIR} \
    --sample_infer_data_file=${INFER_DATA_DIR} \
    --data_size 64 64 \
    --batch_size=16 \
    --dropout=0.01 \
    --init_weight=0.5 \
    --steps_per_stats=10 \
    --num_gpus=1 \
    --steps_per_eval=800 \
    --random_seed=1 \
    --override_loaded_hparams=True \
    --num_intra_threads=16 \
    --num_inter_threads=0
