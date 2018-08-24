#!/bin/bash

# nMOR package directory and base directory
nMOR_DIR=${HOME}/nMOR
BASE_DIR=`pwd`

# Model and data directories from args
OUTPUT_DIR=${BASE_DIR}/$1
DATA_DIR=${BASE_DIR}/$2
NUM_UNITS=$3
NUM_PRETRAIN_STEPS=$4
NUM_STEPS=$5

# Train and test datasets to use
TRAIN_DATA_DIR=${DATA_DIR}/lid_Nx-64_trainset-7978.h5
TEST_DATA_DIR=${DATA_DIR}/lid_Nx-64_testset-886.h5

python3 -m nMOR.nMOR \
    --num_units=${NUM_UNITS} \
    --time_major=True \
    --optimizer=adam \
    --learning_rate=0.00075 \
    --decay_scheme=luong234 \
    --num_pretrain_steps=${NUM_PRETRAIN_STEPS} \
    --num_train_steps=${NUM_STEPS} \
    --out_dir=${OUTPUT_DIR} \
    --data_file=${TRAIN_DATA_DIR} \
    --dev_data_file=${TEST_DATA_DIR} \
    --data_size 64 64 \
    --batch_size=4 \
    --steps_per_stats=1 \
    --num_gpus=0 \
    --steps_per_eval=200 \
    --random_seed=0 \
    --override_loaded_hparams=True \
    --num_intra_threads=16 \
    --num_inter_threads=0
