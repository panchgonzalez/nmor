# nMOR: *neural* Model Order Reduction
## Deep learning framework for model reduction of dynamical systems

*Authors: [Francisco J. Gonzalez](http://franciscojgonzalez.com)*

This code was primarily developed for the paper *[Deep convolutional recurrent autoencoders for learning low-dimensional feature dynamics of fluid systems](https://arxiv.org/abs/1808.01346).*

<p align="center">
<img width="35%" src="nmor/doc/learning_vortices.gif" />
<br>
Figure 1. <b>Learning low-dimensional dynamics</b> - this animation depicts the
process of learning the dynamics of two vortices governed by the Navier-Stokes equations using nMOR.
</p>

- [Introduction](#introduction)
  - [Convolutional Autoencoder](#convolutional-autoencoder)
  - [Evolver RNN](#evolver-rnn)
- [Installing nMOR](#installing-nMOR)
- [Training - *How to build an nMOR model*](#training--how-to-build-an-nMOR-model)
  - [Dataset Construction](#dataset-construction)
  - [Offline Training](#offline-training)
  - [Online Evaluation](#online-evaluation)
- [Other Resources](#other-resources)
- [Acknowledgment](#acknowledgment)
- [References](#references)
- [BibTex](#bibTex)

# Introduction
nMOR (*neural* Model Order Reduction) is a deep convolutional recurrent autoencoder
architecture for completely data-driven model reduction of high-dimensional
dynamical systems. At the heart of this approach is a modular architecture
consisting of a convolutional autoencoder (or encoder-decoder network) and a evolver
RNN (constructed using a modified version of an LSTM), depicted in Figures 2 and 3.

## Convolutional Autoencoder
<p align="center">
<img width="100%" src="nmor/doc/cnn_autoencoder.png" />
<br>
Figure 2. <b>Convolutional Autoencoder</b> - this architecture is used to learn
an efficient low-dimensional representation of the physical data.
</p>

The main idea behind using a convolutional autoencoder method is that it
exploits local, location-invariant correlations present in physical data through
the use of convolutional neural networks. That is, rather than of applying a
fully-connected autoencoder to the high-dimensional input data we instead apply
it to a vectorized feature map produced by a convolutional encoder, and
similarly the reverse is done for reconstruction. The result is the
identification of an expressive low-dimensional manifold obtained at a much
lower cost while offering specific advantages over both traditional POD-based
ROMs and fully-connected autoencoders

## Evolver RNN
<p align="center">
<img width="50%" src="nmor/doc/rnn.png" />
<br>
Figure 3. <b>Modified LSTM network</b> - this RNN architecture is used to
learn the dynamics of low-dimensional representation on its underlying nonlinear manifold.
</p>

We propose a modified LSTM network to model the evolution of low-dimensional
data representations on this manifold that avoids costly state reconstructions
at every step. In doing this, we ensure that the evaluation of new steps scales
only with the size of the low-dimensional representation and not with the size
of the full dimensional data, which may be large for some problems.


# Installing nMOR
To install this code first make sure you have TensorFlow (>= v1.3) installed
on your system. To install TensorFlow visit the [installation instructions here](https://www.tensorflow.org/install/).

Once you have TensorFlow installed, you can download the source code by running:

```shell
git clone https://github.com/franjgonzalez/nMOR
```

# Training - How to build an nMOR model
## Dataset Construction
By default, this code assumes that the data is in some specific format. Since this
approach is targeting computational modeling applications having structured data
on which to train the nMOR model requires only some light preprocessing.

Ultimately, you should have a dataset $\mathcal{X}$ which has the following form

$$
\mathcal{X} = \{ \mathbf{X}^1, \mathbf{X}^2, ..., \mathbf{X}^{N_s} \}\in[0,1]^{N_x\times N_y\times N_t\times N_s}
$$

where each $\mathbf{X}^i=[\mathbf{x}_i^1,...,\mathbf{x}_i^{N_t}]$ is a training sample
consisting of a $N_t$ solution snapshots $\mathbf{x}_i^n\in[0,1]^{N_x\times N_y}$.
In this case $N_s$ represents the number of training samples. It is recommended
that the each solution snapshot is defiend on a uniform grid as this allows
each convolutional filter to act equally over the entire domain. By default, this
code reads datasets saved using the `h5py` library under the dataset name *dataset*.
It is then read by

```python
with h5py.File("datafile.h5", "r") as hf:
    dataset = hf["dataset"][:]
```

Typically, a dataset is split into a training set which is used to make parameter updates,
and a validation set which does not make parameter updates. The performance of the
model on the validation set can be used to tune hyper-parameters or prevent overfitting.
In addition this code assumes the existance of a inference set: a dataset consisting
of a training sample and a validation sample. This can be used for futher analysis
on the training progress.

## Offline Training
For each model you will need to pre-define some environment variables

```bash
# nMOR package directory and base directory
nMOR_DIR=${HOME}/nMOR
BASE_DIR=`pwd`

# Model and data directories from args
OUTPUT_DIR=${BASE_DIR}
DATA_DIR=${HOME}/path/to/the/data
NUM_UNITS=16
NUM_STEPS=100000

# Train and test datasets to use
TRAIN_DATA_DIR=${DATA_DIR}/trainset.h5
TEST_DATA_DIR=${DATA_DIR}/testset.h5
INFER_DATA_DIR=${DATA_DIR}/inferset.h5
```

To start training run the following command:

```bash
python3 -m nMOR.nMOR \
    --num_units=${NUM_UNITS} \
    --time_major=True \
    --optimizer=adam \
    --learning_rate=0.0005 \
    --num_steps=${NUM_STEPS} \
    --out_dir=${OUTPUT_DIR} \
    --data_file=${TRAIN_DATA_DIR} \
    --dev_data_file=${TEST_DATA_DIR} \
    --sample_infer_data_file=${INFER_DATA_DIR} \
    --data_size 64 64 \
    --batch_size=32 \
    --dropout=0.01 \
    --init_weight=0.5 \
    --steps_per_stats=10 \
    --num_gpus=1 \
    --steps_per_eval=800
```

If the setup was done correctly you should eventually see an output that looks like this

```
# First eval, global_step 0 dev loss 5.02981
# Start step 0, lr 0.0005, Sat May 26 09:48:39 2018
# Init train iterator.
# Training for 600000 steps
  step 10 lr 0.0005 step-time 0.44s loss 5.00348 gN 8.99 , Sat May 26 09:48:46 2018
  step 20 lr 0.0005 step-time 0.28s loss 4.90889 gN 8.87 , Sat May 26 09:48:48 2018
  step 30 lr 0.0005 step-time 0.28s loss 4.84085 gN 8.80 , Sat May 26 09:48:51 2018
  step 40 lr 0.0005 step-time 0.28s loss 4.77679 gN 8.74 , Sat May 26 09:48:54 2018
  step 50 lr 0.0005 step-time 0.28s loss 4.70085 gN 8.65 , Sat May 26 09:48:57 2018
  step 60 lr 0.0005 step-time 0.28s loss 4.62944 gN 8.57 , Sat May 26 09:49:00 2018
```

## Online Evaluation
To evaluate a trained model, first define the following environment variables:

```bash
# Number of time steps to run evaluation
NUM_INFER_STEPS=2500

# Desired checkpoint on which to evaluate model
CKPT=100000

# Checkpoint directory
CKPT_DIR=/path/to/checkpoint/directory/ckpt/nMOR_model.ckpt-${CKPT}

# Dataset to perform inference on and output directory
INFER_DATA_DIR=${DATA_DIR}/infer_ic.h5
INFER_OUTPUT=${CASE_DIR}/infer_h-${NUM_UNITS}_ckpt-${CKPT}_s_${NUM_INFER_STEPS}.h5
```

To evaluate run the following command:

```bash
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
    --num_gpus=0 \
    --ckpt=${CKPT_DIR}
```

# BibTex
```
@article{gonzalez18,
  author = {Francisco J. Gonzalez and Maciej Balajewicz},
  title = {Deep convolutional recurrent autoencoders for learning low-dimensional feature dynamics of fluid systems},
  archivePrefix = {arXiv},
  arxivId = {1808.01346}
}
```
