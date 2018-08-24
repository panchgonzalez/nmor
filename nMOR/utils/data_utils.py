"""Utility functions to handle data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import h5py
import numpy as np
import tensorflow as tf

from ..utils import misc_utils as utils


def load_1D_data(data_file):
  """Load 1D spatial hdf5 dataset."""
  with tf.device("/cpu:0"):
    with h5py.File(data_file, "r") as hf:
      dataset = hf["dataset"][:]
  return dataset


def load_2D_data(data_file):
  """load 2D spatial hdf5 dataset."""
  with tf.device("/cpu:0"):
    with h5py.File(data_file, "r") as hf:
      dataset = hf["dataset"][:] # NOTE: might want to parse out BCs.
  return dataset


def check_data(data_file, out_dir):
  """Check data_file, extract spatial dimension of the data."""
  data_file_path = os.path.join(out_dir, data_file)
  if tf.gfile.Exists(data_file_path):
    utils.print_out("# Data file %s exists" % data_file)
    with h5py.File(data_file_path, "r") as hf:
      data_dtype = hf["dataset"].dtype
      data_shape = hf["dataset"].shape
  else:
    raise ValueError("Data file %s does not exist." % data_file)
  return data_dtype, data_shape
