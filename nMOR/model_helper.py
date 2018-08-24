"""Utility functions for building models."""
from __future__ import print_function

import collections
import time
import os

import numpy as np
import tensorflow as tf

from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import data_utils

__all__ = [
    "get_initializer", "get_device_str",
    "create_train_model", "create_eval_model", "create_infer_model",
    "create_rnn_cell", "gradient_clip", "create_or_load_model", "load_model"
]


def get_initializer(init_op, seed=None, init_weight=None):
  """Create an initializer, init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(
        -init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id, num_gpus):
  """Return a device string from multi-GPU setup."""
  if num_gpus == 0:
    return "/cpu:0"
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


# Train Model
class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "data_placeholder"))):
  pass


def create_train_model(model_creator, hparams, scope=None):
  """Create train graph, model, and iterator."""
  out_dir = hparams.out_dir
  data_file = hparams.data_file
  data_dtype, data_shape = data_utils.check_data(data_file, out_dir)

  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "train"):

    # Define dataset from placeholder, will be fed in during training
    data_placeholder = tf.placeholder(data_dtype, data_shape)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(data_placeholder)

    iterator = iterator_utils.get_iterator(
        dataset,
        batch_size=hparams.batch_size,
        random_seed=hparams.random_seed,
        param_var=hparams.param_var,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        output_buffer_size=None)

    model_device_fn = None # if we have a special device name or function
    with tf.device(model_device_fn):
      model = model_creator(
          hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          scope=None)

  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator,
      data_placeholder=data_placeholder)


# Eval Model
class EvalModel(
    collections.namedtuple("EvalModel", ("graph", "model", "iterator",
                                         "data_placeholder"))):
  pass


def create_eval_model(model_creator, hparams, scope=None):
  """Create eval graph, model, and iterator."""
  out_dir = hparams.out_dir
  dev_data_file = hparams.dev_data_file
  data_dtype, data_shape = data_utils.check_data(dev_data_file, out_dir)

  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "eval"):

    # Define dataset from placeholder, will be fed in during evaluation
    data_placeholder = tf.placeholder(data_dtype, data_shape)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(data_placeholder)

    iterator = iterator_utils.get_iterator(
        dataset,
        batch_size=hparams.batch_size,
        random_seed=hparams.random_seed,
        param_var=hparams.param_var,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        output_buffer_size=None)

    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        scope=None)

  return EvalModel(
      graph=graph,
      model=model,
      iterator=iterator,
      data_placeholder=data_placeholder)


# Infer Model
class InferModel(
    collections.namedtuple("InferModel", ("graph", "model", "iterator",
                                          "data_placeholder"))):
  pass


def create_infer_model(model_creator, hparams, scope=None):
  """Create infer graph, model, and iterator."""
  out_dir = hparams.out_dir
  if hparams.infer_data_file:
    infer_data_file = hparams.infer_data_file
  else:
    infer_data_file = hparams.sample_infer_data_file
  data_dtype, data_shape = data_utils.check_data(infer_data_file, out_dir)

  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "infer"):

    # Define dataset from placeholder, will be fed in during inference
    data_placeholder = tf.placeholder(data_dtype, data_shape)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(data_placeholder)

    iterator = iterator_utils.get_infer_iterator(
        dataset,
        batch_size=hparams.infer_batch_size,
        num_infer_steps=hparams.num_infer_steps,
        param_var=hparams.param_var)

    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.INFER,
        scope=None)

  return InferModel(
      graph=graph,
      model=model,
      iterator=iterator,
      data_placeholder=data_placeholder)


def create_rnn_cell(unit_type, num_units, forget_bias, dropout, mode,
                    num_proj=None, use_peepholes=True, device_str=None):
  """Creates an instance of a single RNN cell."""

  # Cell Type
  if unit_type == "lstm":
    utils.print_out("  LSTM, forget_bias=%g" % forget_bias)
    single_cell = tf.contrib.rnn.LSTMCell(
        num_units=num_units,
        use_peepholes=use_peepholes, # diagonal peephole connections to learn timing
        num_proj=num_proj, # linear output projection
        forget_bias=forget_bias)
  elif unit_type == "layer_norm_lstm":
    utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
                    new_line=False)
    single_cell = tf.contrib.LayerNormBasicLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # # Dropout (= 1-keep_prob) is set to 0 for eval and infer modes
  # dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
  # if dropout > 0.0:
  #   single_cell = tf.contrib.rnn.DropoutWrapper(
  #       cell=single_cell, input_keep_prob=(1.0-dropout))
  #   utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout))
  # TODO: Residual
  # TODO: DeviceWrapper
  return single_cell


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  return clipped_gradients, gradient_norm


def load_model(model, ckpt, session, name):
  """Load model from checkpoint."""
  start_time = time.time()
  model.saver.restore(session, ckpt)
  utils.print_out("  loaded %s model parameters from %s, time %.2fs" %
                  (name, ckpt, time.time()-start_time))
  return model


def create_or_load_model(model, ckpt_dir, session, name):
  """Create nMOR model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                   (name, time.time()-start_time))
  global_step = model.global_step.eval(session=session)
  return model, global_step
