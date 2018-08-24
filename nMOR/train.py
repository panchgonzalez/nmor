"""For training nMOR models."""
from __future__ import print_function

import math
import os
import random
import time
import h5py

import numpy as np
import tensorflow as tf

from . import model as nMOR_model
from . import inference
from . import model_helper
from .utils import misc_utils as utils
from .utils import data_utils

# # DEBUG
# from tensorflow.python import debug as tf_debug


utils.check_tensorflow_version()

__all__ = [
    "run_eval", "run_full_eval", "run_sample_decode", "init_stats",
    "update_stats", "process_stats", "print_step_info", "before_train", "train"
]


def run_eval(eval_model, eval_sess, ckpt_dir, dev_data,
             hparams, summary_writer, use_test_set=False):
  """Compute evalution for both dev/test sets."""
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, ckpt_dir, eval_sess, "eval")

  dev_eval_iterator_feed_dict = {
      eval_model.data_placeholder: dev_data}
  dev_loss = _run_eval(loaded_eval_model, global_step, eval_sess,
                       eval_model.iterator, dev_eval_iterator_feed_dict,
                       summary_writer, "dev")

  # TODO: implement evaluation on test set for hyperparameter optimization

  return dev_loss


def run_full_eval(ckpt_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, infer_data, dev_data):
  """Wrapper for eval and sample_decode."""
  run_sample_decode(infer_model, infer_sess, ckpt_dir, hparams, infer_data)
  dev_loss = run_eval(eval_model, eval_sess, ckpt_dir, dev_data,
                      hparams, summary_writer)
  return dev_loss


def run_sample_decode(infer_model, infer_sess, ckpt_dir, hparams, infer_data):
  """Sample decode from fixed inference dataset."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, ckpt_dir, infer_sess, "infer")

  _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, infer_model.data_placeholder, infer_data)


def init_stats():
  """Initialize statistics that we want to keep."""
  return {"step_time": 0.0, "loss": 0.0, "grad_norm": 0.0}


def update_stats(stats, start_time, step_result):
  """Update stats: write summary and accumulate statistics."""
  (_, step_loss, step_summary, global_step,
   batch_size, grad_norm, learning_rate) = step_result

  # Update statistics
  stats["step_time"] += (time.time() - start_time)
  stats["loss"] += step_loss
  stats["grad_norm"] += grad_norm

  return global_step, learning_rate, step_summary


def process_stats(stats, info, global_step, steps_per_stats, log_f):
  """Update info."""
  # Update info
  info["avg_step_time"] = stats["step_time"] / steps_per_stats
  info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
  info["avg_step_loss"] = stats["loss"] / steps_per_stats


def print_step_info(prefix, global_step, info, result_summary, log_f):
  """Print all info at the current global_step."""
  utils.print_out(
      "%sstep %d lr %g step-time %.2fs loss %g gN %.2f %s, %s" %
      (prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["avg_step_loss"], info["avg_grad_norm"], result_summary,
       time.ctime()),
      log_f)


def before_train(loaded_train_model, train_model, train_sess, global_step,
                 train_data, hparams, log_f):
  """Misc tasks to do before training."""
  stats = init_stats()
  info = {"avg_step_time": 0.0, "avg_grad_norm": 0.0, "avg_step_loss": 0.0,
          "learning_rate": loaded_train_model.learning_rate.eval(
              session=train_sess)}
  start_train_time = time.time()
  utils.print_out("# Start step %d, lr %g, %s" %
                  (global_step, info["learning_rate"], time.ctime()), log_f)
  # Initialize all iterators
  utils.print_out("# Init train iterator.")
  # NOTE: if passing the entire dataset to here is too costly, can initialize
  # in the train().
  train_sess.run(train_model.iterator.initializer,
                 feed_dict={train_model.data_placeholder: train_data})
  return stats, info, start_train_time


def train(hparams, scope=None, target_session=""):
  """Train a sequence-to-sequence autoencoder model."""

  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_steps = hparams.num_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_eval = hparams.steps_per_eval

  if not steps_per_eval:
    steps_per_eval = 10 * steps_per_stats

  # Define model creator
  if len(hparams.data_size) == 1:
    model_creator = nMOR_model.Model1D
    load_data = data_utils.load_1D_data
  elif len(hparams.data_size) == 2:
    if hparams.param_var is True:
      # NOTE: ParamVarModel for 2D data only
      model_creator = nMOR_model.ParamVarModel
    else:
      model_creator = nMOR_model.Model2D
    load_data = data_utils.load_2D_data
  else:
    raise ValueError("No model for %d-D data" % (len(hparams.data_size)))

  # Create train and eval models
  train_model = model_helper.create_train_model(model_creator, hparams, scope)
  eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  # Load train, dev, and infer datasets outside tf.graph
  data_file_path = os.path.join(out_dir, hparams.data_file)
  train_data = load_data(data_file_path)
  dev_data_file_path = os.path.join(out_dir, hparams.dev_data_file)
  dev_data = load_data(dev_data_file_path)
  infer_data_file_path = os.path.join(out_dir, hparams.sample_infer_data_file)
  infer_data = load_data(infer_data_file_path)

  summary_name = "train_log"
  ckpt_dir = os.path.join(out_dir, "ckpt")
  if not tf.gfile.Exists(ckpt_dir):
    utils.print_out("# Creating ckpt directory %s ..." % ckpt_dir)
    tf.gfile.MakeDirs(ckpt_dir)

  # Log and output file
  log_file = os.path.join(out_dir, "log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("# log_file=%s" % log_file, log_f)

  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement,
      num_intra_threads=hparams.num_intra_threads,
      num_inter_threads=hparams.num_inter_threads)
  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(
      target=target_session, config=config_proto, graph=infer_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, ckpt_dir, train_sess, "train")

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, summary_name), train_model.graph)

  # First evalution
  dev_loss = run_full_eval(ckpt_dir, infer_model, infer_sess,
                           eval_model, eval_sess, hparams, summary_writer,
                           infer_data, dev_data)
  utils.print_out(
      "# First eval, global_step %d dev loss %g " % (global_step, dev_loss))

  last_stats_step = global_step
  last_eval_step = global_step

  stats, info, start_train_time = before_train(
      loaded_train_model, train_model, train_sess, global_step,
      train_data, hparams, log_f)

  # Training loop
  utils.print_out("# Training for %d steps" % num_steps)
  while global_step < num_steps:
    ### Run a step ###
    start_time = time.time()
    try:
      step_result = loaded_train_model.train(train_sess)
      hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset. Go to next epoch.
      hparams.epoch_step = 0
      utils.print_out("# Finished an epoch at step %d." % global_step)
      train_sess.run(train_model.iterator.initializer,
                     feed_dict={train_model.data_placeholder: train_data})
      continue

    # Process step_result, accumulate_stats, and write_summary
    global_step, info["learning_rate"], step_summary = update_stats(
        stats, start_time, step_result)
    summary_writer.add_summary(step_summary, global_step)

    # Print statistics every steps_per_stats
    if global_step - last_stats_step >= steps_per_stats:
      last_stats_step = global_step
      process_stats(stats, info, global_step, steps_per_stats, log_f)
      print_step_info("  ", global_step, info, "", log_f)
      # Reset statistics
      stats = init_stats()

    # Run eval every steps_per_eval
    if global_step - last_eval_step >= steps_per_eval:
      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(ckpt_dir, "nMOR_model.ckpt"),
          global_step=global_step)
      last_eval_step = global_step
      dev_loss = run_full_eval(ckpt_dir, infer_model, infer_sess,
                               eval_model, eval_sess, hparams, summary_writer,
                               infer_data, dev_data)
      utils.print_out(
          "# Save eval, global_step %d dev loss %g" % (global_step, dev_loss))

  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(ckpt_dir, "nMOR_model.ckpt"),
      global_step=global_step)

  print_step_info("# Final, ", global_step, info, "", log_f)
  utils.print_time("# Done training!", start_train_time)

  summary_writer.close()

  return global_step


def _sample_decode(model, global_step, sess, hparams, iterator,
                  data_placeholder, infer_data):
  """Decode sample inputs and decode."""
  sess.run(iterator.initializer, feed_dict={data_placeholder: infer_data})

  infer_outputs, _ = model.decode(sess)

  infer_dir = os.path.join(hparams.out_dir, "infer")
  if not tf.gfile.Exists(infer_dir):
    utils.print_out("# Creating infer directory %s ..." % infer_dir)
    tf.gfile.MakeDirs(infer_dir)
  infer_file = os.path.join(infer_dir, "infer.h5")
  try:
    with h5py.File(infer_file, "a") as hf:
      hf.create_dataset("decoded/step-%d" % global_step,
                        shape=infer_outputs.shape,
                        dtype=infer_outputs.dtype,
                        data=infer_outputs)
  except:
    utils.print_out("# Infer step %d already saved ..." % global_step)
    pass


def _run_eval(model, global_step, sess, iterator, iterator_feed_dict,
              summary_writer, label):
  """Computing evalution."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  eval_loss, epoch_step = 0, 0

  while True:
    try:
      eval_step_loss, batch_size = model.eval(sess)
      eval_loss += eval_step_loss
      epoch_step += 1
    except tf.errors.OutOfRangeError:
      break

  eval_loss = eval_loss / epoch_step
  utils.add_summary(summary_writer, global_step, "%s_loss" % label, eval_loss)
  return eval_loss
