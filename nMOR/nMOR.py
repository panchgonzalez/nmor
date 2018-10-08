"""Neural Model Order Reduction with Tensorflow"""
from __future__ import print_function

import os
import sys
import random
import argparse

import numpy as np
import tensorflow as tf

from . import train
from . import inference
from .utils import misc_utils as utils

utils.check_tensorflow_version()

FLAGS = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # Network
  parser.add_argument("--num_units", type=int, default=32,
                      help="Number of units to in each layer for the bi-RNN.")
  parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                      help="Whether to use time-major mode.")

  # Optimizer
  parser.add_argument("--optimizer", type=str, default="sgd", help="adam | sgd")
  parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate.")
  parser.add_argument("--decay_scheme", type=str, default="",
                      help="""\
                      How we decay learning rate. Options include:
                        luong234: after 2/3 num train steps, we start halving
                          the learning rate for 4 times before finishing.
                        luong5: after 1/2 num train steps, we start halving
                          the learning rate for 5 times before finishing.
                        luong10: after 1/2 num train steps, we start halving
                          the learning rate for 10 times before finishing.\
                      """)
  parser.add_argument("--num_steps", type=int, default=12000,
                      help="Number of steps to train.")
  parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                      const=True, default=True,
                      help="""\
                      Wheter try colocating gradients with corresponding op\
                      """)

  # Initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                       help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help="""\
                      For uniform init_op, initialize weights between
                      [-init_weight, init_weight].\
                      """)

  # Data
  parser.add_argument("--out_dir", type=str, default=None,
                      help="Store log/model files.")
  parser.add_argument("--data_file", type=str, default=None,
                      help="Data file path.")
  parser.add_argument("--dev_data_file", type=str, default=None,
                      help="Dev (i.e., validation) data file path.")
  parser.add_argument("--test_data_file", type=str, default=None,
                      help="Test data file path.")
  parser.add_argument("--sample_infer_data_file", type=str, default=None,
                      help="Sample infer data file path.")
  parser.add_argument("--data_size", type=int, nargs="+", default=None,
                      help="Spatial dimension(s) of the data.")

  # sequence lengths
  parser.add_argument("--src_max_len", type=int, default=50,
                      help="Max length of src sequences during training.")
  parser.add_argument("--tgt_max_len", type=int, default=50,
                      help="Max length of tgt sequences during training.")

  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  parser.add_argument("--infer_data_file", type=str, default=None,
                      help="Set to the spatiotemporal data file to decode.")
  parser.add_argument("--infer_batch_size", type=int, default=32,
                      help="Batch size for inference mode.")
  parser.add_argument("--infer_output_file", type=str, default=None,
                      help="Output file to store decoding results.")
  parser.add_argument("--inference_ref_file", type=str, default=None,
                      help="""\
                      Reference file to compute evaluation scores (if given).\
                      """)
  parser.add_argument("--num_infer_steps", type=int, default=50,
                      help="Number of steps to decode.")

  # Misc
  parser.add_argument("--num_gpus", type=int, default=1,
                      help="Number of gpus in each worker.")
  parser.add_argument("--unit_type", type=str, default="lstm",
                      help="lstm | layer_norm_lstm")
  parser.add_argument("--forget_bias", type=float, default=1.0,
                      help="Forget bias for LSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (== 1 - keep_prob)")
  parser.add_argument("--loss_weight", type=float, default=1.0,
                      help="Weight of squared loss vs. energy loss.")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
  parser.add_argument("--steps_per_stats", type=int, default=100,
                      help="""\
                      How many training steps to do per stats logging. Save
                      checkpoint every 10x steps_per_stats.\
                      """)
  parser.add_argument("--log_device_placement", type="bool", nargs="?",
                      const=True, default=False, help="Debug GPU allocation.")
  parser.add_argument("--steps_per_eval", type=int, default=None,
                      help="""\
                      How many steps to do per evaluation on dev set.
                      Automatically set based on data if None.\
                      """)
  parser.add_argument("--scope", type=str, default=None,
                      help="scope to put variables under.")
  parser.add_argument("--hparams_path", type=str, default=None,
                      help="""\
                      Path to standard hparams json file that overrides hparams
                      values from FLAGS.\
                      """)
  parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed (>0, set a specific seed).")
  parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                      const=True, default=False,
                      help="Override loaded hparams with values specified.")
  parser.add_argument("--num_keep_ckpts", type=int, default=5,
                      help="Max number of checkpoints to keep.")


  # Job info
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers (inference only)")
  parser.add_argument("--num_intra_threads", type=int, default=0,
                      help="""\
                      Number of intra_op_parallelism_threads.
                      This should be set to the number of physical cores.\
                      """)
  parser.add_argument("--num_inter_threads", type=int, default=0,
                      help="""\
                      Number of inter_op_parallelism_threads.
                      This should be set to the number of sockets.\
                      """)


def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Network
      num_units=flags.num_units,
      time_major=flags.time_major,

      # Optimizer
      optimizer=flags.optimizer,
      learning_rate=flags.learning_rate,
      decay_scheme=flags.decay_scheme,
      num_steps=flags.num_steps,
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

      # Initializer
      init_op=flags.init_op,
      init_weight=flags.init_weight,

      # Data
      out_dir=flags.out_dir,
      data_file=flags.data_file,
      dev_data_file=flags.dev_data_file,
      test_data_file=flags.test_data_file,
      sample_infer_data_file=flags.sample_infer_data_file,
      data_size=flags.data_size,

      # Sequence lengths
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,

      #Inference
      infer_data_file=flags.infer_data_file,
      infer_batch_size=flags.infer_batch_size,
      infer_output_file=flags.infer_output_file,
      num_infer_steps=flags.num_infer_steps,

      # Misc
      num_gpus=flags.num_gpus,
      epoch_step=0, # record where we were within an epoch
      unit_type=flags.unit_type,
      forget_bias=flags.forget_bias,
      dropout=flags.dropout,
      loss_weight=flags.loss_weight,
      max_gradient_norm=flags.max_gradient_norm,
      batch_size=flags.batch_size,
      log_device_placement=flags.log_device_placement,
      steps_per_stats=flags.steps_per_stats,
      steps_per_eval=flags.steps_per_eval,
      random_seed=flags.random_seed,
      override_loaded_hparams=flags.override_loaded_hparams,
      num_keep_ckpts=flags.num_keep_ckpts,
      num_intra_threads=flags.num_intra_threads,
      num_inter_threads=flags.num_inter_threads,)


def create_or_load_hparams(
    out_dir, default_hparams, hparams_path, save_hparams=True):
  """Create hparams or load hparams from out_dir."""
  hparams = utils.load_hparams(out_dir)
  # Use default hparams, otherwise ensure compatibility
  if not hparams:
    hparams = default_hparams
    hparams = utils.maybe_parse_standard_hparams(hparams, hparams_path)
  else:
    hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)
  # Save hparams
  if save_hparams:
    utils.save_hparams(out_dir, hparams)
  # Print hparams
  utils.print_hparams(hparams)
  return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
  """Make sure the loaded hparams is compatible with the new changes."""
  default_hparams = utils.maybe_parse_standard_hparams(
      default_hparams, hparams_path)

  # For compatibility reasons, if there are new fields in default_hparams,
  # we add them to the current hparams
  default_config = default_hparams.values()
  config = hparams.values()
  for key in default_config:
    if key not in config:
      hparams.add_hparam(key, default_config[key])

  # Update all hparams' keys if override_loaded_hparams=True
  if default_hparams.override_loaded_hparams:
    for key in default_config:
      if getattr(hparams, key) != default_config[key]:
        utils.print_out("# Updating hparams.%s: %s -> %s" %
                        (key, str(getattr(hparams, key)),
                         str(default_config[key])))
        setattr(hparams, key, default_config[key])
  return hparams


def run_main(flags, default_hparams, train_fn, inference_fn, target_sesion=""):
  """Run main."""
  # Job
  jobid = flags.jobid
  num_workers = flags.num_workers
  utils.print_out("# Job id %d" % jobid)

  # Random
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)

  out_dir = flags.out_dir
  if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

  if flags.infer_data_file:
    # Inference
    hparams = create_or_load_hparams(
        out_dir, default_hparams, flags.hparams_path, save_hparams=False)
    out_file = flags.infer_output_file
    ckpt = flags.ckpt
    if not ckpt:
      ckpt_dir = os.path.join(out_dir, "ckpt")
      ckpt = tf.train.latest_checkpoint(ckpt_dir)
    inference_fn(ckpt, flags.infer_data_file,
                 out_file, hparams, num_workers, jobid)

  else:
    # Train
    hparams = create_or_load_hparams(
        out_dir, default_hparams, flags.hparams_path, save_hparams=(jobid == 0))
    train_fn(hparams, target_session=target_sesion)


def main(unused_argv):
  """Main entry point to program."""
  # Default hparams set to flags
  default_hparams = create_hparams(FLAGS)

  train_fn = train.train
  inference_fn = inference.inference
  run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
  # Create nMOR package argument parser
  nMOR_parser = argparse.ArgumentParser()

  # Add predefined arguments to nMOR_parser
  add_arguments(nMOR_parser)

  # Parse arguments and known args to FLAGS
  FLAGS, unparsed = nMOR_parser.parse_known_args()

  # Run the main program
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
