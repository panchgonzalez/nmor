from __future__ import print_function

import time
import h5py

import numpy as np
import tensorflow as tf

from . import model as nMOR_model
from . import model_helper
from .utils import misc_utils as utils
from .utils import data_utils


__all__ =["inference", "single_worker_inference", "multi_worker_inference"]


def inference(ckpt,
              inference_input_file,
              inference_output_file,
              hparams,
              num_workers=1,
              jobid=0,
              scope=None):
  """Perform decoding."""

  if len(hparams.data_size) == 1:
    model_creator = nMOR_model.Model1D
  elif len(hparams.data_size) == 2:
    model_creator = nMOR_model.Model2D
  else:
    raise ValueError("No model for %d-D data" (len(hparams.data_size)))
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  if num_workers == 1:
    single_worker_inference(
        infer_model,
        ckpt,
        inference_input_file,
        inference_output_file,
        hparams)
  else:
    multi_worker_inference(
        infer_model,
        ckpt,
        inference_input_file,
        inference_output_file,
        hparams,
        num_workers=num_workers,
        jobid=jobid)


def single_worker_inference(infer_model,
                            ckpt,
                            inference_input_file,
                            inference_output_file,
                            hparams):
  """Inference with a single worker."""
  output_infer = inference_output_file

  # Read data
  if len(hparams.data_size) == 1:
    infer_data = data_utils.load_1D_data(inference_input_file)
  elif len(hparams.data_size) == 2:
    infer_data = data_utils.load_2D_data(inference_input_file)
  else:
    raise ValueError("No data loader for %d-D data" (len(hparams.data_size)))


  with tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto()) as sess:
    loaded_infer_model = model_helper.load_model(
        infer_model.model, ckpt, sess, "infer")
    sess.run(infer_model.iterator.initializer,
             feed_dict={infer_model.data_placeholder: infer_data})

    # Decode
    utils.print_out("# Start decoding")
    decode_and_save(loaded_infer_model, sess, output_infer)


def multi_worker_inference(infer_model,
                           ckpt,
                           inference_input_file,
                           inference_output_file,
                           hparams,
                           num_workers,
                           jobid):
  """Inference using multiple workers."""
  # TODO: Implement multiworker inference
  raise NotImplementedError("Multi-worker inference to be implemented...")


def decode_and_save(model,
                    sess,
                    decoded_file,
                    ref_file=None,
                    save=True,
                    global_step=None):
  """Decode a test set and optionally save output tensor as numpy array."""
  # Decode
  utils.print_out("  decoding to %s" % decoded_file)

  start_time = time.time()
  num_sequences = 0
  decoded_sequences = []
  while True:
    try:
      infer_outputs, _ = model.decode(sess)
      batch_size = tf.shape(infer_outputs)[0] # should be in batch-major
      num_sequences += batch_size

      # Convert to NumPy array
      # NOTE: if infer_outputs is a tensor (e.g. tf.transpose) we need to
      #   append infer_outputs.eval().
      decoded_sequences.append(infer_outputs)
    except tf.errors.OutOfRangeError:
      utils.print_time("  done, num sequences %d" %
          num_sequences.eval(), start_time)
      break

  # Save decoded sequences
  if save:
    utils.print_out("  saving decoded sequences to %s" % decoded_file)
    # Concatenate decoded sequences
    decoded_sequences = np.concatenate(decoded_sequences, axis=0)

    with h5py.File(decoded_file, "a") as hf:
      hf.create_dataset("decoded",
                        shape=decoded_sequences.shape,
                        dtype=decoded_sequences.dtype,
                        data=decoded_sequences)
