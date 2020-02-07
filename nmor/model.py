"""Basic dynamic convolutional autoencoder nMOR model."""
from __future__ import absolute_import
from __future__ import print_function

import os
import abc

import numpy as np
import tensorflow as tf

from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model1D", "Model2D"]


class BaseModel(object):
  """Convolutional autoencoder nMOR base class."""

  def __init__(self, hparams, iterator, mode, scope=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      iterator: Dataset Iterator that feeds sequential data.
      mode: TRAIN | EVAL | INFER
      scope: scope of the model.
    """
    assert isinstance(iterator, iterator_utils.BatchedInput)
    self.iterator = iterator
    self.mode = mode

    self.data_size = hparams.data_size
    self.num_gpus = hparams.num_gpus
    self.time_major = hparams.time_major
    self.batch_size = tf.shape(self.iterator.source)[0]
    self.loss_weight = hparams.loss_weight

    # Initializer
    initializer = model_helper.get_initializer(
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Train graph
    res = self.build_graph(hparams, scope=scope)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[0]
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[0]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_outputs = res[1]

    self.global_step = tf.Variable(0, trainable=False)

    # Get trainable parameters
    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Get optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)

      # Calculate gradients
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)
      clipped_grads, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm = grad_norm

      # Graph update
      self.update =opt.apply_gradients(
          zip(clipped_grads, params),
          global_step=self.global_step)

      # Summary
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", self.learning_rate),
          tf.summary.scalar("train_loss", self.train_loss),
          tf.summary.scalar("grad_norm", self.grad_norm),
          tf.summary.scalar("clipped_grads", tf.global_norm(clipped_grads))])

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Saver
    self.saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    # Print trainable variables
    utils.print_out("# Autoencoder trainable variables")
    total_params = 0
    for param in tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "nMOR/ae"):
      total_params += np.prod(param.get_shape().as_list())
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))
    utils.print_out("  Total autoencoder trainable variables: %d" % total_params)
    utils.print_out("# Evolver trainable variables")
    total_params = 0
    for param in tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "nMOR/evolver"):
      total_params += np.prod(param.get_shape().as_list())
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))
    utils.print_out("  Total evolver trainable variables: %d" % total_params)


  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      if hparams.decay_scheme == "luong5":
        start_decay_step = int(hparams.num_steps / 2)
        decay_times = 5
      elif hparams.decay_scheme == "luong10":
        start_decay_step = int(hparams.num_steps / 2)
        decay_times = 10
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(hparams.num_steps * 2 / 3)
        decay_times = 4
      remain_steps = hparams.num_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = hparams.num_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (hparams.decay_scheme,
                                         start_decay_step,
                                         decay_steps,
                                         decay_factor))
    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")


  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update,
                     self.train_loss,
                     self.train_summary,
                     self.global_step,
                     self.batch_size,
                     self.grad_norm,
                     self.learning_rate])


  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss, self.batch_size])


  @abc.abstractmethod
  def build_graph(self):
    """Subclass must implement this.
    Creates a convolutional autoencoder nMOR model with raw_rnn RNN API.

    Args:
      defined by subclass

    Returns:
      loss: train loss
      outputs: outputs of the network for inference
    """
    pass


  def _conv2d(self, _input, filters, kernel_size, strides, dilation_rate,
              batch_norm=False, activation=None, name=None):
    """Wrapper for 2d convolutional layer."""
    with tf.variable_scope(name) as scope:
      # Full convolutional layer
      c1 = tf.layers.conv2d(
          _input, filters=filters, kernel_size=kernel_size, strides=strides,
          dilation_rate=dilation_rate, padding="same", activation=activation)
      if batch_norm:
        c1 = tf.layers.batch_normalization(c1, training=True)
    return c1


  def _conv3d(self, _input, filters, kernel_size, strides, dilation_rate,
               batch_norm=False, activation=None, name=None):
    """Wrapper for 3d convolutional layer."""
    with tf.variable_scope(name) as scope:
      # Full 3d convolutional layer
      c1 = tf.layers.conv3d(
          _input, filters=filters, kernel_size=kernel_size, strides=strides,
          dilation_rate=dilation_rate, padding="same", activation=activation)
      if batch_norm:
        c1 = tf.layers.batch_normalization(c1, training=True)
    return c1


  def _dconv2d(self, _input, filters, kernel_size, strides,
               batch_norm=False, activation=None, name=None):
    """Wrapper for 2D "deconvolution" layer"""
    with tf.variable_scope(name) as scope:
      # Upsample convolutional layer
      c1 = tf.layers.conv2d_transpose(
          _input, filters=filters, kernel_size=kernel_size,
          strides=strides , padding="same", activation=activation)
      if batch_norm:
        c1 = tf.layers.batch_normalization(c1, training=True)
    return c1


  def _build_rnn(self, hparams, enc_state, tgt_states=None):
    """
    Build and run the RNN that evolves the hidden states.

    Args:
      enc_state: the initial hidden state to evolve.
      hparams: Hyperparameter configurations.

    Returns:
      A tuple of the form (rnn_outputs, rnn_final_state) where
        rnn_outputs: a float32 Tensor [batch_size, num_steps, num_units]
        rnn_final_state: a float32 Tensor [batch_size, num_steps, num_units]
    """
    # Convert enc_state to LSTMStateTuple
    enc_state = tf.contrib.rnn.LSTMStateTuple(c=enc_state, h=enc_state)

    # # Use tgt_states random teacher forcing
    # if tgt_states is not None:
    #   tgt_states = tf.reshape(
    #       tgt_states, (self.batch_size, -1, hparams.num_units))
    #   tgt_states = tf.transpose(tgt_states, [1,0,2])

    # Define start and end steps
    start_step = tf.ones([self.batch_size, hparams.num_units], dtype=tf.float32)
    end_step = tf.zeros([self.batch_size, hparams.num_units], dtype=tf.float32)

    # Raw rnn looping functions
    def _loop_fn_initial():
      initial_elements_finished = (0 >= self.iterator.sequence_length)
      initial_input = start_step
      initial_cell_state = enc_state
      initial_cell_output = None
      initial_loop_state = None
      return (initial_elements_finished,
              initial_input,
              initial_cell_state,
              initial_cell_output,
              initial_loop_state)

    def _loop_fn_transition(time, previous_output, previous_state,
                            previous_loop_state):
      def get_next_input():
        # if tgt_states is not None: # random teacher forcing
        #   teach = tf.cast(np.random.randint(0,2)<1, tf.bool)
        #   next_input = tf.cond(
        #       teach, lambda: tgt_states[time], lambda: previous_output)
        # else: # no teacher forcing during inference
        #   next_input = previous_output
        return previous_output
      elements_finished = (time >= self.iterator.sequence_length)
      finished = tf.reduce_all(elements_finished)
      input_ = tf.cond(finished, lambda: end_step, get_next_input)
      state = previous_state
      output = previous_output
      loop_state = None
      return (elements_finished, input_, state, output, loop_state)

    def _loop_fn(time, previous_output, previous_state, previous_loop_state):
      if previous_state is None: # time == 0
        assert previous_output is None and previous_state is None
        return _loop_fn_initial()
      else:
        return _loop_fn_transition(time, previous_output, previous_state,
                                   previous_loop_state)

    with tf.variable_scope("rnn") as scope:
      cell = model_helper.create_rnn_cell(
          unit_type=hparams.unit_type,
          num_units=hparams.num_units,
          forget_bias=hparams.forget_bias,
          dropout=hparams.dropout,
          mode=self.mode,
          # num_proj=hparams.num_units,
          use_peepholes=True)
      rnn_outputs_ta, rnn_final_state, _ = tf.nn.raw_rnn(cell, _loop_fn)
      rnn_outputs = rnn_outputs_ta.stack()
      # Back to batch-major
      rnn_outputs = tf.transpose(rnn_outputs, [1,0,2])
    return rnn_outputs, rnn_final_state


  @abc.abstractmethod
  def _compute_loss(self, ae_out, ae_enc, rnn_out, rnn_enc,
                    alpha=0.5, beta=0.5):
    """Subclass must implement this.
    Computes the unsupervised training loss.
    """
    pass


  def _get_infer_summary(self, hparams):
    return tf.no_op()


  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    return sess.run([self.infer_outputs, self.infer_summary])


  def decode(self, sess):
    """Decode a batch."""
    infer_outputs, infer_summary = self.infer(sess)
    return infer_outputs, infer_summary


class Model1D(BaseModel):
  """Deep convolutional sequence-to-sequence model.

  This class implements a deep convolutional sequence-to-sequence model for
  2D spatio-temporal data (space, time).
  """

  def build_graph(self, hparams, scope=None):
    """Creates a convolutional autoencoder for 2D data with an evolver RNN.

    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default is "conv_seq2seq".

    Returns:
      A tuple of the form (loss, outputs)
      where
        loss: total loss
        outputs: float32 Tensor [batch_size, num_steps, data_size]
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    def _reshape_outputs(dec_outputs_flat):
      dec_outputs = tf.reshape(
          dec_outputs_flat, (self.batch_size, -1, self.data_size[0]))
      return dec_outputs

    with tf.variable_scope(scope or "nMOR", dtype=dtype):
      # Training or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # Build convolutional autoencoder
        with tf.variable_scope("ae"):
          ae_enc_states = self._build_encoder(hparams, train_ae=True)
          ae_dec_outputs_flat = self._build_decoder(ae_enc_states, hparams)
          ae_dec_outputs = _reshape_outputs(ae_dec_outputs_flat)

        # Build convolutional encoder to feed evolver
        with tf.variable_scope("ae", reuse=True):
          rnn_enc_state = self._build_encoder(
              hparams, train_ae=False, reuse=True)

        # Build evolver RNN
        with tf.variable_scope("evolver"):
          rnn_outputs, _ = self._build_rnn(hparams, rnn_enc_state)

        # Reconstruct the full state from evolved rnn states
        rnn_outputs_flat = tf.reshape(rnn_outputs, (-1, hparams.num_units))
        with tf.variable_scope("ae", reuse=True):
          rnn_dec_outputs_flat = self._build_decoder(
              rnn_outputs_flat, hparams, reuse=True)
        rnn_dec_outputs = _reshape_outputs(rnn_dec_outputs_flat)

        with tf.device(model_helper.get_device_str(1, self.num_gpus)):
          # Compute ae and evolver loss jointly
          loss = self._compute_loss(ae_dec_outputs, rnn_dec_outputs)

      # Inferece
      else:
        # Build convolutional encoder to feed evolver
        with tf.variable_scope("ae"):
          rnn_enc_state = self._build_encoder(hparams, train_ae=False)
          _ = self._build_decoder(rnn_enc_state, hparams)

        # Build evolver RNN
        with tf.variable_scope("evolver"):
          rnn_outputs, _ = self._build_rnn(hparams, rnn_enc_state)

        # Reconstruct the full state from evolved rnn states
        rnn_outputs_flat = tf.reshape(rnn_outputs, (-1, hparams.num_units))
        with tf.variable_scope("ae", reuse=True):
          rnn_dec_outputs_flat = self._build_decoder(rnn_outputs_flat, hparams)
        rnn_dec_outputs = _reshape_outputs(rnn_dec_outputs_flat)

        loss = None

      return loss, rnn_dec_outputs


  def _build_encoder(self, hparams, train_ae=False, reuse=None):
    """Build and run a CNN encoder.

    Args:
      hparams: Hyperparameter configurations.
      train_ae: set to true if training autoencoder, otherwise training evolver
      resuse: set to true if reusing for rnn.

    Returns:
      enc_state: float32 Tensor [num_samples, num_unts], where num_samples
        is either batch_size*num_steps if training autoencoder or batch_size
        if training entire graph
    """
    source = self.iterator.source
    num_samples = self.batch_size
    if train_ae:
      source = tf.reshape(self.iterator.target_output, (-1, self.data_size[0]))
      num_samples, _ = tf.unstack(tf.shape(source))
    source = tf.cast(source, dtype=tf.float32)

    with tf.variable_scope("encoder", reuse=reuse) as scope:

      enc_dense_1 = tf.layers.dense(
          source,
          units=512,
          activation=tf.nn.sigmoid,
          name="enc_dense_1")

      enc_dense_2 = tf.layers.dense(
          enc_dense_1,
          units=256,
          activation=tf.nn.sigmoid,
          name="enc_dense_2")

      # Dropout for better generalization
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        enc_dense_2 = tf.nn.dropout(
            enc_dense_2,
            keep_prob=(1.0-hparams.dropout),
            seed=hparams.random_seed)

      enc_state = tf.layers.dense(
          enc_dense_2,
          units=hparams.num_units,
          activation=tf.nn.sigmoid,
          name="enc_dense_3")

    return enc_state


  def _build_decoder(self, enc_state, hparams, reuse=None):
    """Build and run an RNN decoder.

    Args:
      enc_state: low-dim state to decode to full dim.
      hparams: Hyperparameter configurations.
      resuse: set to true if reusing for rnn.

    Returns:
      outputs: full dim reconstruction
    """

    with tf.variable_scope("decoder", reuse=reuse) as scope:

      dec_dense_1 = tf.layers.dense(
          enc_state,
          units=256,
          activation=tf.nn.sigmoid,
          name="dec_dense_1")

      # Dropout for better generalization
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        dec_dense_1 = tf.nn.dropout(
            dec_dense_1,
            keep_prob=(1.0-hparams.dropout),
            seed=hparams.random_seed)

      dec_dense_2 = tf.layers.dense(
          dec_dense_1,
          units=512,
          activation=tf.nn.sigmoid,
          name="dec_dense_2")

      outputs = tf.layers.dense(
          dec_dense_2,
          units=self.data_size[0],
          activation=tf.nn.sigmoid,
          name="dec_dense_3")

    return outputs


  def _compute_loss(self, ae_out, rnn_out, alpha=0.5, beta=0.5):
    """Compute optimization loss as weighted sum of autoencoder loss and loss
    between autoencoder states and predicted rnn states.

    Args:
      ae_out: autoencoder output; float64 Tensor [batch_size, num_steps, Nx,...]
      rnn_out: rnn output; float32 Tensor [batch_size, num_steps, Nx, ...]
      alpha: autoencoder loss weight; default 0.5
      beta: rnn loss weightl; default 0.5

    Returns:
      loss = alpha*ae_loss + beta*rnn_loss
    """
    epsilon = 0.0001
    reduce_axis = 2 # Reduce sum over spatial axes

    # Autoencoder loss
    ae_tgt = tf.cast(self.iterator.target_output, dtype=tf.float32)
    ae_sqr_diff = tf.squared_difference(ae_tgt, ae_out)
    ae_sqr_l2 = tf.reduce_sum(ae_sqr_diff, axis=reduce_axis)
    ae_norm_factor = tf.reduce_sum(tf.square(ae_tgt), axis=reduce_axis)
    ae_loss = tf.reduce_mean(ae_sqr_l2/(ae_norm_factor+epsilon))

    # RNN output loss
    rnn_sqr_diff = tf.squared_difference(ae_tgt, rnn_out)
    rnn_sqr_l2 = tf.reduce_sum(rnn_sqr_diff, axis=reduce_axis)
    rnn_loss = tf.reduce_mean(rnn_sqr_l2/(ae_norm_factor+epsilon))

    # Weighted sum loss
    loss = alpha*ae_loss + beta*rnn_loss

    return loss


class Model2D(BaseModel):
  """Deep convolutional sequence-to-sequence model.

  This class implements a deep convolutional sequence-to-sequence model for
  3D spatio-temporal data (space1, space2, time).
  """

  def build_graph(self, hparams, scope=None):
    """Creates a convolutional autoencoder for 2D data with an evolver RNN.

    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default is "conv_seq2seq".

    Returns:
      A tuple of the form (loss, outputs)
      where
        loss: total loss
        outputs: float32 Tensor [batch_size, num_steps, data_size]
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    def _reshape_outputs(dec_outputs_flat):
      dec_outputs = tf.reshape(
          dec_outputs_flat,
          (self.batch_size, -1, self.data_size[0], self.data_size[1]))
      return dec_outputs

    with tf.variable_scope(scope or "nMOR", dtype=dtype):
      # Training or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # Build convolutional autoencoder
        with tf.variable_scope("ae"):
          ae_enc_states = self._build_encoder(hparams, train_ae=True)
          ae_dec_outputs_flat = self._build_decoder(ae_enc_states, hparams)
          ae_dec_outputs = _reshape_outputs(ae_dec_outputs_flat)

        # Build convolutional encoder to feed evolver
        with tf.variable_scope("ae", reuse=True):
          rnn_enc_state = self._build_encoder(
              hparams, train_ae=False, reuse=True)

        # Build evolver RNN
        with tf.variable_scope("evolver"):
          rnn_outputs, _ = self._build_rnn(hparams, rnn_enc_state)

        # Reconstruct the full state from evolved rnn states
        rnn_outputs_flat = tf.reshape(rnn_outputs, (-1, hparams.num_units))
        with tf.variable_scope("ae", reuse=True):
          rnn_dec_outputs_flat = self._build_decoder(
              rnn_outputs_flat, hparams, reuse=True)
        rnn_dec_outputs = _reshape_outputs(rnn_dec_outputs_flat)

        with tf.device(model_helper.get_device_str(1, self.num_gpus)):
          # Compute ae and evolver loss jointly
          loss = self._compute_loss(ae_dec_outputs, rnn_dec_outputs)

      # Inferece
      else:
        # Build convolutional encoder to feed evolver
        with tf.variable_scope("ae"):
          rnn_enc_state = self._build_encoder(hparams, train_ae=False)
          _ = self._build_decoder(rnn_enc_state, hparams)

        # Build evolver RNN
        with tf.variable_scope("evolver"):
          rnn_outputs, _ = self._build_rnn(
              hparams, rnn_enc_state)

        # Reconstruct the full state from evolved rnn states
        rnn_outputs_flat = tf.reshape(rnn_outputs, (-1, hparams.num_units))
        with tf.variable_scope("ae", reuse=True):
          rnn_dec_outputs_flat = self._build_decoder(rnn_outputs_flat, hparams)
        rnn_dec_outputs = _reshape_outputs(rnn_dec_outputs_flat)

        loss = None

      return loss, rnn_dec_outputs


  def _build_encoder(self, hparams, train_ae=False, reuse=None):
    """Build and run a CNN encoder.

    Args:
      hparams: Hyperparameter configurations.
      train_ae: set to true if training autoencoder, otherwise training evolver
      resuse: set to true if reusing for rnn.

    Returns:
      enc_state: float32 Tensor [num_samples, num_unts], where num_samples
        is either batch_size*num_steps if training autoencoder or batch_size
        if training entire graph
    """
    source = tf.expand_dims(self.iterator.source, -1)
    num_samples = self.batch_size
    if train_ae:
      source = tf.expand_dims(self.iterator.target_output, -1)
      source = tf.reshape(source, (-1, self.data_size[0], self.data_size[1], 1))
      num_samples, _, _, _ = tf.unstack(tf.shape(source))
    source = tf.cast(source, dtype=tf.float32)

    with tf.variable_scope("encoder", reuse=reuse) as scope:

      # first convolutional block for 128 grid size
      kernel_size = (10,10) if self.data_size[0] == 128 else (5,5)
      strides = (4,4) if self.data_size[0] == 128 else (2,2)

      # First convolutional block
      sconv1 = self._conv2d(
          source,
          filters=4,
          kernel_size=kernel_size,
          strides=strides,
          dilation_rate=(1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_1")

      sconv2 = self._conv2d(
          sconv1,
          filters=8,
          kernel_size=(5, 5),
          strides=(2,2),
          dilation_rate=(1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_2")

      sconv3 = self._conv2d(
          sconv2,
          filters=16,
          kernel_size=(5,5),
          strides=(2,2),
          dilation_rate=(1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_3")

      sconv4 = self._conv2d(
          sconv3,
          filters=32,
          kernel_size=(5,5),
          strides=(2,2),
          dilation_rate=(1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_4")

      conv_output = tf.reshape(sconv4, (num_samples, 512))

      # Dropout for better generalization
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        conv_output = tf.nn.dropout(
            conv_output,
            keep_prob=(1.0-hparams.dropout),
            seed=hparams.random_seed)

      enc_dense_1 = tf.layers.dense(
          conv_output,
          units=256,
          activation=tf.nn.sigmoid,
          name="enc_dense_1")

      enc_state = tf.layers.dense(
          enc_dense_1,
          units=hparams.num_units,
          activation=tf.nn.sigmoid,
          name="enc_dense_2")

    return enc_state


  def _build_decoder(self, enc_state, hparams, reuse=None):
    """Build and run an RNN decoder.

    Args:
      enc_state: low-dim state to decode to full dim.
      hparams: Hyperparameter configurations.
      resuse: set to true if reusing for rnn.

    Returns:
      outputs: full dim reconstruction
    """

    with tf.variable_scope("decoder", reuse=reuse) as scope:

      dec_dense_1 = tf.layers.dense(
          enc_state,
          units=256,
          activation=tf.nn.sigmoid,
          name="dec_dense_1")

      dec_dense_2 = tf.layers.dense(
          dec_dense_1,
          units=512,
          activation=tf.nn.sigmoid,
          name="dec_dense_2")

      # Dropout for better generalization
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        dec_dense_2 = tf.nn.dropout(
            dec_dense_2,
            keep_prob=(1.0-hparams.dropout),
            seed=hparams.random_seed)

      num_samples, _ = tf.unstack(tf.shape(dec_dense_2))
      tconv_input = tf.reshape(dec_dense_2, (num_samples, 4, 4, 32))

      # change strides of the first block to control feature map size
      strides = (4,4) if self.data_size[0] == 128 else (2,2)

      # First "deconvolutional" block
      tconv1 = self._dconv2d(
          tconv_input,
          filters=16,
          kernel_size=(5,5),
          strides=strides,
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_1")

      tconv2 = self._dconv2d(
          tconv1,
          filters=8,
          kernel_size=(5,5),
          strides=(2,2),
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_2")

      tconv3 = self._dconv2d(
          tconv2,
          filters=4,
          kernel_size=(5,5),
          strides=(2,2),
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_3")

      tconv4 = self._dconv2d(
          tconv3,
          filters=1,
          kernel_size=(5,5),
          strides=(2,2),
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_4")

      outputs = tf.reshape(tconv4,
                           (num_samples, self.data_size[0], self.data_size[1]))

    return outputs


  def _compute_loss(self, ae_out, rnn_out, alpha=0.5, beta=0.5):
    """Compute optimization loss as weighted sum of autoencoder loss and loss
    between autoencoder states and predicted rnn states.

    Args:
      ae_out: autoencoder output; float64 Tensor [batch_size, num_steps, Nx,...]
      rnn_out: rnn output; float32 Tensor [batch_size, num_steps, Nx, ...]
      alpha: autoencoder loss weight; default 0.5
      beta: rnn loss weightl; default 0.5

    Returns:
      loss = alpha*ae_loss + beta*rnn_loss
    """
    epsilon = 0.0001
    reduce_axis = [2,3] # Reduce sum over spatial axes

    # Autoencoder loss
    ae_tgt = tf.cast(self.iterator.target_output, dtype=tf.float32)
    ae_sqr_diff = tf.squared_difference(ae_tgt, ae_out)
    ae_sqr_l2 = tf.reduce_sum(ae_sqr_diff, axis=reduce_axis)
    ae_norm_factor = tf.reduce_sum(tf.square(ae_tgt), axis=reduce_axis)
    ae_loss = tf.reduce_mean(ae_sqr_l2/(ae_norm_factor+epsilon))

    # RNN output loss
    rnn_sqr_diff = tf.squared_difference(ae_tgt, rnn_out)
    rnn_sqr_l2 = tf.reduce_sum(rnn_sqr_diff, axis=reduce_axis)
    rnn_loss = tf.reduce_mean(rnn_sqr_l2/(ae_norm_factor+epsilon))

    # Weighted sum loss
    loss = alpha*ae_loss + beta*rnn_loss

    return loss


class ParamVarModel(BaseModel):
  """Deep convolutional recurrent autoencoder for parameter-varying systems.

  This class implements a 3d convolutional neural network for parameter to
  identify the dynamics of and reduce the input data.
  """

  def build_graph(self, hparams, scope=None):
    """Creates a convolutional autoencoder for 2D data with an evolver RNN.

    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default is "conv_seq2seq".

    Returns:
      A tuple of the form (loss, outputs)
      where
        loss: total loss
        outputs: float32 Tensor [batch_size, num_steps, data_size]
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    def _reshape_outputs(dec_outputs_flat):
      dec_outputs = tf.reshape(
          dec_outputs_flat,
          (self.batch_size, -1, self.data_size[0], self.data_size[1]))
      return dec_outputs

    with tf.variable_scope(scope or "nMOR", dtype=dtype):

      # Build convolutional encoder to feed evolver
      with tf.variable_scope("ae"):
        rnn_enc_state = self._build_encoder(hparams)

      # Build evolver RNN
      with tf.variable_scope("evolver"):
        rnn_outputs, _ = self._build_rnn(hparams, rnn_enc_state)

      # Reconstruct the full state from evolved rnn states
      rnn_outputs_flat = tf.reshape(rnn_outputs, (-1, hparams.num_units))
      with tf.variable_scope("ae", reuse=None):
        rnn_dec_outputs_flat = self._build_decoder(rnn_outputs_flat, hparams)
      rnn_dec_outputs = _reshape_outputs(rnn_dec_outputs_flat)

      # Training or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device(model_helper.get_device_str(1, self.num_gpus)):
          # Compute ae and evolver loss jointly
          loss = self._compute_loss(rnn_dec_outputs)
      # Inferece
      else:
        loss = None

      return loss, rnn_dec_outputs


  def _build_encoder(self, hparams, reuse=None):
    """Build and run a CNN encoder.

    Args:
      hparams: Hyperparameter configurations.
      resuse: set to true if reusing for rnn.

    Returns:
      enc_state: float32 Tensor [num_samples, num_unts], where num_samples
        is either batch_size*num_steps if training autoencoder or batch_size
        if training entire graph
    """
    source = tf.expand_dims(self.iterator.source, -1)
    num_samples = self.batch_size
    source = tf.cast(source, dtype=tf.float32)

    with tf.variable_scope("encoder", reuse=reuse) as scope:

      # first convolutional block for 128 grid size
      kernel_size = (2,10,10) if self.data_size[0] == 128 else (2,5,5)
      strides = (1,4,4) if self.data_size[0] == 128 else (1,2,2)

      # First convolutional block
      sconv1 = self._conv3d(
          source,
          filters=4,
          kernel_size=kernel_size,
          strides=strides,
          dilation_rate=(1,1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_1")

      sconv2 = self._conv3d(
          sconv1,
          filters=8,
          kernel_size=(2,5,5),
          strides=(2,2,2),
          dilation_rate=(1,1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_2")

      sconv3 = self._conv3d(
          sconv2,
          filters=16,
          kernel_size=(2,5,5),
          strides=(1,2,2),
          dilation_rate=(1,1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_3")

      sconv4 = self._conv3d(
          sconv3,
          filters=32,
          kernel_size=(2,5,5),
          strides=(2,2,2),
          dilation_rate=(1,1,1),
          batch_norm=False,
          activation=tf.nn.sigmoid,
          name="enc_sconv_4")

      conv_output = tf.reshape(sconv4, (num_samples, 512))

      # Dropout for better generalization
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        conv_output = tf.nn.dropout(
            conv_output,
            keep_prob=(1.0-hparams.dropout),
            seed=hparams.random_seed)

      enc_dense_1 = tf.layers.dense(
          conv_output,
          units=256,
          activation=tf.nn.sigmoid,
          name="enc_dense_1")

      enc_state = tf.layers.dense(
          enc_dense_1,
          units=hparams.num_units,
          activation=tf.nn.sigmoid,
          name="enc_dense_2")

    return enc_state


  def _build_decoder(self, enc_state, hparams, reuse=None):
    """Build and run an RNN decoder.

    Args:
      enc_state: low-dim state to decode to full dim.
      hparams: Hyperparameter configurations.
      resuse: set to true if reusing for rnn.

    Returns:
      outputs: full dim reconstruction
    """

    with tf.variable_scope("decoder", reuse=reuse) as scope:

      dec_dense_1 = tf.layers.dense(
          enc_state,
          units=256,
          activation=tf.nn.sigmoid,
          name="dec_dense_1")

      dec_dense_2 = tf.layers.dense(
          dec_dense_1,
          units=512,
          activation=tf.nn.sigmoid,
          name="dec_dense_2")

      # Dropout for better generalization
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        dec_dense_2 = tf.nn.dropout(
            dec_dense_2,
            keep_prob=(1.0-hparams.dropout),
            seed=hparams.random_seed)

      num_samples, _ = tf.unstack(tf.shape(dec_dense_2))
      tconv_input = tf.reshape(dec_dense_2, (num_samples, 4, 4, 32))

      # change strides of the first block to control feature map size
      strides = (4,4) if self.data_size[0] == 128 else (2,2)

      # First "deconvolutional" block
      tconv1 = self._dconv2d(
          tconv_input,
          filters=16,
          kernel_size=(5,5),
          strides=strides,
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_1")

      tconv2 = self._dconv2d(
          tconv1,
          filters=8,
          kernel_size=(5,5),
          strides=(2,2),
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_2")

      tconv3 = self._dconv2d(
          tconv2,
          filters=4,
          kernel_size=(5,5),
          strides=(2,2),
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_3")

      tconv4 = self._dconv2d(
          tconv3,
          filters=1,
          kernel_size=(5,5),
          strides=(2,2),
          batch_norm=True,
          activation=tf.nn.sigmoid,
          name="dec_tconv_4")

      outputs = tf.reshape(tconv4,
                           (num_samples, self.data_size[0], self.data_size[1]))

    return outputs


  def _compute_loss(self, rnn_out):
    """Compute optimization loss as weighted sum of autoencoder loss and loss
    between autoencoder states and predicted rnn states.

    Args:
      rnn_out: rnn output; float32 Tensor [batch_size, num_steps, Nx, ...]

    Returns:
      loss = alpha*ae_loss + beta*rnn_loss
    """
    epsilon = 0.0001
    reduce_axis = [2,3] # Reduce sum over spatial axes

    # RNN output loss
    rnn_tgt = tf.cast(self.iterator.target_output, dtype=tf.float32)
    rnn_sqr_diff = tf.squared_difference(rnn_tgt, rnn_out)
    rnn_sqr_l2 = tf.reduce_sum(rnn_sqr_diff, axis=reduce_axis)
    rnn_norm_factor = tf.reduce_sum(tf.square(rnn_tgt), axis=reduce_axis)
    rnn_loss = tf.reduce_mean(rnn_sqr_l2/(rnn_norm_factor+epsilon))

    return rnn_loss
