# Copyright 2024, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adafactor optimizer."""

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.learning.optimizers import optimizer

_BETA_2_DECAY_KEY = 'beta_2_decay'
_EPSILON_1_KEY = 'epsilon_1'
_EPSILON_2_KEY = 'epsilon_2'
_CLIP_THRESHOLD_KEY = 'clip_threshold'
_RELATIVE_STEP_KEY = 'relative_step'

_NestedTensorSpecs = Union[
    Mapping[str, '_NestedTensorSpecs'],
    Sequence['_NestedTensorSpecs'],
    tf.TensorSpec,
]
_NestedTensors = Union[
    Mapping[str, '_NestedTensorSpecs'],
    Sequence['_NestedTensorSpecs'],
    tf.TensorSpec,
]


class _AdaFactorMoment(NamedTuple):
  """An internal representation of factorized moment (state for optimizer)."""

  r: tf.Tensor
  c: tf.Tensor
  v: tf.Tensor


class _AdafactorOptimizer(
    optimizer.Optimizer[optimizer.State, optimizer.Weights, optimizer.Hparams]
):
  """Adafactor optimizer, see `build_adafactor` for details."""

  def __init__(
      self,
      learning_rate: optimizer.Float,
      beta_2_decay: optimizer.Float,
      epsilon_1: optimizer.Float,
      epsilon_2: optimizer.Float,
      clip_threshold: optimizer.Float,
      relative_step: bool,
  ):
    """Initializes Adafactor optimizer."""
    if learning_rate < 0.0:
      raise ValueError(
          'Adafactor must have a learning_rate greater than 0.0, got'
          f' {learning_rate=}'
      )
    self._learning_rate = learning_rate
    self._beta_2_decay = beta_2_decay
    self._epsilon_1 = epsilon_1
    self._epsilon_2 = epsilon_2
    self._clip_threshold = clip_threshold
    self._relative_step = relative_step

  def initialize(self, specs: _NestedTensorSpecs) -> optimizer.State:
    def initialize_moment(tensor_spec: tf.TensorSpec):
      # Only factor moments for tensors with 2 or more dimensions.
      if tensor_spec.shape.rank < 2:
        r_shape = (0,)
        c_shape = (0,)
      else:
        r_shape = tensor_spec.shape[:-1]
        c_shape = tensor_spec.shape[:-2] + tensor_spec.shape[-1]
      r = tf.zeros(shape=r_shape, dtype=tensor_spec.dtype)
      c = tf.zeros(shape=c_shape, dtype=tensor_spec.dtype)
      v = tf.zeros(shape=tensor_spec.shape, dtype=tensor_spec.dtype)
      return _AdaFactorMoment(r, c, v)

    return {
        'steps': 0,
        'moments': tuple(
            initialize_moment(spec) for spec in tf.nest.flatten(specs)
        ),
        'hparams': {
            optimizer.LEARNING_RATE_KEY: self._learning_rate,
            _BETA_2_DECAY_KEY: self._beta_2_decay,
            _EPSILON_1_KEY: self._epsilon_1,
            _EPSILON_2_KEY: self._epsilon_2,
            _CLIP_THRESHOLD_KEY: self._clip_threshold,
            _RELATIVE_STEP_KEY: self._relative_step,
        },
    }

  def next(
      self,
      state: optimizer.State,
      weights: optimizer.Weights,
      gradients: _NestedTensors,
  ) -> tuple[optimizer.State, optimizer.Weights]:
    local_step = tf.cast(state['steps'] + 1, dtype=tf.float32)
    hparams = self.get_hparams(state)
    lr = hparams[optimizer.LEARNING_RATE_KEY]
    beta_2_decay = hparams[_BETA_2_DECAY_KEY]
    epsilon_1 = hparams[_EPSILON_1_KEY]
    epsilon_2 = hparams[_EPSILON_2_KEY]
    clip_threshold = hparams[_CLIP_THRESHOLD_KEY]
    one = tf.constant(1.0, dtype=tf.float32)

    relative_step = hparams[_RELATIVE_STEP_KEY]
    if relative_step:
      lr = tf.math.minimum(
          lr, tf.math.rsqrt(tf.cast(local_step, dtype=tf.float32))
      )
    rho_t = tf.math.minimum(
        lr, 1.0 / tf.math.rsqrt(tf.cast(local_step, dtype=tf.float32))
    )

    def _rms(t: tf.Tensor) -> tf.Tensor:
      return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(t)))

    def update(
        state: _AdaFactorMoment,
        weight: tf.Tensor,
        gradient: Union[tf.Tensor, None],
    ) -> tuple[_AdaFactorMoment, tf.Tensor]:
      if gradient is None:
        return state, weight
      alpha_t = tf.math.maximum(epsilon_2, _rms(weight)) * rho_t
      regulated_gradient_squared = tf.math.square(gradient) + epsilon_1
      beta_2_t = 1.0 - tf.math.pow(local_step, beta_2_decay)
      if weight.shape.rank < 2:
        new_r = state.r
        new_c = state.c
        new_v = (
            beta_2_t * state.v + (1.0 - beta_2_t) * regulated_gradient_squared
        )
      else:
        new_r = beta_2_t * state.r + (1.0 - beta_2_t) * tf.reduce_mean(
            regulated_gradient_squared, axis=-1
        )
        new_c = beta_2_t * state.c + (1.0 - beta_2_t) * tf.reduce_mean(
            regulated_gradient_squared, axis=-2
        )
        new_v = tf.expand_dims(
            new_r / tf.reduce_mean(new_r, axis=-1, keepdims=1), axis=-1
        ) * tf.expand_dims(new_c, axis=-2)
      new_moment = _AdaFactorMoment(r=new_r, c=new_c, v=new_v)
      u_t = gradient * tf.math.rsqrt(new_moment.v)
      u_t_hat = u_t / tf.maximum(one, (_rms(u_t) / clip_threshold))
      new_weight = weight + -alpha_t * u_t_hat
      return new_moment, new_weight

    if not tf.nest.flatten(weights):
      new_moments = state['moments']
      new_weights = weights
    else:
      new_moments, new_weights = zip(
          *tuple(
              update(moment, weight, gradient)
              for moment, weight, gradient in zip(
                  state['moments'],
                  tf.nest.flatten(weights),
                  tf.nest.flatten(gradients),
              )
          )
      )
      new_weights = tf.nest.pack_sequence_as(weights, new_weights)

    return {
        'steps': local_step,
        'moments': new_moments,
        'hparams': hparams,
    }, new_weights

  def get_hparams(self, state: optimizer.State) -> optimizer.Hparams:
    return state['hparams']

  def set_hparams(
      self, state: optimizer.State, hparams: dict[str, optimizer.Float]
  ) -> optimizer.State:
    # We use `tff.structure.update_struct` (rather than something like
    # `copy.deepcopy`) to ensure that this can be called within a
    # `tff.Computation`.
    return structure.update_struct(state['hparams'], **hparams)


def build_adafactor(
    learning_rate: optimizer.Float,
    *,
    beta_2_decay: optimizer.Float = -0.8,
    epsilon_1: optimizer.Float = 1e-30,
    epsilon_2: optimizer.Float = 1e-3,
    clip_threshold: optimizer.Float = 1.0,
    relative_step: bool = True,
) -> optimizer.Optimizer:
  """Builds an Adafactor optimizer.

  An implementation of Adafactor from Shazeer, Noam et al described in
  https://arxiv.org/abs/1804.04235.

  Args:
    learning_rate: Initial value of the learning rate.
    beta_2_decay: The decay rate of `beta_2`.
    epsilon_1: A small offset to keep denomiantor away from zero.
    epsilon_2: A small offset to avoid learning rate becoming two small over
      time.
    clip_threshold: The clipping threshold of the Adafactor algorithm.
    relative_step: If `True`, learning rate is adjusted based on number of
      iterations. This is the default Adafactor learning rate decay.

  Returns:
    A `tff.learning.optimizers.Optimizer` that implements the Adafactor
    optimizer.
  """
  return _AdafactorOptimizer(
      learning_rate,
      beta_2_decay=beta_2_decay,
      epsilon_1=epsilon_1,
      epsilon_2=epsilon_2,
      clip_threshold=clip_threshold,
      relative_step=relative_step,
  )
