# Copyright 2021, The TensorFlow Federated Authors.
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
"""Yogi optimizer."""

import collections
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning.optimizers import optimizer

_LEARNING_RATE_KEY = 'learning_rate'
_BETA_1_KEY = 'beta_1'
_BETA_2_KEY = 'beta_2'
_EPSILON_KEY = 'epsilon'
_STEP_KEY = 'step'
_ACCUMULATOR_KEY = 'accumulator'
_PRECONDITIONER_KEY = 'preconditioner'


class _Yogi(optimizer.Optimizer):
  """Yogi optimizer, see `build_yogi` for details."""

  def __init__(self,
               learning_rate: float,
               beta_1: float = 0.9,
               beta_2: float = 0.999,
               epsilon: float = 1e-3,
               initial_preconditioner_value=1e-6):
    """Initializes Yogi optimizer."""
    py_typecheck.check_non_negative_float(learning_rate, 'learning rate')
    _check_beta(beta_1)
    _check_beta(beta_2)
    py_typecheck.check_non_negative_float(epsilon, 'epsilon')
    self._lr = learning_rate
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._epsilon = epsilon
    self._initial_preconditioner_value = initial_preconditioner_value

  def initialize(self, specs):
    initial_accumulator = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype), specs)
    initial_preconditioner = tf.nest.map_structure(
        lambda s: tf.ones(s.shape, s.dtype) * self.
        _initial_preconditioner_value, specs)
    state = collections.OrderedDict([
        (_LEARNING_RATE_KEY, self._lr),
        (_BETA_1_KEY, self._beta_1),
        (_BETA_2_KEY, self._beta_2),
        (_EPSILON_KEY, self._epsilon),
        (_STEP_KEY, 0),
        (_ACCUMULATOR_KEY, initial_accumulator),
        (_PRECONDITIONER_KEY, initial_preconditioner),
    ])
    return state

  def next(self, state, weights, gradients):
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    optimizer.check_weights_gradients_match(weights, gradients)
    lr = state[_LEARNING_RATE_KEY]
    beta_1 = state[_BETA_1_KEY]
    beta_2 = state[_BETA_2_KEY]
    epsilon = state[_EPSILON_KEY]
    step = state[_STEP_KEY] + 1
    accumulator = state[_ACCUMULATOR_KEY]
    preconditioner = state[_PRECONDITIONER_KEY]
    optimizer.check_weights_state_match(weights, accumulator, 'accumulator')
    optimizer.check_weights_state_match(weights, preconditioner,
                                        'preconditioner')

    updated_accumulator = tf.nest.map_structure(
        lambda a, g: a + (g - a) * (1 - beta_1), accumulator, gradients)

    def preconditioner_update(s, g):
      g2 = tf.math.square(g)
      sign = tf.sign(g2 - s)
      return s + (1 - beta_2) * sign * g2

    updated_preconditioner = tf.nest.map_structure(preconditioner_update,
                                                   preconditioner, gradients)
    normalized_lr = lr * tf.math.sqrt(
        (1 - tf.math.pow(beta_2, step))) / (1 - tf.math.pow(beta_1, step))
    updated_weights = tf.nest.map_structure(
        lambda w, g, a, s: w - normalized_lr * a / (tf.math.sqrt(s) + epsilon),
        weights, gradients, updated_accumulator, updated_preconditioner)

    updated_state = collections.OrderedDict([
        (_LEARNING_RATE_KEY, lr),
        (_BETA_1_KEY, beta_1),
        (_BETA_2_KEY, beta_2),
        (_EPSILON_KEY, epsilon),
        (_STEP_KEY, step),
        (_ACCUMULATOR_KEY, updated_accumulator),
        (_PRECONDITIONER_KEY, updated_preconditioner),
    ])
    return updated_state, updated_weights


def _check_beta(beta):
  py_typecheck.check_type(beta, float)
  if beta < 0.0 or beta >= 1.0:
    raise ValueError('Beta must be equal to 0.0 or more, and less than 1.0.')


def build_yogi(learning_rate: float,
               beta_1: float = 0.9,
               beta_2: float = 0.999,
               epsilon: float = 1e-3,
               initial_preconditioner_value=1e-6) -> optimizer.Optimizer:
  """Returns a `tff.learning.optimizers.Optimizer` for Yogi.

  The Yogi optimizer is based on [Adaptive methods for nonconvex optimization](
  https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf)

  The update rule given learning rate `lr`, epsilon `eps`, accumulator `acc`,
  preconditioner `s`, iteration `t`, weights `w` and gradients `g` is:

  ```
  acc = beta_1 * acc + (1 - beta_1) * g
  s = s + (1 - beta_2) * sign(g - s) * (g ** 2)
  normalized_lr = lr * sqrt(1 - beta_2**t) / (1 - beta_1**t)
  w = w - normalized_lr * acc / (sqrt(s) + eps)
  ```

  Implementation of Yogi is based on additive updates, as opposed to
  multiplicative updates (as in Adam). Experiments show better performance
  across NLP and Vision tasks both in centralized and federated settings.

  Typically use 10x the learning rate used for Adam.

  Args:
    learning_rate: A positive `float` for learning rate.
    beta_1: A `float` between `0.0` and `1.0` for the decay used to track the
      previous gradients.
    beta_2: A `float` between `0.0` and `1.0` for the decay used to track the
      magnitude (second moment) of previous gradients.
    epsilon: A constant trading off adaptivity and noise..
    initial_preconditioner_value: The starting value for preconditioner. Only
      positive values are allowed.
  """
  return _Yogi(learning_rate, beta_1, beta_2, epsilon,
               initial_preconditioner_value)
