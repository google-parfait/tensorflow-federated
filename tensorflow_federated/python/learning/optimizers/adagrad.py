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
"""Adagrad optimizer."""

import collections
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning.optimizers import optimizer

_LEARNING_RATE_KEY = 'learning_rate'
_PRECONDITIONER_KEY = 'preconditioner'
_EPSILON_KEY = 'epsilon'


class _Adagrad(optimizer.Optimizer):
  """Adagrad optimizer, see `build_adagrad` for details."""

  def __init__(self,
               learning_rate: float,
               initial_preconditioner_value: float = 0.1,
               epsilon: float = 1e-7):
    """Initializes SGD optimizer."""
    py_typecheck.check_non_negative_float(learning_rate, 'learning rate')
    py_typecheck.check_non_negative_float(initial_preconditioner_value,
                                          'initial preconditioner value')
    py_typecheck.check_non_negative_float(epsilon, 'epsilon')
    self._lr = learning_rate
    self._initial_precond = initial_preconditioner_value
    self._epsilon = epsilon

  def initialize(self, specs):
    initial_preconditioner = tf.nest.map_structure(
        lambda s: tf.ones(s.shape, s.dtype) * self._initial_precond, specs)
    state = collections.OrderedDict([
        (_LEARNING_RATE_KEY, self._lr),
        (_EPSILON_KEY, self._epsilon),
        (_PRECONDITIONER_KEY, initial_preconditioner),
    ])
    return state

  def next(self, state, weights, gradients):
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    optimizer.check_weights_gradients_match(weights, gradients)
    lr = state[_LEARNING_RATE_KEY]
    epsilon = state[_EPSILON_KEY]
    preconditioner = state[_PRECONDITIONER_KEY]
    optimizer.check_weights_state_match(weights, preconditioner,
                                        'preconditioner')

    updated_preconditioner = tf.nest.map_structure(
        lambda a, g: a + tf.math.square(g), preconditioner, gradients)
    updated_weights = tf.nest.map_structure(
        lambda w, g, a: w - lr * g / tf.math.sqrt(a + epsilon), weights,
        gradients, updated_preconditioner)

    updated_state = collections.OrderedDict([
        (_LEARNING_RATE_KEY, lr),
        (_EPSILON_KEY, epsilon),
        (_PRECONDITIONER_KEY, updated_preconditioner),
    ])
    return updated_state, updated_weights


def build_adagrad(learning_rate: float,
                  initial_preconditioner_value: float = 0.1,
                  epsilon: float = 1e-7) -> optimizer.Optimizer:
  """Returns a `tff.learning.optimizers.Optimizer` for Adagrad.

  The Adagrad optimizer is based on [Adaptive Subgradient Methods for Online
  Learning and Stochastic Optimization](
  https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

  The update rule given learning rate `lr`, epsilon `eps`, preconditioner `s`,
  weights `w` and gradients `g` is:

  ```
  s = s + g**2
  w = w - lr * g / sqrt(s + eps)
  ```

  Args:
    learning_rate: A positive float for learning rate.
    initial_preconditioner_value: A non-negative float, initial value for the
      preconditioner.
    epsilon: A small non-negative float, used to maintain numerical stability.
  """
  return _Adagrad(learning_rate, initial_preconditioner_value, epsilon)
