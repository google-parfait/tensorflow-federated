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
"""Gradient descent optimizer."""

import collections
from typing import Optional
import tensorflow as tf

from tensorflow_federated.python.learning.optimizers import optimizer


LEARNING_RATE_KEY = 'lr'
MOMENTUM_KEY = 'momentum'
ACCUMULATOR_KEY = 'accumulator'


class _SGD(optimizer.Optimizer):
  """Gradient descent optimizer, see `build_sgdm` for details."""

  def __init__(self, learning_rate: float, momentum: Optional[float] = None):
    """Initializes SGD optimizer."""
    optimizer.check_non_negative_float(learning_rate, 'learning rate')
    if momentum is not None:
      optimizer.check_momentum(momentum)
    self._lr = learning_rate
    self._momentum = momentum

  def initialize(self, specs):
    state = collections.OrderedDict([(LEARNING_RATE_KEY, self._lr)])
    if self._momentum is not None and self._momentum > 0:
      state[MOMENTUM_KEY] = self._momentum
      state[ACCUMULATOR_KEY] = tf.nest.map_structure(
          lambda s: tf.zeros(s.shape, s.dtype), specs)
    return state

  def next(self, state, weights, gradients):
    optimizer.check_weights_gradients_match(weights, gradients)
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    lr = state[LEARNING_RATE_KEY]

    if MOMENTUM_KEY not in state:
      updated_weights = tf.nest.map_structure(lambda w, g: w - lr * g, weights,
                                              gradients)
      updated_state = collections.OrderedDict([(LEARNING_RATE_KEY, lr)])
    else:
      momentum = state[MOMENTUM_KEY]
      accumulator = state[ACCUMULATOR_KEY]
      optimizer.check_weights_state_match(weights, accumulator, 'accumulator')
      updated_accumulator = tf.nest.map_structure(lambda a, g: momentum * a + g,
                                                  accumulator, gradients)
      updated_weights = tf.nest.map_structure(lambda w, m: w - lr * m, weights,
                                              updated_accumulator)
      updated_state = collections.OrderedDict([
          (LEARNING_RATE_KEY, lr),
          (MOMENTUM_KEY, momentum),
          (ACCUMULATOR_KEY, updated_accumulator),
      ])
    return updated_state, updated_weights


def build_sgdm(learning_rate: float = 0.01,
               momentum: Optional[float] = None) -> _SGD:
  """Returns a `tff.learning.optimizers.Optimizer` for momentum SGD.

  This class supports the simple gradient descent and its variant with momentum.

  If momentum is not used, the update rule given learning rate `lr`, weights `w`
  and gradients `g` is:

  ```
  w = w - lr * g
  ```

  If momentum `m` (a float between `0.0` and `1.0`) is used, the update rule is

  ```
  v = m * v + g
  w = w - lr * v
  ```

  where `v` is the velocity from previous steps of the optimizer.

  Args:
    learning_rate: A positive float for learning rate, default to 0.01.
    momentum: A float between 0.0 and 1.0 for momentum.
  """
  return _SGD(learning_rate=learning_rate, momentum=momentum)
