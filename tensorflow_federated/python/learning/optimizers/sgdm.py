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

from typing import Optional
import tensorflow as tf

from tensorflow_federated.python.learning.optimizers import optimizer


class _SGD(optimizer.Optimizer):
  """Gradient descent optimizer, see `build_sgdm` for details."""

  def __init__(self, learning_rate: float, momentum: Optional[float] = None):
    """Initializes SGD optimizer."""
    optimizer.check_learning_rate(learning_rate)
    if momentum is not None:
      optimizer.check_momentum(momentum)
    self._lr = learning_rate
    self._momentum = momentum

  def initialize(self, specs):
    if self._momentum is None or self._momentum == 0:
      return ()
    else:
      return tf.nest.map_structure(lambda s: tf.zeros(s.shape, s.dtype), specs)

  @tf.function
  def next(self, state, weights, gradients):
    optimizer.check_weights_gradients_match(weights, gradients)
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    if self._momentum is None or self._momentum == 0:
      updated_state = state
      updated_weights = tf.nest.map_structure(lambda w, g: w - self._lr * g,
                                              weights, gradients)
    else:
      _check_momentum_matches_weights(state, weights)
      updated_state = tf.nest.map_structure(lambda m, g: self._momentum * m + g,
                                            state, gradients)
      updated_weights = tf.nest.map_structure(lambda w, m: w - self._lr * m,
                                              weights, updated_state)
    return updated_state, updated_weights


def _check_momentum_matches_weights(state, weights):
  try:
    tf.nest.assert_same_structure(state, weights)
  except (TypeError, ValueError):
    # Raises a more informative error message.
    raise ValueError(
        'Provided state and weigths do not match. The momentum term in state '
        'the and weights must be collections of tensors of the same structure '
        'and the tensors must have the same shapes and dtypes. A possible '
        'reason is that the `initialize` method was invoked with `spect` not '
        f'matching the weights being optimized.\n'
        f'Provided state: {state}\n'
        f'Provided weights: {weights}')


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
