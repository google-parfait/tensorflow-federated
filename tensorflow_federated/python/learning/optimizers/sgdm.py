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
from typing import Any, Optional, TypeVar

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.learning.optimizers import nest_utils
from tensorflow_federated.python.learning.optimizers import optimizer

_MOMENTUM_KEY = 'momentum'
_ACCUMULATOR_KEY = 'accumulator'

State = TypeVar('State', bound=collections.OrderedDict[str, Any])
Hparams = TypeVar('Hparams', bound=collections.OrderedDict[str, Any])


class _SGD(optimizer.Optimizer[State, optimizer.Weights, Hparams]):
  """Gradient descent optimizer, see `build_sgdm` for details."""

  def __init__(
      self,
      learning_rate: optimizer.Float,
      momentum: Optional[optimizer.Float] = None,
  ):
    """Initializes SGD optimizer."""
    if not tf.is_symbolic_tensor(learning_rate) and learning_rate < 0.0:
      raise ValueError(
          f'SGD `learning_rate` must be nonnegative, found {learning_rate}.'
      )
    if momentum:
      # We should only track momentum as a hparam in the case that it is both
      # specified and nonzero.
      if not tf.is_symbolic_tensor(momentum) and (
          momentum < 0.0 or momentum > 1.0
      ):
        raise ValueError(
            'SGD `momentum` must be `None` or in the range [0, 1], found '
            f'{momentum}.'
        )
      self._hparams_keys = [optimizer.LEARNING_RATE_KEY, _MOMENTUM_KEY]
    else:
      self._hparams_keys = [optimizer.LEARNING_RATE_KEY]
    self._lr = learning_rate
    self._momentum = momentum

  def initialize(self, specs: Any) -> State:
    state = collections.OrderedDict([(optimizer.LEARNING_RATE_KEY, self._lr)])
    if self._momentum is not None and self._momentum > 0:
      state[_MOMENTUM_KEY] = self._momentum
      state[_ACCUMULATOR_KEY] = tf.nest.map_structure(
          lambda s: tf.zeros(s.shape, s.dtype), specs
      )
    return state

  def next(
      self, state: State, weights: optimizer.Weights, gradients: Any
  ) -> tuple[State, optimizer.Weights]:
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    optimizer.check_weights_gradients_match(weights, gradients)
    lr = state[optimizer.LEARNING_RATE_KEY]

    if _MOMENTUM_KEY not in state:

      def _sgd_update(w, g):
        if g is None:
          return w
        return w - tf.cast(lr, dtype=g.dtype) * g

      updated_weights = tf.nest.map_structure(
          _sgd_update,
          weights,
          gradients,
      )
      updated_state = collections.OrderedDict(
          [(optimizer.LEARNING_RATE_KEY, lr)]
      )
    else:
      momentum = state[_MOMENTUM_KEY]
      accumulator = state[_ACCUMULATOR_KEY]
      optimizer.check_weights_state_match(weights, accumulator, 'accumulator')

      def _sgdm_update(w, a, g):
        if g is None:
          return w, a
        a = momentum * a + g
        w = w - lr * a
        return w, a

      updated_weights, updated_accumulator = nest_utils.map_at_leaves(
          _sgdm_update,
          weights,
          accumulator,
          gradients,
          # We have to tell `map_at_leaves` how many outputs to yield in case
          # `weights` has no leaves.
          num_outputs=2,
      )
      updated_state = collections.OrderedDict([
          (optimizer.LEARNING_RATE_KEY, lr),
          (_MOMENTUM_KEY, momentum),
          (_ACCUMULATOR_KEY, updated_accumulator),
      ])
    return updated_state, updated_weights

  def get_hparams(self, state: State) -> Hparams:
    return collections.OrderedDict([(k, state[k]) for k in self._hparams_keys])

  def set_hparams(self, state: State, hparams: Hparams) -> State:
    return structure.update_struct(state, **hparams)


def build_sgdm(
    learning_rate: optimizer.Float = 0.01,
    momentum: Optional[optimizer.Float] = None,
) -> optimizer.Optimizer:
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
    momentum: An optional float between 0.0 and 1.0. If `None`, no momentum is
      used.
  """
  return _SGD(learning_rate=learning_rate, momentum=momentum)
