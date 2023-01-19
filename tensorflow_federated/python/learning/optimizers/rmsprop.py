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
"""RMSprop optimizer."""

import collections
from typing import Any, TypeVar

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.learning.optimizers import optimizer

_DECAY_KEY = 'decay'
_PRECONDITIONER_KEY = 'preconditioner'
_EPSILON_KEY = 'epsilon'
_HPARAMS_KEYS = [optimizer.LEARNING_RATE_KEY, _DECAY_KEY, _EPSILON_KEY]

State = TypeVar('State', bound=collections.OrderedDict[str, Any])
Hparams = TypeVar('Hparams', bound=collections.OrderedDict[str, float])


class _RmsProp(optimizer.Optimizer[State, optimizer.Weights, Hparams]):
  """RMSprop optimizer, see `build_rmsprop` for details."""

  def __init__(
      self, learning_rate: float, decay: float = 0.9, epsilon: float = 1e-7
  ):
    """Initializes RMSprop optimizer."""
    py_typecheck.check_non_negative_float(learning_rate, 'learning rate')
    _check_decay(decay)
    py_typecheck.check_non_negative_float(epsilon, 'epsilon')
    self._lr = learning_rate
    self._decay = decay
    self._epsilon = epsilon

  def initialize(self, specs: Any) -> State:
    initial_preconditioner = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype), specs
    )
    state = collections.OrderedDict([
        (optimizer.LEARNING_RATE_KEY, self._lr),
        (_DECAY_KEY, self._decay),
        (_EPSILON_KEY, self._epsilon),
        (_PRECONDITIONER_KEY, initial_preconditioner),
    ])
    return state

  def next(
      self, state: State, weights: optimizer.Weights, gradients: Any
  ) -> tuple[State, optimizer.Weights]:
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    optimizer.check_weights_gradients_match(weights, gradients)
    lr = state[optimizer.LEARNING_RATE_KEY]
    decay = state[_DECAY_KEY]
    epsilon = state[_EPSILON_KEY]
    preconditioner = state[_PRECONDITIONER_KEY]
    optimizer.check_weights_state_match(
        weights, preconditioner, 'preconditioner'
    )

    updated_preconditioner = tf.nest.map_structure(
        lambda p, g: p + (tf.math.square(g) - p) * (1 - decay),
        preconditioner,
        gradients,
    )
    updated_weights = tf.nest.map_structure(
        lambda w, g, p: w - lr * g / (tf.math.sqrt(p) + epsilon),
        weights,
        gradients,
        updated_preconditioner,
    )

    updated_state = collections.OrderedDict([
        (optimizer.LEARNING_RATE_KEY, lr),
        (_DECAY_KEY, decay),
        (_EPSILON_KEY, epsilon),
        (_PRECONDITIONER_KEY, updated_preconditioner),
    ])
    return updated_state, updated_weights

  def get_hparams(self, state: State) -> Hparams:
    return collections.OrderedDict([(k, state[k]) for k in _HPARAMS_KEYS])

  def set_hparams(self, state: State, hparams: Hparams) -> State:
    # TODO(b/245962555): Find an alternative to `update_struct` if it interferes
    # with typing guarantees.
    # We use `tff.structure.update_struct` (rather than something like
    # `copy.deepcopy`) to ensure that this can be called within a
    # `tff.Computation`.
    return structure.update_struct(state, **hparams)


def _check_decay(decay):
  py_typecheck.check_type(decay, float)
  if decay < 0.0 or decay >= 1.0:
    raise ValueError('Decay must be equal to 0.0 or more, and less than 1.0.')


def build_rmsprop(
    learning_rate: float, decay: float = 0.9, epsilon: float = 1e-7
) -> optimizer.Optimizer:
  """Returns a `tff.learning.optimizers.Optimizer` for RMSprop.

  The RMSprop optimizer is based on [Tieleman and Hinton, 2012](
  http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf).

  The update rule given learning rate `lr`, epsilon `eps`, decay `d`,
  preconditioner `s`, weights `w` and gradients `g` is:

  ```
  s = d * s + (1 - d) * g**2
  w = w - lr * g / (sqrt(s) + eps)
  ```

  Args:
    learning_rate: A positive float for learning rate, default to 0.01.
    decay: A float between 0.0 and 1.0 for the decay used to track the magnitude
      of previous gradients.
    epsilon: A small non-negative float, used to maintain numerical stability.
  """
  return _RmsProp(learning_rate, decay, epsilon)
