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
"""AdamW optimizer."""

import collections
from typing import Any, TypeVar

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.learning.optimizers import nest_utils
from tensorflow_federated.python.learning.optimizers import optimizer

_BETA_1_KEY = 'beta_1'
_BETA_2_KEY = 'beta_2'
_EPSILON_KEY = 'epsilon'
_STEP_KEY = 'step'
_ACCUMULATOR_KEY = 'accumulator'
_PRECONDITIONER_KEY = 'preconditioner'
_WEIGHT_DECAY_KEY = 'weight_decay'
_HPARAMS_KEYS = [
    optimizer.LEARNING_RATE_KEY,
    _BETA_1_KEY,
    _BETA_2_KEY,
    _EPSILON_KEY,
    _WEIGHT_DECAY_KEY,
]

State = TypeVar('State', bound=collections.OrderedDict[str, Any])
Hparams = TypeVar('Hparams', bound=collections.OrderedDict[str, float])


class _AdamW(optimizer.Optimizer[State, optimizer.Weights, Hparams]):
  """AdamW optimizer, see `build_adamw` for details."""

  def __init__(
      self,
      learning_rate: optimizer.Float,
      beta_1: optimizer.Float = 0.9,
      beta_2: optimizer.Float = 0.999,
      epsilon: optimizer.Float = 1e-7,
      weight_decay: optimizer.Float = 0.004,  # tf.keras default
  ):
    """Initializes AdamW optimizer."""
    if not tf.is_symbolic_tensor(learning_rate) and learning_rate < 0.0:
      raise ValueError(
          f'AdamW `learning_rate` must be nonnegative, found {learning_rate}.'
      )
    if not tf.is_symbolic_tensor(beta_1) and (beta_1 < 0.0 or beta_1 > 1.0):
      raise ValueError(
          f'AdamW `beta_1` must be in the range [0.0, 1.0], found {beta_1}.'
      )
    if not tf.is_symbolic_tensor(beta_2) and (beta_2 < 0.0 or beta_2 > 1.0):
      raise ValueError(
          f'AdamW `beta_2` must be in the range [0.0, 1.0], found {beta_2}.'
      )
    if not tf.is_symbolic_tensor(epsilon) and epsilon < 0.0:
      raise ValueError(f'AdamW `epsilon` must be nonnegative, found {epsilon}.')
    if weight_decay < 0.0:
      raise ValueError(
          f'AdamW `weight_decay` must be nonnegative, found {weight_decay}.'
      )
    self._lr = learning_rate
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._epsilon = epsilon
    self._weight_decay = weight_decay

  def initialize(self, specs: Any) -> State:
    initial_accumulator = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype), specs
    )
    initial_preconditioner = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype), specs
    )
    state = collections.OrderedDict([
        (optimizer.LEARNING_RATE_KEY, self._lr),
        (_BETA_1_KEY, self._beta_1),
        (_BETA_2_KEY, self._beta_2),
        (_EPSILON_KEY, self._epsilon),
        (_STEP_KEY, 0),
        (_ACCUMULATOR_KEY, initial_accumulator),
        (_PRECONDITIONER_KEY, initial_preconditioner),
        (_WEIGHT_DECAY_KEY, self._weight_decay),
    ])
    return state

  def next(
      self, state: State, weights: optimizer.Weights, gradients: Any
  ) -> tuple[State, optimizer.Weights]:
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    optimizer.check_weights_gradients_match(weights, gradients)
    lr = state[optimizer.LEARNING_RATE_KEY]
    beta_1 = state[_BETA_1_KEY]
    beta_2 = state[_BETA_2_KEY]
    epsilon = state[_EPSILON_KEY]
    step = state[_STEP_KEY] + 1
    accumulator = state[_ACCUMULATOR_KEY]
    preconditioner = state[_PRECONDITIONER_KEY]
    weight_decay = state[_WEIGHT_DECAY_KEY]
    optimizer.check_weights_state_match(weights, accumulator, 'accumulator')
    optimizer.check_weights_state_match(
        weights, preconditioner, 'preconditioner'
    )

    if tf.is_tensor(beta_1):
      casted_step = tf.cast(step, beta_1.dtype)
    else:
      casted_step = step

    normalization = tf.math.sqrt((1.0 - tf.math.pow(beta_2, casted_step))) / (
        1.0 - tf.math.pow(beta_1, casted_step)
    )

    def _adamw_update(w, a, p, g):
      if g is None:
        return w, a, p
      a = a + (g - a) * (1.0 - tf.cast(beta_1, g.dtype))
      p = p + (tf.math.square(g) - p) * (1.0 - tf.cast(beta_2, g.dtype))
      w = w - tf.cast(lr, g.dtype) * (
          tf.cast(normalization, g.dtype)
          * a
          / (tf.math.sqrt(p) + tf.cast(epsilon, g.dtype))
          + weight_decay * w
      )
      return w, a, p

    updated_weights, updated_accumulator, updated_preconditioner = (
        nest_utils.map_at_leaves(
            _adamw_update,
            weights,
            accumulator,
            preconditioner,
            gradients,
            # We have to tell `map_at_leaves` how many outputs to yield in case
            # `weights` has no leaves.
            num_outputs=3,
        )
    )

    updated_state = collections.OrderedDict([
        (optimizer.LEARNING_RATE_KEY, lr),
        (_BETA_1_KEY, beta_1),
        (_BETA_2_KEY, beta_2),
        (_EPSILON_KEY, epsilon),
        (_STEP_KEY, step),
        (_ACCUMULATOR_KEY, updated_accumulator),
        (_PRECONDITIONER_KEY, updated_preconditioner),
        (_WEIGHT_DECAY_KEY, weight_decay),
    ])
    return updated_state, updated_weights

  def get_hparams(self, state: State) -> Hparams:
    return collections.OrderedDict([(k, state[k]) for k in _HPARAMS_KEYS])

  def set_hparams(self, state: State, hparams: Hparams) -> State:
    # TODO: b/245962555 - Find an alternative to `update_struct` if it
    # interferes with typing guarantees.
    # We use `tff.structure.update_struct` (rather than something like
    # `copy.deepcopy`) to ensure that this can be called within a
    # `tff.Computation`.
    return structure.update_struct(state, **hparams)


def build_adamw(
    learning_rate: optimizer.Float,
    beta_1: optimizer.Float = 0.9,
    beta_2: optimizer.Float = 0.999,
    epsilon: optimizer.Float = 1e-7,
    weight_decay: optimizer.Float = 0.004,  # tf.keras default
) -> optimizer.Optimizer:
  """Returns a `tff.learning.optimizers.Optimizer` for AdamW.

  The AdamW optimizer is based on [Decoupled Weight Decay
  Regularization](https://arxiv.org/abs/1711.05101)

  The update rule given learning rate `lr`, epsilon `eps`, accumulator `acc`,
  preconditioner `s`, weigh decay `lambda`, iteration `t`, weights `w` and
  gradients `g` is:

  ```
  acc = beta_1 * acc + (1 - beta_1) * g
  s = beta_2 * s + (1 - beta_2) * g**2
  normalization = sqrt(1 - beta_2**t) / (1 - beta_1**t)
  w = w - lr * (normalization * acc / (sqrt(s) + eps) + lambda * w)
  ```

  Args:
    learning_rate: A positive `float` for learning rate.
    beta_1: A `float` between `0.0` and `1.0` for the decay used to track the
      previous gradients.
    beta_2: A `float` between `0.0` and `1.0` for the decay used to track the
      magnitude (second moment) of previous gradients.
    epsilon: A small non-negative `float`, used to maintain numerical stability.
    weight_decay: A non-negative `float`, governing the amount of weight decay.
      When set to 0, this recovers Adam.
  """
  return _AdamW(learning_rate, beta_1, beta_2, epsilon, weight_decay)
