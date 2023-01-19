# Copyright 2022, The TensorFlow Federated Authors.
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
"""Helpers for learning rate scheduling."""

import collections
from collections.abc import Callable

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base

_LEARNING_RATE_KEY = optimizer_base.LEARNING_RATE_KEY


def schedule_learning_rate(
    optimizer: optimizer_base.Optimizer, schedule_fn: Callable[[int], float]
) -> optimizer_base.Optimizer:
  """Returns an optimizer with scheduled learning rate.

  The returned optimizer will use a learning rate of `schedule_fn(i)` for the
  `i`-th invocation of its `next` method, indexing from 0.

  Args:
    optimizer: A `tff.learning.optimizers.Optimizer` which uses a learning rate.
    schedule_fn: A callable mapping integer round number to a floating point
      learning rate. To be invoked in the cotext of a `tff.tf_computation`, thus
      should support a `tf.Tensor` input.

  Returns:
    A `tff.learning.optimizers.Optimizer`.

  Raises:
    KeyError: If the provided `optimizer`'s state is not a dictionary with
      learning rate stored under the `tff.learning.optimizers.LEARNING_RATE_KEY`
      key.
  """
  return _ScheduledLROptimizer(optimizer, schedule_fn)


class _ScheduledLROptimizer(optimizer_base.Optimizer):
  """Optimizer with scheduled learning rate."""

  def __init__(
      self,
      optimizer: optimizer_base.Optimizer,
      schedule_fn: Callable[[int], float],
  ):
    py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
    py_typecheck.check_callable(schedule_fn)
    self._optimizer = optimizer
    self._schedule_fn = schedule_fn

  def initialize(self, specs):
    optimizer_state = self._optimizer.initialize(specs)
    _check_lr_exists(optimizer_state)
    round_num = tf.constant(0, tf.int32)
    optimizer_state[_LEARNING_RATE_KEY] = self._schedule_fn(round_num)
    return collections.OrderedDict(
        round_num=round_num, optimizer=optimizer_state
    )

  def next(self, state, weights, gradients):
    optimizer_state, weights = self._optimizer.next(
        state['optimizer'], weights, gradients
    )
    round_num = state['round_num'] + 1
    optimizer_state[_LEARNING_RATE_KEY] = self._schedule_fn(round_num)
    new_state = collections.OrderedDict(
        round_num=round_num, optimizer=optimizer_state
    )
    return new_state, weights


def _check_lr_exists(optimizer_state):
  if _LEARNING_RATE_KEY not in optimizer_state:
    raise KeyError(
        'Optimizer to be scheduled must have learning rate under '
        '`tff.learning.optimizer.LEARNING_RATE_KEY` key in its state. Found '
        f'optimizer state: {optimizer_state}'
    )
