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
"""Optimizer adapter for Keras optimizer."""

from collections.abc import Callable
from typing import Any, Optional, Union

import tensorflow as tf

from tensorflow_federated.python.learning.optimizers import optimizer


class KerasOptimizer(optimizer.Optimizer):
  """Adapter for keras optimzier as `tff.learning.optimizers.Optimizer`.

  This class is expected to be instantiated in the context of
  `tff.tensorflow.computation` in which it is to be used. This is because the
  `optimizer_fn` provided in constructor is going to be invoked, which will
  create a keras optimizer instance, and we will force the creation of
  `tf.Variable` objects which store the state of that keras optimizer.

  If this class is supposed to be used as a "server optimizer", set
  `disjoint_init_and_next` to True, which means that the `initialize` and `next`
  methods are going to be invoked in the context of *different*
  `tff.tensorflow.computation`s, and TFF needs to carry the optimizer variables
  between the invocations.

  If this class is supposed to be used as a "client optimizer", set
  `disjoint_init_and_next` to False, which means that the `initialize` and
  `next` methods are going to be invoked in the context of *the same*
  `tff.tensorflow.computation` and we don't need to pass the variables of keras
  optimizer to TFF to handle.

  NOTE: This class is not meant to be exposed in public API for now. Rather, it
  is used to convert the previous default support for keras optimizers to the
  tff.learning.optimizers format.
  """

  def __init__(
      self,
      optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
      weights: Any,
      disjoint_init_and_next: bool,
  ):
    """Initializes `KerasOptimizer`.

    Args:
      optimizer_fn: A no-arg callable that creates and returns a
        `tf.keras.optimizers.Optimizer`.
      weights: A (possibly nested) structure of `tf.Variable` objects which are
        supposed to be modified during call to the `next` method of the
        optimizer.
      disjoint_init_and_next: A boolean, determining whether the `initialize`
        and `next` methods are going to be invoked in the context of the same
        `tff.tensorflow.computation`.
    """
    self._optimizer = optimizer_fn()
    self._disjoint_init_and_next = disjoint_init_and_next

    def mock_apply_gradients(opt, variables):
      opt.apply_gradients(
          [(tf.zeros_like(w), w) for w in tf.nest.flatten(variables)]
      )

    # Force the creation of tf.Variables controlled by the keras optimizer but
    # keep the variables unmodified. For instance, the "step" variable will be
    # 0, not 1, after this operation.
    tf.function(mock_apply_gradients).get_concrete_function(
        self._optimizer, weights
    )

  def initialize(self, specs):
    del specs  # Unused.
    if self._disjoint_init_and_next:
      return self._optimizer.variables()
    else:
      return ()

  def next(self, state, weights, gradients):
    if self._disjoint_init_and_next:
      tf.nest.map_structure(
          lambda v, s: v.assign(s), self._optimizer.variables(), state
      )

    self._optimizer.apply_gradients(
        list(zip(tf.nest.flatten(gradients), tf.nest.flatten(weights)))
    )

    if self._disjoint_init_and_next:
      return self._optimizer.variables(), weights
    else:
      return (), weights


def build_or_verify_tff_optimizer(
    optimizer_fn: Union[
        Callable[[], tf.keras.optimizers.Optimizer], optimizer.Optimizer
    ],
    trainable_weights: Optional[Any] = None,
    disjoint_init_and_next: Optional[bool] = None,
) -> optimizer.Optimizer:
  """Returns `tff.learning.optimizers.Optimizer` for `optimizer_fn`.

  This helper function is used for `tff.learning` to provide backward
  compatibility of accepting an argument of a no-arg callable returns a
  `tf.keras.optimizers.Optimizer`. Keras optimizer has to be eagerly created in
  each TFF computation function. If the input `optimizer_fn` is already
  a `tff.learning.optimizers.Optimizer`, it will be directly returned.

  Args:
    optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-argument
      callable that constructs and returns a `tf.keras.optimizers.Optimizer`.
    trainable_weights: Optional if `optimizer_fn` is a
      `tff.learning.optimizers.Optimizer`. A (possibly nested) structure of
      `tf.Variable` objects used to eagerly initialize Keras optimizers if
      `optimizer_fn` is a callable.
    disjoint_init_and_next: Optional if `optimizer_fn` is a
      `tff.learning.optimizers.Optimizer`. A boolean, determining whether the
      `initialize` and `next` methods are going to be invoked in the context of
      the same `tff.tensorflow.computation` if `optimizer_fn` is a callable.

  Raises:
    TypeError: Input `optimizer_fn` is not `tff.learning.optimizers.Optimizer`
      or a callable.

  Returns:
    A `tff.learning.optimizers.Optimizer`.
  """
  if isinstance(optimizer_fn, optimizer.Optimizer):
    return optimizer_fn
  elif callable(optimizer_fn):
    return KerasOptimizer(
        optimizer_fn, trainable_weights, disjoint_init_and_next
    )
  else:
    raise TypeError(
        '`optimizer_fn` must be a callable or '
        f'`tff.learning.optimizers.Optimizer`, got {type(optimizer_fn)}'
    )
