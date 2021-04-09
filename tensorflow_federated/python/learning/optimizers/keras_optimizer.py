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

import tensorflow as tf

from tensorflow_federated.python.learning.optimizers import optimizer


class KerasOptimizer(optimizer.Optimizer):
  """Adapter for keras optimzier as `tff.learning.optimizers.Optimizer`.

  This class is expected to be instantiated in the context of
  `tff.tf_computation` in which it is to be used. This is because the
  `optimizer_fn` provided in constructor is going to be invoked, which will
  create a keras optimizer instance, and we will force the creation of
  `tf.Variable` objects which store the state of that keras optimizer.

  If this class is supposed to be used as a "server optimizer", set
  `disjoint_init_and_next` to True, which means that the `initialize` and `next`
  methods are going to be invoked in the context of *different*
  `tff.tf_computations`, and TFF needs to carry the optimizer variables between
  the invocations.

  If this class is supposed to be used as a "client optimizer", set
  `disjoint_init_and_next` to False, which means that the `initialize` and
  `next` methods are going to be invoked in the context of *the same*
  `tff.tf_compuation` and we don't need to pass the variables of keras optimizer
  to TFF to handle.

  NOTE: This class is not meant to be exposed in public API for now. Rather, it
  is used to convert the previous default support for keras optimizers to the
  tff.learning.optimizers format.
  """

  def __init__(self, optimizer_fn, weights, disjoint_init_and_next):
    """Initializes `KerasOptimizer`.

    Args:
      optimizer_fn: A no-arg callable that creates and returns a
        `tf.keras.optimizers.Optimizer`.
      weights: A (possibly nested) structure of `tf.Variable` objects which are
        supposed to be modified during call to the `next` method of the
        optimizer.
      disjoint_init_and_next: A boolean, determining whether the `initialize`
        and `next` methods are going to be invoked in the context of the same
        `tff.tf_computation`.
    """
    self._optimizer = optimizer_fn()
    self._disjoint_init_and_next = disjoint_init_and_next

    def mock_apply_gradients(opt, variables):
      opt.apply_gradients([
          (tf.zeros_like(w), w) for w in tf.nest.flatten(variables)
      ])

    # Force the creation of tf.Variables controlled by the keras optimizer but
    # keep the variables unmodified. For instance, the "step" variable will be
    # 0, not 1, after this operation.
    tf.function(mock_apply_gradients).get_concrete_function(
        self._optimizer, weights)

  def initialize(self, specs):
    del specs  # Unused.
    if self._disjoint_init_and_next:
      return self._optimizer.variables()
    else:
      return ()

  def next(self, state, weights, gradients):
    if self._disjoint_init_and_next:
      tf.nest.map_structure(lambda v, s: v.assign(s),
                            self._optimizer.variables(), state)

    self._optimizer.apply_gradients(
        list(zip(tf.nest.flatten(gradients), tf.nest.flatten(weights))))

    if self._disjoint_init_and_next:
      return self._optimizer.variables(), weights
    else:
      return (), weights
