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
"""Abstractions for optimizers used in federated learning."""

import abc
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


class Optimizer(abc.ABC):
  """Represents an optimizer for use in TensorFlow Federated.

  Its pair of `initialize` and `next` methods define the optimization
  algorithm, with `next` corresponding to a step of the optimizer.

  This class captures iterative optimization algorithms where the same operation
  is applied in every optimization step. The `next` method should be a
  computation that can be implemented as a `tf.function`. Each method will
  generally only be traced once to create the corresponding TensorFlow graph
  functions. Thus, the methods should not create or use `tf.Variable` objects.

  Instead, any dependency between steps of the algorithm should be included
  as tensors in a state. For instance, a momentum term for momentum SGD is
  created in the `initialize` method as all-zeros tensor, which is then both
  an input and an output of the `next` method.
  """

  @abc.abstractmethod
  def initialize(self, specs):
    """Returns the initial state of the optimizer.

    Args:
      specs: A (possibly nested) structure of `tf.TensorSpec`s describing the
        weights to be optimized. The `weights` and `grads` argument of `next`
        must match the structure and (shape, dtype) of `specs`.

    Returns:
      Initial state of the optimizer. A (possibly nested) structure of tensors.
    """
    pass

  @abc.abstractmethod
  def next(self, state, weights, gradients):
    """Takes a step of the optimizer.

    Args:
      state: State of the optimizer. A structure of tensors matching the
        structure returned by `initialize` method.
      weights: The weights to be updated by the optimizer. A collection of
        Tensors matching the structure of `specs` provided in the `initialize`
        method.
      gradients: The gradients to use for the update by the optimizer. A
        collection of tensors matching the structure of `specs` provided in the
        `initialize` method.

    Returns:
      A (state, weights) tuple representing the updated `state` and `weights`.
    """
    pass


def check_learning_rate(lr):
  py_typecheck.check_type(lr, float)
  if lr <= 0.0:
    raise ValueError('Learning rate must be positive.')


def check_momentum(momentum):
  py_typecheck.check_type(momentum, float)
  if momentum <= 0.0 or momentum >= 1.0:
    raise ValueError('Momentum must be between 0.0 and 1.0.')


def check_weights_gradients_match(weights, gradients):
  try:
    tf.nest.assert_same_structure(weights, gradients, check_types=True)
  except (TypeError, ValueError):
    # Raises a more informative error message specific for optimizers.
    raise ValueError(
        f'Provided weights and gradients must be collections of tensors of the '
        f'same structure and the tensors must have the same shapes and dtypes. '
        f'Provided weights: {weights}\n'
        f'Provided gradients: {gradients}')


def handle_indexed_slices_gradients(gradients):
  """Converts any tf.IndexedSlices to tensors.

  The `tf.IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients (e.g. `tf.gather`). See
  also tf.GradientTape documentation. This method is an elementary utility
  converting the slices to a tensor, which can be used to make an optimizer
  immediately compatible with such gradients. All other values are left
  unmodified.

  Note however, this operation may be expensive in some situations. For more
  details, see
  https://github.com/tensorflow/tensorflow/blob/2b44549aca184ae0eb986a8bd46feef2b17004ab/tensorflow/python/framework/indexed_slices.py#L406

  Args:
    gradients: A collection of gradients to be used by an optimizer.

  Returns:
    The same collection with `tf.IndexedSlices` replaced by tensors.
  """

  def slices_to_tensor(value):
    if isinstance(value, tf.IndexedSlices):
      return tf.convert_to_tensor(value)
    return value

  return tf.nest.map_structure(slices_to_tensor, gradients)
