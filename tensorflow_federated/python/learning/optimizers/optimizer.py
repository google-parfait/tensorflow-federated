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
        tensors matching the structure of `specs` provided in the `initialize`
        method.
      gradients: The gradients to use for the update by the optimizer. A
        collection of tensors matching the structure of `specs` provided in the
        `initialize` method.

    Returns:
      A (state, weights) tuple representing the updated `state` and `weights`.
    """
    pass


def _check_shape_dtype_match(x, y):
  if not x.shape.is_compatible_with(y.shape) or x.dtype != y.dtype:
    raise TypeError('Provided tensors do not have the same shapes and dtypes.')


def check_weights_gradients_match(weights, gradients):
  """Checks that weights and gradients match.

  This check is meant to be used in the `next` method of implemented
  `tff.learning.optimizers.Optimizer` to check whether the provided weights and
  gradients match, and provide easy and more informative error message.

  Args:
    weights: A structure of tensors.
    gradients: A structure of tensors.

  Raises:
    ValueError: If `weights` and `gradients` do not have the same structure, or
      if the tensors in the structures do not have the same shapes and dtypes.
  """
  try:
    tf.nest.assert_same_structure(weights, gradients, check_types=True)
    tf.nest.map_structure(_check_shape_dtype_match, weights, gradients)
  except (TypeError, ValueError):
    # Raises a more informative error message specific for optimizers.
    raise ValueError(
        'Provided weights and gradients must be collections of tensors of the '
        'same structure and the tensors must have the same shapes and dtypes.\n'
        f'Provided weights: {weights}\n'
        f'Provided gradients: {gradients}')


def check_weights_state_match(weights, state, name):
  try:
    tf.nest.assert_same_structure(state, weights, check_types=True)
    tf.nest.map_structure(_check_shape_dtype_match, weights, state)
  except (TypeError, ValueError):
    # Raises a more informative error message.
    raise ValueError(
        f'Provided {name} and weigths do not match. The {name} term in '
        f'the state and weights must be collections of tensors of the same '
        f'structure and the tensors must have the same shapes and dtypes. A '
        f'possible reason is that the `initialize` method was invoked with '
        f'`specs` not matching the weights being optimized.\n'
        f'Provided {name}: {state}\n'
        f'Provided weights: {weights}')


def handle_indexed_slices_gradients(gradients):
  """Converts any `tf.IndexedSlices` to tensors.

  The `tf.IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients (e.g. `tf.gather`). See
  also `tf.GradientTape` documentation. This method is an elementary utility
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
