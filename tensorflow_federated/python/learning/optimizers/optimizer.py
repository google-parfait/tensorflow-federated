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
import collections
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar, Union

import numpy as np
import tensorflow as tf

# Types related to optimizer hyperparameters
Int = Union[int, tf.Tensor, np.number]
Float = Union[float, tf.Tensor, np.number]
State = TypeVar('State')
Weights = TypeVar('Weights')
Gradients = Any
Hparams = TypeVar('Hparams', bound=collections.OrderedDict[str, Any])
T = TypeVar('T')
_Structure = Union[
    T,
    Sequence['_Structure[T]'],
    Mapping[str, '_Structure[T]'],
]

# Common attribute names for optimizers
LEARNING_RATE_KEY = 'learning_rate'


class Optimizer(abc.ABC, Generic[State, Weights, Hparams]):
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
  an input and an output of the `next` method. These aspects should be accessed
  and changed via `get_hparams` and `set_hparams`, respectively.

  As a best practice, any implementation using learning rate, should store it in
  its state under the key `tff.learning.optimizers.LEARNING_RATE_KEY`.
  """

  @abc.abstractmethod
  def initialize(self, specs: Any) -> State:
    """Returns the initial state of the optimizer.

    Args:
      specs: A (possibly nested) structure of `tf.TensorSpec`s describing the
        weights to be optimized. The `weights` and `grads` argument of `next`
        must match the structure and (shape, dtype) of `specs`.

    Returns:
      Initial state of the optimizer. A (possibly nested) structure of tensors.
    """

  @abc.abstractmethod
  def next(
      self, state: State, weights: Weights, gradients: Any
  ) -> tuple[State, Weights]:
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

  def get_hparams(self, state: State) -> Hparams:
    """Returns a dictionary containing the optimizer state hyperparameters.

    Args:
      state: The state of the optimizer. Must match the structure returned by
        the `initialize` method.

    Returns:
      An ordered dictionary representing the hyperparameters in the given state.
    """
    del state
    return collections.OrderedDict()

  def set_hparams(self, state: State, hparams: Hparams) -> State:
    """Returns an optimizer state with updated hyperparameters.

    Args:
      state: The state of the optimizer. Must match the structure returned by
        the `initialize` method.
      hparams: A dictionary matching the output of `get_hparams` containing the
        updated hyperparameters to use.

    Returns:
      An ordered dictionary representing the hyperparameters in the given state.
    """
    del hparams
    return state


def _check_shape_dtype_match(x: tf.Tensor, y: Union[tf.Tensor, None]) -> None:
  if y is None:
    return
  if not x.shape.is_compatible_with(y.shape) or x.dtype != y.dtype:
    raise TypeError('Provided tensors do not have the same shapes and dtypes.')


def check_weights_gradients_match(
    weights: _Structure[tf.Tensor],
    gradients: _Structure[Union[tf.Tensor, None]],
) -> None:
  """Checks that weights and non-none gradients match.

  This check is meant to be used in the `next` method of implemented
  `tff.learning.optimizers.Optimizer` to check whether the provided weights and
  gradients match, and provide easy and more informative error message.

  To match behavior of `tf.keras.optimizers`, this check will only be applied
  to gradient leaves that are not `None`.

  Args:
    weights: A structure of tensors.
    gradients: A structure of tensors.

  Raises:
    ValueError: If `weights` and `gradients` do not have the same structure, or
      if the tensors in the structures do not have the same shapes and dtypes,
      at some leaf where `gradients` is not `None`.
  """
  try:
    tf.nest.assert_same_structure(weights, gradients, check_types=True)
    tf.nest.map_structure(_check_shape_dtype_match, weights, gradients)
  except (TypeError, ValueError) as e:

    def _type_and_shape(nest):
      return tf.nest.map_structure(lambda x: (x.shape, x.dtype), nest)

    # Raises a more informative error message specific for optimizers.
    raise ValueError(
        'Provided weights and gradients must be collections of tensors of the '
        'same structure and the tensors must have the same shapes and dtypes.\n'
        'Provided weights have type '
        f'{_type_and_shape(weights)}, value {weights}\n'
        'Provided gradients have type '
        f'{_type_and_shape(gradients)}, value {gradients}\n'
    ) from e


def check_weights_state_match(
    weights: _Structure[tf.Tensor], state: _Structure[tf.Tensor], name: str
) -> None:
  try:
    tf.nest.assert_same_structure(state, weights, check_types=True)
    tf.nest.map_structure(_check_shape_dtype_match, weights, state)
  except (TypeError, ValueError) as e:
    # Raises a more informative error message.
    raise ValueError(
        f'Provided {name} and weigths do not match. The {name} term in '
        'the state and weights must be collections of tensors of the same '
        'structure and the tensors must have the same shapes and dtypes. A '
        'possible reason is that the `initialize` method was invoked with '
        '`specs` not matching the weights being optimized.\n'
        f'Provided {name}: {state}\n'
        f'Provided weights: {weights}'
    ) from e


def handle_indexed_slices_gradients(
    gradients: _Structure[T],
) -> _Structure[Union[T, tf.IndexedSlices]]:
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

  def slices_to_tensor(value: T) -> Union[T, tf.IndexedSlices]:
    if isinstance(value, tf.IndexedSlices):
      return tf.convert_to_tensor(value)
    return value

  return tf.nest.map_structure(slices_to_tensor, gradients)
