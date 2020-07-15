# Copyright 2019, The TensorFlow Federated Authors.
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
"""ExecutorFactory interface and simple implementation."""

import abc
from typing import Callable, Mapping, List, Tuple, Any, Dict

import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import sizing_executor
from tensorflow_federated.python.core.impl.types import placement_literals

CardinalitiesType = Mapping[placement_literals.PlacementLiteral, int]


@attr.s(auto_attribs=True, eq=False, order=False, frozen=True)
class SizeInfo(object):
  """Structure for size information from SizingExecutorFactory.get_size_info().

  Fields:
  -   `broadcast_history`: 2D ragged list of 2-tuples which represents the
      broadcast history.
  -   `aggregate_history`: 2D ragged list of 2-tuples which represents the
      aggregate history.
  -   `broadcast_bits`: A list of shape [number_of_execs] representing the
      number of broadcasted bits passed through each executor.
  -   `aggregate_bits`: A list of shape [number_of_execs] representing the
      number of aggregated bits passed through each executor.
  """
  broadcast_history: Dict[Any, sizing_executor.SizeAndDTypes]
  aggregate_history: Dict[Any, sizing_executor.SizeAndDTypes]
  broadcast_bits: List[int]
  aggregate_bits: List[int]


class ExecutorFactory(metaclass=abc.ABCMeta):
  """Interface defining executor factories.

  `ExecutorFactory` should be considered to own the executors it creates; it
  is responsible for their instantiation and management.

  `ExecutorFactory` exposes two methods, `create_executor` and
  `clean_up_executors`. There is a particular coupling between these two
  methods; any executor returned by `create_executor` should not be used
  after `clean_up_executors` has been called without reinitialization. That is,
  `create_executor` should be called again, and `ExecutorFactory` will ensure
  that the returned executor is safe for use.
  """

  @abc.abstractmethod
  def create_executor(
      self, cardinalities: CardinalitiesType) -> executor_base.Executor:
    """Abstract method to construct instance of `executor_base.Executor`.

    `create_executor` must accept a dict mapping
    `placement_literals.PlacementLiterals` to `ints`, and return an
    `executor_base.Executor`.

    Args:
      cardinalities: a dict mapping instances of
        `placement_literals.PlacementLiteral` to ints, specifying the population
        size at each placement.

    Returns:
      Instance of `executor_base.Executor`.
    """
    pass

  @abc.abstractmethod
  def clean_up_executors(self):
    """Releases any resources held by the factory.

    Note that calling this method may invalidate the state of any executors
    which have previously been returned by the factory; `create_executor`
    should be called again if a new executor which is safe to use is desired.
    """
    pass


def create_executor_factory(
    executor_stack_fn: Callable[[CardinalitiesType], executor_base.Executor]
) -> ExecutorFactory:
  """Create an `ExecutorFactory` for a given executor stack function."""
  py_typecheck.check_callable(executor_stack_fn)
  return ExecutorFactoryImpl(executor_stack_fn)


def _get_hashable_key(cardinalities: CardinalitiesType):
  return tuple(sorted((str(k), v) for k, v in cardinalities.items()))


class ExecutorFactoryImpl(ExecutorFactory):
  """Implementation of executor factory holding an executor per cardinality."""

  def __init__(self, executor_stack_fn: Callable[[CardinalitiesType],
                                                 executor_base.Executor]):
    """Initializes `ExecutorFactoryImpl`.

    Args:
      executor_stack_fn: Callable taking a mapping from
        `placement_literals.PlacementLiteral` to integers, and returning an
        `executor_base.Executor`. The returned executor will be configured to
        handle these cardinalities.
    """

    py_typecheck.check_callable(executor_stack_fn)
    self._executor_stack_fn = executor_stack_fn
    self._executors = {}

  def create_executor(
      self, cardinalities: CardinalitiesType) -> executor_base.Executor:
    """Constructs or gets existing executor.

    Returns a previously-constructed executor if this method has already been
    invoked with `cardinalities`. If not, invokes `self._executor_stack_fn`
    with `cardinalities` and returns the result.

    Args:
      cardinalities: `dict` with `placement_literals.PlacementLiteral` keys and
        integer values, specifying the population size at each placement. The
        executor stacks returned from this method are not themselves
        polymorphic; a concrete stack must have fixed sizes at each placement.

    Returns:
      Instance of `executor_base.Executor` as described above.
    """
    py_typecheck.check_type(cardinalities, dict)
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is not None:
      return ex
    ex = self._executor_stack_fn(cardinalities)
    py_typecheck.check_type(ex, executor_base.Executor)
    self._executors[key] = ex
    return ex

  def clean_up_executors(self):
    """Calls `close` on all constructed executors, resetting internal cache.

    If a caller holds a name bound to any of the executors returned from
    `create_executor`, this executor should be assumed to be in an invalid
    state, and should not be used after this method is called. Instead, callers
    should again invoke `create_executor`.
    """
    for _, ex in self._executors.items():
      ex.close()
    self._executors = {}


class SizingExecutorFactory(ExecutorFactoryImpl):
  """A executor factory holding an executor per cardinality."""

  def __init__(
      self,
      executor_stack_fn: Callable[[CardinalitiesType],
                                  Tuple[executor_base.Executor,
                                        List[sizing_executor.SizingExecutor]]]):
    """Initializes `SizingExecutorFactory`.

    Args:
      executor_stack_fn: Similar to base class but the second return value of
        the callable is used to expose the SizingExecutors.
    """

    super().__init__(executor_stack_fn)
    self._sizing_executors = {}

  def create_executor(
      self, cardinalities: CardinalitiesType) -> executor_base.Executor:
    """See base class."""

    py_typecheck.check_type(cardinalities, dict)
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is not None:
      return ex
    ex, sizing_executors = self._executor_stack_fn(cardinalities)
    self._sizing_executors[key] = []
    for executor in sizing_executors:
      if not isinstance(executor, sizing_executor.SizingExecutor):
        raise ValueError('Expected all input executors to be sizing executors')
      self._sizing_executors[key].append(executor)
    py_typecheck.check_type(ex, executor_base.Executor)
    self._executors[key] = ex
    return ex

  def get_size_info(self) -> SizeInfo:
    """Returns information about the transferred data of each SizingExecutor.

    Returns the history of broadcast and aggregation for each executor as well
    as the number of aggregated bits that has been passed through.

    Returns:
      An instance of `SizeInfo`.
    """
    size_ex_dict = self._sizing_executors

    def _extract_history(sizing_exs: List[sizing_executor.SizingExecutor]):
      broadcast_history, aggregate_history = [], []
      for ex in sizing_exs:
        broadcast_history.extend(ex.broadcast_history)
        aggregate_history.extend(ex.aggregate_history)
      return broadcast_history, aggregate_history

    broadcast_history, aggregate_history = {}, {}
    for key, size_exs in size_ex_dict.items():
      current_broadcast_history, current_aggregate_history = _extract_history(
          size_exs)
      broadcast_history[key] = current_broadcast_history
      aggregate_history[key] = current_aggregate_history

    broadcast_bits = [
        self._calculate_bit_size(hist) for hist in broadcast_history.values()
    ]
    aggregate_bits = [
        self._calculate_bit_size(hist) for hist in aggregate_history.values()
    ]
    return SizeInfo(
        broadcast_history=broadcast_history,
        aggregate_history=aggregate_history,
        broadcast_bits=broadcast_bits,
        aggregate_bits=aggregate_bits)

  def _bits_per_element(self, dtype: tf.DType) -> int:
    """Returns the number of bits that a tensorflow DType uses per element."""
    if dtype == tf.string:
      return 8
    elif dtype == tf.bool:
      return 1
    return dtype.size * 8

  def _calculate_bit_size(self, history: sizing_executor.SizeAndDTypes) -> int:
    """Takes a list of 2 element lists and calculates the number of bits represented.

    The input list should follow the format of self.broadcast_history or
    self.aggregate_history. That is, each 2 element list should be
    [num_elements, dtype].

    Args:
      history: The history of values passed through the executor.

    Returns:
      The number of bits represented in the history.
    """
    bit_size = 0
    for num_elements, dtype in history:
      bit_size += num_elements * self._bits_per_element(dtype)
    return bit_size
