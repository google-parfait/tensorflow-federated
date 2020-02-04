# Lint as: python3
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
from typing import Callable, Mapping

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl.compiler import placement_literals

CardinalitiesType = Mapping[placement_literals.PlacementLiteral, int]


class ExecutorFactory(metaclass=abc.ABCMeta):
  """Interface defining executor factories.

  Executor factories will be returned to users via execution constructors
  exposed in `tff.framework`, and are the objects accepted by
  `tff.framework.set_default_executor`.

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

  def _get_hashable_key(self, cardinalities: CardinalitiesType):
    return tuple(sorted(cardinalities.items()))

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
    key = self._get_hashable_key(cardinalities)
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
