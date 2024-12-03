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
"""A collection of constructors for basic types of executor stacks."""

from collections.abc import Callable

import cachetools
import federated_language

from tensorflow_federated.python.common_libs import py_typecheck

# Place a limit on the maximum size of the executor caches managed by the
# ExecutorFactories, to prevent unbounded thread and memory growth in the case
# of rapidly-changing cross-round cardinalities.
_EXECUTOR_CACHE_SIZE = 10


def _get_hashable_key(
    cardinalities: federated_language.framework.CardinalitiesType,
):
  return tuple(sorted((str(k), v) for k, v in cardinalities.items()))


class ResourceManagingExecutorFactory(
    federated_language.framework.ExecutorFactory
):
  """Implementation of executor factory holding an executor per cardinality."""

  def __init__(
      self,
      executor_stack_fn: Callable[
          [federated_language.framework.CardinalitiesType],
          federated_language.framework.Executor,
      ],
  ):
    """Initializes `ResourceManagingExecutorFactory`.

    `ResourceManagingExecutorFactory` manages a mapping from `cardinalities`
    to `federated_language.framework.Executors`, closing and destroying the
    executors in this
    mapping when asked.

    Args:
      executor_stack_fn: Callable taking a mapping from
        `federated_language.framework.PlacementLiteral` to integers, and
        returning an `federated_language.framework.Executor`. The returned
        executor will be configured to handle these cardinalities.
    """
    self._executor_stack_fn = executor_stack_fn
    self._executors = cachetools.LRUCache(_EXECUTOR_CACHE_SIZE)

  def create_executor(
      self, cardinalities: federated_language.framework.CardinalitiesType
  ) -> federated_language.framework.Executor:
    """Constructs or gets existing executor.

    Returns a previously-constructed executor if this method has already been
    invoked with `cardinalities`. If not, invokes `self._executor_stack_fn`
    with `cardinalities` and returns the result.

    Args:
      cardinalities: `dict` with `federated_language.framework.PlacementLiteral`
        keys and integer values, specifying the population size at each
        placement. The executor stacks returned from this method are not
        themselves polymorphic; a concrete stack must have fixed sizes at each
        placement.

    Returns:
      Instance of `federated_language.framework.Executor` as described above.
    """
    py_typecheck.check_type(cardinalities, dict)
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is not None:
      return ex
    ex = self._executor_stack_fn(cardinalities)
    py_typecheck.check_type(ex, federated_language.framework.Executor)
    self._executors[key] = ex
    return ex

  def clean_up_executor(
      self, cardinalities: federated_language.framework.CardinalitiesType
  ):
    """Calls `close` on constructed executors, resetting internal cache.

    If a caller holds a name bound to any of the executors returned from
    `create_executor` with the specified cardinalities, this executor
    should be assumed to be in an invalid state, and should not be used after
    this method is called. Instead, callers should again invoke
    `create_executor`.

    Args:
      cardinalities: The cardinalities of the executor to be cleaned up.
    """
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is None:
      return
    ex.close()
    del self._executors[key]
