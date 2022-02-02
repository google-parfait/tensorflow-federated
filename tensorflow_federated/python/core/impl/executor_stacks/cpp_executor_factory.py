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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A collection of constructors for basic types of executor stacks."""

from typing import Optional, Sequence

import cachetools

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executor_stacks import executor_stack_bindings
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.types import placements


def _get_hashable_key(cardinalities: executor_factory.CardinalitiesType):
  return tuple(sorted((str(k), v) for k, v in cardinalities.items()))


class CPPExecutorFactory(executor_factory.ExecutorFactory):
  """An ExcutorFactory which wraps a simple executor_fn."""

  def __init__(self, executor_fn, executor_cache_size: int = 5):
    self._executor_fn = executor_fn
    self._cache_size = executor_cache_size
    self._executors = cachetools.LRUCache(self._cache_size)

  def create_executor(self, cardinalities):
    cardinalities_key = _get_hashable_key(cardinalities)
    if self._executors.get(cardinalities_key):
      return self._executors[cardinalities_key]
    executor = self._executor_fn(cardinalities)
    self._executors[cardinalities_key] = executor
    return executor

  def clean_up_executors(self):
    self._executors = cachetools.LRUCache(self._cache_size)


def local_cpp_executor_factory(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: Optional[int] = None
) -> executor_factory.ExecutorFactory:
  """Local ExecutorFactory backed by C++ Executor bindings."""
  py_typecheck.check_type(default_num_clients, int)

  def _executor_fn(
      cardinalities: executor_factory.CardinalitiesType
  ) -> executor_bindings.Executor:
    if cardinalities.get(placements.CLIENTS) is None:
      cardinalities[placements.CLIENTS] = default_num_clients
    tf_executor = executor_bindings.create_tensorflow_executor(
        max_concurrent_computation_calls)
    sub_federating_reference_resolving_executor = executor_bindings.create_reference_resolving_executor(
        tf_executor)
    federating_ex = executor_bindings.create_federating_executor(
        sub_federating_reference_resolving_executor, cardinalities)
    top_level_reference_resolving_ex = executor_bindings.create_reference_resolving_executor(
        federating_ex)
    return top_level_reference_resolving_ex

  return CPPExecutorFactory(_executor_fn)


def remote_cpp_executor_factory(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0) -> executor_factory.ExecutorFactory:
  """ExecutorFactory backed by C++ Executor bindings."""
  py_typecheck.check_type(default_num_clients, int)

  def _executor_fn(
      cardinalities: executor_factory.CardinalitiesType
  ) -> executor_bindings.Executor:
    if cardinalities.get(placements.CLIENTS) is None:
      cardinalities[placements.CLIENTS] = default_num_clients
    return executor_stack_bindings.create_remote_executor_stack(
        channels, cardinalities)

  return CPPExecutorFactory(_executor_fn, executor_cache_size=1)
