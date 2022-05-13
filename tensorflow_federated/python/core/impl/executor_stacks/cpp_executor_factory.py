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

import math
from typing import Optional, Sequence

from absl import logging
import cachetools

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executor_stacks import executor_stack_bindings
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.types import placements


# Users likely do not intend to run 4 or more TensorFlow functions sequentially;
# we special-case to warn users explicitly in this case, in addition to
# logging in the case of any implied sequential execution.
_CONCURRENCY_LEVEL_TO_WARN = 4


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


def _log_and_warn_on_sequential_execution(max_concurrent_computation_calls: int,
                                          num_clients: int,
                                          expected_concurrency_factor: int):
  """Logs warnings that users may be using the runtime in an unexpected way."""

  if expected_concurrency_factor >= _CONCURRENCY_LEVEL_TO_WARN:
    logging.warning(
        'Running %s clients with max concurrency %s will result in significant '
        'serialization of execution; running %s TensorFlow functions '
        'sequentially. This invocation could benefit significantly from more '
        'resources (e.g. more GPUs), or moving to TFF\'s distributed runtime.')
  else:
    logging.info(
        'TFF-C++ local executor configured to maximally run %s '
        'calls into TensorFlow in  parallel; asked to run %s '
        'clients. This will result in %s invocations running '
        'sequentially, indicating that this invocation will run '
        'faster when equipped with increased resources or invoked '
        'against the distributed TFF runtime.',
        max_concurrent_computation_calls, num_clients,
        expected_concurrency_factor)


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
    num_clients = cardinalities[placements.CLIENTS]
    if max_concurrent_computation_calls is not None and num_clients > max_concurrent_computation_calls:
      expected_concurrency_factor = math.ceil(num_clients /
                                              max_concurrent_computation_calls)
      _log_and_warn_on_sequential_execution(max_concurrent_computation_calls,
                                            num_clients,
                                            expected_concurrency_factor)
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
