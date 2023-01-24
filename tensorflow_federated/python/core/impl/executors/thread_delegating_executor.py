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
"""A concurrent executor that does work asynchronously in multiple threads."""

from typing import Optional

from tensorflow_federated.python.common_libs import async_utils
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executor_base as eb
from tensorflow_federated.python.core.impl.executors import executor_value_base as evb


def _delegate_with_trace_ctx(coro, async_runner):
  coro_with_trace_ctx = tracing.wrap_coroutine_in_current_trace_context(coro)
  return async_runner.await_coro_and_return_result(coro_with_trace_ctx)


class ThreadDelegatingExecutorValue(evb.ExecutorValue):
  """An ExecutorValue which delegates `compute` to an external event loop."""

  def __init__(
      self,
      value: evb.ExecutorValue,
      async_runner: async_utils.AsyncThreadRunner,
  ):
    self._value = value
    self._async_runner = async_runner

  @property
  def reference(self) -> evb.ExecutorValue:
    return self._value

  @property
  def type_signature(self):
    return self.reference.type_signature

  async def compute(self):
    return await _delegate_with_trace_ctx(
        self._value.compute(), self._async_runner
    )


class ThreadDelegatingExecutor(eb.Executor):
  """The concurrent executor delegates work to a separate thread.

  This executor only handles threading. It delegates all execution to an
  underlying pool of target executors.
  """

  # TODO(b/134543154): Upgrade this to a threadpool with multiple workers,
  # possibly one that could be shared among multiple of these executors.

  def __init__(self, target_executor: eb.Executor):
    """Creates a concurrent executor backed by a target executor.

    Args:
      target_executor: The executor that does all the work.
    """
    py_typecheck.check_type(target_executor, eb.Executor)
    self._target_executor = target_executor
    self._async_runner = async_utils.AsyncThreadRunner()

  def close(self):
    # Close does not clean up the event loop or thread.
    # Using the executor again after cleanup used to lazily re-initialized the
    # event loop, but this resulted in bugs related to the persistence of values
    # associated with the old event loop ("futures are tied to different event
    # loops"). See the closed bug b/148288711 for more information.
    self._target_executor.close()

  async def _delegate(self, coro):
    """Runs a coroutine which returns an executor value on the event loop."""
    result_value = await _delegate_with_trace_ctx(coro, self._async_runner)
    return ThreadDelegatingExecutorValue(result_value, self._async_runner)

  @tracing.trace
  async def create_value(self, value, type_spec=None) -> evb.ExecutorValue:
    return await self._delegate(
        self._target_executor.create_value(value, type_spec)
    )

  @tracing.trace
  async def create_call(
      self, comp: evb.ExecutorValue, arg: Optional[evb.ExecutorValue] = None
  ) -> evb.ExecutorValue:
    comp = comp.reference
    arg = arg.reference if arg else None
    return await self._delegate(self._target_executor.create_call(comp, arg))

  @tracing.trace
  async def create_struct(self, elements):
    elements_as_structure = structure.from_container(elements)
    elements_iter = structure.iter_elements(elements_as_structure)
    pairs = ((n, v.reference) for (n, v) in elements_iter)
    inner_elements = structure.Struct(pairs)
    return await self._delegate(
        self._target_executor.create_struct(inner_elements)
    )

  @tracing.trace
  async def create_selection(self, source, index):
    source = source.reference
    return await self._delegate(
        self._target_executor.create_selection(source, index)
    )
