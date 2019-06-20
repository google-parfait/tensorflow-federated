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
"""A concurrent executor that does work asynchronously in multiple threads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import asyncio
import functools
import threading

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import executor_base


class ConcurrentExecutor(executor_base.Executor):
  """The concurrent executor delegates work to a separate thread.

  This executor only handles threading. It delegates all execution to an
  underlying pool of target executors.
  """

  # TODO(b/134543154): Upgrade this to a threadpool with multiple workers,
  # possibly one that could be shared among multiple of these executors.

  def __init__(self, target_executor):
    """Creates a concurrent executor backed by a target executor.

    Args:
      target_executor: The executor that does all the work.
    """
    py_typecheck.check_type(target_executor, executor_base.Executor)
    self._target_executor = target_executor
    self._event_loop = asyncio.new_event_loop()

    def run_loop(loop):
      loop.run_forever()
      loop.close()

    self._thread = threading.Thread(
        target=functools.partial(run_loop, self._event_loop))
    self._thread.start()

  def __del__(self):
    self._event_loop.call_soon_threadsafe(self._event_loop.stop)
    self._thread.join()

  def _delegate(self, coro):
    return asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(coro, self._event_loop))

  async def create_value(self, value, type_spec=None):
    return await self._delegate(
        self._target_executor.create_value(value, type_spec))

  async def create_call(self, comp, arg=None):
    return await self._delegate(self._target_executor.create_call(comp, arg))

  async def create_tuple(self, elements):
    return await self._delegate(self._target_executor.create_tuple(elements))

  async def create_selection(self, source, index=None, name=None):
    return await self._delegate(
        self._target_executor.create_selection(source, index=index, name=name))
