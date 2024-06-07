## Copyright 2022, The TensorFlow Federated Authors.
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
"""Concurrency utilities for use with Python `async`."""

import asyncio
import threading

from tensorflow_federated.python.common_libs import tracing


class AsyncThreadRunner:
  """Class which bridges async and synchronous synchronous interfaces.

  This class serves as a resource and logic container, starting an event loop
  in a separate thread and managing dispatching of coroutine functions to this
  event loop in both synchronous and asynchronous interfaces.

  There are two main uses of this class. First, this class can be used to wrap
  interfaces which use `asyncio` in a synchronous 'run this coroutine'
  interface in a manner which is compatible with integrating with other async
  libraries. This feature is generally useful for backwards-compatibility (e.g.,
  introducing asyncio in some component which sits on top of the synchronous
  function calls this interface exposes), but should generally be viewed as
  suboptimal--it is preferable in a situation like this to simply expose the
  underlying async interfaces.

  Second, this class can be used to delegate asynchronous work from one thread
  to another, using its asynchronous interface.
  """

  def __init__(self):
    self._event_loop = asyncio.new_event_loop()
    self._event_loop.set_task_factory(
        tracing.propagate_trace_context_task_factory
    )

    def target_fn():
      self._event_loop.run_forever()

    self._thread = threading.Thread(target=target_fn, daemon=True)
    self._thread.start()

    def finalizer(loop, thread):
      loop.call_soon_threadsafe(loop.stop)
      thread.join()

    self._finalizer = finalizer

  def __del__(self):
    self._finalizer(self._event_loop, self._thread)

  def run_coro_and_return_result(self, coro):
    """Runs coroutine in the managed event loop, returning the result."""
    future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
    return future.result()

  async def await_coro_and_return_result(self, coro):
    """Runs coroutine in the managed event loop, returning the result."""
    return await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(coro, self._event_loop)
    )
