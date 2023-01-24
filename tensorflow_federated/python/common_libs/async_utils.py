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
from collections.abc import Callable
import contextlib
import sys
import threading

from absl import logging

from tensorflow_federated.python.common_libs import tracing


async def _log_error(awaitable):
  try:
    await awaitable
  except:  # pylint: disable=bare-except
    logging.exception("Error running task")


async def _then(first, second):
  await first
  await second


class OrderedTasks:
  """A group of asynchronous tasks to be run sequentially.

  Newly added awaitables or functions will be added to a new task to be run as
  soon as all previously-scheduled tasks have completed.

  Any exceptions encountered during the execution of an awaitable or task will
  be logged, and concurrent or subsequent tasks will continue to run unaffected.
  """

  def __init__(self):
    async def empty_coroutine():
      pass

    self._last_task: asyncio.Task = asyncio.create_task(empty_coroutine())

  def add(self, awaitable):
    """Add an awaitable to run after previously-added tasks have completed."""
    self._last_task = asyncio.create_task(
        _then(self._last_task, _log_error(awaitable))
    )

  def add_all(self, *awaitables):
    """Add awaitables run concurrently after previous tasks have completed."""
    awaitables = (_log_error(x) for x in awaitables)
    gather = asyncio.gather(*awaitables)
    self._last_task = asyncio.create_task(_then(self._last_task, gather))

  def add_callable(self, function: Callable[[], None]):
    """Add function to run after previous tasks have completed."""
    # Save `_last_task` to a variable so that `_then_run` references the
    # current value of `_last_task`, rather than the value after it is updated.
    previous = self._last_task

    async def _then_run():
      await previous
      try:
        function()
      except:  # pylint: disable=bare-except
        logging.exception("Error running synchronous task")

    self._last_task = asyncio.create_task(_then_run())

  async def wait(self):
    """Waits for all previously-scheduled tasks to complete."""
    await self._last_task

  async def aclose(self):
    """Alias for `wait` to be used with Python 3.10+'s `aclosing` function."""
    await self.wait()


@contextlib.asynccontextmanager
async def ordered_tasks():
  """A context manager ensuring that all ordered tasks complete before exit."""
  tasks = OrderedTasks()
  try:
    yield tasks
  finally:
    await tasks.wait()


class SharedAwaitable:
  """A wrapper allowing `async` functions to be `await`ed from multiple places.

  `async` functions (those that start with `async def`) are typically `await`ed
  immediately at their callsite, as in `await foo()`. However, if users want to
  `await` this value from multiple `async` functions without running `foo()`
  twice, it can be useful to write something like this:

  ```python
  foo_coroutine = foo()

  async def fn_one():
    ...
    x = await foo_coroutine
    ...

  async def fn_two():
    ...
    x = await foo_coroutine
    ...
  ```

  Unfortunately, directly `await`ing the result of an `async` function multiple
  times is not supported, and will fail with an exception:

  `RuntimeError: cannot reuse already awaited coroutine`

  `SharedAwaitable` fixes this problem:

  ```python
  foo_coroutine = SharedAwaitable(foo())

  async def fn_one():
    ...
    x = await foo_coroutine
    ...

  async def fn_two():
    ...
    x = await foo_coroutine
    ...
  ```
  """

  def __init__(self, awaitable):
    """Creates a new `SharedAwaitable` from an existing `awaitable`."""
    self._awaitable = awaitable
    self._event = None
    self._result = None
    self._exception = None

  def __await__(self):
    # If it's the first await, spawn a separate task to actually run the
    # awaitable and report back with the result.
    if self._event is None:
      self._event = asyncio.Event()

      async def get_result():
        try:
          self._result = await self._awaitable
        except:  # pylint: disable=bare-except
          self._exception = sys.exc_info()
        finally:
          self._event.set()

      asyncio.create_task(get_result())

    # Then wait for the result to be reported back.

    async def waiter():
      await self._event.wait()
      if self._exception is not None:
        _, exception, traceback = self._exception
        raise exception.with_traceback(traceback)
      return self._result

    return waiter().__await__()


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
