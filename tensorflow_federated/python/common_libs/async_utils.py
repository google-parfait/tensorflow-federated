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
import sys


class SharedAwaitable():
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
        e_type, value, traceback = self._exception
        exception = e_type(value)
        exception.__traceback__ = traceback
        raise exception
      return self._result

    return waiter().__await__()
