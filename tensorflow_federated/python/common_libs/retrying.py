# Copyright 2021, The TensorFlow Federated Authors.
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
"""Library of pure-python retrying decorators."""

import asyncio
from collections.abc import Callable
import functools
import inspect
import time
from typing import Union

from tensorflow_federated.python.common_libs import py_typecheck


def retry(
    fn=None,
    *,
    retry_on_exception_filter: Callable[[Exception], bool] = lambda x: True,
    retry_on_result_filter: Callable[[object], bool] = lambda x: False,
    wait_max_ms: Union[float, int] = 30000,
    wait_multiplier: Union[float, int] = 2,
):
  """Pure Python decorator that retries functions or coroutine functions.

  `retry` starts at some delay between function invocations, and backs
  off exponentialy with factor `wait_multiplier` until the max of
  `max_wait_ms`, at which point `retry` will continue to retry `fn` at intervals
  of `max_wait_ms` until `retry_on_exception_filter` returns `False`.

  Args:
    fn: Optional Python function or coroutine function to wrap in retrying
      logic. If None, `retry` will return a callable which decorates a function
      or corofunc to be passed later.
    retry_on_exception_filter: Function accepting a Python `Exception`, and
      returning a Boolean indicating whether or not to retry the invocation.
    retry_on_result_filter: Function accepting a function result or coroutine
      function result, and returning a Boolean indicating whether or not to
      retry the invocation.
    wait_max_ms: Maximum time `retry` is allowed to wait between invocations of
      `fn`, in milliseconds. Must be positive.
    wait_multiplier: Number determining the exponential backoff multiplier to
      use. Must be positive.

  Returns:
    In the case that `fn` is provided, a decorated version of `fn` respecting
    the semantics above. If `fn` is not provided, returns a callable which can
    be used to decorate a function or coroutine function at a later time.
  """
  py_typecheck.check_type(wait_max_ms, (float, int))
  py_typecheck.check_type(wait_multiplier, (float, int))
  if not inspect.isfunction(retry_on_exception_filter):
    raise TypeError(
        'Expected function to be passed as retry_on_exception_filter; '
        'encountered {} of type {}.'.format(
            retry_on_exception_filter, type(retry_on_exception_filter)
        )
    )
  if not inspect.isfunction(retry_on_result_filter):
    raise TypeError(
        'Expected function to be passed as retry_on_result_filter; '
        'encountered {} of type {}.'.format(
            retry_on_result_filter, type(retry_on_result_filter)
        )
    )
  if wait_max_ms <= 0:
    raise ValueError(
        'wait_max_ms required to be positive; encountered value {}.'.format(
            wait_max_ms
        )
    )
  if wait_multiplier <= 0:
    raise ValueError(
        'wait_multiplier required to be positive; encountered value {}.'.format(
            wait_multiplier
        )
    )

  if fn is None:
    # Called with arguments; delay decoration until `fn` is passed in.
    return functools.partial(
        retry,
        retry_on_exception_filter=retry_on_exception_filter,
        retry_on_result_filter=retry_on_result_filter,
        wait_max_ms=wait_max_ms,
        wait_multiplier=wait_multiplier,
    )

  if inspect.iscoroutinefunction(fn):
    # Similar to the logic in tracing.py, we case on corofunction versus vanilla
    # function.

    @functools.wraps(fn)
    async def retry_coro_fn(*args, **kwargs):
      retry_wait_ms = 1.0

      while True:
        try:
          result = await fn(*args, **kwargs)
          if retry_on_result_filter(result):
            retry_wait_ms = min(wait_max_ms, retry_wait_ms * wait_multiplier)
            # time.sleep takes arguments in seconds.
            await asyncio.sleep(retry_wait_ms / 1000)
            continue
          else:
            return result
        except Exception as e:  # pylint: disable=broad-except
          if not retry_on_exception_filter(e):
            raise e
          retry_wait_ms = min(wait_max_ms, retry_wait_ms * wait_multiplier)
          # asyncio.sleep takes arguments in seconds.
          await asyncio.sleep(retry_wait_ms / 1000)

    return retry_coro_fn

  elif inspect.isfunction(fn):
    # Vanilla Python function; decorate as normal.

    @functools.wraps(fn)
    def retry_fn(*args, **kwargs):
      retry_wait_ms = 1.0

      while True:
        try:
          result = fn(*args, **kwargs)
          if retry_on_result_filter(result):
            retry_wait_ms = min(wait_max_ms, retry_wait_ms * wait_multiplier)
            # time.sleep takes arguments in seconds.
            time.sleep(retry_wait_ms / 1000)
            continue
          else:
            return result
        except Exception as e:  # pylint: disable=broad-except
          if not retry_on_exception_filter(e):
            raise e
          retry_wait_ms = min(wait_max_ms, retry_wait_ms * wait_multiplier)
          # time.sleep takes arguments in seconds.
          time.sleep(retry_wait_ms / 1000)

    return retry_fn

  else:
    raise TypeError(
        'Retrying expects Python function or coroutine function; '
        'passed {} of type {}.'.format(fn, type(fn))
    )
