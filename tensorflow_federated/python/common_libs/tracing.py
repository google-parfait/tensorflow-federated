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
"""Utility functions for instrumenting code with timing and tracing data.

This module provides several functions for preserving trace context across
various boundaries, namely between asyncio and regular python code:

  * task_trace_context should be used when executing synchronous code
    from within an async function.
  * run_coroutine_threadsafe_in_{ambient,task}_trace_context should be used as a
    drop-in replacement for asyncio.run_coroutine_threadsafe. Choose 'task'
    if running in an async context, 'ambient' otherwise.
  * run_coroutine_in_ambient_trace_context wraps a coroutine such that it
    inherits the ambient trace context. It should be used when executing a
    coroutine from non-async code, e.g. EventLoop.run_until_complete.
  * EventLoops should use the Task factory provided by
    propagate_trace_context_task_factory by calling
    `set_task_factory(propagate_trace_context_task_factory)`.
"""

import abc
import asyncio
import contextlib
import functools
import inspect
import random
import time

from absl import logging

from tensorflow_federated.python.common_libs import py_typecheck


class TracingProvider(metaclass=abc.ABCMeta):
  """Abstract base class for tracing providers.

  The tracing provider is responsible for both creating the tracing spans and
  providing functions to preserve this context across various boundaries, namely
  when transitioning between asyncio and regular python code. See module
  documentation for more information.
  """

  @abc.abstractmethod
  def trace(self, fn=None, **kwargs):
    """Decorate a method with tracing.

    Works with async or regular functions. If no kwargs specified, can be used
    without parens. Parameters accepted in kwargs depends on the specific
    tracing provider.

    Args:
      fn: The function to decorate.
      **kwargs: Options for configuring the trace.

    Returns:
      The decorated function.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def span(self, *unused_args):
    """"Create a Context which records its lifespan in a monitoring trace."""

  @abc.abstractmethod
  def task_trace_context(self):
    """Returns a context manager which installs the Task's trace context.

    This function should be used to inherit the trace context when calling a
    synchronous function from async code.

    """
    raise NotImplementedError

  @abc.abstractmethod
  def propagate_trace_context_task_factory(self, loop, coro):
    """Task factory which preserves trace context across Tasks.

    Suitable for use with EventLoop.set_task_factory.

    Args:
      loop: The EventLoop for the newly-created Task.
      coro: The underlying coroutine for the task.  This method is currently a
        placeholder for tracing functionality such as OpenTelemetry.

    Returns:
      The newly-created task.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def run_coroutine_threadsafe_in_ambient_trace_context(self, coro, loop):
    """Propagate the ambient trace context into an asyncio coroutine.

    Args:
      coro: Corouting to execute.
      loop: Loop to execute in.  This method is currently a placeholder for
        tracing functionality such as OpenTelemetry.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def run_coroutine_threadsafe_in_task_trace_context(self, coro, loop):
    """Propagate the task trace context into an asyncio coroutine.

    Args:
      coro: Corouting to execute.
      loop: Loop to execute in.  This method is currently a placeholder for
        tracing functionality such as OpenTelemetry.
    """
    return NotImplementedError

  @abc.abstractmethod
  def run_coroutine_in_ambient_trace_context(self, coro):
    """Wrap coro to run in the current ambient trace context.

    This method is currently a placeholder for tracing
    functionality such as OpenTelemetry.

    Args:
      coro: The coroutine to wrap.

    Returns:
      The input coroutine.
    """
    return NotImplementedError


class LoggingTracingProvider(TracingProvider):
  """Implements TracingProvider and outputs the results via logging.

  This implementation does not require storing additional trace context state,
  so most methods are no-ops.
  """

  def trace(self, fn=None, **kwargs):
    """Decorate a method with tracing.

    Works with async or regular functions. If no kwargs specified, can be used
    without parens. Parameters accepted in kwargs depends on the implementation
    vary across platforms.

    Args:
      fn: The function to decorate.
      **kwargs: Options for configuring the trace.

    Returns:
      The decorated function.
    """

    class_name, method_name = func_to_class_and_method(fn)

    logging.debug('Decorating sync method: %s.%s', class_name, method_name)

    def _pre_fn():
      # This nonce is used to correlate log messages for a single invocation.
      nonce = random.randrange(1000000000)
      start_time = time.time()
      logging.debug('(%s) Entering %s.%s', nonce, class_name, method_name)
      return nonce, start_time

    def _post_fn(nonce, start_time):
      logging.debug('(%s) Exiting %s.%s. Elapsed time %f', nonce, class_name,
                    method_name,
                    time.time() - start_time)

    if inspect.iscoroutinefunction(fn):

      @functools.wraps(fn)
      async def async_fn(*args, **kwargs):
        nonce, start_time = _pre_fn()
        retval = await fn(*args, **kwargs)
        _post_fn(nonce, start_time)
        return retval

      return async_fn

    else:

      @functools.wraps(fn)
      def sync_fn(*args, **kwargs):
        nonce, start_time = _pre_fn()
        retval = fn(*args, **kwargs)
        _post_fn(nonce, start_time)
        return retval

      return sync_fn

  def span(self, unused_scope, unused_sub_scope):
    """"No-op implementation of span."""

    class NoSuchError(Exception):
      pass

    # contextlib.suppress(NoSuchError) is used to create a no-op Context and
    # should be replaced by contextlib.nullcontext when it becomes available.
    return contextlib.suppress(NoSuchError)

  def task_trace_context(self):
    """No-op implementation of task_trace_context."""

    class NoSuchError(Exception):
      pass

    # contextlib.suppress(NoSuchError) is used to create a no-op Context and
    # should be replaced by contextlib.nullcontext when it is available.
    return contextlib.suppress(NoSuchError)

  def propagate_trace_context_task_factory(self, loop, coro):
    """No-op implementation of propagate_trace_context_task_factory."""
    return asyncio.tasks.Task(coro, loop=loop)

  def run_coroutine_threadsafe_in_ambient_trace_context(self, coro, loop):
    """No-op implementation of run_coroutine_threadsafe_in_ambient_trace_context."""
    return asyncio.run_coroutine_threadsafe(coro, loop)

  def run_coroutine_threadsafe_in_task_trace_context(self, coro, loop):
    """No-op implementation of run_coroutine_threadsafe_in_ambient_trace_context."""
    return asyncio.run_coroutine_threadsafe(coro, loop)

  def run_coroutine_in_ambient_trace_context(self, coro):
    """No-op implementation of run_coroutine_in_ambient_trace_context."""
    return coro


_global_tracing_provider = LoggingTracingProvider()


def trace(fn=None, **trace_kwargs):
  """Delegates to the current global `TracingProvider`.

  Note that this function adds a layer of indirection so that the decoration
  happens when the method is executed. This is necessary so that the current
  TracingProvider is used.

  Args:
    fn: Function to decorate.
    **trace_kwargs: Tracing options. Supported options differ by tracing
      provider.

  Returns:
    Decorated instance of fn.
  """
  if fn is None:
    return functools.partial(trace, **trace_kwargs)

  class MemoizingDecorator():
    """Decorates a function using the global tracing provider.

    This happens once when the function is invoked and the decorated funciton
    is reused on future invocations.
    """

    def __init__(self, decorated):
      self._decorated = decorated
      self._fn = None

    def __call__(self, *fn_args, **fn_kwargs):
      if self._fn is None:
        self._fn = _global_tracing_provider.trace(self._decorated,
                                                  **trace_kwargs)
      return self._fn(*fn_args, **fn_kwargs)

  memoizing_decorator = MemoizingDecorator(fn)

  if inspect.iscoroutinefunction(fn):
    # The decorator of an async fn needs to return an async fn, so create one
    # that simply delegates to the memoizing_decorator.
    async def async_trace(*fn_args, **fn_kwargs):
      return await memoizing_decorator(*fn_args, **fn_kwargs)

    return async_trace
  else:
    return memoizing_decorator


def span(scope, sub_scope):
  """Delegates to the current global `TracingProvider`."""
  return _global_tracing_provider.span(scope, sub_scope)


def task_trace_context():
  """Delegates to the current global `TracingProvider`."""
  return _global_tracing_provider.task_trace_context()


def propagate_trace_context_task_factory(loop, coro):
  """Delegates to the current global `TracingProvider`."""
  return _global_tracing_provider.propagate_trace_context_task_factory(
      loop, coro)


def run_coroutine_threadsafe_in_ambient_trace_context(coro, loop):
  """Delegates to the current global `TracingProvider`."""
  return _global_tracing_provider.run_coroutine_threadsafe_in_ambient_trace_context(
      coro, loop)


def run_coroutine_threadsafe_in_task_trace_context(coro, loop):
  """Delegates to the current global `TracingProvider`."""
  return _global_tracing_provider.run_coroutine_threadsafe_in_task_trace_context(
      coro, loop)


def run_coroutine_in_ambient_trace_context(coro):
  """Delegates to the current global `TracingProvider`."""
  return _global_tracing_provider.run_coroutine_in_ambient_trace_context(coro)


def set_tracing_provider(tracing_provider):
  """Set the global tracing provider."""
  py_typecheck.check_type(tracing_provider, TracingProvider)
  global _global_tracing_provider
  _global_tracing_provider = tracing_provider


def func_to_class_and_method(fn):
  module_name = fn.__module__
  split = fn.__qualname__.split('.')
  if len(split) >= 2:
    class_name = split[-2]
    method_name = split[-1]
  else:
    class_name = module_name.split('.')[-1]
    method_name = fn.__name__
  return class_name, method_name
