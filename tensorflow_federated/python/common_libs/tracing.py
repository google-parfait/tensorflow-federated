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

  * wrap_coroutine_in_trace_context wraps a coroutine such that it
    inherits the ambient trace context. It should be used when executing a
    coroutine that should inherit trace context from the current thread or
    task.
  * EventLoops should use the Task factory provided by
    propagate_trace_context_task_factory by calling
    `set_task_factory(propagate_trace_context_task_factory)`.
"""

import abc
import asyncio
from collections.abc import Generator
import contextlib
import functools
import inspect
import random
import sys
import threading
import time
from typing import Generic, Optional, TypeVar, Union

from absl import logging

from tensorflow_federated.python.common_libs import py_typecheck


class TracedSpan:
  """The trace was wrapping a non-function span.

  This value will be given back from `TracingProvider::span`'s first `yield`
  if the trace was being used to wrap a `span` rather than a whole function.
  """

  pass


class TracedFunctionReturned:
  """The traced function returned successfully.

  This value will be given back from `TracingProvider::span`'s first `yield`
  if the function being traced returned normally. The return value will be kept
  in the `value` field.
  """

  def __init__(self, value):
    self.value = value


class TracedFunctionThrew:
  """The traced function threw an exception.

  This value will be given back from `TracingProvider::span`'s first `yield`
  if the function being traced threw an exception.
  """

  def __init__(self, error_type, error_value, traceback):
    self.error_type = error_type
    self.error_value = error_value
    self.traceback = traceback


TraceResult = Union[TracedSpan, TracedFunctionReturned, TracedFunctionThrew]

T = TypeVar('T')


class TracingProvider(Generic[T], metaclass=abc.ABCMeta):
  """Abstract base class for tracers."""

  @abc.abstractmethod
  def span(
      self,
      scope: str,
      sub_scope: str,
      nonce: int,
      parent_span_yield: Optional[T],
      fn_args: Optional[tuple[object, ...]],
      fn_kwargs: Optional[dict[str, object]],
      trace_opts: dict[str, object],
  ) -> Generator[T, TraceResult, None]:
    """Create a new tracing span.

    Args:
      scope: String name of the scope, often the class name.
      sub_scope: String name of the sub-scope, often the function name.
      nonce: Number used to correlate tracing messages relating to the same
        function invocation.
      parent_span_yield: The value yielded by the most recently started (and not
        exited) call to `span` on this `TracingProvider` on the current
        `asyncio.Task` or thread (when running outside of an async context).
      fn_args: When this tracing provider wraps a function, this will be a tuple
        containing all of the non-keyword function arguments.
      fn_kwargs: When this tracing provider wraps a function, this will be a
        dict containing all of the keyword function arguments.
      trace_opts: User-provided options to the span constructor.
        `TracingProvider`s should ignore unknown options.

    Returns:
      A `Generator` which will be immediately started and run up until it
      yields for the first time. The value yielded by this `Generator`
      will be passed on to nested calls to `span`. When the spanned code ends,
      a `TraceResult` will be passed back through the `yield`.
    """
    raise NotImplementedError

  def wrap_rpc(
      self, parent_span_yield: Optional[T]
  ) -> contextlib.AbstractContextManager[None]:
    """Wrap an RPC call so that it can carry over the `parent_span_yield`."""
    del parent_span_yield
    return contextlib.nullcontext()

  def receive_rpc(self) -> Optional[T]:
    """Unpack `parent_span_yield` from the receiving end of an RPC."""
    return None


class LoggingTracingProvider(TracingProvider[None]):
  """Implements TracingProvider and outputs the results via logging.

  This implementation does not require storing additional trace context state,
  so most methods are no-ops.
  """

  def span(
      self,
      scope: str,
      sub_scope: str,
      nonce: int,
      parent_span_yield: Optional[None],
      fn_args: Optional[tuple[object, ...]],
      fn_kwargs: Optional[dict[str, object]],
      trace_opts: dict[str, object],
  ) -> Generator[None, TraceResult, None]:
    assert parent_span_yield is None
    del parent_span_yield, fn_args, fn_kwargs, trace_opts
    start_time = time.time()
    logging.debug('(%s) Entering %s.%s', nonce, scope, sub_scope)
    yield None
    logging.debug(
        '(%s) Exiting %s.%s. Elapsed time %f',
        nonce,
        scope,
        sub_scope,
        time.time() - start_time,
    )


_global_tracing_providers = [LoggingTracingProvider()]


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

  scope, sub_scope = _func_to_class_and_method(fn)

  # Note: in a classic "what color is your function" situation,
  # we unfortunately have to duplicate the wrapping of the
  # underlying function in order to cover both the sync and async cases.
  if inspect.iscoroutinefunction(fn):

    @functools.wraps(fn)
    async def async_trace(*fn_args, **fn_kwargs):
      # Produce the span generator
      span_gen = _span_generator(
          scope, sub_scope, trace_kwargs, fn_args=fn_args, fn_kwargs=fn_kwargs
      )
      # Run up until the first yield
      next(span_gen)
      completed = False
      # Run the underlying function, recording the resulting value or exception
      # and passing it back to the span generator
      try:
        result = await fn(*fn_args, **fn_kwargs)
        completed = True
        try:
          span_gen.send(TracedFunctionReturned(result))
        except StopIteration:
          pass
        return result
      except:
        if not completed:
          error_type, error_value, traceback = sys.exc_info()
          try:
            span_gen.send(
                TracedFunctionThrew(error_type, error_value, traceback)
            )
          except StopIteration:
            pass
        raise

    return async_trace
  else:

    @functools.wraps(fn)
    def sync_trace(*fn_args, **fn_kwargs):
      span_gen = _span_generator(
          scope, sub_scope, trace_kwargs, fn_args=fn_args, fn_kwargs=fn_kwargs
      )
      next(span_gen)
      completed = False
      try:
        result = fn(*fn_args, **fn_kwargs)
        completed = True
        try:
          span_gen.send(TracedFunctionReturned(result))
        except StopIteration:
          pass
        return result
      except:
        if not completed:
          error_type, error_value, traceback = sys.exc_info()
          try:
            span_gen.send(
                TracedFunctionThrew(error_type, error_value, traceback)
            )
          except StopIteration:
            pass
        raise

    return sync_trace


# The code below manages the active "span yields" for a task or thread.
# Here's a quick summary of how that works.
#
# A "span yield" is a value `yield`ed by the `TracingProvider.span` function.
# The span yields for the current encompassing span need to be tracked so that
# they can be passed to new calls to `span` as the `parent_span_yield`.
#
# Typically, these would be tracked with a thread-local. However, async tasks
# can interleave on a single thread, so it makes more sense for them to track
# "task locals".
#
# `_current_span_yields` and `_set_span_yields` below handle the logic of
# tracking these spans. If we're in an async context, they'll read and write
# to the current async tasks, but fall back to using a thread local if we're
# in a synchronous context.

# A single yielded value for each currently-active TracingProvider.
SpanYields = list[Optional[object]]


class ThreadLocalSpanYields(threading.local):
  """The span set for the current thread.

  This is only used when outside of an async context.
  """

  def __init__(self):
    super().__init__()
    self._span_yields: Optional[SpanYields] = None

  def set(self, span_yields: Optional[SpanYields]):
    self._span_yields = span_yields

  def get(self) -> Optional[SpanYields]:
    return self._span_yields


_non_async_span_yields = ThreadLocalSpanYields()


def _current_task() -> Optional[asyncio.Task]:
  """Get the current running task, or `None` if no task is running."""
  # Note: `current_task` returns `None` if there is no current task, but it
  # throws if no currently running async loop.
  try:
    return asyncio.current_task()
  except RuntimeError:
    return None


def _current_span_yields() -> SpanYields:
  """Returns the current parent span yield list."""
  task = _current_task()
  if task is None:
    # There is no current task, so we're not running in an async context.
    # Grab the spans from the current thread.
    spans = _non_async_span_yields.get()
  else:
    spans = getattr(task, 'trace_span_yields', None)
  if spans is None:
    spans = [None for _ in range(len(_global_tracing_providers))]
  assert len(_global_tracing_providers) == len(spans)
  return spans


def _set_span_yields(span_yields: Optional[SpanYields]):
  """Sets the current parent span list."""
  task = _current_task()
  if task is None:
    # There is no current task, so we're not running in an async context.
    # Set the spans for the current thread.
    _non_async_span_yields.set(span_yields)
  else:
    setattr(task, 'trace_span_yields', span_yields)


@contextlib.contextmanager
def _with_span_yields(span_yields: Optional[SpanYields]):
  """Context manager which sets and unsets the current parent span list."""
  old_span_yields = _current_span_yields()
  _set_span_yields(span_yields)
  yield None
  _set_span_yields(old_span_yields)


@contextlib.contextmanager
def span(scope, sub_scope, **trace_opts):
  """Creates a `ContextManager` that wraps the code in question with a span."""
  span_gen = _span_generator(scope, sub_scope, trace_opts)
  next(span_gen)
  yield
  try:
    span_gen.send(TracedSpan())
  except StopIteration:
    pass


def _span_generator(
    scope, sub_scope, trace_opts, fn_args=None, fn_kwargs=None
) -> Generator[None, TraceResult, None]:
  """Wraps up all the `TracingProvider.span` generators into one."""
  # Create a nonce so that all of the traces from this span can be associated
  # with one another.
  nonce = random.randrange(1000000000)
  # Call `span` on all the global `TraceProvider`s and run it up until `yield`.
  span_generators = []
  new_span_yields: SpanYields = []
  for tp, parent_span_yield in zip(
      _global_tracing_providers, _current_span_yields()
  ):
    new_span_gen = tp.span(
        scope,
        sub_scope,
        nonce,
        parent_span_yield,
        fn_args,
        fn_kwargs,
        trace_opts,
    )
    new_span_yield = next(new_span_gen)
    span_generators.append(new_span_gen)
    new_span_yields.append(new_span_yield)
  # Set the values yielded by the `span` calls above to be the current span
  # yields, and yield so that the function can be run to completion.
  with _with_span_yields(new_span_yields):
    result = yield None
  # Send the result of the function to all of the generators so that they can
  # complete.
  for span_gen in reversed(span_generators):
    try:
      span_gen.send(result)
    except StopIteration:
      pass


def propagate_trace_context_task_factory(loop, coro):
  """Creates a new task on `loop` to run `coro`, inheriting current spans."""
  child_task = asyncio.tasks.Task(coro, loop=loop)
  trace_span_yields = _current_span_yields()
  setattr(child_task, 'trace_span_yields', trace_span_yields)
  return child_task


def wrap_coroutine_in_current_trace_context(coro):
  """Wraps the coroutine in the currently active span."""
  trace_span_yields = _current_span_yields()

  async def _wrapped():
    with _with_span_yields(trace_span_yields):
      return await coro

  return _wrapped()


@contextlib.contextmanager
def wrap_rpc_in_trace_context():
  """Attempts to record the trace context into the enclosed RPC call."""
  with contextlib.ExitStack() as stack:
    for tp, parent_span_yield in zip(
        _global_tracing_providers, _current_span_yields()
    ):
      stack.enter_context(tp.wrap_rpc(parent_span_yield))
    yield None


@contextlib.contextmanager
def with_trace_context_from_rpc():
  """Attempts to pick up the trace context from the receiving RPC call."""
  span_yields_from_rpc = [tp.receive_rpc() for tp in _global_tracing_providers]
  with _with_span_yields(span_yields_from_rpc):
    yield None


def add_tracing_provider(tracing_provider: TracingProvider):
  """Add to the global list of tracing providers."""
  py_typecheck.check_type(tracing_provider, TracingProvider)
  _global_tracing_providers.append(tracing_provider)


def set_tracing_providers(tracing_providers: list[TracingProvider]):
  """Set the global list of tracing providers, replacing any existing."""
  py_typecheck.check_type(tracing_providers, list)
  for tp in tracing_providers:
    py_typecheck.check_type(tp, TracingProvider)
  global _global_tracing_providers
  _global_tracing_providers = tracing_providers


def _func_to_class_and_method(fn) -> tuple[str, str]:
  """Returns the names of the function's class and method."""
  split = fn.__qualname__.split('.')
  if len(split) >= 2:
    class_name = split[-2]
    method_name = split[-1]
  else:
    module_name = fn.__module__
    class_name = module_name.split('.')[-1]
    method_name = fn.__name__
  return class_name, method_name
