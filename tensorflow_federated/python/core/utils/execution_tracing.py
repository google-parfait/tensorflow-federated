import asyncio
import contextlib
from enum import Enum
import functools
import inspect
import logging
import re

from tensorflow.python.framework.ops import EagerTensor as tf_EagerTensor

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.executors import caching_executor
from tensorflow_federated.python.core.impl.executors import composing_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.executors import sizing_executor
from tensorflow_federated.python.core.impl.executors import thread_delegating_executor
from tensorflow_federated.python.core.impl.executors import transforming_executor


class TraceFilterMode(Enum):
  VALUES = 0
  INTRINSICS = 1
  ALL = 2


def pattern_union(*patterns):
  pattern_strs = map(lambda p: p.pattern, patterns)
  return re.compile("|".join(pattern_strs))


_VALUE_PATTERN = re.compile('^create_([a-z]+)$')
_INTRINSIC_PATTERN = re.compile('^_compute_intrinsic_([a-z_]+)$')

_TRACE_FILTER_PATTERNS = {
    TraceFilterMode.VALUES:
        _VALUE_PATTERN,
    TraceFilterMode.INTRINSICS:
        pattern_union(_VALUE_PATTERN, _INTRINSIC_PATTERN),
    TraceFilterMode.ALL:
        None,
}

DEFAULT_FORMAT_STRATEGY = lambda x: "{} {}".format(x, type(x))


class ExecutionTracingProvider(tracing.TracingProvider):
  """Traces a subset of functions appearing in the current execution context.

  This implementation does not require storing additional trace context state,
  so most methods are no-ops.
  """

  def __init__(self,
               trace_filter_mode=TraceFilterMode.INTRINSICS,
               default_format_strategy=DEFAULT_FORMAT_STRATEGY):
    assert isinstance(trace_filter_mode, TraceFilterMode)

    # establish what we're tracing
    self._is_traceable_regex = _TRACE_FILTER_PATTERNS[trace_filter_mode]

    # fill in formatting strategies
    self._default_strategy = default_format_strategy
    self._formatting_strategies = dict()
    self._make_formatting_strategies()

    # indentation management
    # start at -1 so the first call prints at 0
    self._current_call_level = -1

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

    class_name, method_name = tracing.func_to_class_and_method(fn)

    regex = self._is_traceable_regex
    force_trace = regex is None
    if not self._is_traceable(method_name, regex, override=force_trace):
      return fn

    def _pre_fn(*args, **kwargs):
      self._current_call_level += 1
      self._trace_method(fn, class_name, method_name, "call")(*args, **kwargs)

    def _post_fn(*args, **kwargs):
      self._trace_method(fn, class_name, method_name, "retr")(*args, **kwargs)
      self._current_call_level -= 1

    if inspect.iscoroutinefunction(fn):

      @functools.wraps(fn)
      async def async_fn(*args, **kwargs):
        _pre_fn(*args, **kwargs)
        retval = await fn(*args, **kwargs)
        _post_fn(*args, **kwargs)
        return retval

      return async_fn

    else:

      @functools.wraps(fn)
      def sync_fn(*args, **kwargs):
        _pre_fn(*args, **kwargs)
        retval = fn(*args, **kwargs)
        _post_fn(*args, **kwargs)
        return retval

      return sync_fn

  def span(self, unused_scope, unused_sub_scope, **unused_trace_opts):
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

  @classmethod
  def _is_traceable(cls, method_name, regex, override=False):
    if override:
      return True
    match = regex.search(method_name)
    if match is None:
      return False
    return True

  def _trace_method(self, fn, class_name, method_name, call_or_retr):
    # rough workaround -- Python can't distinguish that fn is a method here
    fn_sig = inspect.signature(fn)
    has_self = 'self' in fn_sig.parameters

    def _log_fn(*args, **kwargs):
      nonlocal has_self

      ctx = None
      if has_self:
        ctx = args[0]
        args = args[1:]

      prefix_str = self._prefix(ctx, class_name)
      main_str = "{m} {c}".format(m=method_name, c=call_or_retr)
      suffix_args = " ".join(self.format_object(a) for a in args)
      suffix_kwargs = " ".join(self.format_object(v) for k, v in kwargs.items())
      logging.debug("{p} {m} {sa} {sk}".format(
          p=prefix_str, m=main_str, sa=suffix_args, sk=suffix_kwargs))

    return _log_fn

  def _prefix(self, ctx, type_):
    indentation = '  ' * self._current_call_level
    template = "{indentation}{type} {hash}"

    if ctx is None:
      return template.format(indentation=indentation, type=type_, hash="NO_ID")
    return template.format(indentation=indentation, type=type_, hash=hash(ctx))

  def _format_object(self, x):
    strategy = self._formatting_strategies.get(type(x), self._default_strategy)
    return strategy(x)

  def _format_anon_tuple(self, tup):
    names = tup._name_array
    values = tup._element_array
    fmt_strs = []
    for n, v in zip(names, values):
      fmt_strs.append("{}: {}".format(n, self._format_object(v)))
    return "({})".format(", ".join(fmt_strs))

  def _register_formatting_strategy(self, type_, formatting_strategy):
    self._formatting_strategies[type_] = formatting_strategy

  # Register all types here
  def _make_formatting_strategies(self):
    self.register_executors()
    self.register_executor_values()
    self.register_computations()
    self.register_computation_types()
    self.register_primitives()

  def register_executors(self):
    register = self._register_formatting_strategy

    for type_ in [
        caching_executor.CachingExecutor,
        composing_executor.ComposingExecutor,
        eager_tf_executor.EagerTFExecutor,
        federating_executor.FederatingExecutor,
        reference_resolving_executor.ReferenceResolvingExecutor,
        remote_executor.RemoteExecutor,
        sizing_executor.SizingExecutor,
        thread_delegating_executor.ThreadDelegatingExecutor,
        transforming_executor.TransformingExecutor,
    ]:
      register(type_, lambda x: "<{} @{}>".format(type_.__name__, id(x)))

  def register_executor_values(self):
    register = self._register_formatting_strategy

    for type_ in [
        caching_executor.CachedValue,
        composing_executor.CompositeValue,
        eager_tf_executor.EagerValue,
        federating_executor.FederatingExecutorValue,
        reference_resolving_executor.ReferenceResolvingExecutorValue,
        remote_executor.RemoteValue,
    ]:
      register(
          type_, lambda x: "<{} @{} : {}>".format(type_.__name__, id(x),
                                                  str(x.type_signature)))

  def register_computation_types(self):
    register = self._register_formatting_strategy

    for type_ in [
        computation_types.TensorType,
        computation_types.FederatedType,
        computation_types.FunctionType,
        computation_types.SequenceType,
    ]:
      register(type_, lambda x: "<{}>".format(x))

  def register_primitives(self):
    register = self._register_formatting_strategy

    register(float, lambda x: "<{} : float>".format(x))
    register(int, lambda x: "<{} : int>".format(x))
    register(type(None), lambda _: "-")
    register(
        list,
        lambda x: "<" + ", ".join([self._format_object(xi) for xi in x]) + ">")
    register(
        tuple,
        lambda x: "<" + ", ".join(tuple(self._format_object(xi)
                                        for xi in x)) + ">")

    register(tf_EagerTensor,
             lambda x: "<{} @{}>".format(tf_EagerTensor.__name__, id(x)))

    register(
        anonymous_tuple.AnonymousTuple,
        lambda x: "<AnonymousTuple @{} : {}>".format(
            id(x), self._format_anon_tuple(x)))

  def register_computations(self):
    register = self._register_formatting_strategy

    register(
        computation_impl.ComputationImpl,
        lambda x: "<ComputationImpl {} @{} : {}>".format(
            x._computation_proto.WhichOneof('computation'), id(x), x.
            type_signature))

    register(
        computation_pb2.Computation,
        lambda x: "<ComputationPb {} @{} : {}>".format(
            x.WhichOneof('computation'), id(x), x.type.WhichOneof('type')))

    register(
        intrinsic_defs.IntrinsicDef,
        lambda x: "<IntrinsicDef {} {} @{}>".format(x.uri, x.type_signature,
                                                    id(x)))
