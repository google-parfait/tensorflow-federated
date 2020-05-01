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
"""Tracing provider that creates human-readable logging for computations.

This module is helpful for understanding how computations are handled by an
established set of executor stacks. This process can be opaque if a user is
unfamiliar with a particular executor handles or modifies a specific
Computation as it makes its way through the executor stack. This
TracingProvider logs information about the computation in a presentable manner.
Note that it does not attempt to correlate asynchronous calls, and so using
this in conjunction with a ThreadDelegatingExecutor will not preserve logging
order as it otherwise would.
"""
from enum import Enum
import logging
import re
from typing import Any, Dict, Optional, Tuple

from tensorflow_federated.python.common_libs import tracing


class TraceFilterMode(Enum):
  VALUES = 0
  INTRINSICS = 1
  ALL = 2


def _pattern_union(*patterns):
  pattern_strs = map(lambda p: p.pattern, patterns)
  return re.compile('|'.join(pattern_strs))


_VALUE_PATTERN = re.compile('^create_([a-z]+)$')
_INTRINSIC_PATTERN = re.compile('^_compute_intrinsic_([a-z_]+)$')

_TRACE_FILTER_PATTERNS = {
    TraceFilterMode.VALUES:
        _VALUE_PATTERN,
    TraceFilterMode.INTRINSICS:
        _pattern_union(_VALUE_PATTERN, _INTRINSIC_PATTERN),
    TraceFilterMode.ALL:
        None,
}

DEFAULT_FORMAT_STRATEGY = lambda x: '{} {}'.format(x, type(x))


class ExecutionTracingProvider(tracing.TracingProvider):
  """Traces a subset of functions appearing in the current execution context.

  Note: This TracingProvider does not attempt to correlate log messages across
  threads, so executor stacks using ThreadDelegatingExecutor will generally lose
  their nesting structure in the logs.

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

  def span(
      self,
      scope: str,
      sub_scope: str,
      nonce: int,
      parent_span_yield: Optional[None],
      fn_args: Optional[Tuple[Any, ...]],
      fn_kwargs: Optional[Dict[str, Any]],
      trace_opts: Dict[str, Any],
  ):
    assert parent_span_yield is None
    del parent_span_yield, nonce, trace_opts

    regex = self._is_traceable_regex
    force_trace = regex is None
    if not self._is_traceable(sub_scope, regex, override=force_trace):
      # exit span early
      yield None

    else:
      # Pre-fn
      self._current_call_level += 1
      self._log_trace('call', scope, sub_scope, fn_args, fn_kwargs)
      yield None

      # Post-fn
      self._current_call_level -= 1
      self._log_trace('retr', scope, sub_scope, fn_args, fn_kwargs)

  @classmethod
  def _is_traceable(cls, method_name, regex, override=False):
    if override:
      return True
    match = regex.search(method_name)
    if match is None:
      return False
    return True

  def _log_trace(self, call_or_retr, scope, sub_scope, fn_args, fn_kwargs):
    """Logs a function trace on call in human-readable form.

    This function logs the scope and arguments with which a function has been
    called. Nested calls via `tracing.trace` preserve depth information, so the
    resulting log can be used to trace the AST for a given computation.
    """
    fn_args = fn_args or [None]
    fn_kwargs = fn_kwargs or {}

    ctx = fn_args[0]

    if ctx is not None:
      has_self = ctx.__class__.__name__ == scope

      if has_self and len(fn_args) > 1:
        fn_args = fn_args[1:]

    prefix_str = self._prefix(ctx, scope)
    main_str = '{m} {c}'.format(m=sub_scope, c=call_or_retr)
    suffix_args = ' '.join(self._format_object(a) for a in fn_args)
    suffix_kwargs = ' '.join(
        self._format_object(v) for k, v in fn_kwargs.items())
    final = '{p} {m} {sa} {sk}'.format(
        p=prefix_str, m=main_str, sa=suffix_args, sk=suffix_kwargs)

    logging.debug(final)

  def _prefix(self, ctx, type_string):
    indentation = '  ' * self._current_call_level
    template = '{indentation}{type} {hash}'

    if ctx is None:
      return template.format(
          indentation=indentation, type=type_string, hash='NO_ID')
    return template.format(
        indentation=indentation, type=type_string, hash=hash(ctx))

  def _format_object(self, x):
    strategy = self._formatting_strategies.get(
        type(x).__name__, self._default_strategy)
    return strategy(x)

  def _format_anon_tuple(self, tup):
    # pylint: disable=protected-access
    names = tup._name_array
    values = tup._element_array
    # pylint: enable=protected-access
    fmt_strs = []
    for n, v in zip(names, values):
      fmt_strs.append('{}: {}'.format(n, self._format_object(v)))
    return '({})'.format(', '.join(fmt_strs))

  def _register_formatting_strategy(self, type_string, formatting_strategy):
    self._formatting_strategies[type_string] = formatting_strategy

  # Register all types here
  def _make_formatting_strategies(self):
    self.register_executors()
    self.register_executor_values()
    self.register_computations()
    self.register_computation_types()
    self.register_primitives()

  # pylint: disable=unnecessary-lambda,missing-function-docstring
  def register_executors(self):
    for type_string in [
        'CachingExecutor',
        'ComposingExecutor',
        'EagerTFExecutor',
        'FederatingExecutor',
        'ReferenceResolvingExecutor',
        'RemoteExecutor',
        'SizingExecutor',
        'ThreadDelegatingExecutor',
        'TransformingExecutor',
    ]:
      self._register_formatting_strategy(
          type_string, lambda x, t=type_string: '<{} @{}>'.format(t, id(x)))

  def register_executor_values(self):
    for type_string in [
        'CachedValue',
        'CompositeValue',
        'EagerValue',
        'FederatingExecutorValue',
        'ReferenceResolvingExecutorValue',
        'RemoteValue',
    ]:
      self._register_formatting_strategy(
          type_string,
          lambda x, t=type_string: '<{} @{} : {}>'.format(
              t, id(x), str(x.type_signature)))

  def register_computation_types(self):
    for type_string in [
        'TensorType',
        'FederatedType',
        'FunctionType',
        'SequenceType',
    ]:
      self._register_formatting_strategy(type_string,
                                         lambda x: '<{}>'.format(x))

  def register_primitives(self):
    self._register_formatting_strategy(float,
                                       lambda x: '<{} : float>'.format(x))
    self._register_formatting_strategy(int, lambda x: '<{} : int>'.format(x))
    self._register_formatting_strategy(type(None), lambda _: '-')
    self._register_formatting_strategy(
        list,
        lambda x: '<' + ', '.join([self._format_object(xi) for xi in x]) + '>')
    self._register_formatting_strategy(
        tuple,
        lambda x: '<' + ', '.join(tuple(self._format_object(xi)
                                        for xi in x)) + '>')

    self._register_formatting_strategy(
        'EagerTensor', lambda x: '<{} @{}>'.format('EagerTensor', id(x)))

    self._register_formatting_strategy(
        'AnonymousTuple', lambda x: '<AnonymousTuple @{} : {}>'.format(
            id(x), self._format_anon_tuple(x)))

  def register_computations(self):
    self._register_formatting_strategy(
        'ComputationImpl',
        lambda x: '<ComputationImpl {} @{} : {}>'.format(
            x._computation_proto.WhichOneof('computation'),  # pylint: disable=protected-access
            id(x),
            x.type_signature))

    self._register_formatting_strategy(
        'ComputationPb', lambda x: '<ComputationPb {} @{} : {}>'.format(
            x.WhichOneof('computation'), id(x), x.type.WhichOneof('type')))

    self._register_formatting_strategy(
        'IntrinsicDef', lambda x: '<IntrinsicDef {} {} @{}>'.format(
            x.uri, x.type_signature, id(x)))

  # pylint: enable=unnecessary-lambda,missing-function-docstring
