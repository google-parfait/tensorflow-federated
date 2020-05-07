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
"""A context for execution based on an embedded executor instance."""

import asyncio
import contextlib

import retrying
import tensorflow.compat.v2 as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_value_base


class RetryableError(Exception):
  """Raised when execution fails and can be retried."""


def _is_retryable_error(exception):
  return isinstance(exception, RetryableError)


def _unwrap(value):
  if isinstance(value, tf.Tensor):
    return value.numpy()
  elif isinstance(value, anonymous_tuple.AnonymousTuple):
    return anonymous_tuple.AnonymousTuple(
        (k, _unwrap(v)) for k, v in anonymous_tuple.iter_elements(value))
  else:
    return value


class ExecutionContextValue(typed_object.TypedObject):
  """Wrapper class for values produced by `ExecutionContext`."""

  def __init__(self, value, type_spec):
    py_typecheck.check_type(type_spec, computation_types.Type)
    self._value = value
    self._type_spec = type_spec

  @property
  def type_signature(self):
    return self._type_spec

  @property
  def value(self):
    return self._value


async def _ingest(executor, val, type_spec):
  """A coroutine that handles ingestion.

  Args:
    executor: An instance of `executor_base.Executor`.
    val: The first argument to `context_base.Context.ingest()`.
    type_spec: The second argument to `context_base.Context.ingest()`.

  Returns:
    The result of the ingestion.

  Raises:
    TypeError: If the `val` is not a value of type `type_spec`.
  """
  if isinstance(val, executor_value_base.ExecutorValue):
    return val
  elif (isinstance(val, anonymous_tuple.AnonymousTuple) and
        not isinstance(type_spec, computation_types.FederatedType)):
    py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
    v_elem = anonymous_tuple.to_elements(val)
    t_elem = anonymous_tuple.to_elements(type_spec)
    if ([k for k, _ in v_elem] != [k for k, _ in t_elem]):
      raise TypeError('Value {} does not match type {}.'.format(val, type_spec))
    ingested = []
    for (_, v), (_, t) in zip(v_elem, t_elem):
      ingested.append(_ingest(executor, v, t))
    ingested = await asyncio.gather(*ingested)
    return await executor.create_tuple(
        anonymous_tuple.AnonymousTuple(
            (name, val) for (name, _), val in zip(t_elem, ingested)))
  else:
    return await executor.create_value(val, type_spec)


async def _invoke(executor, comp, arg):
  """A coroutine that handles invocation.

  Args:
    executor: An instance of `executor_base.Executor`.
    comp: The first argument to `context_base.Context.invoke()`.
    arg: The optional second argument to `context_base.Context.invoke()`.

  Returns:
    The result of the invocation.
  """
  py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
  result_type = comp.type_signature.result
  if arg is not None:
    py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
  comp = await executor.create_value(comp)
  result = await executor.create_call(comp, arg)
  py_typecheck.check_type(result, executor_value_base.ExecutorValue)
  result_val = _unwrap(await result.compute())
  if type_utils.is_anon_tuple_with_py_container(result_val, result_type):
    return type_utils.convert_to_py_container(result_val, result_type)
  else:
    return result_val


def _unwrap_execution_context_value(val):
  """Recursively removes wrapping from `val` under anonymous tuples."""
  if isinstance(val, anonymous_tuple.AnonymousTuple):
    value_elements_iter = anonymous_tuple.iter_elements(val)
    return anonymous_tuple.AnonymousTuple(
        (name, _unwrap_execution_context_value(elem))
        for name, elem in value_elements_iter)
  elif isinstance(val, ExecutionContextValue):
    return _unwrap_execution_context_value(val.value)
  else:
    return val


class ExecutionContext(context_base.Context):
  """Represents an execution context backed by an `executor_base.Executor`."""

  def __init__(self, executor_fn: executor_factory.ExecutorFactory):
    """Initializes execution context.

    Args:
      executor_fn: Instance of `executor_factory.ExecutorFactory`.
    """
    py_typecheck.check_type(executor_fn, executor_factory.ExecutorFactory)
    self._executor_factory = executor_fn

  def ingest(self, val, type_spec):
    return ExecutionContextValue(val, type_spec)

  @retrying.retry(
      retry_on_exception=_is_retryable_error,
      wait_exponential_max=300000,  # in milliseconds
      wait_exponential_multiplier=1000,  # in milliseconds
      wait_jitter_max=1000  # in milliseconds
  )
  def invoke(self, comp, arg):

    with tracing.span('ExecutionContext', 'Invoke', span=True):

      @contextlib.contextmanager
      def executor_closer(wrapped_executor):
        """Wraps an Executor into a closeable resource."""
        try:
          yield wrapped_executor
        finally:
          wrapped_executor.close()

      if arg is not None:
        py_typecheck.check_type(arg, ExecutionContextValue)
        unwrapped_arg = _unwrap_execution_context_value(arg)
        cardinalities = cardinalities_utils.infer_cardinalities(
            unwrapped_arg, arg.type_signature)
      else:
        cardinalities = {}

      with executor_closer(
          self._executor_factory.create_executor(cardinalities)) as executor:
        py_typecheck.check_type(executor, executor_base.Executor)

        def get_event_loop():
          new_loop = asyncio.new_event_loop()
          new_loop.set_task_factory(
              tracing.propagate_trace_context_task_factory)
          return new_loop

        event_loop = get_event_loop()

        if arg is not None:
          arg = event_loop.run_until_complete(
              tracing.wrap_coroutine_in_current_trace_context(
                  _ingest(executor, unwrapped_arg, arg.type_signature)))

        return event_loop.run_until_complete(
            tracing.wrap_coroutine_in_current_trace_context(
                _invoke(executor, comp, arg)))
