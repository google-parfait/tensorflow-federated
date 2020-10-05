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
from typing import Any, Callable, Optional

import retrying
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl.compiler import compiler_pipeline
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import type_conversions


class RetryableError(Exception):
  """Raised when execution fails and can be retried."""


def _is_retryable_error(exception):
  return isinstance(exception, RetryableError)


def _unwrap(value):
  if isinstance(value, tf.Tensor):
    return value.numpy()
  elif isinstance(value, structure.Struct):
    return structure.Struct(
        (k, _unwrap(v)) for k, v in structure.iter_elements(value))
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
  elif (isinstance(val, structure.Struct) and not type_spec.is_federated()):
    type_spec.check_struct()
    v_elem = structure.to_elements(val)
    t_elem = structure.to_elements(type_spec)
    if ([k for k, _ in v_elem] != [k for k, _ in t_elem]):
      raise TypeError('Value {} does not match type {}.'.format(val, type_spec))
    ingested = []
    for (_, v), (_, t) in zip(v_elem, t_elem):
      ingested.append(_ingest(executor, v, t))
    ingested = await asyncio.gather(*ingested)
    return await executor.create_struct(
        structure.Struct(
            (name, val) for (name, _), val in zip(t_elem, ingested)))
  else:
    return await executor.create_value(val, type_spec)


async def _invoke(executor, comp, arg, result_type: computation_types.Type):
  """A coroutine that handles invocation.

  Args:
    executor: An instance of `executor_base.Executor`.
    comp: The first argument to `context_base.Context.invoke()`.
    arg: The optional second argument to `context_base.Context.invoke()`.
    result_type: The type signature of the result. This is used to convert the
      execution result into the proper container types.

  Returns:
    The result of the invocation.
  """
  if arg is not None:
    py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
  comp = await executor.create_value(comp)
  result = await executor.create_call(comp, arg)
  py_typecheck.check_type(result, executor_value_base.ExecutorValue)
  result_val = _unwrap(await result.compute())
  return type_conversions.type_to_py_container(result_val, result_type)


def _unwrap_execution_context_value(val):
  """Recursively removes wrapping from `val` under anonymous tuples."""
  if isinstance(val, structure.Struct):
    value_elements_iter = structure.iter_elements(val)
    return structure.Struct((name, _unwrap_execution_context_value(elem))
                            for name, elem in value_elements_iter)
  elif isinstance(val, ExecutionContextValue):
    return _unwrap_execution_context_value(val.value)
  else:
    return val


class ExecutionContext(context_base.Context):
  """Represents an execution context backed by an `executor_base.Executor`."""

  def __init__(self,
               executor_fn: executor_factory.ExecutorFactory,
               compiler_fn: Optional[Callable[[computation_base.Computation],
                                              Any]] = None):
    """Initializes an execution context.

    Args:
      executor_fn: Instance of `executor_factory.ExecutorFactory`.
      compiler_fn: A Python function that will be used to compile a computation.
    """
    py_typecheck.check_type(executor_fn, executor_factory.ExecutorFactory)
    self._executor_factory = executor_fn

    self._event_loop = asyncio.new_event_loop()
    self._event_loop.set_task_factory(
        tracing.propagate_trace_context_task_factory)
    if compiler_fn is not None:
      py_typecheck.check_callable(compiler_fn)
      self._compiler_pipeline = compiler_pipeline.CompilerPipeline(compiler_fn)
    else:
      self._compiler_pipeline = None

  @property
  def executor_factory(self):
    return self._executor_factory

  def ingest(self, val, type_spec):
    return ExecutionContextValue(val, type_spec)

  @retrying.retry(
      retry_on_exception=_is_retryable_error,
      wait_exponential_max=300000,  # in milliseconds
      wait_exponential_multiplier=1000,  # in milliseconds
      wait_jitter_max=1000  # in milliseconds
  )
  def invoke(self, comp, arg):
    comp.type_signature.check_function()
    # Save the type signature before compiling. Compilation currently loses
    # container types, so we must remember them here so that they can be
    # restored in the output.
    result_type = comp.type_signature.result
    if self._compiler_pipeline is not None:
      with tracing.span('ExecutionContext', 'Compile', span=True):
        comp = self._compiler_pipeline.compile(comp)

    with tracing.span('ExecutionContext', 'Invoke', span=True):

      @contextlib.contextmanager
      def executor_closer(ex_factory, cardinalities):
        """Wraps an Executor into a closeable resource."""
        ex = ex_factory.create_executor(cardinalities)
        try:
          yield ex
        except Exception as e:
          ex_factory.clean_up_executors()
          raise e

      if arg is not None:
        py_typecheck.check_type(arg, ExecutionContextValue)
        unwrapped_arg = _unwrap_execution_context_value(arg)
        cardinalities = cardinalities_utils.infer_cardinalities(
            unwrapped_arg, arg.type_signature)
      else:
        cardinalities = {}

      with executor_closer(self._executor_factory, cardinalities) as executor:
        py_typecheck.check_type(executor, executor_base.Executor)

        if arg is not None:
          arg = self._event_loop.run_until_complete(
              tracing.wrap_coroutine_in_current_trace_context(
                  _ingest(executor, unwrapped_arg, arg.type_signature)))

        return self._event_loop.run_until_complete(
            tracing.wrap_coroutine_in_current_trace_context(
                _invoke(executor, comp, arg, result_type)))
