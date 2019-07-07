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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import asyncio

import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_value_base
from tensorflow_federated.python.core.impl import type_utils


def _unwrap(value):
  if isinstance(value, tf.Tensor):
    return value.numpy()
  elif isinstance(value, anonymous_tuple.AnonymousTuple):
    return anonymous_tuple.map_structure(_unwrap, value)
  else:
    return value


async def _ingest(executor, val, type_spec):
  """A coroutine that handles ingestion.

  Args:
    executor: An instance of `executor_base.Executor`.
    val: The first argument to `context_base.Context.ingest()`.
    type_spec: The second argument to `context_base.Context.ingest()`.

  Returns:
    The result of the ingestion.

  Raises:
    ValueError: If the value does not match the type.
  """
  if isinstance(val, executor_value_base.ExecutorValue):
    return val
  elif (isinstance(val, anonymous_tuple.AnonymousTuple) and
        not isinstance(type_spec, computation_types.FederatedType)):
    py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
    v_elem = anonymous_tuple.to_elements(val)
    t_elem = anonymous_tuple.to_elements(type_spec)
    if ([k for k, _ in v_elem] != [k for k, _ in t_elem]):
      raise ValueError('Value {} does not match type {}.'.format(
          str(val), str(type_spec)))
    ingested = []
    for (_, v), (_, t) in zip(v_elem, t_elem):
      ingested.append(_ingest(executor, v, t))
    ingested = await asyncio.gather(*ingested)
    return await executor.create_tuple(
        anonymous_tuple.AnonymousTuple([
            (name, val) for (name, _), val in zip(t_elem, ingested)
        ]))
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


class ExecutionContext(context_base.Context):
  """Represents an execution context backed by an `executor_base.Executor`."""

  def __init__(self, executor):
    """Constructs a new execution context backed by `executor`.

    Args:
      executor: An instance of `executor_base.Executor`.
    """
    py_typecheck.check_type(executor, executor_base.Executor)
    self._executor = executor

  def ingest(self, val, type_spec):
    result = asyncio.get_event_loop().run_until_complete(
        _ingest(self._executor, val, type_spec))
    py_typecheck.check_type(result, executor_value_base.ExecutorValue)
    return result

  def invoke(self, comp, arg):
    return asyncio.get_event_loop().run_until_complete(
        _invoke(self._executor, comp, arg))
