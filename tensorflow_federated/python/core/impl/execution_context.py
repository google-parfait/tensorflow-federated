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
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import type_utils


def _unwrap(value):
  if isinstance(value, tf.Tensor):
    return value.numpy()
  elif isinstance(value, anonymous_tuple.AnonymousTuple):
    return anonymous_tuple.map_structure(_unwrap, value)
  else:
    return value


async def _invoke(executor, comp, arg):
  """A coroutine that handles invocation.

  Args:
    executor: An instance of `executor_base.Executor`.
    comp: The first argument to `context_base.Context.invoke()`.
    arg: The optional second argument to `context_base.Context.invoke()`.

  Returns:
    The result of the invocation.
  """
  result_type = comp.type_signature.result
  elements = [executor.create_value(comp)]
  if isinstance(arg, anonymous_tuple.AnonymousTuple):
    elements.append(executor.create_tuple(arg))
  elements = await asyncio.gather(*elements)
  comp = elements[0]
  if len(elements) > 1:
    arg = elements[1]
  result = await executor.create_call(comp, arg)
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
    return asyncio.get_event_loop().run_until_complete(
        self._executor.create_value(val, type_spec))

  def invoke(self, comp, arg):
    return asyncio.get_event_loop().run_until_complete(
        _invoke(self._executor, comp, arg))
