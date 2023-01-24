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
"""An executor responsible for the `data` building block."""

import asyncio
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import data_backend_base
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


class DataExecutor(executor_base.Executor):
  """The data executor is responsible for the `data` building block."""

  def __init__(
      self,
      target_executor: executor_base.Executor,
      data_backend: data_backend_base.DataBackend,
  ):
    """Creates a new instance of this executor.

    Args:
      target_executor: Downstream executor that this executor delegates to.
      data_backend: The data backend responsible for materializing payloads.

    Raises:
      TypeError: if arguments are of the wrong types.
    """
    py_typecheck.check_type(target_executor, executor_base.Executor)
    py_typecheck.check_type(data_backend, data_backend_base.DataBackend)
    self._target_executor = target_executor
    self._data_backend = data_backend

  @tracing.trace(span=True)
  async def create_value(self, value, type_spec=None):
    """Creates a value in this executor.

    The following kinds of `value` are supported as the input:

    * An instance of a TFF computation proto that represents a `data` building
      block, to be handled natively by this executor.

    * Anything that is supported by the target executor (as a pass-through).

    * A nested structure of any of the above.

    Args:
      value: The input for which to create a value.
      type_spec: An optional TFF type of `value`.

    Returns:
      A value embedded in the target executor.
    """
    if isinstance(value, pb.Computation):
      if value.WhichOneof('computation') == 'data':
        value_type = type_serialization.deserialize_type(value.type)
        if type_spec is not None:
          type_spec.check_equivalent_to(value_type)
        else:
          type_spec = value_type
        payload = await self._data_backend.materialize(value.data, type_spec)
        return await self._target_executor.create_value(payload, type_spec)
      else:
        return await self._target_executor.create_value(value, type_spec)
    elif isinstance(type_spec, computation_types.StructType):
      if not isinstance(value, structure.Struct):
        value = structure.from_container(value)
      elements = structure.flatten(value)
      element_types = structure.flatten(type_spec)
      flat_embedded_vals = await asyncio.gather(
          *[
              self.create_value(el, el_type)
              for el, el_type in zip(elements, element_types)
          ]
      )
      embedded_struct = structure.pack_sequence_as(value, flat_embedded_vals)
      return await self._target_executor.create_struct(embedded_struct)
    else:
      return await self._target_executor.create_value(value, type_spec)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    return await self._target_executor.create_call(comp, arg=arg)

  @tracing.trace
  async def create_struct(self, elements):
    return await self._target_executor.create_struct(elements)

  @tracing.trace
  async def create_selection(self, source, index):
    return await self._target_executor.create_selection(source, index)

  def close(self):
    self._target_executor.close()
