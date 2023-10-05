# Copyright 2022, The TensorFlow Federated Authors.
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
"""Implementation of Python executor interface backed by a C++ executor."""

import asyncio
from collections.abc import Sequence
import concurrent
from typing import NoReturn, Optional

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.types import computation_types


def _handle_error(exception: Exception) -> NoReturn:
  if executors_errors.is_absl_status_retryable_error(exception):
    raise executors_errors.RetryableAbslStatusError() from exception
  else:
    raise exception


class CppToPythonExecutorValue(executor_value_base.ExecutorValue):
  """ExecutorValue representation of values embedded in C++ executors.

  Instances of this class represent ownership of the resources which back
  the values they point to in C++; when these objects are garbage collected,
  these resource will be freed. Values of this type should be treated as
  immutable; effectively, python representations of unique pointers to values
  embedded in C++ executors.
  """

  def __init__(
      self,
      owned_value_id: executor_bindings.OwnedValueId,
      type_signature: computation_types.Type,
      cpp_executor: executor_bindings.Executor,
      futures_executor: concurrent.futures.Executor,
  ):
    self._owned_value_id = owned_value_id
    self._type_signature = type_signature
    self._cpp_executor = cpp_executor
    self._futures_executor = futures_executor

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature

  @property
  def reference(self) -> int:
    return self._owned_value_id.ref

  @tracing.trace
  async def compute(self) -> object:
    """Pulls protocol buffer out of C++ into Python, and deserializes."""
    running_loop = asyncio.get_running_loop()

    def _materialize():
      try:
        return self._cpp_executor.materialize(self._owned_value_id.ref)
      except Exception as e:  # pylint: disable=broad-except
        _handle_error(e)

    result_pb = await running_loop.run_in_executor(
        self._futures_executor, _materialize
    )
    deserialized_value, _ = value_serialization.deserialize_value(
        result_pb, self._type_signature
    )
    return deserialized_value


class CppToPythonExecutorBridge(executor_base.Executor):
  """Implementation of Python executor interface in terms of C++ executor.

  This class implements a thin layer integrating the
  `executor_bindings.Executor` interface exposed directly from C++ with asyncio
  and the Python executor interface. Instances of this class hold references to
  the C++ executor taken as an initialization parameter; values created from
  the `createX` methods implemented here will hold references to this executor,
  ensuring that the C++ executor lives as long as both the instance of this
  bridge class and the values it constructs.
  """

  def __init__(
      self,
      cpp_executor: executor_bindings.Executor,
      futures_executor: concurrent.futures.Executor,
  ):
    self._cpp_executor = cpp_executor
    self._futures_executor = futures_executor

  @tracing.trace
  async def create_value(
      self, value: object, type_signature: computation_types.Type
  ) -> CppToPythonExecutorValue:
    serialized_value, _ = value_serialization.serialize_value(
        value, type_signature
    )
    try:
      owned_id = self._cpp_executor.create_value(serialized_value)
    except Exception as e:  # pylint: disable=broad-except
      _handle_error(e)
    return CppToPythonExecutorValue(
        owned_id, type_signature, self._cpp_executor, self._futures_executor
    )

  @tracing.trace
  async def create_call(
      self,
      fn: CppToPythonExecutorValue,
      arg: Optional[CppToPythonExecutorValue] = None,
  ) -> CppToPythonExecutorValue:
    fn_ref = fn.reference
    if arg is not None:
      arg_ref = arg.reference
    else:
      arg_ref = None
    try:
      owned_call_id = self._cpp_executor.create_call(fn_ref, arg_ref)
    except Exception as e:  # pylint: disable=broad-except
      _handle_error(e)
    return CppToPythonExecutorValue(
        owned_call_id,
        fn.type_signature.result,  # pytype: disable=attribute-error
        self._cpp_executor,
        self._futures_executor,
    )

  @tracing.trace
  async def create_struct(
      self, elements: Sequence[CppToPythonExecutorValue]
  ) -> CppToPythonExecutorValue:
    executor_value_struct = structure.from_container(elements)
    id_list = []
    type_list = []
    for name, value in structure.iter_elements(executor_value_struct):
      id_list.append(value.reference)
      type_list.append((name, value.type_signature))
    try:
      struct_id = self._cpp_executor.create_struct(id_list)
    except Exception as e:  # pylint: disable=broad-except
      _handle_error(e)
    return CppToPythonExecutorValue(
        struct_id,
        computation_types.StructType(type_list),
        self._cpp_executor,
        self._futures_executor,
    )

  @tracing.trace
  async def create_selection(
      self, source: CppToPythonExecutorValue, index: int
  ) -> CppToPythonExecutorValue:
    try:
      selection_id = self._cpp_executor.create_selection(
          source.reference, index
      )
    except Exception as e:  # pylint: disable=broad-except
      _handle_error(e)
    selection_type = source.type_signature[index]  # pytype: disable=unsupported-operands
    return CppToPythonExecutorValue(
        selection_id, selection_type, self._cpp_executor, self._futures_executor
    )

  def close(self):
    # We pass on close; though we could release the reference we hold to the C++
    # executor, all the values we return also hold references to it, so it may
    # not immediately release resources in the manner we expect. Rather than
    # deleting this exceutor and immediately setting ourselves into an invalid
    # state, we pass.
    pass
