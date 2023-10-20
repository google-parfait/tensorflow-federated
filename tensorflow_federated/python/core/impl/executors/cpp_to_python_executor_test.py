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
import concurrent
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.executors import cpp_to_python_executor
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.types import computation_types


class CcToPythonExecutorTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self._mock_executor = unittest.mock.create_autospec(
        executor_bindings.Executor
    )
    self._test_executor = cpp_to_python_executor.CppToPythonExecutorBridge(
        self._mock_executor,
        concurrent.futures.ThreadPoolExecutor(max_workers=None),
    )

  @parameterized.named_parameters(
      ('integer', 1, computation_types.to_type(np.int32)),
      ('float', 1.0, computation_types.to_type(np.float32)),
      (
          'mixed_structure',
          structure.Struct.unnamed(0, 1.0),
          computation_types.to_type([np.int32, np.float32]),
      ),
      (
          'nested_structure',
          structure.Struct.unnamed(0, structure.Struct.unnamed(1, 2)),
          computation_types.to_type([np.int32, [np.int32, np.int32]]),
      ),
  )
  async def test_create_value(self, value, type_spec):
    _ = await self._test_executor.create_value(value, type_spec)
    serialized_value, _ = value_serialization.serialize_value(value, type_spec)
    self._mock_executor.create_value.assert_called_once_with(serialized_value)

  async def test_create_call_tensorflow_function_noarg(self):
    owned_call_id = unittest.mock.create_autospec(
        executor_bindings.OwnedValueId
    )
    self._mock_executor.create_call.return_value = owned_call_id
    fn = unittest.mock.create_autospec(
        cpp_to_python_executor.CppToPythonExecutorValue
    )
    fn.type_signature = computation_types.FunctionType(
        None, computation_types.to_type(np.int32)
    )
    fn.reference = 1

    constructed_call = await self._test_executor.create_call(fn)

    self._mock_executor.create_call.assert_called_with(1, None)
    self.assertIs(constructed_call.reference, owned_call_id.ref)

  async def test_create_call_tensorflow_function_with_arg(self):
    owned_call_id = unittest.mock.create_autospec(
        executor_bindings.OwnedValueId
    )
    self._mock_executor.create_call.return_value = owned_call_id
    fn = unittest.mock.create_autospec(
        cpp_to_python_executor.CppToPythonExecutorValue
    )
    fn.type_signature = computation_types.FunctionType(
        None, computation_types.to_type(np.int32)
    )
    fn.reference = 1
    arg = unittest.mock.create_autospec(
        cpp_to_python_executor.CppToPythonExecutorValue
    )
    arg.type_signature = computation_types.to_type(np.int32)
    arg.reference = 2

    constructed_call = await self._test_executor.create_call(fn, arg)

    self._mock_executor.create_call.assert_called_with(1, 2)
    self.assertIs(constructed_call.reference, owned_call_id.ref)

  async def test_create_struct(self):
    owned_struct_id = unittest.mock.create_autospec(
        executor_bindings.OwnedValueId
    )
    self._mock_executor.create_struct.return_value = owned_struct_id
    struct_element = unittest.mock.create_autospec(
        cpp_to_python_executor.CppToPythonExecutorValue
    )
    struct_element.type_signature = computation_types.to_type(np.int32)
    struct_element.reference = 1

    constructed_struct = await self._test_executor.create_struct(
        [struct_element]
    )

    self._mock_executor.create_struct.assert_called_with([1])
    self.assertIs(constructed_struct.reference, owned_struct_id.ref)

  async def test_create_selection(self):
    owned_selection_id = unittest.mock.create_autospec(
        executor_bindings.OwnedValueId
    )
    self._mock_executor.create_selection.return_value = owned_selection_id
    source = unittest.mock.create_autospec(
        cpp_to_python_executor.CppToPythonExecutorValue
    )
    source.type_signature = computation_types.to_type([np.int32])
    source.reference = 1

    selected_element = await self._test_executor.create_selection(source, 0)

    self._mock_executor.create_selection.assert_called_with(1, 0)
    self.assertIs(selected_element.reference, owned_selection_id.ref)

  async def test_compute(self):
    owned_id = unittest.mock.create_autospec(executor_bindings.OwnedValueId)
    owned_id.ref = 1
    serialized_two, _ = value_serialization.serialize_value(
        2, computation_types.to_type(np.int32)
    )
    self._mock_executor.materialize.return_value = serialized_two
    type_signature = computation_types.to_type(np.int32)
    executor_value = cpp_to_python_executor.CppToPythonExecutorValue(
        owned_id,
        type_signature,
        self._mock_executor,
        concurrent.futures.ThreadPoolExecutor(),
    )

    computed_value = await executor_value.compute()

    self._mock_executor.materialize.assert_called_with(1)
    self.assertEqual(computed_value, 2)


if __name__ == '__main__':
  absltest.main()
