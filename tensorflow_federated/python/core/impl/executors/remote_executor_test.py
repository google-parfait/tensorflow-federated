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

import asyncio
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grpc
import numpy as np

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.executors import remote_executor_stub
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _raise_non_retryable_grpc_error(*args):
  del args  # Unused
  error = grpc.RpcError()
  error.code = lambda: grpc.StatusCode.ABORTED
  raise error


def _set_cardinalities_with_mock(
    executor: remote_executor.RemoteExecutor, mock_stub: mock.Mock
):
  mock_stub.get_executor.return_value = executor_pb2.GetExecutorResponse(
      executor=executor_pb2.ExecutorId(id='id')
  )
  executor.set_cardinalities({placements.CLIENTS: 3})


class RemoteValueTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_compute_returns_result_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    comp = computation_factory.create_lambda_empty_struct()
    value = executor_pb2.Value(computation=comp)
    mock_stub.compute.return_value = executor_pb2.ComputeResponse(value=value)
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    executor.set_cardinalities({placements.CLIENTS: 3})
    type_signature = computation_types.FunctionType(None, np.int32)
    remote_value = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    result = asyncio.run(remote_value.compute())
    mock_stub.compute.assert_called_once()
    self.assertEqual(result, comp)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_compute_reraises_grpc_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.compute = mock.Mock(side_effect=_raise_non_retryable_grpc_error)
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.FunctionType(None, np.int32)
    comp = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(grpc.RpcError) as context:
      asyncio.run(comp.compute())

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_compute_reraises_type_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.compute = mock.Mock(side_effect=TypeError)
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.FunctionType(None, np.int32)
    comp = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(TypeError):
      asyncio.run(comp.compute())


class RemoteExecutorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_set_cardinalities_returns_none_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.get_executor.return_value = executor_pb2.GetExecutorResponse(
        executor=executor_pb2.ExecutorId(id='test_id')
    )
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    result = executor.set_cardinalities({placements.CLIENTS: 3})
    self.assertIsNone(result)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_value_returns_remote_value_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_value.return_value = executor_pb2.CreateValueResponse()
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)

    result = asyncio.run(executor.create_value(1, np.int32))

    mock_stub.create_value.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_value_reraises_grpc_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_value = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error
    )
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)

    with self.assertRaises(grpc.RpcError) as context:
      asyncio.run(executor.create_value(1, np.int32))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_value_reraises_type_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_value = mock.Mock(side_effect=TypeError)
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)

    with self.assertRaises(TypeError):
      asyncio.run(executor.create_value(1, np.int32))

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_value_for_nested_struct_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    if stream_structs:
      self.skipTest(
          'b/263261613 - Support multiple return_value types in mock_stub'
      )
      mock_stub.create_value.return_value = executor_pb2.CreateStructResponse()
    else:
      mock_stub.create_value.return_value = executor_pb2.CreateValueResponse()

    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    tensor_shape = (2, 10)
    struct_value = structure.Struct([
        ('a', np.zeros(shape=tensor_shape, dtype=np.int32)),
        (
            'b',
            structure.Struct([
                ('b0', np.zeros(shape=tensor_shape, dtype=np.int32)),
                ('b1', np.zeros(shape=tensor_shape, dtype=np.int32)),
            ]),
        ),
        ('c', np.zeros(shape=tensor_shape, dtype=np.int32)),
    ])

    type_signature = computation_types.StructType([
        (
            'a',
            computation_types.TensorType(shape=tensor_shape, dtype=np.int32),
        ),
        (
            'b',
            computation_types.StructType([
                (
                    'b0',
                    computation_types.TensorType(
                        shape=tensor_shape, dtype=np.int32
                    ),
                ),
                (
                    'b1',
                    computation_types.TensorType(
                        shape=tensor_shape, dtype=np.int32
                    ),
                ),
            ]),
        ),
        (
            'c',
            computation_types.TensorType(shape=tensor_shape, dtype=np.int32),
        ),
    ])

    result = asyncio.run(executor.create_value(struct_value, type_signature))

    self.assertIsInstance(result, remote_executor.RemoteValue)
    if stream_structs:
      self.assertEqual(mock_stub.create_value.call_count, 5)
    else:
      self.assertEqual(mock_stub.create_value.call_count, 1)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_call_returns_remote_value_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_call.return_value = executor_pb2.CreateCallResponse()
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.FunctionType(None, np.int32)
    fn = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    result = asyncio.run(executor.create_call(fn, None))

    mock_stub.create_call.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_call_reraises_grpc_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_call = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error
    )
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.FunctionType(None, np.int32)
    comp = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(grpc.RpcError) as context:
      asyncio.run(executor.create_call(comp, None))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_call_reraises_type_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_call = mock.Mock(side_effect=TypeError)
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.FunctionType(None, np.int32)
    comp = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(TypeError):
      asyncio.run(executor.create_call(comp))

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_struct_returns_remote_value_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_struct.return_value = executor_pb2.CreateStructResponse()
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.TensorType(np.int32)
    value_1 = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )
    value_2 = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    result = asyncio.run(executor.create_struct([value_1, value_2]))

    mock_stub.create_struct.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_struct_reraises_grpc_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_struct = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error
    )
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.TensorType(np.int32)
    value_1 = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )
    value_2 = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(grpc.RpcError) as context:
      asyncio.run(executor.create_struct([value_1, value_2]))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_struct_reraises_type_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_struct = mock.Mock(side_effect=TypeError)
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.TensorType(np.int32)
    value_1 = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )
    value_2 = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(TypeError):
      asyncio.run(executor.create_struct([value_1, value_2]))

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_selection_returns_remote_value_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_selection.return_value = (
        executor_pb2.CreateSelectionResponse()
    )
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.StructType([np.int32, np.int32])
    source = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    result = asyncio.run(executor.create_selection(source, 0))

    mock_stub.create_selection.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_selection_reraises_non_retryable_grpc_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_selection = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error
    )
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.StructType([np.int32, np.int32])
    source = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(grpc.RpcError) as context:
      asyncio.run(executor.create_selection(source, 0))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  @parameterized.named_parameters(
      ('false', False),
      ('true', True),
  )
  @mock.patch.object(remote_executor_stub, 'RemoteExecutorStub')
  def test_create_selection_reraises_type_error_with_stream_structs(
      self, stream_structs, mock_stub
  ):
    mock_stub.create_selection = mock.Mock(side_effect=TypeError)
    executor = remote_executor.RemoteExecutor(
        mock_stub, stream_structs=stream_structs
    )
    _set_cardinalities_with_mock(executor, mock_stub)
    type_signature = computation_types.StructType([np.int32, np.int32])
    source = remote_executor.RemoteValue(
        executor_pb2.ValueRef(), type_signature, executor
    )

    with self.assertRaises(TypeError):
      asyncio.run(executor.create_selection(source, 0))


if __name__ == '__main__':
  absltest.main()
