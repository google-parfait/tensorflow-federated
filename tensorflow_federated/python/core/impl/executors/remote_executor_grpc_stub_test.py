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
"""Tests for remote_executor_grpc_stub."""

from unittest import mock

from absl.testing import absltest
import grpc
import portpicker
import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.core.impl.executors import executor_serialization
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import remote_executor_grpc_stub


def create_stub():
  port = portpicker.pick_unused_port()
  channel = grpc.insecure_channel('localhost:{}'.format(port))
  return remote_executor_grpc_stub.RemoteExecutorGrpcStub(channel)


def _raise_grpc_error_unavailable(*args):
  del args  # Unused
  error = grpc.RpcError()
  error.code = lambda: grpc.StatusCode.UNAVAILABLE
  raise error


def _raise_non_retryable_grpc_error(*args):
  del args  # Unused
  error = grpc.RpcError()
  error.code = lambda: grpc.StatusCode.ABORTED
  raise error


class GrpcConnectivityTest(absltest.TestCase):

  def fake_channel_subscribe(self, callback, try_to_connect=True):
    if try_to_connect:
      callback(grpc.ChannelConnectivity.READY)

  def test_grpc_connectivity(self):
    channel = mock.create_autospec(grpc.Channel, instance=True)
    channel.subscribe.side_effect = self.fake_channel_subscribe
    stub = remote_executor_grpc_stub.RemoteExecutorGrpcStub(channel)
    self.assertTrue(stub.is_ready)


@mock.patch.object(executor_pb2_grpc, 'ExecutorStub')
class RemoteExecutorGrpcStubTest(absltest.TestCase):

  def test_compute_returns_result(self, mock_executor_grpc_stub):
    tensor_proto = tf.make_tensor_proto(1)
    any_pb = any_pb2.Any()
    any_pb.Pack(tensor_proto)
    value = executor_pb2.Value(tensor=any_pb)
    response = executor_pb2.ComputeResponse(value=value)
    instance = mock_executor_grpc_stub.return_value
    instance.Compute = mock.Mock(side_effect=[response])

    request = executor_pb2.ComputeRequest(value_ref=executor_pb2.ValueRef())

    stub = create_stub()
    result = stub.compute(request)

    instance.Compute.assert_called_once()

    value, _ = executor_serialization.deserialize_value(result.value)
    self.assertEqual(value, 1)

  def test_compute_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.Compute = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    stub = create_stub()

    with self.assertRaises(executors_errors.RetryableError):
      stub.compute(
          executor_pb2.ComputeRequest(value_ref=executor_pb2.ValueRef()))

  def test_compute_reraises_grpc_error(self, grpc_stub):
    instance = grpc_stub.return_value
    instance.Compute = mock.Mock(side_effect=_raise_non_retryable_grpc_error)

    stub = create_stub()
    request = executor_pb2.ComputeRequest(value_ref=executor_pb2.ValueRef())

    with self.assertRaises(grpc.RpcError) as context:
      stub.compute(request)

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_set_cardinalities(self, mock_executor_grpc_stub):
    response = executor_pb2.SetCardinalitiesResponse()
    instance = mock_executor_grpc_stub.return_value
    instance.SetCardinalities = mock.Mock(side_effect=[response])
    stub = create_stub()
    result = stub.set_cardinalities(
        request=executor_pb2.SetCardinalitiesRequest())
    self.assertEqual(result, response)

  def test_create_value_returns_value(self, mock_executor_grpc_stub):
    response = executor_pb2.CreateValueRequest()
    instance = mock_executor_grpc_stub.return_value
    instance.CreateValue = mock.Mock(side_effect=[response])
    stub = create_stub()
    result = stub.create_value(request=executor_pb2.CreateValueRequest())
    self.assertEqual(result, response)

  def test_create_value_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateValue = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    stub = create_stub()

    with self.assertRaises(executors_errors.RetryableError):
      stub.create_value(request=executor_pb2.CreateValueRequest())

  def test_create_value_reraises_grpc_error(self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateValue = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    stub = create_stub()

    with self.assertRaises(grpc.RpcError) as context:
      stub.create_value(request=executor_pb2.CreateValueRequest())

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_value_reraises_type_error(self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateValue = mock.Mock(side_effect=TypeError)
    stub = create_stub()

    with self.assertRaises(TypeError):
      stub.create_value(request=executor_pb2.CreateValueRequest())

  def test_create_call_returns_remote_value(self, mock_executor_grpc_stub):
    response = executor_pb2.CreateCallResponse()
    instance = mock_executor_grpc_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=[response])
    stub = create_stub()
    result = stub.create_call(request=executor_pb2.CreateCallRequest())

    instance.CreateCall.assert_called_once()
    self.assertEqual(result, response)

  def test_create_call_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    stub = create_stub()

    with self.assertRaises(executors_errors.RetryableError):
      stub.create_call(request=executor_pb2.CreateCallRequest())

  def test_create_call_reraises_grpc_error(self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=_raise_non_retryable_grpc_error)
    stub = create_stub()

    with self.assertRaises(grpc.RpcError) as context:
      stub.create_call(request=executor_pb2.CreateCallRequest())

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_call_reraises_type_error(self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=TypeError)
    stub = create_stub()

    with self.assertRaises(TypeError):
      stub.create_call(request=executor_pb2.CreateCallRequest())

  def test_create_struct_returns_value(self, mock_executor_grpc_stub):
    response = executor_pb2.CreateStructResponse()
    instance = mock_executor_grpc_stub.return_value
    instance.CreateStruct = mock.Mock(side_effect=[response])
    stub = create_stub()

    result = stub.create_struct(request=executor_pb2.CreateStructRequest())

    instance.CreateStruct.assert_called_once()
    self.assertEqual(result, response)

  def test_create_struct_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateStruct = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    stub = create_stub()

    with self.assertRaises(executors_errors.RetryableError):
      stub.create_struct(request=executor_pb2.CreateStructRequest())

  def test_create_struct_reraises_grpc_error(self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateStruct = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    stub = create_stub()

    with self.assertRaises(grpc.RpcError) as context:
      stub.create_struct(request=executor_pb2.CreateStructRequest())

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_struct_reraises_type_error(self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateStruct = mock.Mock(side_effect=TypeError)
    stub = create_stub()

    with self.assertRaises(TypeError):
      stub.create_struct(request=executor_pb2.CreateStructRequest())

  def test_create_selection_returns_value(self, mock_executor_grpc_stub):
    response = executor_pb2.CreateSelectionResponse()
    instance = mock_executor_grpc_stub.return_value
    instance.CreateSelection = mock.Mock(side_effect=[response])
    stub = create_stub()

    result = stub.create_selection(
        request=executor_pb2.CreateSelectionRequest())

    instance.CreateSelection.assert_called_once()
    self.assertEqual(result, response)

  def test_create_selection_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateSelection = mock.Mock(
        side_effect=_raise_grpc_error_unavailable)
    stub = create_stub()

    with self.assertRaises(executors_errors.RetryableError):
      stub.create_selection(request=executor_pb2.CreateSelectionRequest())

  def test_create_selection_reraises_non_retryable_grpc_error(
      self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateSelection = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    stub = create_stub()
    with self.assertRaises(grpc.RpcError) as context:
      stub.create_selection(request=executor_pb2.CreateSelectionRequest())

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_selection_reraises_type_error(self, mock_executor_grpc_stub):
    instance = mock_executor_grpc_stub.return_value
    instance.CreateSelection = mock.Mock(side_effect=TypeError)
    stub = create_stub()

    with self.assertRaises(TypeError):
      stub.create_selection(request=executor_pb2.CreateSelectionRequest())


if __name__ == '__main__':
  absltest.main()
