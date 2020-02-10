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

import asyncio
import collections
import contextlib
from unittest import mock

from absl.testing import absltest
import grpc
from grpc.framework.foundation import logging_pool
import portpicker
import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import execution_context
from tensorflow_federated.python.core.impl import executor_service
from tensorflow_federated.python.core.impl import executor_stacks
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl import remote_executor
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import lambda_executor
from tensorflow_federated.python.core.impl.wrappers import set_default_executor


def create_remote_executor():
  port = portpicker.pick_unused_port()
  channel = grpc.insecure_channel('localhost:{}'.format(port))
  executor = remote_executor.RemoteExecutor(channel, 'REQUEST_REPLY')
  return executor


@contextlib.contextmanager
def test_context(rpc_mode='REQUEST_REPLY'):
  port = portpicker.pick_unused_port()
  server_pool = logging_pool.pool(max_workers=1)
  server = grpc.server(server_pool)
  server.add_insecure_port('[::]:{}'.format(port))
  target_executor = executor_stacks.local_executor_factory(
      num_clients=3).create_executor({})
  tracer = executor_test_utils.TracingExecutor(target_executor)
  service = executor_service.ExecutorService(tracer)
  executor_pb2_grpc.add_ExecutorServicer_to_server(service, server)
  server.start()
  channel = grpc.insecure_channel('localhost:{}'.format(port))
  remote_exec = remote_executor.RemoteExecutor(channel, rpc_mode)
  executor = lambda_executor.LambdaExecutor(remote_exec)
  set_default_executor.set_default_executor(
      executor_factory.ExecutorFactoryImpl(lambda _: executor))
  try:
    yield collections.namedtuple('_', 'executor tracer')(executor, tracer)
  finally:
    set_default_executor.set_default_executor()
    try:
      channel.close()
    except AttributeError:
      pass  # Public gRPC channel doesn't support close()
    finally:
      server.stop(None)


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


@mock.patch(
    'tensorflow_federated.proto.v0.executor_pb2_grpc.ExecutorStub'
)
class RemoteValueTest(absltest.TestCase):

  def test_compute_returns_result(self, mock_stub):
    tensor_proto = tf.make_tensor_proto(1)
    any_pb = any_pb2.Any()
    any_pb.Pack(tensor_proto)
    value = executor_pb2.Value(tensor=any_pb)
    response = executor_pb2.ComputeResponse(value=value)
    instance = mock_stub.return_value
    instance.Compute = mock.Mock(side_effect=[response])
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    comp = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                       executor)

    result = loop.run_until_complete(comp.compute())

    instance.Compute.assert_called_once()
    self.assertEqual(result, 1)

  def test_compute_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_stub):
    instance = mock_stub.return_value
    instance.Compute = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    comp = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                       executor)

    with self.assertRaises(execution_context.RetryableError):
      loop.run_until_complete(comp.compute())

  def test_compute_reraises_grpc_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.Compute = mock.Mock(side_effect=_raise_non_retryable_grpc_error)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    comp = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                       executor)

    with self.assertRaises(grpc.RpcError) as context:
      loop.run_until_complete(comp.compute())

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_compute_reraises_type_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.Compute = mock.Mock(side_effect=TypeError)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    comp = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                       executor)

    with self.assertRaises(TypeError):
      loop.run_until_complete(comp.compute())


@mock.patch(
    'tensorflow_federated.proto.v0.executor_pb2_grpc.ExecutorStub'
)
class RemoteExecutorTest(absltest.TestCase):

  def test_create_value_returns_remote_value(self, mock_stub):
    response = executor_pb2.CreateValueResponse()
    instance = mock_stub.return_value
    instance.CreateValue = mock.Mock(side_effect=[response])
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()

    result = loop.run_until_complete(executor.create_value(1, tf.int32))

    instance.CreateValue.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  def test_create_value_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateValue = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()

    with self.assertRaises(execution_context.RetryableError):
      loop.run_until_complete(executor.create_value(1, tf.int32))

  def test_create_value_reraises_grpc_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateValue = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()

    with self.assertRaises(grpc.RpcError) as context:
      loop.run_until_complete(executor.create_value(1, tf.int32))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_value_reraises_type_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateValue = mock.Mock(side_effect=TypeError)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()

    with self.assertRaises(TypeError):
      loop.run_until_complete(executor.create_value(1, tf.int32))

  def test_create_call_returns_remote_value(self, mock_stub):
    response = executor_pb2.CreateCallResponse()
    instance = mock_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=[response])
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    fn = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                     executor)

    result = loop.run_until_complete(executor.create_call(fn, None))

    instance.CreateCall.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  def test_create_call_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    comp = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                       executor)

    with self.assertRaises(execution_context.RetryableError):
      loop.run_until_complete(executor.create_call(comp, None))

  def test_create_call_reraises_grpc_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=_raise_non_retryable_grpc_error)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    comp = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                       executor)

    with self.assertRaises(grpc.RpcError) as context:
      loop.run_until_complete(executor.create_call(comp, None))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_call_reraises_type_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateCall = mock.Mock(side_effect=TypeError)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.FunctionType(None, tf.int32)
    comp = remote_executor.RemoteValue(executor_pb2.ValueRef(), type_signature,
                                       executor)

    with self.assertRaises(TypeError):
      loop.run_until_complete(executor.create_call(comp))

  def test_create_tuple_returns_remote_value(self, mock_stub):
    response = executor_pb2.CreateTupleResponse()
    instance = mock_stub.return_value
    instance.CreateTuple = mock.Mock(side_effect=[response])
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    result = loop.run_until_complete(executor.create_tuple([value_1, value_2]))

    instance.CreateTuple.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  def test_create_tuple_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateTuple = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    with self.assertRaises(execution_context.RetryableError):
      loop.run_until_complete(executor.create_tuple([value_1, value_2]))

  def test_create_tuple_reraises_grpc_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateTuple = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    with self.assertRaises(grpc.RpcError) as context:
      loop.run_until_complete(executor.create_tuple([value_1, value_2]))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_tuple_reraises_type_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateTuple = mock.Mock(side_effect=TypeError)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    with self.assertRaises(TypeError):
      loop.run_until_complete(executor.create_tuple([value_1, value_2]))

  def test_create_selection_returns_remote_value(self, mock_stub):
    response = executor_pb2.CreateSelectionResponse()
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(side_effect=[response])
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.NamedTupleType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    result = loop.run_until_complete(executor.create_selection(source, index=0))

    instance.CreateSelection.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  def test_create_selection_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(
        side_effect=_raise_grpc_error_unavailable)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.NamedTupleType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    with self.assertRaises(execution_context.RetryableError):
      loop.run_until_complete(executor.create_selection(source, index=0))

  def test_create_selection_reraises_non_retryable_grpc_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.NamedTupleType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    with self.assertRaises(grpc.RpcError) as context:
      loop.run_until_complete(executor.create_selection(source, index=0))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_selection_reraises_type_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(side_effect=TypeError)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.NamedTupleType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    with self.assertRaises(TypeError):
      loop.run_until_complete(executor.create_selection(source, index=0))


class RemoteExecutorIntegrationTest(absltest.TestCase):

  def test_no_arg_tf_computation(self):
    with test_context():

      @computations.tf_computation
      def comp():
        return 10

      self.assertEqual(comp(), 10)

  def test_one_arg_tf_computation(self):
    with test_context():

      @computations.tf_computation(tf.int32)
      def comp(x):
        return x + 1

      self.assertEqual(comp(10), 11)

  def test_two_arg_tf_computation(self):
    with test_context():

      @computations.tf_computation(tf.int32, tf.int32)
      def comp(x, y):
        return x + y

      self.assertEqual(comp(10, 20), 30)

  def test_with_selection(self):
    with test_context() as context:
      self._test_with_selection(context)

  def test_with_selection_streaming(self):
    with test_context(rpc_mode='STREAMING') as context:
      self._test_with_selection(context)

  def _test_with_selection(self, context):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return collections.OrderedDict([('A', x + 10), ('B', x + 20)])

    @computations.tf_computation(tf.int32, tf.int32)
    def bar(x, y):
      return x + y

    @computations.federated_computation(tf.int32)
    def baz(x):
      return bar(foo(x).A, foo(x).B)

    self.assertEqual(baz(100), 230)

    # Make sure exactly two selections happened.
    self.assertLen(
        [x for x in context.tracer.trace if x[0] == 'create_selection'], 2)

  def test_runs_tf(self):
    with test_context() as context:
      executor_test_utils.test_runs_tf(self, context.executor)

  def test_runs_tf_streaming_rpc(self):
    with test_context(rpc_mode='STREAMING') as context:
      executor_test_utils.test_runs_tf(self, context.executor)

  def test_with_federated_computations(self):
    with test_context():

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.CLIENTS))
      def foo(x):
        return intrinsics.federated_sum(x)

      self.assertEqual(foo([10, 20, 30]), 60)

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def bar(x):
        return intrinsics.federated_broadcast(x)

      self.assertEqual(bar(50), 50)

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def baz(x):
        return intrinsics.federated_map(
            computations.tf_computation(lambda y: y + 1, tf.int32),
            intrinsics.federated_broadcast(x))

      self.assertEqual(baz(50), [51, 51, 51])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
