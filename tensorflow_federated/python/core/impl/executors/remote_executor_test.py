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
from absl.testing import parameterized
import grpc
from grpc.framework.foundation import logging_pool
import portpicker
import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_service
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import placements


def create_remote_executor():
  port = portpicker.pick_unused_port()
  channel = grpc.insecure_channel('localhost:{}'.format(port))
  return remote_executor.RemoteExecutor(channel)


@contextlib.contextmanager
def test_context():
  port = portpicker.pick_unused_port()
  server_pool = logging_pool.pool(max_workers=1)
  server = grpc.server(server_pool)
  server.add_insecure_port('[::]:{}'.format(port))
  target_factory = executor_stacks.local_executor_factory(num_clients=3)
  tracers = []

  def _tracer_fn(cardinalities):
    tracer = executor_test_utils.TracingExecutor(
        target_factory.create_executor(cardinalities))
    tracers.append(tracer)
    return tracer

  service = executor_service.ExecutorService(
      executor_stacks.ResourceManagingExecutorFactory(_tracer_fn))
  executor_pb2_grpc.add_ExecutorServicer_to_server(service, server)
  server.start()

  channel = grpc.insecure_channel('localhost:{}'.format(port))

  remote_exec = remote_executor.RemoteExecutor(channel)
  asyncio.get_event_loop().run_until_complete(
      remote_exec.set_cardinalities({placements.CLIENTS: 3}))
  executor = reference_resolving_executor.ReferenceResolvingExecutor(
      remote_exec)
  try:
    yield collections.namedtuple('_', 'executor tracers')(executor, tracers)
  finally:
    executor.close()
    for tracer in tracers:
      tracer.close()
    try:
      channel.close()
    except AttributeError:
      pass  # Public gRPC channel doesn't support close()
    finally:
      server.stop(None)


def _invoke(ex, comp, arg=None):
  loop = asyncio.get_event_loop()
  v1 = loop.run_until_complete(ex.create_value(comp))
  if arg is not None:
    type_spec = v1.type_signature.parameter
    v2 = loop.run_until_complete(ex.create_value(arg, type_spec))
  else:
    v2 = None
  v3 = loop.run_until_complete(ex.create_call(v1, v2))
  return loop.run_until_complete(v3.compute())


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

  def test_set_cardinalities_returns_none(self, mock_stub):
    response = executor_pb2.SetCardinalitiesResponse()
    instance = mock_stub.return_value
    instance.SetCardinalities = mock.Mock(side_effect=[response])
    executor = create_remote_executor()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        executor.set_cardinalities({placements.CLIENTS: 3}))
    self.assertIsNone(result)

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

  def test_create_struct_returns_remote_value(self, mock_stub):
    response = executor_pb2.CreateStructResponse()
    instance = mock_stub.return_value
    instance.CreateStruct = mock.Mock(side_effect=[response])
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    result = loop.run_until_complete(executor.create_struct([value_1, value_2]))

    instance.CreateStruct.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  def test_create_struct_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateStruct = mock.Mock(side_effect=_raise_grpc_error_unavailable)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    with self.assertRaises(execution_context.RetryableError):
      loop.run_until_complete(executor.create_struct([value_1, value_2]))

  def test_create_struct_reraises_grpc_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateStruct = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    with self.assertRaises(grpc.RpcError) as context:
      loop.run_until_complete(executor.create_struct([value_1, value_2]))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_struct_reraises_type_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateStruct = mock.Mock(side_effect=TypeError)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.TensorType(tf.int32)
    value_1 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)
    value_2 = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                          type_signature, executor)

    with self.assertRaises(TypeError):
      loop.run_until_complete(executor.create_struct([value_1, value_2]))

  def test_create_selection_returns_remote_value(self, mock_stub):
    response = executor_pb2.CreateSelectionResponse()
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(side_effect=[response])
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.StructType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    result = loop.run_until_complete(executor.create_selection(source, 0))

    instance.CreateSelection.assert_called_once()
    self.assertIsInstance(result, remote_executor.RemoteValue)

  def test_create_selection_raises_retryable_error_on_grpc_error_unavailable(
      self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(
        side_effect=_raise_grpc_error_unavailable)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.StructType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    with self.assertRaises(execution_context.RetryableError):
      loop.run_until_complete(executor.create_selection(source, 0))

  def test_create_selection_reraises_non_retryable_grpc_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(
        side_effect=_raise_non_retryable_grpc_error)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.StructType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    with self.assertRaises(grpc.RpcError) as context:
      loop.run_until_complete(executor.create_selection(source, 0))

    self.assertEqual(context.exception.code(), grpc.StatusCode.ABORTED)

  def test_create_selection_reraises_type_error(self, mock_stub):
    instance = mock_stub.return_value
    instance.CreateSelection = mock.Mock(side_effect=TypeError)
    loop = asyncio.get_event_loop()
    executor = create_remote_executor()
    type_signature = computation_types.StructType([tf.int32, tf.int32])
    source = remote_executor.RemoteValue(executor_pb2.ValueRef(),
                                         type_signature, executor)

    with self.assertRaises(TypeError):
      loop.run_until_complete(executor.create_selection(source, 0))


class RemoteExecutorIntegrationTest(parameterized.TestCase):

  def test_no_arg_tf_computation(self):
    with test_context() as context:

      @computations.tf_computation
      def comp():
        return 10

      result = _invoke(context.executor, comp)
      self.assertEqual(result, 10)

  def test_one_arg_tf_computation(self):
    with test_context() as context:

      @computations.tf_computation(tf.int32)
      def comp(x):
        return x + 1

      result = _invoke(context.executor, comp, 10)
      self.assertEqual(result, 11)

  def test_two_arg_tf_computation(self):
    with test_context() as context:

      @computations.tf_computation(tf.int32, tf.int32)
      def comp(x, y):
        return x + y

      result = _invoke(context.executor, comp, (10, 20))
      self.assertEqual(result, 30)

  def test_with_selection(self):
    with test_context() as context:
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

    result = _invoke(context.executor, baz, 100)
    self.assertEqual(result, 230)

    # Make sure exactly two selections happened.
    seletions = [
        x for x in context.tracers[0].trace if x[0] == 'create_selection'
    ]
    self.assertLen(seletions, 2)

  def test_execution_of_tensorflow(self):

    @computations.tf_computation
    def comp():
      return tf.math.add(5, 5)

    with test_context() as context:
      result = _invoke(context.executor, comp)

    self.assertEqual(result, 10)

  def test_with_federated_computations(self):
    with test_context() as context:

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.CLIENTS))
      def foo(x):
        return intrinsics.federated_sum(x)

      result = _invoke(context.executor, foo, [10, 20, 30])
      self.assertEqual(result, 60)

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def bar(x):
        return intrinsics.federated_broadcast(x)

      result = _invoke(context.executor, bar, 50)
      self.assertEqual(result, 50)

      @computations.tf_computation(tf.int32)
      def add_one(x):
        return x + 1

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def baz(x):
        value = intrinsics.federated_broadcast(x)
        return intrinsics.federated_map(add_one, value)

      result = _invoke(context.executor, baz, 50)
      self.assertEqual(result, [51, 51, 51])


if __name__ == '__main__':
  absltest.main()
