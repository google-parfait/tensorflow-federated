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

import threading
from unittest import mock

from absl.testing import absltest
import grpc
from grpc.framework.foundation import logging_pool
import portpicker
import tensorflow as tf

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_serialization
from tensorflow_federated.python.core.impl.executors import executor_service
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import placements


class TestEnv(object):
  """A test environment that consists of a single client and backend service."""

  def __init__(self,
               ex_factory: executor_factory.ExecutorFactory,
               num_clients: int = 0):
    port = portpicker.pick_unused_port()
    self._server_pool = logging_pool.pool(max_workers=1)
    self._server = grpc.server(self._server_pool)
    self._server.add_insecure_port('[::]:{}'.format(port))
    self._service = executor_service.ExecutorService(ex_factory=ex_factory)
    executor_pb2_grpc.add_ExecutorGroupServicer_to_server(
        self._service, self._server)
    self._server.start()
    self._channel = grpc.insecure_channel('localhost:{}'.format(port))
    self._stub = executor_pb2_grpc.ExecutorGroupStub(self._channel)

    serialized_cards = executor_serialization.serialize_cardinalities(
        {placements.CLIENTS: num_clients})
    self._executor_pb = self._stub.GetExecutor(
        executor_pb2.GetExecutorRequest(
            cardinalities=serialized_cards)).executor

  def __del__(self):
    self._channel.close()
    self._server_pool.shutdown(wait=False)
    self._server.stop(None)

  @property
  def executor_pb(self):
    return self._executor_pb

  @property
  def stub(self):
    return self._stub

  def get_value(self, value_id: str):
    """Retrieves a value using the `Compute` endpoint."""
    response = self._stub.Compute(
        executor_pb2.ComputeRequest(
            executor=self._executor_pb,
            value_ref=executor_pb2.ValueRef(id=value_id)))
    py_typecheck.check_type(response, executor_pb2.ComputeResponse)
    value, _ = executor_serialization.deserialize_value(response.value)
    return value

  def get_value_future_directly(self, value_id: str):
    """Retrieves value by reaching inside the service object."""
    with self._service._lock:
      return self._service._values[value_id]

  def close_channel(self):
    self._channel.close()


class ExecutorServiceTest(absltest.TestCase):

  def test_executor_service_slowly_create_tensor_value(self):

    class SlowExecutorValue(executor_value_base.ExecutorValue):

      def __init__(self, v, t):
        self._v = v
        self._t = t

      @property
      def type_signature(self):
        return self._t

      async def compute(self):
        return self._v

    class SlowExecutor(executor_base.Executor):

      def __init__(self):
        self.status = 'idle'
        self.busy = threading.Event()
        self.done = threading.Event()

      async def create_value(self, value, type_spec=None):
        self.status = 'busy'
        self.busy.set()
        self.done.wait()
        self.status = 'done'
        return SlowExecutorValue(value, type_spec)

      async def create_call(self, comp, arg=None):
        raise NotImplementedError

      async def create_struct(self, elements):
        raise NotImplementedError

      async def create_selection(self, source, index):
        raise NotImplementedError

      def close(self):
        pass

    ex = SlowExecutor()
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(lambda _: ex)
    env = TestEnv(ex_factory)
    self.assertEqual(ex.status, 'idle')
    value_proto, _ = executor_serialization.serialize_value(10, tf.int32)
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    ex.busy.wait()
    self.assertEqual(ex.status, 'busy')
    ex.done.set()
    value = env.get_value(response.value_ref.id)
    self.assertEqual(ex.status, 'done')
    self.assertEqual(value, 10)

  def test_executor_service_create_tensor_value(self):
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: eager_tf_executor.EagerTFExecutor())
    env = TestEnv(ex_factory)
    value_proto, _ = executor_serialization.serialize_value(
        tf.constant(10.0).numpy(), tf.float32)
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    value_id = str(response.value_ref.id)
    value = env.get_value(value_id)
    self.assertEqual(value, 10.0)
    del env

  def test_executor_service_create_no_arg_computation_value_and_call(self):
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: eager_tf_executor.EagerTFExecutor())
    env = TestEnv(ex_factory)

    @tensorflow_computation.tf_computation
    def comp():
      return tf.constant(10)

    value_proto, _ = executor_serialization.serialize_value(comp)
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    response = env.stub.CreateCall(
        executor_pb2.CreateCallRequest(
            executor=env.executor_pb, function_ref=response.value_ref))
    self.assertIsInstance(response, executor_pb2.CreateCallResponse)
    value_id = str(response.value_ref.id)
    value = env.get_value(value_id)
    self.assertEqual(value, 10)
    del env

  def test_executor_service_value_unavailable_after_dispose(self):
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: eager_tf_executor.EagerTFExecutor())
    env = TestEnv(ex_factory)
    value_proto, _ = executor_serialization.serialize_value(
        tf.constant(10.0).numpy(), tf.float32)
    # Create the value
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    value_id = str(response.value_ref.id)
    # Check that the value appears in the _values map
    env.get_value_future_directly(value_id)
    # Dispose of the value
    dispose_request = executor_pb2.DisposeRequest(executor=env.executor_pb)
    dispose_request.value_ref.append(response.value_ref)
    response = env.stub.Dispose(dispose_request)
    self.assertIsInstance(response, executor_pb2.DisposeResponse)
    # Check that the value is gone from the _values map
    # get_value_future_directly is used here so that we can catch the
    # exception rather than having it occur on the GRPC thread.
    with self.assertRaises(KeyError):
      env.get_value_future_directly(value_id)

  def test_dispose_does_not_trigger_cleanup(self):

    class MockFactory(executor_factory.ExecutorFactory, mock.MagicMock):

      def create_executor(self, *args, **kwargs):
        return mock.MagicMock()

      def clean_up_executors(self):
        return

    ex_factory = MockFactory()
    ex_factory.clean_up_executors = mock.MagicMock()

    env = TestEnv(ex_factory)
    value_proto, _ = executor_serialization.serialize_value(
        tf.constant(10.0).numpy(), tf.float32)
    # Create the value
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    value_id = str(response.value_ref.id)
    # Check that the value appears in the _values map
    env.get_value_future_directly(value_id)
    # Dispose of the value
    dispose_request = executor_pb2.DisposeRequest(executor=env.executor_pb)
    dispose_request.value_ref.append(response.value_ref)
    response = env.stub.Dispose(dispose_request)
    # We shouldn't be propagating close down the executor stack on Dispose--this
    # would close the bidi stream and cause a hang in the streaming case with
    # intermediate aggregation. Python GC takes care of pushing Dispose requests
    # from the aggregators to the workers.
    ex_factory.clean_up_executors.assert_not_called()

  def test_executor_service_create_one_arg_computation_value_and_call(self):
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: eager_tf_executor.EagerTFExecutor())
    env = TestEnv(ex_factory)

    @tensorflow_computation.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 1)

    value_proto, _ = executor_serialization.serialize_value(comp)
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    comp_ref = response.value_ref

    value_proto, _ = executor_serialization.serialize_value(10, tf.int32)
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    arg_ref = response.value_ref

    response = env.stub.CreateCall(
        executor_pb2.CreateCallRequest(
            executor=env.executor_pb,
            function_ref=comp_ref,
            argument_ref=arg_ref))
    self.assertIsInstance(response, executor_pb2.CreateCallResponse)
    value_id = str(response.value_ref.id)
    value = env.get_value(value_id)
    self.assertEqual(value, 11)
    del env

  def test_executor_service_create_and_select_from_tuple(self):
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: eager_tf_executor.EagerTFExecutor())
    env = TestEnv(ex_factory)

    value_proto, _ = executor_serialization.serialize_value(10, tf.int32)
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    ten_ref = response.value_ref
    self.assertEqual(env.get_value(ten_ref.id), 10)

    value_proto, _ = executor_serialization.serialize_value(20, tf.int32)
    response = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)
    twenty_ref = response.value_ref
    self.assertEqual(env.get_value(twenty_ref.id), 20)

    response = env.stub.CreateStruct(
        executor_pb2.CreateStructRequest(
            executor=env.executor_pb,
            element=[
                executor_pb2.CreateStructRequest.Element(
                    name='a', value_ref=ten_ref),
                executor_pb2.CreateStructRequest.Element(
                    name='b', value_ref=twenty_ref)
            ]))
    self.assertIsInstance(response, executor_pb2.CreateStructResponse)
    tuple_ref = response.value_ref
    self.assertEqual(str(env.get_value(tuple_ref.id)), '<a=10,b=20>')

    for index, result_val in [(0, 10), (1, 20)]:
      response = env.stub.CreateSelection(
          executor_pb2.CreateSelectionRequest(
              executor=env.executor_pb, source_ref=tuple_ref, index=index))
      self.assertIsInstance(response, executor_pb2.CreateSelectionResponse)
      selection_ref = response.value_ref
      self.assertEqual(env.get_value(selection_ref.id), result_val)

    del env

  def test_dispose_executor_calls_close(self):

    class MockExecutor(executor_base.Executor):

      async def create_value(self, value, type_spec=None):
        raise NotImplementedError

      async def create_call(self, comp, arg=None):
        raise NotImplementedError

      async def create_struct(self, elements):
        raise NotImplementedError

      async def create_selection(self, source, index):
        raise NotImplementedError

      def close(self):
        raise NotImplementedError

    mock_executor = MockExecutor()
    mock_executor.close = mock.MagicMock()
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: mock_executor)
    env = TestEnv(ex_factory)
    env.stub.DisposeExecutor(
        executor_pb2.DisposeExecutorRequest(executor=env.executor_pb))
    mock_executor.close.assert_called_once()

  def test_raising_failed_precondition_destroys_executor(self):

    # A simple clas to mock out the exceptions raised by GRPC.
    class GrpcFailedPrecondition(grpc.RpcError):
      pass

      def code(self):
        return grpc.StatusCode.FAILED_PRECONDITION

    def _return_error():
      return GrpcFailedPrecondition('Raising failed precondition')

    raising_ex = executor_test_utils.RaisingExecutor(_return_error)
    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: raising_ex)
    env = TestEnv(ex_factory)
    value_proto, _ = executor_serialization.serialize_value(10, tf.int32)
    value_id = env.stub.CreateValue(
        executor_pb2.CreateValueRequest(
            executor=env.executor_pb, value=value_proto))
    # When CreateValue is forced to run, the executor's error will be raised. We
    # raise lazily because the service operates future-to-future.
    with self.assertRaises(grpc.RpcError):
      env.get_value(value_id.value_ref.id)

    with self.assertRaises(grpc.RpcError) as rpc_err:
      # The next CreateValue call, or indeed any call, should eagerly raise,
      # since the executor has been destroyed.
      env.stub.CreateValue(
          executor_pb2.CreateValueRequest(
              executor=env.executor_pb, value=value_proto))
    self.assertEqual(rpc_err.exception.code(),
                     grpc.StatusCode.FAILED_PRECONDITION)
    self.assertIn('No executor found', rpc_err.exception.exception().details())


if __name__ == '__main__':
  absltest.main()
