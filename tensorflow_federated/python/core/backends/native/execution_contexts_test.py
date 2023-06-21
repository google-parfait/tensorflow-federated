# Copyright 2020, The TensorFlow Federated Authors.
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
import os.path
import subprocess
import sys
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.executors import remote_executor_grpc_stub
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.tensorflow_libs import tensorflow_test_utils


@tensorflow_computation.tf_computation
def return_one():
  return 1


class DatasetsTest(parameterized.TestCase):
  """Tests for Datasets in a native backend.

  These tests ensure that `tf.data.Datasets`s are passed through to TF without
  TFF mutating or changing the data.
  """

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_local_cpp_execution_context()

  @tensorflow_test_utils.skip_test_for_gpu
  def test_takes_dataset(self):
    @tensorflow_computation.tf_computation
    def foo(ds):
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10)
    actual_result = foo(ds)

    expected_result = ds.take(10).reduce(np.int64(0), lambda x, y: x + y)
    self.assertEqual(actual_result, expected_result)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset(self):
    @tensorflow_computation.tf_computation
    def foo():
      return tf.data.Dataset.range(10)

    actual_result = foo()

    expected_result = tf.data.Dataset.range(10)
    self.assertEqual(
        list(actual_result.as_numpy_iterator()),
        list(expected_result.as_numpy_iterator()),
    )

  def test_takes_dataset_infinite(self):
    @tensorflow_computation.tf_computation
    def foo(ds):
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10).repeat()
    actual_result = foo(ds)

    expected_result = ds.take(10).reduce(np.int64(0), lambda x, y: x + y)
    self.assertEqual(actual_result, expected_result)

  def test_returns_dataset_infinite(self):
    @tensorflow_computation.tf_computation
    def foo():
      return tf.data.Dataset.range(10).repeat()

    actual_result = foo()

    expected_result = tf.data.Dataset.range(10).repeat()
    self.assertEqual(
        actual_result.take(100).reduce(np.int64(0), lambda x, y: x + y),
        expected_result.take(100).reduce(np.int64(0), lambda x, y: x + y),
    )

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset_two(self):
    @tensorflow_computation.tf_computation
    def foo():
      return [tf.data.Dataset.range(5), tf.data.Dataset.range(10)]

    actual_result = foo()

    expected_result = [tf.data.Dataset.range(5), tf.data.Dataset.range(10)]
    self.assertEqual(
        list(actual_result[0].as_numpy_iterator()),
        list(expected_result[0].as_numpy_iterator()),
    )
    self.assertEqual(
        list(actual_result[1].as_numpy_iterator()),
        list(expected_result[1].as_numpy_iterator()),
    )

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset_and_tensor(self):
    @tensorflow_computation.tf_computation
    def foo():
      return [tf.data.Dataset.range(5), tf.constant(5)]

    actual_result = foo()

    expected_result = [tf.data.Dataset.range(5), tf.constant(5)]
    self.assertEqual(
        list(actual_result[0].as_numpy_iterator()),
        list(expected_result[0].as_numpy_iterator()),
    )
    self.assertEqual(actual_result[1], expected_result[1])

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_empty_dataset(self):
    @tensorflow_computation.tf_computation
    def foo():
      tensor_slices = collections.OrderedDict([('a', [1, 1]), ('b', [1, 1])])
      ds = tf.data.Dataset.from_tensor_slices(tensor_slices)
      return ds.batch(5).take(0)

    actual_result = foo()

    expected_element_spec = collections.OrderedDict([
        ('a', tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        ('b', tf.TensorSpec(shape=(None,), dtype=tf.int32)),
    ])
    self.assertEqual(actual_result.element_spec, expected_element_spec)
    expected_result = tf.data.Dataset.range(10).batch(5).take(0)
    self.assertEqual(
        list(actual_result.as_numpy_iterator()),
        list(expected_result.as_numpy_iterator()),
    )


def _create_mock_remote_executor_grpc_stub(
    computation: computation_impl.ConcreteComputation,
) -> remote_executor_grpc_stub.RemoteExecutorGrpcStub:
  class _GetExecutorResponse:

    @property
    def executor(self, *args, **kwargs):
      del args, kwargs  # Unused.
      return executor_pb2.ExecutorId(id='0')

  mock_ex = mock.create_autospec(
      remote_executor_grpc_stub.RemoteExecutorGrpcStub
  )
  mock_ex.is_ready = True
  mock_ex.get_executor.return_value = _GetExecutorResponse()
  mock_ex.create_value.return_value = executor_pb2.CreateValueResponse()
  mock_ex.create_call.return_value = executor_pb2.CreateCallResponse()
  value = executor_pb2.Value(computation=computation.to_building_block().proto)
  mock_ex.compute.return_value = executor_pb2.ComputeResponse(value=value)
  return mock_ex


class SyncLocalCPPExecutionContextTest(absltest.TestCase):

  def test_raises_runtime_error_if_no_worker_binarys(self):
    with self.assertRaises(RuntimeError):
      with mock.patch.object(os.path, 'isfile', return_value=False):
        execution_contexts.create_sync_local_cpp_execution_context()

  @mock.patch.object(subprocess, 'Popen')
  def test_process_starts(self, mock_popen):
    context = execution_contexts.create_sync_local_cpp_execution_context()

    mock_remote_ex = _create_mock_remote_executor_grpc_stub(return_one)

    with mock.patch.object(
        remote_executor_grpc_stub,
        'RemoteExecutorGrpcStub',
        return_value=mock_remote_ex,
    ):
      context.invoke(return_one, None)

    expected_args = [
        mock.ANY,
        mock.ANY,
        '--max_concurrent_computation_calls=-1',
    ]
    mock_popen.assert_called_once_with(
        expected_args, stdout=sys.stdout, stderr=sys.stderr
    )

  @mock.patch.object(subprocess, 'Popen')
  def test_stub_going_down_restarts_process(self, mock_popen):
    context = execution_contexts.create_sync_local_cpp_execution_context()
    mock_remote_ex = _create_mock_remote_executor_grpc_stub(return_one)
    mock_remote_ex.is_ready = False

    with mock.patch.object(
        remote_executor_grpc_stub,
        'RemoteExecutorGrpcStub',
        return_value=mock_remote_ex,
    ):
      value_fn = lambda: context.invoke(return_one, None)
      # Set the thread to daemonic, to avoid blocking shutdown.
      thread = threading.Thread(target=value_fn, daemon=True)
      thread.start()
      thread.join(timeout=1)
      # Stub returning false for is_ready will cause the invocation to block;
      # is_alive returning true indicates the thread has not finished.
      self.assertTrue(thread.is_alive())

    expected_args = [
        mock.ANY,
        mock.ANY,
        '--max_concurrent_computation_calls=-1',
    ]
    mock_popen.assert_called_once_with(
        expected_args, stdout=sys.stdout, stderr=sys.stderr
    )


class DistributedExecutionContextIntegrationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('async_context_on_server', True, False, False),
      ('sync_context_on_server', True, False, True),
      ('async_context_on_client', False, True, True),
      ('sync_context_on_client', False, True, True),
      ('async_context_on_server_client', True, True, False),
      ('sync_context_on_server_client', True, True, True),
  )
  def test_runs_tensorflow_on_dtensor_on_server(
      self, dtensor_on_server, dtensor_on_client, is_sync_context
  ):
    mesh = tf.experimental.dtensor.create_mesh(
        devices=['CPU:0'], mesh_dims=[('test_dim', 1)]
    )

    def _create_context():
      server_mesh = mesh if dtensor_on_server else None
      client_mesh = mesh if dtensor_on_client else None
      distributed_config = execution_contexts.DistributedConfiguration(
          server_mesh=server_mesh,
          client_mesh=client_mesh,
      )
      if is_sync_context:
        return execution_contexts.create_sync_experimental_distributed_cpp_execution_context(
            distributed_config=distributed_config
        )
      return execution_contexts.create_async_experimental_distributed_cpp_execution_context(
          distributed_config=distributed_config
      )

    @context_stack_test_utils.with_context(_create_context)
    def test_dtensor_stack():
      @tensorflow_computation.tf_computation(
          collections.OrderedDict(x=tf.int32, y=tf.int32)
      )
      def multiply(ordered_dict):
        return ordered_dict['x'] * ordered_dict['y']

      zero = multiply(collections.OrderedDict(x=0, y=1))
      one = multiply(collections.OrderedDict(x=1, y=1))

      if not is_sync_context:
        self.assertTrue(asyncio.iscoroutine(zero))
        self.assertTrue(asyncio.iscoroutine(one))
        zero = asyncio.run(zero)
        one = asyncio.run(one)

      self.assertEqual(zero, 0)
      self.assertEqual(one, 1)

    test_dtensor_stack()

  def test_error_on_distributed_context_creation(self):
    with self.assertRaisesRegex(
        ValueError,
        (
            'Both server side and client side mesh are unspecified'
            ' in distributed configuration.'
        ),
    ):
      execution_contexts.create_sync_experimental_distributed_cpp_execution_context(
          distributed_config=execution_contexts.DistributedConfiguration()
      )


if __name__ == '__main__':
  absltest.main()
