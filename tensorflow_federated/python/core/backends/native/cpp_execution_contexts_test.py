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

import asyncio
import collections
import time
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import cpp_execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_test_utils
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings


class CreateAsyncLocalCPPExecutionContextTest(absltest.TestCase):

  def test_returns_async_context(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()
    self.assertIsInstance(
        context, federated_language.framework.AsyncExecutionContext
    )

  def test_install_and_execute_in_context(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()

    @tensorflow_computation.tf_computation(np.int32)
    def add_one(x):
      return x + 1

    with federated_language.framework.get_context_stack().install(context):
      val_coro = add_one(1)
      self.assertTrue(asyncio.iscoroutine(val_coro))
      self.assertEqual(asyncio.run(val_coro), 2)

  def test_install_and_execute_computations_with_different_cardinalities(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def repackage_arg(x):
      return [x, x]

    with federated_language.framework.get_context_stack().install(context):
      single_val_coro = repackage_arg([1])
      second_val_coro = repackage_arg([1, 2])
      self.assertTrue(asyncio.iscoroutine(single_val_coro))
      self.assertTrue(asyncio.iscoroutine(second_val_coro))
      self.assertEqual(
          [asyncio.run(single_val_coro), asyncio.run(second_val_coro)],
          [[[1], [1]], [[1, 2], [1, 2]]],
      )


class CreateSyncLocalCPPExecutionContextTest(absltest.TestCase):

  def test_returns_sync_context(self):
    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    self.assertIsInstance(
        context, federated_language.framework.SyncExecutionContext
    )


class LocalCPPExecutionContextTest(absltest.TestCase):

  def test_returns_sync_context(self):
    targets = ['fake_target']
    channels = [
        executor_bindings.create_insecure_grpc_channel(t) for t in targets
    ]
    context = cpp_execution_contexts.create_sync_remote_cpp_execution_context(
        channels=channels
    )
    self.assertIsInstance(
        context, federated_language.framework.SyncExecutionContext
    )

  def test_returns_same_python_structure(self):

    @federated_language.federated_computation(
        collections.OrderedDict(a=np.int32, b=np.float32)
    )
    def identity(x):
      return x

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    with federated_language.framework.get_context_stack().install(context):
      odict = identity(collections.OrderedDict(a=0, b=1.0))

    self.assertIsInstance(odict, collections.OrderedDict)

  def test_runs_tensorflow(self):
    @tensorflow_computation.tf_computation(
        collections.OrderedDict(x=np.int32, y=np.int32)
    )
    def multiply(ordered_dict):
      return ordered_dict['x'] * ordered_dict['y']

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    with federated_language.framework.get_context_stack().install(context):
      zero = multiply(collections.OrderedDict(x=0, y=1))
      one = multiply(collections.OrderedDict(x=1, y=1))

    self.assertEqual(zero, 0)
    self.assertEqual(one, 1)

  def test_async_execution_context_runs_in_parallel(self):
    num_parallel_calls = 4
    sleep_seconds = 3
    max_parallel_exec_seconds = 9
    max_iters_cap = 10

    @tensorflow_computation.tf_computation(
        collections.OrderedDict(x=np.int32, y=np.int32)
    )
    @tf.function
    def sleep_and_multiply(ordered_dict):
      init_time = tf.timestamp()
      n_iters = 0
      # This is a busy-sleep; TF exposes no direct sleep ops.
      while tf.timestamp() - init_time < sleep_seconds:
        n_iters += 1
      return (
          ordered_dict['x']
          * ordered_dict['y']
          * tf.math.minimum(n_iters, max_iters_cap)
      )

    context = cpp_execution_contexts.create_async_local_cpp_execution_context()

    async def multiple_invocations():
      return await asyncio.gather(*[
          context.invoke(sleep_and_multiply, collections.OrderedDict(x=1, y=1))
          for _ in range(num_parallel_calls)
      ])

    loop = asyncio.new_event_loop()
    # This timing-based test seems unfortunate.
    start = time.time()
    result = loop.run_until_complete(multiple_invocations())
    exec_time = time.time() - start
    # Four parallel 3s calls take ~3s on >=4 vCPUs (~6s on 2 vCPUs). Sequential
    # calls take ~12s.
    self.assertLess(exec_time, max_parallel_exec_seconds)
    self.assertLen(result, num_parallel_calls)
    for x in result:
      # 1 * 1 * max_iters_cap
      self.assertEqual(x, max_iters_cap)

  def test_returns_sequence(self):
    @tensorflow_computation.tf_computation
    def create_sequence():
      return tf.data.Dataset.range(5)

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    with federated_language.framework.get_context_stack().install(context):
      with self.subTest('unplaced'):
        sequence = create_sequence()
        self.assertEqual(sequence, [0, 1, 2, 3, 4])

      with self.subTest('federated'):

        @federated_language.federated_computation
        def create_federated_sequence():
          return federated_language.federated_eval(
              create_sequence, federated_language.SERVER
          )

        sequence = create_federated_sequence()
        self.assertEqual(sequence, [0, 1, 2, 3, 4])

      with self.subTest('struct'):

        @tensorflow_computation.tf_computation()
        def create_struct_of_sequences():
          return (create_sequence(), create_sequence())

        sequences = create_struct_of_sequences()
        self.assertLen(sequences, 2)
        self.assertEqual(
            sequences,
            (
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ),
        )


class StreamStructsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'sync_local',
          cpp_execution_contexts.create_sync_local_cpp_execution_context,
      ),
      (
          'async_local',
          cpp_execution_contexts.create_async_local_cpp_execution_context,
      ),
  )
  def test_local_context_warns_when_stream_structs_enabled(self, context_fn):
    with self.assertWarnsRegex(DeprecationWarning, 'stream_structs'):
      context_fn(stream_structs=True)

  @parameterized.named_parameters(
      (
          'sync_local',
          cpp_execution_contexts.create_sync_local_cpp_execution_context,
      ),
      (
          'async_local',
          cpp_execution_contexts.create_async_local_cpp_execution_context,
      ),
  )
  def test_local_context_does_not_warn_when_stream_structs_disabled(
      self, context_fn
  ):
    with warnings.catch_warnings(record=True) as recorded_warnings:
      warnings.simplefilter('always', DeprecationWarning)
      context_fn(stream_structs=False)
    deprecation_warnings = [
        warning
        for warning in recorded_warnings
        if issubclass(warning.category, DeprecationWarning)
    ]
    self.assertEmpty(deprecation_warnings)

  @parameterized.named_parameters(
      (
          'sync_remote_true',
          cpp_execution_contexts.create_sync_remote_cpp_execution_context,
          True,
      ),
      (
          'sync_remote_false',
          cpp_execution_contexts.create_sync_remote_cpp_execution_context,
          False,
      ),
      (
          'async_remote_true',
          cpp_execution_contexts.create_async_remote_cpp_execution_context,
          True,
      ),
      (
          'async_remote_false',
          cpp_execution_contexts.create_async_remote_cpp_execution_context,
          False,
      ),
  )
  def test_remote_context_forwards_stream_structs(
      self, context_fn, stream_structs
  ):
    channels = [executor_bindings.create_insecure_grpc_channel('fake_target')]
    mock_executor_factory = mock.create_autospec(
        federated_language.framework.ExecutorFactory, instance=True
    )
    with mock.patch.object(
        cpp_executor_factory,
        'remote_cpp_executor_factory',
        return_value=mock_executor_factory,
        autospec=True,
    ) as mock_factory:
      context_fn(channels=channels, stream_structs=stream_structs)
    mock_factory.assert_called_once_with(
        channels=channels,
        default_num_clients=0,
        stream_structs=stream_structs,
    )

  def test_set_sync_remote_context_forwards_stream_structs(self):
    channels = [executor_bindings.create_insecure_grpc_channel('fake_target')]
    mock_executor_factory = mock.create_autospec(
        federated_language.framework.ExecutorFactory, instance=True
    )
    with (
        mock.patch.object(
            cpp_executor_factory,
            'remote_cpp_executor_factory',
            return_value=mock_executor_factory,
            autospec=True,
        ) as mock_factory,
        mock.patch.object(
            federated_language.framework,
            'set_default_context',
            autospec=True,
        ) as mock_set_default_context,
    ):
      cpp_execution_contexts.set_sync_remote_cpp_execution_context(
          channels=channels,
          default_num_clients=2,
          stream_structs=True,
      )
    mock_factory.assert_called_once_with(
        channels=channels,
        default_num_clients=2,
        stream_structs=True,
    )
    mock_set_default_context.assert_called_once()


class CreateMergeableCompExecutionContextTest(absltest.TestCase):

  def test_returns_mergeable_comp_context(self):
    async_context = (
        cpp_execution_contexts.create_async_local_cpp_execution_context()
    )
    context = cpp_execution_contexts.create_mergeable_comp_execution_context(
        [async_context]
    )
    self.assertIsInstance(
        context,
        mergeable_comp_execution_context.MergeableCompExecutionContext,
    )


class DatasetsTest(parameterized.TestCase):
  """Tests for Datasets in a native backend.

  These tests ensure that `tf.data.Datasets`s are passed through to TF without
  TFF mutating or changing the data.
  """

  def setUp(self):
    super().setUp()
    cpp_execution_contexts.set_sync_local_cpp_execution_context()

  @tensorflow_test_utils.skip_test_for_gpu
  def test_takes_dataset(self):
    @tensorflow_computation.tf_computation
    def foo(ds):
      return ds

    ds = tf.data.Dataset.range(10)
    actual_result = foo(ds)

    expected_result = list(range(10))
    self.assertEqual(actual_result, expected_result)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_sequence(self):
    @tensorflow_computation.tf_computation
    def foo():
      return tf.data.Dataset.range(10)

    actual_result = foo()

    expected_result = list(range(10))
    self.assertEqual(actual_result, expected_result)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset_two(self):
    @tensorflow_computation.tf_computation
    def foo():
      return [tf.data.Dataset.range(5), tf.data.Dataset.range(10)]

    actual_result = foo()

    expected_result = [list(range(5)), list(range(10))]
    self.assertEqual(actual_result, expected_result)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset_and_tensor(self):
    @tensorflow_computation.tf_computation
    def foo():
      return [tf.data.Dataset.range(5), 5]

    actual_result = foo()

    expected_result = [list(range(5)), 5]
    self.assertEqual(actual_result, expected_result)


class AdditionalNativeCPPExecutionContextTest(absltest.TestCase):

  def test_set_mergeable_comp_execution_context_sets_default_context(self):
    context_stack = federated_language.framework.get_context_stack()
    original_context = context_stack.current
    self.addCleanup(context_stack.set_default_context, original_context)
    async_context = (
        cpp_execution_contexts.create_async_local_cpp_execution_context()
    )
    cpp_execution_contexts.set_mergeable_comp_execution_context([async_context])
    self.assertIsInstance(
        context_stack.current,
        mergeable_comp_execution_context.MergeableCompExecutionContext,
    )

  def test_sync_local_context_uses_cardinality_inference_fn(self):
    recorded_calls = []

    def _infer_cardinalities(value, type_spec):
      recorded_calls.append((value, type_spec))
      return {federated_language.CLIENTS: 2}

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context(
        cardinality_inference_fn=_infer_cardinalities
    )

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def identity(client_values):
      return client_values

    with federated_language.framework.get_context_stack().install(context):
      result = identity([1, 2])
    self.assertEqual(result, [1, 2])
    self.assertLen(recorded_calls, 1)

  def test_sync_local_context_cardinality_inference_fn_mismatch_raises(self):
    def _infer_single_client(value, type_spec):
      del value, type_spec  # Unused.
      return {federated_language.CLIENTS: 1}

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context(
        cardinality_inference_fn=_infer_single_client
    )

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def identity(client_values):
      return client_values

    with federated_language.framework.get_context_stack().install(context):
      with self.assertRaisesRegex(
          Exception, 'configured for 1 clients, but 2 values were provided'
      ):
        identity([1, 2])


if __name__ == '__main__':
  absltest.main()
