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
import inspect
import time

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import cpp_execution_contexts
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _assert_signature_equal(first_obj, second_obj):
  first_signature = inspect.signature(first_obj)
  second_signature = inspect.signature(second_obj)
  # Only assert that the parameters and return type annotations are equal, the
  # entire signature (e.g. the docstring) is not expected to be equal.
  if first_signature.parameters != second_signature.parameters:
    raise AssertionError(
        f'{first_signature.parameters} != {second_signature.parameters}'
    )
  if first_signature.return_annotation != second_signature.return_annotation:
    raise AssertionError(
        f'{first_signature.return_annotation} != '
        f'{second_signature.return_annotation}'
    )


class CreateAsyncLocalCPPExecutionContextTest(tf.test.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.create_async_local_cpp_execution_context,
        execution_contexts.create_async_local_cpp_execution_context,
    )

  def test_returns_async_context(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()
    self.assertIsInstance(
        context, async_execution_context.AsyncExecutionContext
    )

  def test_install_and_execute_in_context(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()

    @tensorflow_computation.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    with get_context_stack.get_context_stack().install(context):
      val_coro = add_one(1)
      self.assertTrue(asyncio.iscoroutine(val_coro))
      self.assertEqual(asyncio.run(val_coro), 2)

  def test_install_and_execute_computations_with_different_cardinalities(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()

    @federated_computation.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS)
    )
    def repackage_arg(x):
      return [x, x]

    with get_context_stack.get_context_stack().install(context):
      single_val_coro = repackage_arg([1])
      second_val_coro = repackage_arg([1, 2])
      self.assertTrue(asyncio.iscoroutine(single_val_coro))
      self.assertTrue(asyncio.iscoroutine(second_val_coro))
      self.assertEqual(
          [asyncio.run(single_val_coro), asyncio.run(second_val_coro)],
          [[[1], [1]], [[1, 2], [1, 2]]],
      )


class SetAsyncLocalCPPExecutionContextTest(tf.test.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.set_async_local_cpp_execution_context,
        execution_contexts.set_async_local_cpp_execution_context,
    )


class CreateSyncLocalCPPExecutionContextTest(absltest.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.create_sync_local_cpp_execution_context,
        execution_contexts.create_sync_local_cpp_execution_context,
    )

  def test_returns_sync_context(self):
    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    self.assertIsInstance(context, sync_execution_context.SyncExecutionContext)


class SetSyncLocalCPPExecutionContextTest(absltest.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.set_sync_local_cpp_execution_context,
        execution_contexts.set_sync_local_cpp_execution_context,
    )


class LocalCPPExecutionContextTest(tf.test.TestCase):

  def test_returns_sync_context(self):
    targets = ['fake_target']
    channels = [
        executor_bindings.create_insecure_grpc_channel(t) for t in targets
    ]
    context = cpp_execution_contexts.create_remote_cpp_execution_context(
        channels=channels
    )
    self.assertIsInstance(context, sync_execution_context.SyncExecutionContext)

  def test_returns_same_python_structure(self):

    @federated_computation.federated_computation(
        collections.OrderedDict(a=tf.int32, b=tf.float32)
    )
    def identity(x):
      return x

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    with get_context_stack.get_context_stack().install(context):
      odict = identity(collections.OrderedDict(a=0, b=1.0))

    self.assertIsInstance(odict, collections.OrderedDict)

  def test_runs_tensorflow(self):

    @tensorflow_computation.tf_computation(
        collections.OrderedDict(x=tf.int32, y=tf.int32)
    )
    def multiply(ordered_dict):
      return ordered_dict['x'] * ordered_dict['y']

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    with get_context_stack.get_context_stack().install(context):
      zero = multiply(collections.OrderedDict(x=0, y=1))
      one = multiply(collections.OrderedDict(x=1, y=1))

    self.assertEqual(zero, 0)
    self.assertEqual(one, 1)

  def test_runs_tensorflow_on_dtensor_on_server(self):
    # Create a single device mesh for testing.
    mesh = tf.experimental.dtensor.create_mesh(
        devices=['CPU:0'], mesh_dims=[('test_dim', 1)]
    )

    @tensorflow_computation.tf_computation(
        collections.OrderedDict(x=tf.int32, y=tf.int32)
    )
    def multiply(ordered_dict):
      return ordered_dict['x'] * ordered_dict['y']

    context = cpp_execution_contexts.create_sync_experimental_distributed_cpp_execution_context(
        distributed_config=cpp_execution_contexts.DistributedConfiguration(
            server_mesh=mesh
        )
    )
    with get_context_stack.get_context_stack().install(context):
      zero = multiply(collections.OrderedDict(x=0, y=1))
      one = multiply(collections.OrderedDict(x=1, y=1))

    self.assertEqual(zero, 0)
    self.assertEqual(one, 1)

  def test_runs_tensorflow_on_dtensor_on_client(self):
    # Create a single device mesh for testing.
    mesh = tf.experimental.dtensor.create_mesh(
        devices=['CPU:0'], mesh_dims=[('test_dim', 1)]
    )

    @tensorflow_computation.tf_computation(
        collections.OrderedDict(x=tf.int32, y=tf.int32)
    )
    def multiply(ordered_dict):
      return ordered_dict['x'] * ordered_dict['y']

    context = cpp_execution_contexts.create_sync_experimental_distributed_cpp_execution_context(
        distributed_config=cpp_execution_contexts.DistributedConfiguration(
            client_mesh=mesh
        )
    )
    with get_context_stack.get_context_stack().install(context):
      zero = multiply(collections.OrderedDict(x=0, y=1))
      one = multiply(collections.OrderedDict(x=1, y=1))

    self.assertEqual(zero, 0)
    self.assertEqual(one, 1)

  def test_error_on_distributed_context_creation(self):
    with self.assertRaisesRegex(
        ValueError,
        (
            'Both server side and client side mesh are unspecified'
            ' in distributed configuration.'
        ),
    ):
      cpp_execution_contexts.create_sync_experimental_distributed_cpp_execution_context(
          distributed_config=cpp_execution_contexts.DistributedConfiguration()
      )
    with self.assertRaisesRegex(
        ValueError,
        'Distributed configuration is unspecified.',
    ):
      cpp_execution_contexts.create_sync_experimental_distributed_cpp_execution_context()

  def test_async_execution_context_runs_in_parallel(self):
    n_parallel_calls = 10
    sleep_time = 5

    @tensorflow_computation.tf_computation(
        collections.OrderedDict(x=tf.int32, y=tf.int32)
    )
    @tf.function
    def sleep_and_multiply(ordered_dict):
      init_time = tf.timestamp()
      n_iters = 0
      # This is a busy-sleep; TF exposes no direct sleep ops.
      while tf.timestamp() - init_time < sleep_time:
        n_iters += 1
      return (
          ordered_dict['x'] * ordered_dict['y'] * tf.math.minimum(n_iters, 10)
      )

    context = cpp_execution_contexts.create_async_local_cpp_execution_context()

    async def multiple_invocations():
      return await asyncio.gather(
          *[
              context.invoke(
                  sleep_and_multiply, collections.OrderedDict(x=1, y=1)
              )
              for _ in range(n_parallel_calls)
          ]
      )

    loop = asyncio.new_event_loop()
    # This timing-based test seems unfortunate.
    start = time.time()
    result = loop.run_until_complete(multiple_invocations())
    exec_time = time.time() - start
    # If these calls are 'truly' parallel, this will run in ~5s. If sequential,
    # 50s. So we experimentally pick a time where flakes should be extremely
    # rare (0 in 10K on testing).
    self.assertLess(exec_time, 10)
    self.assertLen(result, n_parallel_calls)
    for x in result:
      # 1 * 1 * 10
      self.assertEqual(x, 10)

  def test_returns_datasets(self):
    @tensorflow_computation.tf_computation
    def create_dataset():
      return tf.data.Dataset.range(5)

    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    with get_context_stack.get_context_stack().install(context):
      with self.subTest('unplaced'):
        dataset = create_dataset()
        self.assertEqual(
            dataset.element_spec, tf.TensorSpec(shape=[], dtype=tf.int64)
        )
        self.assertEqual(dataset.cardinality(), 5)
      with self.subTest('federated'):

        @federated_computation.federated_computation
        def create_federated_dataset():
          return intrinsics.federated_eval(create_dataset, placements.SERVER)

        dataset = create_federated_dataset()
        self.assertEqual(
            dataset.element_spec, tf.TensorSpec(shape=[], dtype=tf.int64)
        )
        self.assertEqual(dataset.cardinality(), 5)
      with self.subTest('struct'):

        @tensorflow_computation.tf_computation()
        def create_struct_of_datasets():
          return (create_dataset(), create_dataset())

        datasets = create_struct_of_datasets()
        self.assertLen(datasets, 2)
        self.assertEqual(
            [d.element_spec for d in datasets],
            [
                tf.TensorSpec(shape=[], dtype=tf.int64),
                tf.TensorSpec(shape=[], dtype=tf.int64),
            ],
        )
        self.assertEqual([d.cardinality() for d in datasets], [5, 5])


if __name__ == '__main__':
  absltest.main()
