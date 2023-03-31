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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import executor_factory
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _install_executor_in_synchronous_context(executor_factory_instance):
  context = sync_execution_context.SyncExecutionContext(
      executor_factory_instance
  )
  return context_stack_impl.context_stack.install(context)


class ExecutionContextIntegrationTest(parameterized.TestCase):

  def test_simple_no_arg_tf_computation_with_int_result(self):
    @tensorflow_computation.tf_computation
    def comp():
      return tf.constant(10)

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      result = comp()

    self.assertEqual(result, 10)

  def test_one_arg_tf_computation_with_int_param_and_result(self):
    @tensorflow_computation.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 10)

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      result = comp(3)

    self.assertEqual(result, 13)

  def test_three_arg_tf_computation_with_int_params_and_result(self):
    @tensorflow_computation.tf_computation(tf.int32, tf.int32, tf.int32)
    def comp(x, y, z):
      return tf.multiply(tf.add(x, y), z)

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      result = comp(3, 4, 5)

    self.assertEqual(result, 35)

  def test_tf_computation_with_dataset_params_and_int_result(self):

    @tensorflow_computation.tf_computation(
        computation_types.SequenceType(tf.int32)
    )
    def comp(ds):
      return ds.reduce(np.int32(0), lambda x, y: x + y)

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      ds = tf.data.Dataset.range(10).map(lambda x: tf.cast(x, tf.int32))
      result = comp(ds)

    self.assertEqual(result, 45)

  def test_tf_computation_with_structured_result(self):
    @tensorflow_computation.tf_computation
    def comp():
      return collections.OrderedDict([
          ('a', tf.constant(10)),
          ('b', tf.constant(20)),
      ])

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      result = comp()

    self.assertIsInstance(result, collections.OrderedDict)
    self.assertDictEqual(result, {'a': 10, 'b': 20})

  def test_changing_cardinalities_across_calls(self):

    @federated_computation.federated_computation(
        computation_types.at_clients(tf.int32)
    )
    def comp(x):
      return x

    five_ints = list(range(5))
    ten_ints = list(range(10))

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      five = comp(five_ints)
      ten = comp(ten_ints)

    self.assertEqual(five, five_ints)
    self.assertEqual(ten, ten_ints)

  def test_conflicting_cardinalities_within_call(self):
    @federated_computation.federated_computation([
        computation_types.at_clients(tf.int32),
        computation_types.at_clients(tf.int32),
    ])
    def comp(x):
      return x

    five_ints = list(range(5))
    ten_ints = list(range(10))

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
        comp([five_ints, ten_ints])

  def test_tuple_argument_can_accept_unnamed_elements(self):
    @tensorflow_computation.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    executor = executor_factory.local_cpp_executor_factory()
    with _install_executor_in_synchronous_context(executor):
      # pylint:disable=no-value-for-parameter
      result = foo(structure.Struct([(None, 2), (None, 3)]))
      # pylint:enable=no-value-for-parameter

    self.assertEqual(result, 5)

  def test_raises_exception(self):
    factory = executor_factory.local_cpp_executor_factory()

    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)

    @federated_computation.federated_computation(arg_type)
    def identity(x):
      return x

    context = sync_execution_context.SyncExecutionContext(
        factory, cardinality_inference_fn=lambda x, y: {placements.CLIENTS: 1}
    )
    with context_stack_impl.context_stack.install(context):
      # This argument conflicts with the value returned by the
      # cardinality-inference function; we should get an error surfaced.
      data = [0, 1]
      with self.assertRaises(Exception) as e:
        self.assertTrue(hasattr(e, 'status'))
        self.assertTrue(hasattr(e.status, 'code_int'))
        identity(data)

  def test_sync_interface_interops_with_asyncio(self):
    @tensorflow_computation.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    async def sleep_and_add_one(x):
      await asyncio.sleep(0.1)
      return add_one(x)

    factory = executor_factory.local_cpp_executor_factory()
    context = sync_execution_context.SyncExecutionContext(
        factory, cardinality_inference_fn=lambda x, y: {placements.CLIENTS: 1}
    )
    with context_stack_impl.context_stack.install(context):
      one = asyncio.run(sleep_and_add_one(0))
      self.assertEqual(one, 1)


if __name__ == '__main__':
  absltest.main()
