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

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.impl.executors import composing_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_factory

tf.compat.v1.enable_v2_behavior()


def create_test_federating_executor() -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  return federating_executor.FederatingExecutor({
      placement_literals.SERVER: create_bottom_stack(),
      placement_literals.CLIENTS: [create_bottom_stack() for _ in range(3)],
      None: create_bottom_stack()
  })


def create_test_composing_executor() -> composing_executor.ComposingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  def create_worker_stack():
    return federating_executor.FederatingExecutor({
        placement_literals.SERVER: create_bottom_stack(),
        placement_literals.CLIENTS: [create_bottom_stack() for _ in range(3)],
        None: create_bottom_stack()
    })

  def create_aggregation_stack(children):
    return composing_executor.ComposingExecutor(create_bottom_stack(), children)

  return create_aggregation_stack([
      create_aggregation_stack([create_worker_stack() for _ in range(3)]),
      create_aggregation_stack([create_worker_stack() for _ in range(3)]),
      create_aggregation_stack([create_worker_stack() for _ in range(3)]),
  ])


@parameterized.named_parameters([
    ('federating_executor', create_test_federating_executor(), 3),
    ('composing_executor', create_test_composing_executor(), 27),
])
class ComputeIntrinsicFederatedBroadcastTest(executor_test_utils.AsyncTestCase,
                                             parameterized.TestCase):

  def test_returns_value_with_federated_type_at_server(self, executor,
                                                       number_of_clients):
    del number_of_clients  # Unused.
    value, type_signature = executor_test_utils.create_dummy_value_at_server()

    value = self.run_sync(executor.create_value(value, type_signature))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_broadcast(executor, value))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = type_factory.at_clients(
        type_signature.member, all_equal=True)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, 10.0)

  def test_raises_type_error_with_federated_type_at_clients(
      self, executor, number_of_clients):
    value, type_signature = executor_test_utils.create_dummy_value_at_clients(
        number_of_clients)

    value = self.run_sync(executor.create_value(value, type_signature))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_broadcast(executor, value))

  def test_raises_type_error_with_unplaced_type(self, executor,
                                                number_of_clients):
    del number_of_clients  # Unused.
    value, type_signature = executor_test_utils.create_dummy_value_unplaced()

    value = self.run_sync(executor.create_value(value, type_signature))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_broadcast(executor, value))


@parameterized.named_parameters([
    ('federating_executor', create_test_federating_executor()),
    ('composing_executor', create_test_composing_executor()),
])
class ComputeIntrinsicFederatedValueTest(executor_test_utils.AsyncTestCase,
                                         parameterized.TestCase):

  def test_returns_value_with_unplaced_type_and_clients(self, executor):
    value, type_signature = executor_test_utils.create_dummy_value_unplaced()

    value = self.run_sync(executor.create_value(value, type_signature))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_value(
            executor, value, placement_literals.CLIENTS))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = type_factory.at_clients(type_signature, all_equal=True)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, 10.0)

  def test_returns_value_with_unplaced_type_and_server(self, executor):
    value, type_signature = executor_test_utils.create_dummy_value_unplaced()

    value = self.run_sync(executor.create_value(value, type_signature))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_value(
            executor, value, placement_literals.SERVER))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = type_factory.at_server(type_signature)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, 10.0)


if __name__ == '__main__':
  absltest.main()
