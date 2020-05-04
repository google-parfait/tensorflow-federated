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

from tensorflow_federated.python.core.impl.executors import composing_federating_strategy
from tensorflow_federated.python.core.impl.executors import default_federating_strategy
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_factory

tf.compat.v1.enable_v2_behavior()


def create_test_default_strategy_with_default_strategy(
    num_clients=3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  factory = default_federating_strategy.DefaultFederatingStrategy.factory({
      placement_literals.SERVER:
          create_bottom_stack(),
      placement_literals.CLIENTS: [
          create_bottom_stack() for _ in range(num_clients)
      ],
  })
  return federating_executor.FederatingExecutor(factory, create_bottom_stack())


def create_test_default_strategy_with_composing_strategy(
    clients_per_stack=3,
    stacks_per_layer=3,
    num_layers=3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  def create_worker_stack():
    factroy = default_federating_strategy.DefaultFederatingStrategy.factory({
        placement_literals.SERVER:
            create_bottom_stack(),
        placement_literals.CLIENTS: [
            create_bottom_stack() for _ in range(clients_per_stack)
        ],
    })
    return federating_executor.FederatingExecutor(factroy,
                                                  create_bottom_stack())

  def create_aggregation_stack(children):
    factory = composing_federating_strategy.ComposingFederatingStrategy.factory(
        create_bottom_stack(), children)
    return federating_executor.FederatingExecutor(factory,
                                                  create_bottom_stack())

  def create_aggregation_layer(num_stacks):
    return create_aggregation_stack(
        [create_worker_stack() for _ in range(num_stacks)])

  return create_aggregation_stack(
      [create_aggregation_layer(stacks_per_layer) for _ in range(num_layers)])


@parameterized.named_parameters([
    ('default_strategy', create_test_default_strategy_with_default_strategy(),
     3),
    ('composing_strategy_9_clients',
     create_test_default_strategy_with_composing_strategy(
         clients_per_stack=3, stacks_per_layer=3, num_layers=1), 9),
    ('composing_strategy_27_clients',
     create_test_default_strategy_with_composing_strategy(
         clients_per_stack=3, stacks_per_layer=3, num_layers=3), 27),
])
class ComputeIntrinsicFederatedBroadcastTest(executor_test_utils.AsyncTestCase,
                                             parameterized.TestCase):

  def test_returns_value_with_federated_type_at_server(self, executor,
                                                       num_clients):
    del num_clients  # Unused.
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
      self, executor, num_clients):
    value, type_signature = executor_test_utils.create_dummy_value_at_clients(
        num_clients)

    value = self.run_sync(executor.create_value(value, type_signature))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_broadcast(executor, value))

  def test_raises_type_error_with_unplaced_type(self, executor, num_clients):
    del num_clients  # Unused.
    value, type_signature = executor_test_utils.create_dummy_value_unplaced()

    value = self.run_sync(executor.create_value(value, type_signature))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_broadcast(executor, value))


@parameterized.named_parameters([
    ('default_strategy', create_test_default_strategy_with_default_strategy()),
    ('composing_strategy_9_clients',
     create_test_default_strategy_with_composing_strategy(
         clients_per_stack=3, stacks_per_layer=3, num_layers=1)),
    ('composing_strategy_27_clients',
     create_test_default_strategy_with_composing_strategy(
         clients_per_stack=3, stacks_per_layer=3, num_layers=3)),
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


class ComputeIntrinsicFederatedWeightedMeanTest(
    executor_test_utils.AsyncTestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ('default_strategy_3_clients',
       create_test_default_strategy_with_default_strategy(3), 3),
      ('default_strategy_10_clients',
       create_test_default_strategy_with_default_strategy(10), 10),
      ('composing_strategy_9_clients',
       create_test_default_strategy_with_composing_strategy(
           clients_per_stack=3, stacks_per_layer=3, num_layers=1), 9),
      ('composing_strategy_27_clients',
       create_test_default_strategy_with_composing_strategy(
           clients_per_stack=3, stacks_per_layer=3, num_layers=3), 27),
  ])
  def test_computes_weighted_mean(
      self,
      executor,
      num_clients,
  ):
    value, type_signature = executor_test_utils.create_dummy_value_at_clients(
        num_clients)

    # Weighted mean computed in Python
    expected_result = sum([x**2 for x in value]) / sum(value)

    value = self.run_sync(executor.create_value(value, type_signature))
    arg = self.run_sync(executor.create_tuple([value, value]))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_weighted_mean(executor, arg))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = type_factory.at_server(type_signature.member)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters([
      ('default_strategy_unplaced_type',
       create_test_default_strategy_with_default_strategy(),
       executor_test_utils.create_dummy_value_unplaced()),
      ('composing_strategy_unplaced_type',
       create_test_default_strategy_with_composing_strategy(),
       executor_test_utils.create_dummy_value_unplaced()),
      ('default_strategy_server_placement',
       create_test_default_strategy_with_default_strategy(),
       executor_test_utils.create_dummy_value_at_server()),
      ('composing_strategy_server_placement',
       create_test_default_strategy_with_composing_strategy(),
       executor_test_utils.create_dummy_value_at_server()),
  ])
  def test_raises_type_error(self, executor, value_and_type_signature):
    value, type_signature = value_and_type_signature
    value = self.run_sync(executor.create_value(value, type_signature))
    arg = self.run_sync(executor.create_tuple([value, value]))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_weighted_mean(
              executor, arg))

  @parameterized.named_parameters([
      ('default_strategy', create_test_default_strategy_with_default_strategy(),
       3),
      ('composing_strategy',
       create_test_default_strategy_with_composing_strategy(), 27),
  ])
  def test_raises_type_error_with_singleton_tuple(
      self,
      executor,
      num_clients,
  ):
    value, type_signature = executor_test_utils.create_dummy_value_at_clients(
        num_clients)
    value = self.run_sync(executor.create_value(value, type_signature))
    arg = self.run_sync(executor.create_tuple([value]))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_weighted_mean(
              executor, arg))


if __name__ == '__main__':
  absltest.main()
