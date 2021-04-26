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

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federated_composing_strategy
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def create_test_federated_stack(
    num_clients=3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  factory = federated_resolving_strategy.FederatedResolvingStrategy.factory({
      placements.SERVER: create_bottom_stack(),
      placements.CLIENTS: [create_bottom_stack() for _ in range(num_clients)],
  })
  return federating_executor.FederatingExecutor(factory, create_bottom_stack())


def create_test_aggregated_stack(
    clients_per_stack=3,
    stacks_per_layer=3,
    num_layers=3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  def create_worker_stack():
    factroy = federated_resolving_strategy.FederatedResolvingStrategy.factory({
        placements.SERVER:
            create_bottom_stack(),
        placements.CLIENTS: [
            create_bottom_stack() for _ in range(clients_per_stack)
        ],
    })
    return federating_executor.FederatingExecutor(factroy,
                                                  create_bottom_stack())

  def create_aggregation_stack(children):
    factory = federated_composing_strategy.FederatedComposingStrategy.factory(
        create_bottom_stack(), children)
    return federating_executor.FederatingExecutor(factory,
                                                  create_bottom_stack())

  def create_aggregation_layer(num_stacks):
    return create_aggregation_stack(
        [create_worker_stack() for _ in range(num_stacks)])

  return create_aggregation_stack(
      [create_aggregation_layer(stacks_per_layer) for _ in range(num_layers)])


# pyformat: disable
@parameterized.named_parameters([
    ('federated_stack',
     create_test_federated_stack(),
     3),
    ('aggregated_stack_9_clients',
     create_test_aggregated_stack(
         clients_per_stack=3, stacks_per_layer=3, num_layers=1),
     9),
    ('aggregated_stack_27_clients',
     create_test_aggregated_stack(
         clients_per_stack=3, stacks_per_layer=3, num_layers=3),
     27),
])
# pyformat: enable
class ComputeIntrinsicFederatedBroadcastTest(executor_test_utils.AsyncTestCase,
                                             parameterized.TestCase):

  def test_returns_value_with_federated_type_at_server(self, executor,
                                                       num_clients):
    del num_clients  # Unused.
    value, type_signature = executor_test_utils.create_whimsy_value_at_server()

    value = self.run_sync(executor.create_value(value, type_signature))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_broadcast(executor, value))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = computation_types.at_clients(
        type_signature.member, all_equal=True)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, 10.0)

  def test_raises_type_error_with_federated_type_at_clients(
      self, executor, num_clients):
    value, type_signature = executor_test_utils.create_whimsy_value_at_clients(
        num_clients)

    value = self.run_sync(executor.create_value(value, type_signature))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_broadcast(executor, value))

  def test_raises_type_error_with_unplaced_type(self, executor, num_clients):
    del num_clients  # Unused.
    value, type_signature = executor_test_utils.create_whimsy_value_unplaced()

    value = self.run_sync(executor.create_value(value, type_signature))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_broadcast(executor, value))


# pyformat: disable
@parameterized.named_parameters([
    ('federated_stack',
     create_test_federated_stack()),
    ('aggregated_stack_9_clients',
     create_test_aggregated_stack(
         clients_per_stack=3, stacks_per_layer=3, num_layers=1)),
    ('aggregated_stack_27_clients',
     create_test_aggregated_stack(
         clients_per_stack=3, stacks_per_layer=3, num_layers=3)),
])
# pyformat: enable
class ComputeIntrinsicFederatedValueTest(executor_test_utils.AsyncTestCase,
                                         parameterized.TestCase):

  def test_returns_value_with_unplaced_type_and_clients(self, executor):
    value, type_signature = executor_test_utils.create_whimsy_value_unplaced()

    value = self.run_sync(executor.create_value(value, type_signature))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_value(
            executor, value, placements.CLIENTS))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = computation_types.at_clients(type_signature, all_equal=True)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, 10.0)

  def test_returns_value_with_unplaced_type_and_server(self, executor):
    value, type_signature = executor_test_utils.create_whimsy_value_unplaced()

    value = self.run_sync(executor.create_value(value, type_signature))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_value(
            executor, value, placements.SERVER))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = computation_types.at_server(type_signature)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, 10.0)


class ComputeIntrinsicFederatedWeightedMeanTest(
    executor_test_utils.AsyncTestCase, parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('default_strategy_3_clients',
       create_test_federated_stack(3),
       3),
      ('default_strategy_10_clients',
       create_test_federated_stack(10),
       10),
      ('aggregated_stack_9_clients',
       create_test_aggregated_stack(
           clients_per_stack=3, stacks_per_layer=3, num_layers=1),
       9),
      ('aggregated_stack_27_clients',
       create_test_aggregated_stack(
           clients_per_stack=3, stacks_per_layer=3, num_layers=3),
       27),
  ])
  # pyformat: enable
  def test_computes_weighted_mean(
      self,
      executor,
      num_clients,
  ):
    value, type_signature = executor_test_utils.create_whimsy_value_at_clients(
        num_clients)

    # Weighted mean computed in Python
    expected_result = sum([x**2 for x in value]) / sum(value)

    value = self.run_sync(executor.create_value(value, type_signature))
    arg = self.run_sync(executor.create_struct([value, value]))
    result = self.run_sync(
        executor_utils.compute_intrinsic_federated_weighted_mean(executor, arg))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    expected_type = computation_types.at_server(type_signature.member)
    self.assertEqual(result.type_signature.compact_representation(),
                     expected_type.compact_representation())
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, expected_result)

  # pyformat: disable
  @parameterized.named_parameters([
      ('default_strategy_unplaced_type',
       create_test_federated_stack(),
       executor_test_utils.create_whimsy_value_unplaced()),
      ('composing_strategy_unplaced_type',
       create_test_aggregated_stack(),
       executor_test_utils.create_whimsy_value_unplaced()),
      ('default_strategy_server_placement',
       create_test_federated_stack(),
       executor_test_utils.create_whimsy_value_at_server()),
      ('composing_strategy_server_placement',
       create_test_aggregated_stack(),
       executor_test_utils.create_whimsy_value_at_server()),
  ])
  # pyformat: enable
  def test_raises_type_error(self, executor, value_and_type_signature):
    value, type_signature = value_and_type_signature
    value = self.run_sync(executor.create_value(value, type_signature))
    arg = self.run_sync(executor.create_struct([value, value]))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_weighted_mean(
              executor, arg))

  # pyformat: disable
  @parameterized.named_parameters([
      ('federated_stack',
       create_test_federated_stack(),
       3),
      ('composing_strategy',
       create_test_aggregated_stack(),
       27),
  ])
  # pyformat: enable
  def test_raises_type_error_with_singleton_tuple(
      self,
      executor,
      num_clients,
  ):
    value, type_signature = executor_test_utils.create_whimsy_value_at_clients(
        num_clients)
    value = self.run_sync(executor.create_value(value, type_signature))
    arg = self.run_sync(executor.create_struct([value]))

    with self.assertRaises(TypeError):
      self.run_sync(
          executor_utils.compute_intrinsic_federated_weighted_mean(
              executor, arg))


class TypeUtilsTest(test_case.TestCase, parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('buiding_block_and_type_spec',
       building_block_factory.create_compiled_identity(
           computation_types.TensorType(tf.int32)),
       computation_types.FunctionType(tf.int32, tf.int32),
       computation_types.FunctionType(tf.int32, tf.int32)),
      ('buiding_block_and_none',
       building_block_factory.create_compiled_identity(
           computation_types.TensorType(tf.int32)),
       None,
       computation_types.FunctionType(tf.int32, tf.int32)),
      ('int_and_type_spec',
       10,
       computation_types.TensorType(tf.int32),
       computation_types.TensorType(tf.int32)),
  ])
  # pyformat: enable
  def test_reconcile_value_with_type_spec_returns_type(self, value, type_spec,
                                                       expected_type):
    actual_type = executor_utils.reconcile_value_with_type_spec(
        value, type_spec)
    self.assertEqual(actual_type, expected_type)

  # pyformat: disable
  @parameterized.named_parameters([
      ('building_block_and_bad_type_spec',
       building_block_factory.create_compiled_identity(
           computation_types.TensorType(tf.int32)),
       computation_types.TensorType(tf.int32)),
      ('int_and_none', 10, None),
  ])
  # pyformat: enable
  def test_reconcile_value_with_type_spec_raises_type_error(
      self, value, type_spec):
    with self.assertRaises(TypeError):
      executor_utils.reconcile_value_with_type_spec(value, type_spec)

  # pyformat: disable
  @parameterized.named_parameters([
      ('value_type_and_type_spec',
       computation_types.TensorType(tf.int32),
       computation_types.TensorType(tf.int32),
       computation_types.TensorType(tf.int32)),
      ('value_type_and_none',
       computation_types.TensorType(tf.int32),
       None,
       computation_types.TensorType(tf.int32)),
  ])
  # pyformat: enable
  def test_reconcile_value_type_with_type_spec_returns_type(
      self, value_type, type_spec, expected_type):
    actual_type = executor_utils.reconcile_value_type_with_type_spec(
        value_type, type_spec)
    self.assertEqual(actual_type, expected_type)

  def test_reconcile_value_type_with_type_spec_raises_type_error_value_type_and_bad_type_spec(
      self):
    value_type = computation_types.TensorType(tf.int32)
    type_spec = computation_types.TensorType(tf.string)

    with self.assertRaises(TypeError):
      executor_utils.reconcile_value_type_with_type_spec(value_type, type_spec)


if __name__ == '__main__':
  absltest.main()
