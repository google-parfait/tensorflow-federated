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
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.types import computation_types


class FederatedResolvingStrategyValueComputeTest(
    executor_test_utils.AsyncTestCase):

  def test_returns_value_with_embedded_value(self):
    tensor_type = computation_types.TensorType(tf.float32, shape=[])
    value = eager_tf_executor.EagerValue(10.0, tensor_type)
    type_signature = computation_types.TensorType(tf.float32)
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, 10.0)

  def test_returns_value_with_federated_type_at_clients(self):
    tensor_type = computation_types.TensorType(tf.float32, shape=[])
    value = [
        eager_tf_executor.EagerValue(10.0, tensor_type),
        eager_tf_executor.EagerValue(11.0, tensor_type),
        eager_tf_executor.EagerValue(12.0, tensor_type),
    ]
    type_signature = computation_types.at_clients(tf.float32)
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, [10.0, 11.0, 12.0])

  def test_returns_value_with_federated_type_at_clients_all_equal(self):
    tensor_type = computation_types.TensorType(tf.float32, shape=[])
    value = [eager_tf_executor.EagerValue(10.0, tensor_type)]
    type_signature = computation_types.at_clients(tf.float32, all_equal=True)
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, 10.0)

  def test_returns_value_with_federated_type_at_server(self):
    tensor_type = computation_types.TensorType(tf.float32, shape=[])
    value = [eager_tf_executor.EagerValue(10.0, tensor_type)]
    type_signature = computation_types.at_server(tf.float32)
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, 10.0)

  def test_returns_value_with_structure_value(self):
    tensor_type = computation_types.TensorType(tf.float32, shape=[])
    element = eager_tf_executor.EagerValue(10.0, tensor_type)
    element_type = computation_types.TensorType(tf.float32)
    names = ['a', 'b', 'c']
    value = structure.Struct((n, element) for n in names)
    type_signature = computation_types.StructType(
        (n, element_type) for n in names)
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        value, type_signature)

    result = self.run_sync(value.compute())

    expected_result = structure.Struct((n, 10.0) for n in names)
    self.assertEqual(result, expected_result)

  def test_raises_type_error_with_unembedded_federated_type(self):
    value = [10.0, 11.0, 12.0]
    type_signature = computation_types.at_clients(tf.float32)
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        value, type_signature)

    with self.assertRaises(TypeError):
      self.run_sync(value.compute())

  def test_raises_runtime_error_with_unsupported_value_or_type(self):
    value = 10.0
    type_signature = computation_types.TensorType(tf.float32)
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        value, type_signature)

    with self.assertRaises(RuntimeError):
      self.run_sync(value.compute())


if __name__ == '__main__':
  absltest.main()
