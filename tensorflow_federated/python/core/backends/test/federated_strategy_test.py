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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import federated_strategy
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import placement_literals


def create_test_executor(
    number_of_clients: int = 3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  factory = federated_strategy.TestFederatedStrategy.factory({
      placement_literals.SERVER:
          create_bottom_stack(),
      placement_literals.CLIENTS: [
          create_bottom_stack() for _ in range(number_of_clients)
      ],
  })
  return federating_executor.FederatingExecutor(factory, create_bottom_stack())


class FederatingExecutorCreateCallTest(executor_test_utils.AsyncTestCase,
                                       parameterized.TestCase,
                                       test_case.TestCase):

  @parameterized.named_parameters([
      ('sum_greater_than_mask', [10, 11, 12], 10, 33),
      ('sum_less_than_mask', [10, 11, 12], 4, 3),
  ])
  def test_returns_value_with_intrinsic_def_federated_secure_sum(
      self, value, bitwidth, expected_result):
    executor = create_test_executor()
    comp, comp_type = executor_test_utils.create_dummy_intrinsic_def_federated_secure_sum(
    )
    value_type = computation_types.at_clients(tf.int32, all_equal=False)
    bitwidth_type = computation_types.TensorType(tf.int32)

    comp = self.run_sync(executor.create_value(comp, comp_type))
    arg_1 = self.run_sync(executor.create_value(value, value_type))
    arg_2 = self.run_sync(executor.create_value(bitwidth, bitwidth_type))
    args = self.run_sync(executor.create_struct([arg_1, arg_2]))
    result = self.run_sync(executor.create_call(comp, args))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assert_types_identical(result.type_signature, comp_type.result)
    actual_result = self.run_sync(result.compute())
    self.assertEqual(actual_result, expected_result)


if __name__ == '__main__':
  test_case.main()
