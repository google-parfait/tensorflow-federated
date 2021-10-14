# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types


class ExecutionContextsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_test_execution_context()

  @parameterized.named_parameters(
      ('one_client_not_divisible', [1], 1),
      ('two_clients_none_divisible', [1, 2], 3),
      ('three_clients_one_divisible', [1, 2, 10], 3),
      ('all_clients_divisible_by_modulus', [x * 5 for x in range(5)], 0))
  def test_executes_computation_with_modular_secure_sum_integer_modulus(
      self, arg, expected_result):

    modulus = 5

    @computations.federated_computation(computation_types.at_clients(tf.int32))
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    self.assertEqual(expected_result, modular_sum_by_five(arg))

  @parameterized.named_parameters(
      ('one_client_not_divisible', [[1, 2]], [1, 2]),
      ('two_clients_none_divisible', [[1, 2], [3, 4]], [4, 6]),
      ('three_clients_one_divisible', [[1, 2], [3, 4], [10, 14]], [4, 6]),
      ('two_clients_one_partially_divisible', [[1, 2], [3, 4], [10, 15]
                                              ], [4, 0]))
  def test_executes_computation_with_modular_secure_sum_struct_modulus(
      self, arg, expected_result):

    modulus = [5, 7]

    @computations.federated_computation(
        computation_types.at_clients([tf.int32, tf.int32]))
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    self.assertEqual(expected_result, modular_sum_by_five(arg))


if __name__ == '__main__':
  absltest.main()
