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


class SecureModularSumTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_test_execution_context()

  @parameterized.named_parameters(
      ('one_client_not_divisible', [1], 1),
      ('two_clients_none_divisible', [1, 2], 3),
      ('three_clients_one_divisible', [1, 2, 10], 3),
      ('all_clients_divisible_by_modulus', [x * 5 for x in range(5)], 0),
  )
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
                                              ], [4, 0]),
  )
  def test_executes_computation_with_modular_secure_sum_struct_modulus(
      self, arg, expected_result):

    modulus = [5, 7]

    @computations.federated_computation(
        computation_types.at_clients([tf.int32, tf.int32]))
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    self.assertEqual(expected_result, modular_sum_by_five(arg))

  def test_executes_computation_with_modular_secure_sum_nonscalar_struct_arg_integer_modulus(
      self):

    modulus = 5

    dtype = computation_types.to_type(((tf.int32, [2]), tf.int32))

    @computations.federated_computation(computation_types.at_clients(dtype))
    def fn(x):
      return intrinsics.federated_secure_modular_sum(x, modulus)

    with self.assertRaises(TypeError):
      # Currently fails due to mismatch between expectations of serialization
      # code, inserting a scalar tensor into a structure, and expectations of
      # TF-generating code: that once we do this promotion, types must match.
      # b/203424058
      fn([([1, 2], 3), ([4, 5], 6)])


class SecureSumBitwidthTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_test_execution_context()

  @parameterized.named_parameters(
      ('one_client', [1]),
      (
          'three_clients',
          [1, 2, 10],
      ),
      ('five_clients', [x * 5 for x in range(5)]),
  )
  def test_executes_computation_with_bitwidth_secure_sum_large_bitwidth(
      self, arg):

    bitwidth = 32

    expected_result = sum(arg)

    @computations.federated_computation(computation_types.at_clients(tf.int32))
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    self.assertEqual(expected_result, sum_with_bitwidth(arg))

  @parameterized.named_parameters(
      ('two_clients_scalar_tensors', [[1, 2], [3, 4]], [4, 6],
       computation_types.at_clients([tf.int32, tf.int32])),
      ('two_clients_nonscalar_tensors', [[
          tf.ones(shape=[10], dtype=tf.int32), 2
      ], [tf.ones(shape=[10], dtype=tf.int32), 4
         ]], [2 * tf.ones(shape=[10], dtype=tf.int32).numpy(), 6],
       computation_types.at_clients([
           computation_types.TensorType(dtype=tf.int32, shape=[10]), tf.int32
       ])),
  )
  def test_executes_computation_with_argument_structure(self, arg,
                                                        expected_result,
                                                        tff_type):

    bitwidth = 32

    @computations.federated_computation(tff_type)
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    self.assertAllClose(expected_result, sum_with_bitwidth(arg))


class SecureSumMaxValueTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_test_execution_context()

  def test_raises_with_arguments_over_max_value(self):

    max_value = 1

    @computations.federated_computation(computation_types.at_clients(tf.int32))
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      secure_sum([2, 4])

  @parameterized.named_parameters(
      ('one_client', [1]),
      (
          'three_clients',
          [1, 2, 10],
      ),
      ('five_clients', [x * 5 for x in range(5)]),
  )
  def test_executes_computation_with_secure_sum_under_max_values(self, arg):

    max_value = 30

    expected_result = sum(arg)

    @computations.federated_computation(computation_types.at_clients(tf.int32))
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    self.assertEqual(expected_result, secure_sum(arg))

  @parameterized.named_parameters(
      ('two_clients_scalar_tensors', [[1, 2], [3, 4]], [4, 6],
       computation_types.at_clients([tf.int32, tf.int32])),
      ('two_clients_nonscalar_tensors', [[
          tf.ones(shape=[10], dtype=tf.int32), 2
      ], [tf.ones(shape=[10], dtype=tf.int32), 4
         ]], [2 * tf.ones(shape=[10], dtype=tf.int32).numpy(), 6],
       computation_types.at_clients([
           computation_types.TensorType(dtype=tf.int32, shape=[10]), tf.int32
       ])),
  )
  def test_executes_computation_with_argument_structure(self, arg,
                                                        expected_result,
                                                        tff_type):

    max_value = 100

    @computations.federated_computation(tff_type)
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    self.assertAllClose(expected_result, secure_sum(arg))


if __name__ == '__main__':
  absltest.main()
