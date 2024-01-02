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
import grpc
import numpy as np

from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements

_CLIENTS_INT = computation_types.FederatedType(np.int32, placements.CLIENTS)
_CLIENTS_INT_LIST = computation_types.FederatedType(
    [np.int32, np.int32], placements.CLIENTS
)


class SecureModularSumTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  @parameterized.named_parameters(
      ('one_client_not_divisible', [1], 1, _CLIENTS_INT),
      ('two_clients_none_divisible', [1, 2], 3, _CLIENTS_INT),
      ('three_clients_one_divisible', [1, 2, 10], 3, _CLIENTS_INT),
      (
          'all_clients_divisible_by_modulus',
          [x * 5 for x in range(5)],
          0,
          _CLIENTS_INT,
      ),
      (
          'nonscalar_struct_arg',
          [([1, 2], 3), ([4, 5], 6)],
          (np.array([0, 2], dtype=np.int32), 4),
          computation_types.FederatedType(
              ((np.int32, [2]), np.int32), placements.CLIENTS
          ),
      ),
  )
  def test_executes_computation_with_modular_secure_sum_integer_modulus(
      self, arg, expected_result, tff_type
  ):
    modulus = 5

    @federated_computation.federated_computation(tff_type)
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    actual_result = modular_sum_by_five(arg)

    if isinstance(actual_result, tuple) and isinstance(expected_result, tuple):
      for a, b in zip(actual_result, expected_result):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
          np.testing.assert_array_equal(a, b)
        else:
          self.assertEqual(a, b)
    else:
      self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('one_client_not_divisible', [[1, 2]], [1, 2], _CLIENTS_INT_LIST),
      (
          'two_clients_none_divisible',
          [[1, 2], [3, 4]],
          [4, 6],
          _CLIENTS_INT_LIST,
      ),
      (
          'three_clients_one_divisible',
          [[1, 2], [3, 4], [10, 14]],
          [4, 6],
          _CLIENTS_INT_LIST,
      ),
      (
          'two_clients_one_partially_divisible',
          [[1, 2], [3, 4], [10, 15]],
          [4, 0],
          _CLIENTS_INT_LIST,
      ),
  )
  def test_executes_computation_with_modular_secure_sum_struct_modulus(
      self, arg, expected_result, tff_type
  ):
    modulus = [5, 7]

    @federated_computation.federated_computation(tff_type)
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    self.assertEqual(expected_result, modular_sum_by_five(arg))


class SecureSumBitwidthTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  @parameterized.named_parameters(
      ('one_client', [1]),
      (
          'three_clients',
          [1, 2, 10],
      ),
      ('five_clients', [x * 5 for x in range(5)]),
  )
  def test_executes_computation_with_bitwidth_secure_sum_large_bitwidth(
      self, arg
  ):
    bitwidth = 32

    expected_result = sum(arg)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    self.assertEqual(expected_result, sum_with_bitwidth(arg))

  @parameterized.named_parameters(
      (
          'two_clients_scalar_tensors',
          [[1, 2], [3, 4]],
          [4, 6],
          computation_types.FederatedType(
              [np.int32, np.int32], placements.CLIENTS
          ),
      ),
      (
          'two_clients_nonscalar_tensors',
          [
              [np.ones(shape=[10], dtype=np.int32), 2],
              [np.ones(shape=[10], dtype=np.int32), 4],
          ],
          [2 * np.ones(shape=[10], dtype=np.int32), 6],
          computation_types.FederatedType(
              [
                  computation_types.TensorType(dtype=np.int32, shape=[10]),
                  np.int32,
              ],
              placements.CLIENTS,
          ),
      ),
  )
  def test_executes_computation_with_argument_structure(
      self, arg, expected_result, tff_type
  ):
    bitwidth = 32

    @federated_computation.federated_computation(tff_type)
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    actual_result = sum_with_bitwidth(arg)

    for a, b in zip(actual_result, expected_result):
      if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
      else:
        self.assertEqual(a, b)


class SecureSumMaxValueTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  def test_raises_with_arguments_over_max_value(self):
    max_value = 1

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    with self.assertRaises(grpc.RpcError):
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

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    self.assertEqual(expected_result, secure_sum(arg))

  @parameterized.named_parameters(
      (
          'two_clients_scalar_tensors',
          [[1, 2], [3, 4]],
          [4, 6],
          computation_types.FederatedType(
              [np.int32, np.int32], placements.CLIENTS
          ),
      ),
      (
          'two_clients_nonscalar_tensors',
          [
              [np.ones(shape=[10], dtype=np.int32), 2],
              [np.ones(shape=[10], dtype=np.int32), 4],
          ],
          [2 * np.ones(shape=[10], dtype=np.int32), 6],
          computation_types.FederatedType(
              [
                  computation_types.TensorType(dtype=np.int32, shape=[10]),
                  np.int32,
              ],
              placements.CLIENTS,
          ),
      ),
  )
  def test_executes_computation_with_argument_structure(
      self, arg, expected_result, tff_type
  ):
    max_value = 100

    @federated_computation.federated_computation(tff_type)
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    actual_result = secure_sum(arg)

    for a, b in zip(actual_result, expected_result):
      if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
      else:
        self.assertEqual(a, b)


if __name__ == '__main__':
  absltest.main()
