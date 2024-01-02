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

import inspect
from typing import NoReturn, Optional
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tree

from tensorflow_federated.python.core.backends.test import cpp_execution_contexts
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
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


def assert_value_equal(a: object, b: object) -> Optional[NoReturn]:
  def _assert_value_equal(a: object, b: object) -> Optional[NoReturn]:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
      np.testing.assert_array_equal(a, b)
    elif a != b:
      raise AssertionError(f'{a} != {b}')

  tree.map_structure(_assert_value_equal, a, b)


class CreateAsyncTestCPPExecutionContextTest(absltest.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.create_async_test_cpp_execution_context,
        execution_contexts.create_async_test_cpp_execution_context,
    )

  def test_returns_async_context(self):
    context = cpp_execution_contexts.create_async_test_cpp_execution_context()
    self.assertIsInstance(
        context, async_execution_context.AsyncExecutionContext
    )


class SetAsyncTestCPPExecutionContextTest(absltest.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.set_async_test_cpp_execution_context,
        execution_contexts.set_async_test_cpp_execution_context,
    )


class CreateSyncTestCPPExecutionContextTest(absltest.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.create_sync_test_cpp_execution_context,
        execution_contexts.create_sync_test_cpp_execution_context,
    )

  def test_returns_sync_context(self):
    context = cpp_execution_contexts.create_sync_test_cpp_execution_context()
    self.assertIsInstance(context, sync_execution_context.SyncExecutionContext)


class SetSyncTestCPPExecutionContextTest(absltest.TestCase):

  def test_has_same_signature(self):
    _assert_signature_equal(
        cpp_execution_contexts.set_sync_test_cpp_execution_context,
        execution_contexts.set_sync_test_cpp_execution_context,
    )


class SecureModularSumTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      ('one_client_not_divisible', [1], 1,
       computation_types.FederatedType(np.int32, placements.CLIENTS)),
      ('two_clients_none_divisible', [1, 2], 3,
       computation_types.FederatedType(np.int32, placements.CLIENTS)),
      ('three_clients_one_divisible', [1, 2, 10], 3,
       computation_types.FederatedType(np.int32, placements.CLIENTS)),
      ('all_clients_divisible_by_modulus', [x * 5 for x in range(5)], 0,
       computation_types.FederatedType(np.int32, placements.CLIENTS)),
      ('nonscalar_struct_arg', [([1, 2], 3), ([4, 5], 6)],
       (np.array([0, 2], dtype=np.int32), 4),
       computation_types.FederatedType(((np.int32, [2]), np.int32), placements.CLIENTS)),
  )
  # pyformat: enable
  def test_executes_computation_with_modular_secure_sum_integer_modulus(
      self, arg, expected_result, tff_type
  ):
    cpp_execution_contexts.set_sync_test_cpp_execution_context()

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

  # # pyformat: disable
  @parameterized.named_parameters(
      (
          'one_client_not_divisible',
          [1],
          1,
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
      (
          'two_clients_none_divisible',
          [1, 2],
          3,
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
  )
  # pyformat: enable
  async def test_async_executes_computation_with_modular_secure_sum_integer_modulus(
      self, arg, expected_result, tff_type
  ):
    cpp_execution_contexts.set_async_test_cpp_execution_context()

    modulus = 5

    @federated_computation.federated_computation(tff_type)
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    self.assertEqual(expected_result, await modular_sum_by_five(arg))

  @parameterized.named_parameters(
      (
          'one_client_not_divisible',
          [[1, 2]],
          [1, 2],
          computation_types.FederatedType(
              [np.int32, np.int32], placements.CLIENTS
          ),
      ),
      (
          'two_clients_none_divisible',
          [[1, 2], [3, 4]],
          [4, 6],
          computation_types.FederatedType(
              [np.int32, np.int32], placements.CLIENTS
          ),
      ),
      (
          'three_clients_one_divisible',
          [[1, 2], [3, 4], [10, 14]],
          [4, 6],
          computation_types.FederatedType(
              [np.int32, np.int32], placements.CLIENTS
          ),
      ),
      (
          'two_clients_one_partially_divisible',
          [[1, 2], [3, 4], [10, 15]],
          [4, 0],
          computation_types.FederatedType(
              [np.int32, np.int32], placements.CLIENTS
          ),
      ),
  )
  def test_executes_computation_with_modular_secure_sum_struct_modulus(
      self, arg, expected_result, tff_type
  ):
    cpp_execution_contexts.set_sync_test_cpp_execution_context()
    modulus = [5, 7]

    @federated_computation.federated_computation(tff_type)
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    self.assertEqual(expected_result, modular_sum_by_five(arg))


class SecureSumBitwidthTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('one_client', [1]),
      ('three_clients', [1, 2, 10]),
      ('five_clients', [x * 5 for x in range(5)]),
  )
  def test_executes_computation_with_bitwidth_secure_sum_large_bitwidth(
      self, arg
  ):
    cpp_execution_contexts.set_sync_test_cpp_execution_context()
    bitwidth = 32
    expected_result = sum(arg)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    self.assertEqual(expected_result, sum_with_bitwidth(arg))

  async def test_async_executes_computation_with_bitwidth_secure_sum_large_bitwidth(
      self,
  ):
    arg = [1, 2, 10]
    cpp_execution_contexts.set_async_test_cpp_execution_context()
    bitwidth = 32
    expected_result = sum(arg)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    self.assertEqual(expected_result, await sum_with_bitwidth(arg))

  # pyformat: disable
  @parameterized.named_parameters(
      ('two_clients_scalar_tensors', [[1, 2], [3, 4]], [4, 6],
       computation_types.FederatedType([np.int32, np.int32], placements.CLIENTS)),
      ('two_clients_nonscalar_tensors',
       [[np.ones(shape=[10], dtype=np.int32), 2],
        [np.ones(shape=[10], dtype=np.int32), 4]],
       [2 * np.ones(shape=[10], dtype=np.int32), 6],
       computation_types.FederatedType([
           computation_types.TensorType(dtype=np.int32, shape=[10]), np.int32], placements.CLIENTS)
      ),
  )
  # pyformat: enable
  def test_executes_computation_with_argument_structure(
      self, arg, expected_result, tff_type
  ):
    cpp_execution_contexts.set_sync_test_cpp_execution_context()

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


class SecureSumMaxValueTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def test_raises_with_arguments_over_max_value(self):
    cpp_execution_contexts.set_sync_test_cpp_execution_context()

    max_value = 1

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    with self.assertRaisesRegex(
        Exception, 'client value larger than maximum specified for secure sum'
    ):
      secure_sum([2, 4])

  @parameterized.named_parameters(
      ('one_client', [1]),
      ('three_clients', [1, 2, 10]),
      ('five_clients', [x * 5 for x in range(5)]),
  )
  def test_executes_computation_with_secure_sum_under_max_values(self, arg):
    cpp_execution_contexts.set_sync_test_cpp_execution_context()

    max_value = 30

    expected_result = sum(arg)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    self.assertEqual(expected_result, secure_sum(arg))

  async def test_async_executes_computation_with_secure_sum_under_max_values(
      self,
  ):
    cpp_execution_contexts.set_async_test_cpp_execution_context()
    arg = [x * 5 for x in range(5)]
    max_value = 30

    expected_result = sum(arg)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    self.assertEqual(expected_result, await secure_sum(arg))

  # pyformat: disable
  @parameterized.named_parameters(
      ('two_clients_scalar_tensors', [[1, 2], [3, 4]], [4, 6],
       computation_types.FederatedType([np.int32, np.int32], placements.CLIENTS)),
      ('two_clients_nonscalar_tensors',
       [[np.ones(shape=[10], dtype=np.int32), 2],
        [np.ones(shape=[10], dtype=np.int32), 4]],
       [2 * np.ones(shape=[10], dtype=np.int32), 6],
       computation_types.FederatedType([
           computation_types.TensorType(
               dtype=np.int32, shape=[10]), np.int32], placements.CLIENTS)),
  )
  # pyformat: enable
  def test_executes_computation_with_argument_structure(
      self, arg, expected_result, tff_type
  ):
    cpp_execution_contexts.set_sync_test_cpp_execution_context()

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
