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

import asyncio
import collections

from absl.testing import absltest
from absl.testing import parameterized
import grpc
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types

_CLIENTS_INT = computation_types.at_clients(tf.int32)
_CLIENTS_INT_LIST = computation_types.at_clients([tf.int32, tf.int32])


class SecureModularSumTest(parameterized.TestCase, tf.test.TestCase):

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
          computation_types.at_clients(((tf.int32, [2]), tf.int32)),
      ),
  )
  def test_executes_computation_with_modular_secure_sum_integer_modulus(
      self, arg, expected_result, tff_type
  ):
    modulus = 5

    @federated_computation.federated_computation(tff_type)
    def modular_sum_by_five(arg):
      return intrinsics.federated_secure_modular_sum(arg, modulus)

    # assertAllEqual doesnt handle nested structures well, so we use
    # assertAllClose with no tolerance here.
    self.assertAllClose(
        expected_result, modular_sum_by_five(arg), atol=0.0, rtol=0.0
    )

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


class SecureSumBitwidthTest(tf.test.TestCase, parameterized.TestCase):

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
        computation_types.at_clients(tf.int32)
    )
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    self.assertEqual(expected_result, sum_with_bitwidth(arg))

  @parameterized.named_parameters(
      (
          'two_clients_scalar_tensors',
          [[1, 2], [3, 4]],
          [4, 6],
          computation_types.at_clients([tf.int32, tf.int32]),
      ),
      (
          'two_clients_nonscalar_tensors',
          [
              [tf.ones(shape=[10], dtype=tf.int32), 2],
              [tf.ones(shape=[10], dtype=tf.int32), 4],
          ],
          [2 * tf.ones(shape=[10], dtype=tf.int32).numpy(), 6],
          computation_types.at_clients([
              computation_types.TensorType(dtype=tf.int32, shape=[10]),
              tf.int32,
          ]),
      ),
  )
  def test_executes_computation_with_argument_structure(
      self, arg, expected_result, tff_type
  ):
    bitwidth = 32

    @federated_computation.federated_computation(tff_type)
    def sum_with_bitwidth(arg):
      return intrinsics.federated_secure_sum_bitwidth(arg, bitwidth)

    self.assertAllClose(expected_result, sum_with_bitwidth(arg))


class SecureSumMaxValueTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  def test_raises_with_arguments_over_max_value(self):
    max_value = 1

    @federated_computation.federated_computation(
        computation_types.at_clients(tf.int32)
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
        computation_types.at_clients(tf.int32)
    )
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    self.assertEqual(expected_result, secure_sum(arg))

  @parameterized.named_parameters(
      (
          'two_clients_scalar_tensors',
          [[1, 2], [3, 4]],
          [4, 6],
          computation_types.at_clients([tf.int32, tf.int32]),
      ),
      (
          'two_clients_nonscalar_tensors',
          [
              [tf.ones(shape=[10], dtype=tf.int32), 2],
              [tf.ones(shape=[10], dtype=tf.int32), 4],
          ],
          [2 * tf.ones(shape=[10], dtype=tf.int32).numpy(), 6],
          computation_types.at_clients([
              computation_types.TensorType(dtype=tf.int32, shape=[10]),
              tf.int32,
          ]),
      ),
  )
  def test_executes_computation_with_argument_structure(
      self, arg, expected_result, tff_type
  ):
    max_value = 100

    @federated_computation.federated_computation(tff_type)
    def secure_sum(arg):
      return intrinsics.federated_secure_sum(arg, max_value)

    self.assertAllClose(expected_result, secure_sum(arg))


class DistributedExecutionContextIntegrationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('async_context_on_server', True, False, False),
      ('sync_context_on_server', True, False, True),
      ('async_context_on_client', False, True, True),
      ('sync_context_on_client', False, True, True),
      ('async_context_on_server_client', True, True, False),
      ('sync_context_on_server_client', True, True, True),
  )
  def test_runs_tensorflow_on_dtensor_on_server(
      self, dtensor_on_server, dtensor_on_client, is_sync_context
  ):
    mesh = tf.experimental.dtensor.create_mesh(
        devices=['CPU:0'], mesh_dims=[('test_dim', 1)]
    )

    def _create_context():
      server_mesh = mesh if dtensor_on_server else None
      client_mesh = mesh if dtensor_on_client else None
      distributed_config = execution_contexts.DistributedConfiguration(
          server_mesh=server_mesh,
          client_mesh=client_mesh,
      )
      if is_sync_context:
        return execution_contexts.create_sync_experimental_distributed_cpp_execution_context(
            distributed_config=distributed_config
        )
      return execution_contexts.create_async_experimental_distributed_cpp_execution_context(
          distributed_config=distributed_config
      )

    @context_stack_test_utils.with_context(_create_context)
    def test_dtensor_stack():
      @tensorflow_computation.tf_computation(
          collections.OrderedDict(x=tf.int32, y=tf.int32)
      )
      def multiply(ordered_dict):
        return ordered_dict['x'] * ordered_dict['y']

      zero = multiply(collections.OrderedDict(x=0, y=1))
      one = multiply(collections.OrderedDict(x=1, y=1))

      if not is_sync_context:
        self.assertTrue(asyncio.iscoroutine(zero))
        self.assertTrue(asyncio.iscoroutine(one))
        zero = asyncio.run(zero)
        one = asyncio.run(one)

      self.assertEqual(zero, 0)
      self.assertEqual(one, 1)

    test_dtensor_stack()

  def test_error_on_distributed_context_creation(self):
    with self.assertRaisesRegex(
        ValueError,
        (
            'Both server side and client side mesh are unspecified'
            ' in distributed configuration.'
        ),
    ):
      execution_contexts.create_sync_experimental_distributed_cpp_execution_context(
          distributed_config=execution_contexts.DistributedConfiguration()
      )


if __name__ == '__main__':
  absltest.main()
