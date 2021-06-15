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

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import federated_strategy
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions


def create_test_executor(
    number_of_clients: int = 3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  factory = federated_strategy.TestFederatedStrategy.factory({
      placements.SERVER:
          create_bottom_stack(),
      placements.CLIENTS: [
          create_bottom_stack() for _ in range(number_of_clients)
      ],
  })
  return federating_executor.FederatingExecutor(factory, create_bottom_stack())


def create_intrinsic_def_federated_secure_sum_bitwidth(value_type,
                                                       bitwidth_type):
  value = intrinsic_defs.FEDERATED_SECURE_SUM
  type_signature = computation_types.FunctionType([
      computation_types.at_clients(value_type),
      bitwidth_type,
  ], computation_types.at_server(value_type))
  return value, type_signature


class FederatingExecutorCreateCallTest(executor_test_utils.AsyncTestCase,
                                       parameterized.TestCase,
                                       test_case.TestCase):

  @parameterized.named_parameters(
      (
          'scalar_sum_less_than_mask',
          [10, 11, 12],
          10,  # larger than sum and individual client values.
          33,  # 33 mod (2**(10+2))
      ),
      (
          'scalar_sum_less_than_extended_mask',
          [10, 11, 12],
          4,  # larger than indivudal client values, but smaller than sum.
          33,  # 33 mod (2**(4+2))
      ),
      (
          'scalar_sum_greater_than_extend_mask',
          [10, 11, 12],
          2,  # smaller than individual client values.
          1,  # 33 mod (2**(2+2))
      ),
      (
          'structured_scalar_sum_less_than_mask',
          [[10, 0], [11, 1], [12, 2]],
          [10, 10],
          structure.from_container([33, 3]),
      ),
      (
          'structued_scalar_sum_greater_than_mask',
          [[10, 0], [11, 1], [12, 2]],
          [3, 3],
          structure.from_container([1, 3]),
      ),
      (
          'named_structured_scalar_sum_less_than_mask',
          [
              collections.OrderedDict(a=10, b=0),
              collections.OrderedDict(a=11, b=1),
              collections.OrderedDict(a=12, b=2),
          ],
          collections.OrderedDict(a=10, b=10),
          structure.from_container(collections.OrderedDict(a=33, b=3)),
      ),
      (
          'named_structued_scalar_sum_greater_than_mask',
          [
              collections.OrderedDict(a=10, b=0),
              collections.OrderedDict(a=11, b=1),
              collections.OrderedDict(a=12, b=2),
          ],
          collections.OrderedDict(a=3, b=3),
          structure.from_container(collections.OrderedDict(a=1, b=3)),
      ),
      (
          'nested_structured_tensor_sum',
          [
              collections.OrderedDict(
                  a=100, b=[tf.constant([0, 1, 2]),
                            tf.constant([10, 70])]),
              collections.OrderedDict(
                  a=1000, b=[tf.constant([1, 2, 3]),
                             tf.constant([20, 80])]),
              collections.OrderedDict(
                  a=10000, b=[tf.constant([2, 3, 4]),
                              tf.constant([30, 90])]),
          ],
          collections.OrderedDict(a=16, b=[4, 5]),
          structure.from_container(
              collections.OrderedDict(
                  a=11100,
                  b=structure.from_container([
                      tf.constant([3, 6, 9]),
                      tf.constant([60, 240 & (2**7 - 1)]),
                  ]))),
      ),
  )
  def test_returns_value_with_intrinsic_def_federated_secure_sum_bitwidth(
      self, client_values, bitwidth, expected_result):
    executor = create_test_executor()
    value_type = computation_types.at_clients(
        type_conversions.infer_type(client_values[0]))
    bitwidth_type = type_conversions.infer_type(bitwidth)
    comp, comp_type = create_intrinsic_def_federated_secure_sum_bitwidth(
        value_type.member, bitwidth_type)

    comp = self.run_sync(executor.create_value(comp, comp_type))
    arg_1 = self.run_sync(executor.create_value(client_values, value_type))
    arg_2 = self.run_sync(executor.create_value(bitwidth, bitwidth_type))
    args = self.run_sync(executor.create_struct([arg_1, arg_2]))
    result = self.run_sync(executor.create_call(comp, args))

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assert_types_identical(result.type_signature, comp_type.result)
    actual_result = self.run_sync(result.compute())
    if isinstance(expected_result, structure.Struct):
      structure.map_structure(self.assertAllEqual, actual_result,
                              expected_result)
    else:
      self.assertEqual(actual_result, expected_result)


if __name__ == '__main__':
  test_case.main()
