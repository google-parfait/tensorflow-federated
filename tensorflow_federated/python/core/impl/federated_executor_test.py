# Lint as: python3
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
"""Tests for federated_executor.py."""

import asyncio

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl import federated_executor
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl import type_constructors
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks


def _make_test_executor(num_clients=1, use_lambda_executor=False):
  bottom_ex = eager_executor.EagerExecutor()
  if use_lambda_executor:
    bottom_ex = lambda_executor.LambdaExecutor(bottom_ex)
  return federated_executor.FederatedExecutor({
      placements.SERVER: bottom_ex,
      placements.CLIENTS: [bottom_ex for _ in range(num_clients)],
      None: bottom_ex
  })


class FederatedExecutorTest(parameterized.TestCase):

  def test_executor_create_value_with_valid_intrinsic_def(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()
    val = loop.run_until_complete(
        ex.create_value(
            intrinsic_defs.FEDERATED_APPLY,
            computation_types.FunctionType([
                type_constructors.unary_op(tf.int32),
                type_constructors.at_server(tf.int32)
            ], type_constructors.at_server(tf.int32))))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(
        str(val.type_signature),
        '(<(int32 -> int32),int32@SERVER> -> int32@SERVER)')
    self.assertIs(val.internal_representation, intrinsic_defs.FEDERATED_APPLY)

  def test_executor_create_value_with_invalid_intrinsic_def(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()
    with self.assertRaises(TypeError):
      loop.run_until_complete(
          ex.create_value(intrinsic_defs.FEDERATED_APPLY, tf.bool))

  def test_executor_create_value_with_intrinsic_as_pb_computation(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()
    val = loop.run_until_complete(
        ex.create_value(
            pb.Computation(
                intrinsic=pb.Intrinsic(uri='generic_zero'),
                type=type_serialization.serialize_type(tf.int32))))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIs(val.internal_representation, intrinsic_defs.GENERIC_ZERO)

  def test_executor_create_value_with_unbound_reference(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()
    with self.assertRaises(ValueError):
      loop.run_until_complete(
          ex.create_value(
              pb.Computation(
                  reference=pb.Reference(name='a'),
                  type=type_serialization.serialize_type(tf.int32))))

  def test_executor_create_value_with_server_int(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()
    val = loop.run_until_complete(
        ex.create_value(10, type_constructors.at_server(tf.int32)))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    self.assertIsInstance(val.internal_representation[0],
                          eager_executor.EagerValue)
    self.assertEqual(
        val.internal_representation[0].internal_representation.numpy(), 10)

  def test_executor_create_value_with_client_int(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)
    val = loop.run_until_complete(
        ex.create_value([10, 20, 30], type_constructors.at_clients(tf.int32)))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), '{int32}@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_executor.EagerValue)
    self.assertCountEqual([
        v.internal_representation.numpy() for v in val.internal_representation
    ], [10, 20, 30])

  def test_executor_create_value_with_all_equal_client_int(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)
    val = loop.run_until_complete(
        ex.create_value(10,
                        type_constructors.at_clients(tf.int32, all_equal=True)))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 10)

  def test_executor_create_value_with_unplaced_int(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()
    val = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIsInstance(val.internal_representation,
                          eager_executor.EagerValue)
    self.assertEqual(
        val.internal_representation.internal_representation.numpy(), 10)

  def test_executor_create_value_with_placement_literal(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()
    val = loop.run_until_complete(ex.create_value(placements.SERVER))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'placement')
    self.assertIs(val.internal_representation, placements.SERVER)

  def test_executor_create_value_with_no_arg_tf_in_fed_comp(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()

    @computations.federated_computation
    def comp():
      return 10

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIsInstance(val.internal_representation,
                          eager_executor.EagerValue)
    self.assertEqual(
        val.internal_representation.internal_representation.numpy(), 10)

  def test_executor_create_value_with_one_arg_tf_in_fed_comp(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return add_one(add_one(add_one(10)))

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIsInstance(val.internal_representation,
                          eager_executor.EagerValue)
    self.assertEqual(
        val.internal_representation.internal_representation.numpy(), 13)

  def test_federated_value_at_server(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()

    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.SERVER)

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    self.assertIsInstance(val.internal_representation[0],
                          eager_executor.EagerValue)
    self.assertEqual(
        val.internal_representation[0].internal_representation.numpy(), 10)

  def test_federated_value_at_clients(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)

    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.CLIENTS)

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 10)

  def test_federated_apply(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return intrinsics.federated_apply(
          add_one, intrinsics.federated_value(10, placements.SERVER))

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    v = val.internal_representation[0]
    self.assertIsInstance(v, eager_executor.EagerValue)
    self.assertEqual(v.internal_representation.numpy(), 11)
    result = loop.run_until_complete(v.compute())
    self.assertEqual(result.numpy(), 11)

  def test_federated_map(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return intrinsics.federated_map(
          add_one, intrinsics.federated_value(10, placements.CLIENTS))

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), '{int32}@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 11)

  def test_federated_broadcast(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)

    @computations.federated_computation
    def comp():
      return intrinsics.federated_broadcast(
          intrinsics.federated_value(10, placements.SERVER))

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 10)

  def test_federated_zip(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)

    @computations.federated_computation
    def ten_on_server():
      return intrinsics.federated_value(10, placements.SERVER)

    @computations.federated_computation
    def ten_on_clients():
      return intrinsics.federated_value(10, placements.CLIENTS)

    for ten, type_string, cardinality, expected_result in [
        (ten_on_server, '<int32,int32>@SERVER', 1, '<10,10>'),
        (ten_on_clients, '{<int32,int32>}@CLIENTS', 3, ['<10,10>'] * 3)
    ]:
      comp = building_block_factory.create_zip_two_values(
          building_blocks.Tuple([
              building_blocks.ComputationBuildingBlock.from_proto(
                  computation_impl.ComputationImpl.get_proto(ten))
          ] * 2))
      val = loop.run_until_complete(
          ex.create_value(comp.proto, comp.type_signature))
      self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
      self.assertEqual(str(val.type_signature), type_string)
      self.assertIsInstance(val.internal_representation, list)
      self.assertLen(val.internal_representation, cardinality)
      result = loop.run_until_complete(val.compute())

      def _print(x):
        return str(anonymous_tuple.map_structure(lambda v: v.numpy(), x))

      if isinstance(expected_result, list):
        self.assertCountEqual([_print(x) for x in result], expected_result)
      else:
        self.assertEqual(_print(result), expected_result)

  def test_federated_reduce_with_simple_integer_sum(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)

    @computations.tf_computation(tf.int32, tf.int32)
    def add_numbers(x, y):
      return x + y

    @computations.federated_computation
    def comp():
      return intrinsics.federated_reduce(
          intrinsics.federated_value(10, placements.CLIENTS), 0, add_numbers)

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    result = loop.run_until_complete(val.compute())
    self.assertEqual(result.numpy(), 30)

  def test_federated_aggregate_with_simple_integer_sum(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)

    @computations.tf_computation(tf.int32, tf.int32)
    def add_numbers(x, y):
      return x + y

    @computations.tf_computation(tf.int32)
    def add_one_because_why_not(x):
      return x + 1

    @computations.federated_computation
    def comp():
      x = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_aggregate(x, 0, add_numbers, add_numbers,
                                            add_one_because_why_not)

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    result = loop.run_until_complete(val.compute())
    self.assertEqual(result.numpy(), 31)

  def test_federated_sum_with_integers(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(3)

    @computations.federated_computation
    def comp():
      x = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_sum(x)

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federated_executor.FederatedExecutorValue)
    result = loop.run_until_complete(val.compute())
    self.assertEqual(result.numpy(), 30)

  def test_federated_mean_with_floats(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(4)

    v1 = loop.run_until_complete(
        ex.create_value([1.0, 2.0, 3.0, 4.0],
                        type_constructors.at_clients(tf.float32)))
    self.assertEqual(str(v1.type_signature), '{float32}@CLIENTS')

    v2 = loop.run_until_complete(
        ex.create_value(
            intrinsic_defs.FEDERATED_MEAN,
            computation_types.FunctionType(
                type_constructors.at_clients(tf.float32),
                type_constructors.at_server(tf.float32))))
    self.assertEqual(
        str(v2.type_signature), '({float32}@CLIENTS -> float32@SERVER)')

    v3 = loop.run_until_complete(ex.create_call(v2, v1))
    self.assertEqual(str(v3.type_signature), 'float32@SERVER')

    result = loop.run_until_complete(v3.compute())
    self.assertEqual(result.numpy(), 2.5)

  def test_federated_weighted_mean_with_floats(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(num_clients=4, use_lambda_executor=True)

    v1 = loop.run_until_complete(
        ex.create_value([1.0, 2.0, 3.0, 4.0],
                        type_constructors.at_clients(tf.float32)))
    self.assertEqual(str(v1.type_signature), '{float32}@CLIENTS')

    v2 = loop.run_until_complete(
        ex.create_value([5.0, 10.0, 3.0, 2.0],
                        type_constructors.at_clients(tf.float32)))
    self.assertEqual(str(v2.type_signature), '{float32}@CLIENTS')

    v3 = loop.run_until_complete(
        ex.create_tuple(
            anonymous_tuple.AnonymousTuple([(None, v1), (None, v2)])))
    self.assertEqual(
        str(v3.type_signature), '<{float32}@CLIENTS,{float32}@CLIENTS>')

    v4 = loop.run_until_complete(
        ex.create_value(
            intrinsic_defs.FEDERATED_WEIGHTED_MEAN,
            computation_types.FunctionType([
                type_constructors.at_clients(tf.float32),
                type_constructors.at_clients(tf.float32)
            ], type_constructors.at_server(tf.float32))))
    self.assertEqual(
        str(v4.type_signature),
        '(<{float32}@CLIENTS,{float32}@CLIENTS> -> float32@SERVER)')

    v5 = loop.run_until_complete(ex.create_call(v4, v3))
    self.assertEqual(str(v5.type_signature), 'float32@SERVER')

    result = loop.run_until_complete(v5.compute())
    self.assertAlmostEqual(result.numpy(), 2.1, places=3)

  def test_with_mnist_training_example(self):
    executor_test_utils.test_mnist_training(self, _make_test_executor(1))

  def test_with_mnist_training_example_with_lambda_executor(self):
    executor_test_utils.test_mnist_training(
        self, _make_test_executor(1, use_lambda_executor=True))

  @parameterized.parameters(((1, 2, 3, 4),), (set([1, 2, 3, 4]),),
                            (frozenset([1, 2, 3, 4]),))
  def test_with_federated_value_as_a_non_py_list(self, val):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(num_clients=4)
    v = loop.run_until_complete(
        ex.create_value(val, type_constructors.at_clients(tf.int32)))
    self.assertEqual(str(v.type_signature), '{int32}@CLIENTS')
    result = tf.nest.map_structure(lambda x: x.numpy(),
                                   loop.run_until_complete(v.compute()))
    self.assertCountEqual(result, [1, 2, 3, 4])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
