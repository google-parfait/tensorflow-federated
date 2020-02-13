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

import asyncio
from typing import Tuple

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
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor

tf.compat.v1.enable_v2_behavior()


def _make_test_executor(
    num_clients=1,
    use_reference_resolving_executor=False,
) -> federating_executor.FederatingExecutor:
  bottom_ex = eager_tf_executor.EagerTFExecutor()
  if use_reference_resolving_executor:
    bottom_ex = reference_resolving_executor.ReferenceResolvingExecutor(
        bottom_ex)
  return federating_executor.FederatingExecutor({
      placements.SERVER: bottom_ex,
      placements.CLIENTS: [bottom_ex for _ in range(num_clients)],
      None: bottom_ex
  })


Runtime = Tuple[asyncio.AbstractEventLoop,
                federating_executor.FederatingExecutor]


def _make_test_runtime(num_clients=1,
                       use_reference_resolving_executor=False) -> Runtime:
  """Creates a test runtime consisting of an event loop and test executor."""
  loop = asyncio.get_event_loop()
  ex = _make_test_executor(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  return loop, ex


def _run_comp_with_runtime(comp, runtime: Runtime):
  """Runs a computation using the provided runtime."""
  loop, ex = runtime

  async def call_value():
    return await ex.create_call(await ex.create_value(comp))

  return loop.run_until_complete(call_value())


def _run_test_comp(comp, num_clients=1, use_reference_resolving_executor=False):
  """Runs a computation (unapplied TFF function) using a test runtime."""
  runtime = _make_test_runtime(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  return _run_comp_with_runtime(comp, runtime)


def _run_test_comp_produces_federated_value(
    test_instance,
    comp,
    num_clients=1,
    use_reference_resolving_executor=False,
):
  """Runs a computation (unapplied TFF function) using a test runtime.

  This is similar to _run_test_comp, but the result is asserted to be a
  FederatedValue and computed.

  Args:
    test_instance: A class with the standard unit testing assertions.
    comp: The computation to run.
    num_clients: The number of clients to use when computing `comp`.
    use_reference_resolving_executor: Whether or not to include an executor
      to resolve references.

  Returns:
    The result of running the computation.
  """
  loop, ex = _make_test_runtime(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  val = _run_comp_with_runtime(comp, (loop, ex))
  test_instance.assertIsInstance(val,
                                 federating_executor.FederatingExecutorValue)
  return loop.run_until_complete(val.compute())


def _produce_test_value(
    value,
    type_spec=None,
    num_clients=1,
    use_reference_resolving_executor=False,
):
  """Produces a TFF value using a test runtime."""
  loop, ex = _make_test_runtime(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  return loop.run_until_complete(ex.create_value(value, type_spec=type_spec))


class FederatingExecutorTest(parameterized.TestCase):

  def test_executor_create_value_with_valid_intrinsic_def(self):
    val = _produce_test_value(
        intrinsic_defs.FEDERATED_APPLY,
        computation_types.FunctionType(
            [type_factory.unary_op(tf.int32),
             type_factory.at_server(tf.int32)],
            type_factory.at_server(tf.int32)))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(
        str(val.type_signature),
        '(<(int32 -> int32),int32@SERVER> -> int32@SERVER)')
    self.assertIs(val.internal_representation, intrinsic_defs.FEDERATED_APPLY)

  def test_executor_create_value_with_invalid_intrinsic_def(self):
    with self.assertRaises(TypeError):
      _produce_test_value(intrinsic_defs.FEDERATED_APPLY, tf.bool)

  def test_executor_create_value_with_intrinsic_as_pb_computation(self):
    val = _produce_test_value(
        pb.Computation(
            intrinsic=pb.Intrinsic(uri='generic_zero'),
            type=type_serialization.serialize_type(tf.int32)))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIs(val.internal_representation, intrinsic_defs.GENERIC_ZERO)

  def test_executor_create_value_with_unbound_reference(self):
    with self.assertRaises(ValueError):
      _produce_test_value(
          pb.Computation(
              reference=pb.Reference(name='a'),
              type=type_serialization.serialize_type(tf.int32)))

  def test_executor_create_value_with_server_int(self):
    val = _produce_test_value(10, type_spec=type_factory.at_server(tf.int32))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    self.assertIsInstance(val.internal_representation[0],
                          eager_tf_executor.EagerValue)
    self.assertEqual(
        val.internal_representation[0].internal_representation.numpy(), 10)

  def test_executor_create_value_with_client_int(self):
    val = _produce_test_value([10, 20, 30],
                              type_spec=type_factory.at_clients(tf.int32),
                              num_clients=3)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), '{int32}@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
    self.assertCountEqual([
        v.internal_representation.numpy() for v in val.internal_representation
    ], [10, 20, 30])

  def test_executor_create_value_with_all_equal_client_int(self):
    val = _produce_test_value(
        10,
        type_spec=type_factory.at_clients(tf.int32, all_equal=True),
        num_clients=3)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 10)

  def test_executor_create_value_with_unplaced_int(self):
    val = _produce_test_value(10, type_spec=tf.int32)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIsInstance(val.internal_representation,
                          eager_tf_executor.EagerValue)
    self.assertEqual(
        val.internal_representation.internal_representation.numpy(), 10)

  def test_executor_create_value_with_placement_literal(self):
    val = _produce_test_value(placements.SERVER)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'placement')
    self.assertIs(val.internal_representation, placements.SERVER)

  def test_executor_create_value_with_no_arg_tf_in_fed_comp(self):
    @computations.federated_computation
    def comp():
      return 10

    val = _run_test_comp(comp)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIsInstance(val.internal_representation,
                          eager_tf_executor.EagerValue)
    self.assertEqual(
        val.internal_representation.internal_representation.numpy(), 10)

  def test_executor_create_value_with_one_arg_tf_in_fed_comp(self):
    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return add_one(add_one(add_one(10)))

    val = _run_test_comp(comp)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIsInstance(val.internal_representation,
                          eager_tf_executor.EagerValue)
    self.assertEqual(
        val.internal_representation.internal_representation.numpy(), 13)

  def test_federated_value_at_server(self):
    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.SERVER)

    val = _run_test_comp(comp)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    self.assertIsInstance(val.internal_representation[0],
                          eager_tf_executor.EagerValue)
    self.assertEqual(
        val.internal_representation[0].internal_representation.numpy(), 10)

  def test_federated_value_at_client_with_zero_clients_raises_error(self):
    self.skipTest('b/145936344')
    @computations.federated_computation
    def comp():
      return intrinsics.federated_broadcast(
          intrinsics.federated_value(10, placements.SERVER))

    val = _run_test_comp(comp, num_clients=0)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    with self.assertRaisesRegex(RuntimeError, '0 clients'):
      val.compute()

  def test_federated_value_at_server_with_tuple(self):
    @computations.federated_computation
    def comp():
      return intrinsics.federated_value([10, 10], placements.SERVER)

    val = _run_test_comp(comp)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), '<int32,int32>@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    self.assertIsInstance(val.internal_representation[0],
                          eager_tf_executor.EagerValue)
    inner_eager_value = val.internal_representation[0]
    self.assertLen(inner_eager_value.internal_representation, 2)
    self.assertEqual(inner_eager_value.internal_representation[0].numpy(), 10)
    self.assertEqual(inner_eager_value.internal_representation[1].numpy(), 10)

  def test_federated_value_at_clients(self):
    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.CLIENTS)

    val = _run_test_comp(comp, num_clients=3)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 10)

  def test_federated_eval_at_clients_simple_number(self):

    @computations.federated_computation
    def comp():
      return_five = computations.tf_computation(lambda: 5)
      return intrinsics.federated_eval(return_five, placements.CLIENTS)

    num_clients = 3
    val = _run_test_comp(comp, num_clients=num_clients)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), '{int32}@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, num_clients)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 5)

  def test_federated_eval_at_server_simple_number(self):

    @computations.federated_computation
    def comp():
      return_five = computations.tf_computation(lambda: 5)
      return intrinsics.federated_eval(return_five, placements.SERVER)

    num_clients = 3
    val = _run_test_comp(comp, num_clients=num_clients)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    v = val.internal_representation[0]
    self.assertIsInstance(v, eager_tf_executor.EagerValue)
    self.assertEqual(v.internal_representation.numpy(), 5)

  def test_federated_eval_at_clients_random(self):

    @computations.federated_computation
    def comp():
      rand = computations.tf_computation(lambda: tf.random.normal([]))
      return intrinsics.federated_eval(rand, placements.CLIENTS)

    num_clients = 3
    val = _run_test_comp(comp, num_clients=num_clients)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), '{float32}@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, num_clients)
    previous_values = set()
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
      number = v.internal_representation.numpy()
      if number in previous_values:
        raise Exception('Multiple clients returned same random number')
      previous_values.add(number)

  def test_federated_map_at_server(self):
    loop, ex = _make_test_runtime()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return intrinsics.federated_map(
          add_one, intrinsics.federated_value(10, placements.SERVER))

    val = _run_comp_with_runtime(comp, (loop, ex))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    v = val.internal_representation[0]
    self.assertIsInstance(v, eager_tf_executor.EagerValue)
    self.assertEqual(v.internal_representation.numpy(), 11)
    result = loop.run_until_complete(v.compute())
    self.assertEqual(result.numpy(), 11)

  def test_federated_map(self):
    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return intrinsics.federated_map(
          add_one, intrinsics.federated_value(10, placements.CLIENTS))

    val = _run_test_comp(comp, num_clients=3)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), '{int32}@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 11)

  def test_federated_broadcast(self):
    @computations.federated_computation
    def comp():
      return intrinsics.federated_broadcast(
          intrinsics.federated_value(10, placements.SERVER))

    val = _run_test_comp(comp, num_clients=3)
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 10)

  def test_federated_zip(self):
    loop, ex = _make_test_runtime(num_clients=3)

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
              building_blocks.Call(
                  building_blocks.ComputationBuildingBlock.from_proto(
                      computation_impl.ComputationImpl.get_proto(ten)))
          ] * 2))
      val = loop.run_until_complete(
          ex.create_value(comp.proto, comp.type_signature))
      self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
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
    @computations.tf_computation(tf.int32, tf.int32)
    def add_numbers(x, y):
      return x + y

    @computations.federated_computation
    def comp():
      return intrinsics.federated_reduce(
          intrinsics.federated_value(10, placements.CLIENTS), 0, add_numbers)

    result = _run_test_comp_produces_federated_value(self, comp, num_clients=3)
    self.assertEqual(result.numpy(), 30)

  def test_federated_aggregate_with_simple_integer_sum(self):
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

    result = _run_test_comp_produces_federated_value(self, comp, num_clients=3)
    self.assertEqual(result.numpy(), 31)

  def test_federated_sum_with_integers(self):
    @computations.federated_computation
    def comp():
      x = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_sum(x)

    result = _run_test_comp_produces_federated_value(self, comp, num_clients=3)
    self.assertEqual(result.numpy(), 30)

  def test_federated_mean_with_floats(self):
    loop, ex = _make_test_runtime(num_clients=4)

    v1 = loop.run_until_complete(
        ex.create_value([1.0, 2.0, 3.0, 4.0],
                        type_factory.at_clients(tf.float32)))
    self.assertEqual(str(v1.type_signature), '{float32}@CLIENTS')

    v2 = loop.run_until_complete(
        ex.create_value(
            intrinsic_defs.FEDERATED_MEAN,
            computation_types.FunctionType(
                type_factory.at_clients(tf.float32),
                type_factory.at_server(tf.float32))))
    self.assertEqual(
        str(v2.type_signature), '({float32}@CLIENTS -> float32@SERVER)')

    v3 = loop.run_until_complete(ex.create_call(v2, v1))
    self.assertEqual(str(v3.type_signature), 'float32@SERVER')

    result = loop.run_until_complete(v3.compute())
    self.assertEqual(result.numpy(), 2.5)

  def test_federated_weighted_mean_with_floats(self):
    loop, ex = _make_test_runtime(
        num_clients=4, use_reference_resolving_executor=True)

    v1 = loop.run_until_complete(
        ex.create_value([1.0, 2.0, 3.0, 4.0],
                        type_factory.at_clients(tf.float32)))
    self.assertEqual(str(v1.type_signature), '{float32}@CLIENTS')

    v2 = loop.run_until_complete(
        ex.create_value([5.0, 10.0, 3.0, 2.0],
                        type_factory.at_clients(tf.float32)))
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
                type_factory.at_clients(tf.float32),
                type_factory.at_clients(tf.float32)
            ], type_factory.at_server(tf.float32))))
    self.assertEqual(
        str(v4.type_signature),
        '(<{float32}@CLIENTS,{float32}@CLIENTS> -> float32@SERVER)')

    v5 = loop.run_until_complete(ex.create_call(v4, v3))
    self.assertEqual(str(v5.type_signature), 'float32@SERVER')

    result = loop.run_until_complete(v5.compute())
    self.assertAlmostEqual(result.numpy(), 2.1, places=3)

  def test_runs_tf(self):
    executor_test_utils.test_runs_tf(self, _make_test_executor(1))

  def test_runs_tf_with_reference_resolving_executor(self):
    executor_test_utils.test_runs_tf(
        self, _make_test_executor(1, use_reference_resolving_executor=True))

  @parameterized.named_parameters(
      ('tuple', (1, 2, 3, 4),),
      ('set', set([1, 2, 3, 4]),),
      ('frozenset', frozenset([1, 2, 3, 4]),),
  )
  def test_with_federated_value_as_a_non_py_list(self, val):
    loop, ex = _make_test_runtime(num_clients=4)
    v = loop.run_until_complete(
        ex.create_value(val, type_factory.at_clients(tf.int32)))
    self.assertEqual(str(v.type_signature), '{int32}@CLIENTS')
    result = tf.nest.map_structure(lambda x: x.numpy(),
                                   loop.run_until_complete(v.compute()))
    self.assertCountEqual(result, [1, 2, 3, 4])

  def test_create_selection_by_index_anonymous_tuple_backed(self):
    loop = asyncio.get_event_loop()
    ex = _make_test_executor(num_clients=4)

    v1 = loop.run_until_complete(
        ex.create_value([1.0, 2.0, 3.0, 4.0],
                        type_factory.at_clients(tf.float32)))
    self.assertEqual(str(v1.type_signature), '{float32}@CLIENTS')

    v2 = loop.run_until_complete(
        ex.create_value([5.0, 10.0, 3.0, 2.0],
                        type_factory.at_clients(tf.float32)))
    self.assertEqual(str(v2.type_signature), '{float32}@CLIENTS')

    v3 = loop.run_until_complete(
        ex.create_tuple(
            anonymous_tuple.AnonymousTuple([(None, v1), (None, v2)])))
    self.assertEqual(
        str(v3.type_signature), '<{float32}@CLIENTS,{float32}@CLIENTS>')

    v4 = loop.run_until_complete(ex.create_selection(v3, index=0))
    self.assertEqual(str(v4.type_signature), '{float32}@CLIENTS')
    result = tf.nest.map_structure(lambda x: x.numpy(),
                                   loop.run_until_complete(v4.compute()))
    self.assertCountEqual(result, [1, 2, 3, 4])

  def test_create_selection_by_name_anonymous_tuple_backed(self):
    loop, ex = _make_test_runtime(num_clients=4)

    v1 = loop.run_until_complete(
        ex.create_value([1.0, 2.0, 3.0, 4.0],
                        type_factory.at_clients(tf.float32)))
    self.assertEqual(str(v1.type_signature), '{float32}@CLIENTS')

    v2 = loop.run_until_complete(
        ex.create_value([5.0, 10.0, 3.0, 2.0],
                        type_factory.at_clients(tf.float32)))
    self.assertEqual(str(v2.type_signature), '{float32}@CLIENTS')

    v3 = loop.run_until_complete(
        ex.create_tuple(anonymous_tuple.AnonymousTuple([('a', v1), ('b', v2)])))
    self.assertEqual(
        str(v3.type_signature), '<a={float32}@CLIENTS,b={float32}@CLIENTS>')

    v4 = loop.run_until_complete(ex.create_selection(v3, name='b'))
    self.assertEqual(str(v4.type_signature), '{float32}@CLIENTS')
    result = tf.nest.map_structure(lambda x: x.numpy(),
                                   loop.run_until_complete(v4.compute()))
    self.assertCountEqual(result, [5, 10, 3, 2])

  def test_create_selection_by_index_eager_tf_executor_backed(self):
    loop, ex = _make_test_runtime()

    @computations.tf_computation()
    def comp():
      return (1, 2)

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    v1 = loop.run_until_complete(ex.create_call(val, None))
    self.assertEqual(str(v1.type_signature), '<int32,int32>')
    selected = loop.run_until_complete(ex.create_selection(v1, index=0))
    self.assertEqual(str(selected.type_signature), 'int32')
    result = loop.run_until_complete(selected.compute())
    self.assertEqual(result, 1)

  def test_create_selection_by_index_reference_resolving_executor_backed(self):
    loop, ex = _make_test_runtime(use_reference_resolving_executor=True)

    @computations.tf_computation()
    def comp():
      return (1, 2)

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    v1 = loop.run_until_complete(ex.create_call(val, None))
    self.assertEqual(str(v1.type_signature), '<int32,int32>')
    selected = loop.run_until_complete(ex.create_selection(v1, index=0))
    self.assertEqual(str(selected.type_signature), 'int32')
    result = loop.run_until_complete(selected.compute())
    self.assertEqual(result, 1)

  def test_create_selection_by_name_eager_tf_executor_backed(self):
    loop, ex = _make_test_runtime()

    @computations.tf_computation()
    def comp():
      return anonymous_tuple.AnonymousTuple([('a', 1), ('b', 2)])

    val = loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    v1 = loop.run_until_complete(ex.create_call(val, None))
    self.assertEqual(str(v1.type_signature), '<a=int32,b=int32>')
    selected = loop.run_until_complete(ex.create_selection(v1, name='b'))
    self.assertEqual(str(selected.type_signature), 'int32')
    result = loop.run_until_complete(selected.compute())
    self.assertEqual(result, 2)

  def test_federated_collect(self):
    loop, ex = _make_test_runtime(num_clients=3)

    @computations.federated_computation
    def comp():
      x = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_collect(x)

    val = _run_comp_with_runtime(comp, (loop, ex))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    result = loop.run_until_complete(val.compute())
    self.assertEqual([x.numpy() for x in result], [10, 10, 10])

    new_ex = _make_test_executor(5)
    val = _run_comp_with_runtime(comp, (loop, new_ex))
    self.assertIsInstance(val, federating_executor.FederatingExecutorValue)
    result = loop.run_until_complete(val.compute())
    self.assertEqual([x.numpy() for x in result], [10, 10, 10, 10, 10])

  def test_federated_collect_with_map_call(self):
    @computations.tf_computation()
    def make_dataset():
      return tf.data.Dataset.range(5)

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def foo(x):
      return x.reduce(tf.constant(0, dtype=tf.int64), lambda a, b: a + b)

    @computations.federated_computation()
    def bar():
      x = intrinsics.federated_value(make_dataset(), placements.CLIENTS)
      return intrinsics.federated_map(
          foo, intrinsics.federated_collect(intrinsics.federated_map(foo, x)))

    result = _run_test_comp_produces_federated_value(self, bar, num_clients=5)
    self.assertEqual(result.numpy(), 50)


if __name__ == '__main__':
  absltest.main()
