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

from collections.abc import Iterable
from typing import Any
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization


def all_isinstance(objs: Iterable[Any], classinfo: type[Any]) -> bool:
  return all(isinstance(x, classinfo) for x in objs)


def create_test_executor(
    number_of_clients: int = 3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  factory = federated_resolving_strategy.FederatedResolvingStrategy.factory({
      placements.SERVER:
          create_bottom_stack(),
      placements.CLIENTS: [
          create_bottom_stack() for _ in range(number_of_clients)
      ],
  })
  return federating_executor.FederatingExecutor(factory, create_bottom_stack())


def get_named_parameters_for_supported_intrinsics() -> list[tuple[str, Any]]:
  # pyformat: disable
  return [
      ('intrinsic_def_federated_aggregate',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_aggregate()),
      ('intrinsic_def_federated_apply',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_apply()),
      ('intrinsic_def_federated_broadcast',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_broadcast()),
      ('intrinsic_def_federated_eval_at_clients',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_eval_at_clients()),
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_eval_at_server()),
      ('intrinsic_def_federated_map',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_map()),
      ('intrinsic_def_federated_map_all_equal',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_map_all_equal()),
      ('intrinsic_def_federated_mean',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_mean()),
      ('intrinsic_def_federated_select',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_select()),
      ('intrinsic_def_federated_sum',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_sum()),
      ('intrinsic_def_federated_value_at_clients',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_value_at_clients()),
      ('intrinsic_def_federated_value_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_value_at_server()),
      ('intrinsic_def_federated_weighted_mean',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_weighted_mean()),
      ('intrinsic_def_federated_zip_at_clients',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_zip_at_clients()),
      ('intrinsic_def_federated_zip_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_zip_at_server()),
  ]
  # pyformat: enable


class FederatingExecutorCreateValueTest(unittest.IsolatedAsyncioTestCase,
                                        parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('placement_literal',
       *executor_test_utils.create_whimsy_placement_literal()),
      ('computation_intrinsic',
       *executor_test_utils.create_whimsy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_whimsy_computation_lambda_empty()),
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_empty()),
      ('federated_type_at_clients',
       *executor_test_utils.create_whimsy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_whimsy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_whimsy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_whimsy_value_unplaced()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  async def test_returns_value_with_value_and_type(self, value, type_signature):
    executor = create_test_executor()

    result = await executor.create_value(value, type_signature)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())

  # pyformat: disable
  @parameterized.named_parameters([
      ('placement_literal',
       *executor_test_utils.create_whimsy_placement_literal()),
      ('computation_intrinsic',
       *executor_test_utils.create_whimsy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_whimsy_computation_lambda_empty()),
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_empty()),
  ])
  # pyformat: enable
  async def test_returns_value_with_value_only(self, value, type_signature):
    executor = create_test_executor()

    result = await executor.create_value(value)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_intrinsic',
       *executor_test_utils.create_whimsy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_whimsy_computation_lambda_empty()),
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_empty()),
  ])
  # pyformat: enable
  async def test_returns_value_with_computation_impl(self, proto,
                                                     type_signature):
    executor = create_test_executor()
    value = computation_impl.ConcreteComputation(
        proto, context_stack_impl.context_stack)

    result = await executor.create_value(value, type_signature)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())

  # pyformat: disable
  @parameterized.named_parameters([
      ('federated_type_at_clients',
       *executor_test_utils.create_whimsy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_whimsy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_whimsy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_whimsy_value_unplaced()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  async def test_raises_type_error_with_value_only(self, value, type_signature):
    del type_signature  # Unused.
    executor = create_test_executor()

    with self.assertRaises(TypeError):
      await executor.create_value(value)

  # pyformat: disable
  @parameterized.named_parameters([
      ('placement_literal',
       *executor_test_utils.create_whimsy_placement_literal()),
      ('computation_call',
       *executor_test_utils.create_whimsy_computation_call()),
      ('computation_intrinsic',
       *executor_test_utils.create_whimsy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_whimsy_computation_lambda_empty()),
      ('computation_selection',
       *executor_test_utils.create_whimsy_computation_selection()),
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_empty()),
      ('computation_tuple',
       *executor_test_utils.create_whimsy_computation_tuple()),
      ('federated_type_at_clients',
       *executor_test_utils.create_whimsy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_whimsy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_whimsy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_whimsy_value_unplaced()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  async def test_raises_type_error_with_value_and_bad_type(
      self, value, type_signature):
    del type_signature  # Unused.
    executor = create_test_executor()
    bad_type_signature = computation_types.TensorType(tf.string)

    with self.assertRaises(TypeError):
      await executor.create_value(value, bad_type_signature)

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_call',
       *executor_test_utils.create_whimsy_computation_call()),
      ('computation_placement',
       *executor_test_utils.create_whimsy_computation_placement()),
      ('computation_reference',
       *executor_test_utils.create_whimsy_computation_reference()),
      ('computation_selection',
       *executor_test_utils.create_whimsy_computation_selection()),
      ('computation_tuple',
       *executor_test_utils.create_whimsy_computation_tuple()),
  ])
  # pyformat: enable
  async def test_raises_value_error_with_value(self, value, type_signature):
    executor = create_test_executor()

    with self.assertRaises(ValueError):
      await executor.create_value(value, type_signature)

  async def test_raises_value_error_with_unrecognized_computation_intrinsic(
      self):
    executor = create_test_executor()
    type_signature = computation_types.TensorType(tf.int32)
    # A `ValueError` will be raised because `create_value` can not recognize the
    # following intrinsic, because it has not been added to the intrinsic
    # registry.
    type_signature = computation_types.TensorType(tf.int32)
    value = pb.Computation(
        type=type_serialization.serialize_type(type_signature),
        intrinsic=pb.Intrinsic(uri='unregistered_intrinsic'))

    with self.assertRaises(ValueError):
      await executor.create_value(value, type_signature)

  async def test_raises_value_error_with_unrecognized_computation_selection(
      self):
    executor = create_test_executor()
    source, _ = executor_test_utils.create_whimsy_computation_tuple()
    type_signature = computation_types.StructType([])
    # A `ValueError` will be raised because `create_value` can not handle the
    # following `pb.Selection`, because does not set either a name or an index
    # field.
    value = pb.Computation(
        type=type_serialization.serialize_type(type_signature),
        selection=pb.Selection(source=source))

    with self.assertRaises(ValueError):
      await executor.create_value(value, type_signature)

  # pyformat: disable
  @parameterized.named_parameters([
      ('intrinsic_def_federated_aggregate',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_aggregate()),
      ('intrinsic_def_federated_apply',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_apply()),
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_eval_at_server()),
      ('intrinsic_def_federated_mean',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_mean()),
      ('intrinsic_def_federated_sum',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_sum()),
      ('intrinsic_def_federated_value_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_value_at_server()),
      ('intrinsic_def_federated_weighted_mean',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_weighted_mean()),
      ('intrinsic_def_federated_zip_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_zip_at_server()),
      ('federated_type_at_server',
       *executor_test_utils.create_whimsy_value_at_server()),
  ])
  # pyformat: enable
  async def test_raises_value_error_with_no_target_executor_server(
      self, value, type_signature):
    factory = federated_resolving_strategy.FederatedResolvingStrategy.factory({
        placements.CLIENTS: eager_tf_executor.EagerTFExecutor(),
    })
    executor = federating_executor.FederatingExecutor(
        factory, eager_tf_executor.EagerTFExecutor())
    value, type_signature = executor_test_utils.create_whimsy_value_at_server()

    with self.assertRaises(ValueError):
      await executor.create_value(value, type_signature)

  async def test_raises_value_error_with_unexpected_federated_type_at_clients(
      self):
    executor = create_test_executor()
    value = [10, 20]
    type_signature = computation_types.at_clients(tf.int32)

    with self.assertRaises(executors_errors.CardinalityError):
      await executor.create_value(value, type_signature)

  async def test_raises_type_error_with_unexpected_federated_type_at_clients_all_equal(
      self):
    executor = create_test_executor()
    value = [10] * 3
    type_signature = computation_types.at_clients(tf.int32, all_equal=True)

    with self.assertRaises(TypeError):
      await executor.create_value(value, type_signature)


class FederatingExecutorCreateCallTest(unittest.IsolatedAsyncioTestCase,
                                       parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('intrinsic_def_federated_aggregate',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_aggregate(),
       [executor_test_utils.create_whimsy_value_at_clients(),
        executor_test_utils.create_whimsy_value_unplaced(),
        executor_test_utils.create_whimsy_computation_tensorflow_add(),
        executor_test_utils.create_whimsy_computation_tensorflow_add(),
        executor_test_utils.create_whimsy_computation_tensorflow_identity()],
       43.0),
      ('intrinsic_def_federated_apply',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_apply(),
       [executor_test_utils.create_whimsy_computation_tensorflow_identity(),
        executor_test_utils.create_whimsy_value_at_server()],
       10.0),
      ('intrinsic_def_federated_broadcast',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_broadcast(),
       [executor_test_utils.create_whimsy_value_at_server()],
       10.0),
      ('intrinsic_def_federated_eval_at_clients',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_eval_at_clients(),
       [executor_test_utils.create_whimsy_computation_tensorflow_constant()],
       [10.0] * 3),
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_eval_at_server(),
       [executor_test_utils.create_whimsy_computation_tensorflow_constant()],
       10.0),
      ('intrinsic_def_federated_map',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_map(),
       [executor_test_utils.create_whimsy_computation_tensorflow_identity(),
        executor_test_utils.create_whimsy_value_at_clients()],
       [10.0, 11.0, 12.0]),
      ('intrinsic_def_federated_map_all_equal',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_map_all_equal(),
       [executor_test_utils.create_whimsy_computation_tensorflow_identity(),
        executor_test_utils.create_whimsy_value_at_clients_all_equal()],
       10.0),
      ('intrinsic_def_federated_mean',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_mean(),
       [executor_test_utils.create_whimsy_value_at_clients()],
       11.0),
      ('intrinsic_def_federated_select',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_select(),
       executor_test_utils.create_whimsy_federated_select_args(),
       executor_test_utils.create_whimsy_federated_select_expected_result(),
       ),
      ('intrinsic_def_federated_sum',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_sum(),
       [executor_test_utils.create_whimsy_value_at_clients()],
       33.0),
      ('intrinsic_def_federated_value_at_clients',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_value_at_clients(),
       [executor_test_utils.create_whimsy_value_unplaced()],
       10.0),
      ('intrinsic_def_federated_value_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_value_at_server(),
       [executor_test_utils.create_whimsy_value_unplaced()],
       10.0),
      ('intrinsic_def_federated_weighted_mean',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_weighted_mean(),
       [executor_test_utils.create_whimsy_value_at_clients(),
        executor_test_utils.create_whimsy_value_at_clients()],
       11.060606),
      ('intrinsic_def_federated_zip_at_clients',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_zip_at_clients(),
       [executor_test_utils.create_whimsy_value_at_clients(),
        executor_test_utils.create_whimsy_value_at_clients()],
       [structure.Struct([(None, 10.0), (None, 10.0)]),
        structure.Struct([(None, 11.0), (None, 11.0)]),
        structure.Struct([(None, 12.0), (None, 12.0)])]),
      ('intrinsic_def_federated_zip_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_zip_at_server(),
       [executor_test_utils.create_whimsy_value_at_server(),
        executor_test_utils.create_whimsy_value_at_server()],
       structure.Struct([(None, 10.0), (None, 10.0)])),
      ('computation_intrinsic',
       *executor_test_utils.create_whimsy_computation_intrinsic(),
       [executor_test_utils.create_whimsy_computation_tensorflow_constant()],
       10.0),
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_identity(),
       [executor_test_utils.create_whimsy_value_unplaced()],
       10.0),
  ])
  # pyformat: enable
  async def test_returns_value_with_comp_and_arg(self, comp, comp_type, args,
                                                 expected_result):
    executor = create_test_executor()

    comp = await executor.create_value(comp, comp_type)
    elements = [await executor.create_value(*x) for x in args]
    if len(elements) > 1:
      arg = await executor.create_struct(elements)
    else:
      arg = elements[0]
    result = await executor.create_call(comp, arg)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     comp_type.result.compact_representation())
    actual_result = await result.compute()
    self.assert_maybe_list_equal(actual_result, expected_result)

  def assert_maybe_list_equal(self, actual_result, expected_result):
    if (all_isinstance([actual_result, expected_result], list) or
        all_isinstance([actual_result, expected_result], tf.data.Dataset)):
      for actual_element, expected_element in zip(actual_result,
                                                  expected_result):
        self.assert_maybe_list_equal(actual_element, expected_element)
    else:
      self.assertEqual(actual_result, expected_result)

  async def test_returns_value_with_intrinsic_def_federated_eval_at_clients_and_random(
      self):
    executor = create_test_executor(number_of_clients=3)
    comp, comp_type = executor_test_utils.create_whimsy_intrinsic_def_federated_eval_at_clients(
    )
    arg, arg_type = executor_test_utils.create_whimsy_computation_tensorflow_random(
    )

    comp = await executor.create_value(comp, comp_type)
    arg = await executor.create_value(arg, arg_type)
    result = await executor.create_call(comp, arg)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     comp_type.result.compact_representation())
    actual_result = await result.compute()
    unique_results = set([x.numpy() for x in actual_result])
    if len(actual_result) != len(unique_results):
      self.fail(
          'Expected the result to contain different random numbers, found {}.'
          .format(actual_result))

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_empty()),
  ])
  # pyformat: enable
  async def test_returns_value_with_comp_only(self, comp, comp_type):
    executor = create_test_executor()

    comp = await executor.create_value(comp, comp_type)
    result = await executor.create_call(comp)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     comp_type.result.compact_representation())
    actual_result = await result.compute()
    expected_result = []
    self.assertCountEqual(actual_result, expected_result)

  async def test_raises_type_error_with_unembedded_comp(self):
    executor = create_test_executor()
    comp, _ = executor_test_utils.create_whimsy_computation_tensorflow_identity(
    )
    arg, arg_type = executor_test_utils.create_whimsy_value_unplaced()

    arg = await executor.create_value(arg, arg_type)
    with self.assertRaises(TypeError):
      await executor.create_call(comp, arg)

  async def test_raises_type_error_with_unembedded_arg(self):
    executor = create_test_executor()
    comp, comp_type = executor_test_utils.create_whimsy_computation_tensorflow_identity(
    )
    arg, _ = executor_test_utils.create_whimsy_value_unplaced()

    comp = await executor.create_value(comp, comp_type)
    with self.assertRaises(TypeError):
      await executor.create_call(comp, arg)

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_intrinsic',
       *executor_test_utils.create_whimsy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_whimsy_computation_lambda_identity()),
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_identity()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  async def test_raises_type_error_with_comp_and_bad_arg(self, comp, comp_type):
    executor = create_test_executor()
    bad_arg = 'string'
    bad_arg_type = computation_types.TensorType(tf.string)

    comp = await executor.create_value(comp, comp_type)
    arg = await executor.create_value(bad_arg, bad_arg_type)
    with self.assertRaises(TypeError):
      await executor.create_call(comp, arg)

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_lambda',
       *executor_test_utils.create_whimsy_computation_lambda_empty()),
      ('federated_type_at_clients',
       *executor_test_utils.create_whimsy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_whimsy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_whimsy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_whimsy_value_unplaced()),
  ])
  # pyformat: enable
  async def test_raises_value_error_with_comp(self, comp, comp_type):
    executor = create_test_executor()

    comp = await executor.create_value(comp, comp_type)
    with self.assertRaises(ValueError):
      await executor.create_call(comp)

  async def test_raises_not_implemented_error_with_intrinsic_def_federated_secure_sum_bitwidth(
      self):
    executor = create_test_executor()
    comp, comp_type = executor_test_utils.create_whimsy_intrinsic_def_federated_secure_sum_bitwidth(
    )
    arg_1 = [10, 11, 12]
    arg_1_type = computation_types.at_clients(tf.int32, all_equal=False)
    arg_2 = 10
    arg_2_type = computation_types.TensorType(tf.int32)

    comp = await executor.create_value(comp, comp_type)
    arg_1 = await executor.create_value(arg_1, arg_1_type)
    arg_2 = await executor.create_value(arg_2, arg_2_type)
    args = await executor.create_struct([arg_1, arg_2])
    with self.assertRaises(NotImplementedError):
      await executor.create_call(comp, args)

  async def test_raises_not_implemented_error_with_unimplemented_intrinsic(
      self):
    executor = create_test_executor()
    # `whimsy_intrinsic` definition is needed to allow lookup.
    whimsy_intrinsic = intrinsic_defs.IntrinsicDef(
        'WHIMSY_INTRINSIC', 'whimsy_intrinsic',
        computation_types.AbstractType('T'))
    type_signature = computation_types.TensorType(tf.int32)
    comp = pb.Computation(
        intrinsic=pb.Intrinsic(uri='whimsy_intrinsic'),
        type=type_serialization.serialize_type(type_signature))
    del whimsy_intrinsic

    comp = await executor.create_value(comp)
    with self.assertRaises(NotImplementedError):
      await executor.create_call(comp)


class FederatingExecutorCreateStructTest(unittest.IsolatedAsyncioTestCase,
                                         parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('federated_type_at_clients',
       *executor_test_utils.create_whimsy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_whimsy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_whimsy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_whimsy_value_unplaced()),
  ])
  # pyformat: enable
  async def test_returns_value_with_elements_value(self, value, type_signature):
    executor = create_test_executor()

    element = await executor.create_value(value, type_signature)
    elements = [element] * 3
    type_signature = computation_types.StructType([type_signature] * 3)
    result = await executor.create_struct(elements)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())
    actual_result = await result.compute()
    expected_result = [await element.compute()] * 3
    self.assertCountEqual(actual_result, expected_result)

  async def test_returns_value_with_elements_value_placement_literal(self):
    executor = create_test_executor()
    value, type_signature = executor_test_utils.create_whimsy_placement_literal(
    )

    element = await executor.create_value(value, type_signature)
    elements = [element] * 3
    type_signature = computation_types.StructType([type_signature] * 3)
    result = await executor.create_struct(elements)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())

  # pyformat: disable
  @parameterized.named_parameters([
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_whimsy_intrinsic_def_federated_eval_at_server(),
       *executor_test_utils.create_whimsy_computation_tensorflow_constant()),
      ('computation_intrinsic',
       *executor_test_utils.create_whimsy_computation_intrinsic(),
       *executor_test_utils.create_whimsy_computation_tensorflow_constant()),
  ])
  # pyformat: enable
  async def test_returns_value_with_elements_fn_and_arg(self, fn, fn_type, arg,
                                                        arg_type):
    executor = create_test_executor()

    fn = await executor.create_value(fn, fn_type)
    arg = await executor.create_value(arg, arg_type)
    element = await executor.create_call(fn, arg)
    elements = [element] * 3
    type_signature = computation_types.StructType([fn_type.result] * 3)
    result = await executor.create_struct(elements)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())
    actual_result = await result.compute()
    expected_result = [await element.compute()] * 3
    self.assertCountEqual(actual_result, expected_result)

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_tensorflow',
       *executor_test_utils.create_whimsy_computation_tensorflow_empty()),
  ])
  # pyformat: enable
  async def test_returns_value_with_elements_fn_only(self, fn, fn_type):
    executor = create_test_executor()

    fn = await executor.create_value(fn, fn_type)
    element = await executor.create_call(fn)
    elements = [element] * 3
    type_signature = computation_types.StructType([fn_type.result] * 3)
    result = await executor.create_struct(elements)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())
    actual_result = await result.compute()
    expected_result = [await element.compute()] * 3
    self.assertCountEqual(actual_result, expected_result)

  async def test_raises_type_error_with_unembedded_elements(self):
    executor = create_test_executor()
    element, _ = executor_test_utils.create_whimsy_value_unplaced()

    elements = [element] * 3
    with self.assertRaises(TypeError):
      await executor.create_struct(elements)


class FederatingExecutorCreateSelectionTest(unittest.IsolatedAsyncioTestCase):

  async def test_returns_value_with_source_and_index_computation_tensorflow(
      self):
    executor = create_test_executor()
    source, type_signature = executor_test_utils.create_whimsy_computation_tensorflow_tuple(
    )

    source = await executor.create_value(source, type_signature)
    source = await executor.create_call(source)
    result = await executor.create_selection(source, 0)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.result[0].compact_representation())
    actual_result = await result.compute()
    expected_result = (await source.compute())[0]
    self.assertEqual(actual_result, expected_result)

  async def test_returns_value_with_source_and_index_structure(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_whimsy_value_unplaced()

    element = await executor.create_value(element, element_type)
    elements = [element] * 3
    type_signature = computation_types.StructType([element_type] * 3)
    source = await executor.create_struct(elements)
    result = await executor.create_selection(source, 0)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature[0].compact_representation())
    actual_result = await result.compute()
    expected_result = (await source.compute())[0]
    self.assertEqual(actual_result, expected_result)

  async def test_returns_value_with_source_and_name_computation_tensorflow(
      self):
    executor = create_test_executor()
    source, type_signature = executor_test_utils.create_whimsy_computation_tensorflow_tuple(
    )

    source = await executor.create_value(source, type_signature)
    source = await executor.create_call(source)
    result = await executor.create_selection(source, 0)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.result['a'].compact_representation())
    actual_result = await result.compute()
    expected_result = (await source.compute())['a']
    self.assertEqual(actual_result, expected_result)

  async def test_returns_value_with_source_and_name_structure(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_whimsy_value_unplaced()

    names = ['a', 'b', 'c']
    element = await executor.create_value(element, element_type)
    elements = structure.Struct((n, element) for n in names)
    type_signature = computation_types.StructType(
        (n, element_type) for n in names)
    source = await executor.create_struct(elements)
    result = await executor.create_selection(source, 0)

    self.assertIsInstance(result, executor_value_base.ExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature['a'].compact_representation())
    actual_result = await result.compute()
    expected_result = (await source.compute())['a']
    self.assertEqual(actual_result, expected_result)

  async def test_raises_type_error_with_unembedded_source(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_whimsy_value_unplaced()

    element = await executor.create_value(element, element_type)
    source = [element] * 3
    with self.assertRaises(TypeError):
      await executor.create_selection(source, 0)

  async def test_raises_type_error_with_not_tuple_type(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_whimsy_value_unplaced()

    source = await executor.create_value(element, element_type)
    with self.assertRaises(TypeError):
      await executor.create_selection(source, 0)

  async def test_raises_value_error_with_unrecognized_generic_zero(self):
    executor = create_test_executor()

    value = intrinsic_defs.GENERIC_ZERO
    type_signature = computation_types.StructType(
        [computation_types.TensorType(tf.int32)] * 3)

    source = await executor.create_value(value, type_signature)
    with self.assertRaises(ValueError):
      await executor.create_selection(source, 0)


if __name__ == '__main__':
  absltest.main()
