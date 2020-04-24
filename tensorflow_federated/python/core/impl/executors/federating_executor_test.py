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

from typing import Any, Iterable, List, Tuple, Type

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor


tf.compat.v1.enable_v2_behavior()


def all_isinstance(objs: Iterable[Any], classinfo: Type[Any]) -> bool:
  return all(isinstance(x, classinfo) for x in objs)


def create_test_executor(
    number_of_clients: int = 3) -> federating_executor.FederatingExecutor:

  def create_bottom_stack():
    executor = eager_tf_executor.EagerTFExecutor()
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  return federating_executor.FederatingExecutor({
      placement_literals.SERVER: create_bottom_stack(),
      placement_literals.CLIENTS: [
          create_bottom_stack() for _ in range(number_of_clients)
      ],
      None: create_bottom_stack()
  })


def get_named_parameters_for_supported_intrinsics() -> List[Tuple[str, Any]]:
  # pyformat: disable
  return [
      ('intrinsic_def_federated_aggregate',
       *executor_test_utils.create_dummy_intrinsic_def_federated_aggregate()),
      ('intrinsic_def_federated_apply',
       *executor_test_utils.create_dummy_intrinsic_def_federated_apply()),
      ('intrinsic_def_federated_broadcast',
       *executor_test_utils.create_dummy_intrinsic_def_federated_broadcast()),
      ('intrinsic_def_federated_collect',
       *executor_test_utils.create_dummy_intrinsic_def_federated_collect()),
      ('intrinsic_def_federated_eval_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_clients()),
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_server()),
      ('intrinsic_def_federated_map',
       *executor_test_utils.create_dummy_intrinsic_def_federated_map()),
      ('intrinsic_def_federated_map_all_equal',
       *executor_test_utils.create_dummy_intrinsic_def_federated_map_all_equal()),
      ('intrinsic_def_federated_mean',
       *executor_test_utils.create_dummy_intrinsic_def_federated_mean()),
      ('intrinsic_def_federated_sum',
       *executor_test_utils.create_dummy_intrinsic_def_federated_sum()),
      ('intrinsic_def_federated_reduce',
       *executor_test_utils.create_dummy_intrinsic_def_federated_reduce()),
      ('intrinsic_def_federated_value_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_value_at_clients()),
      ('intrinsic_def_federated_value_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_value_at_server()),
      ('intrinsic_def_federated_weighted_mean',
       *executor_test_utils.create_dummy_intrinsic_def_federated_weighted_mean()),
      ('intrinsic_def_federated_zip_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_zip_at_clients()),
      ('intrinsic_def_federated_zip_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_zip_at_server()),
  ]
  # pyformat: enable


class FederatingExecutorValueConstructionTest(absltest.TestCase):

  def test_raises_type_error_with_value_and_bad_type(self):
    value = eager_tf_executor.EagerValue(10.0, None, tf.float32)
    bad_type_signature = None

    with self.assertRaises(TypeError):
      federating_executor.FederatingExecutorValue(value, bad_type_signature)


class FederatingExecutorValueComputeTest(executor_test_utils.AsyncTestCase):

  def test_returns_value_with_embedded_value(self):
    value = eager_tf_executor.EagerValue(10.0, None, tf.float32)
    type_signature = computation_types.TensorType(tf.float32)
    value = federating_executor.FederatingExecutorValue(value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, 10.0)

  def test_returns_value_with_federated_type_at_clients(self):
    value = [
        eager_tf_executor.EagerValue(10.0, None, tf.float32),
        eager_tf_executor.EagerValue(11.0, None, tf.float32),
        eager_tf_executor.EagerValue(12.0, None, tf.float32),
    ]
    type_signature = type_factory.at_clients(tf.float32)
    value = federating_executor.FederatingExecutorValue(value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, [10.0, 11.0, 12.0])

  def test_returns_value_with_federated_type_at_clients_all_equal(self):
    value = [eager_tf_executor.EagerValue(10.0, None, tf.float32)]
    type_signature = type_factory.at_clients(tf.float32, all_equal=True)
    value = federating_executor.FederatingExecutorValue(value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, 10.0)

  def test_returns_value_with_federated_type_at_server(self):
    value = [eager_tf_executor.EagerValue(10.0, None, tf.float32)]
    type_signature = type_factory.at_server(tf.float32)
    value = federating_executor.FederatingExecutorValue(value, type_signature)

    result = self.run_sync(value.compute())

    self.assertEqual(result, 10.0)

  def test_returns_value_with_anonymous_tuple_value(self):
    element = eager_tf_executor.EagerValue(10.0, None, tf.float32)
    element_type = computation_types.TensorType(tf.float32)
    names = ['a', 'b', 'c']
    value = anonymous_tuple.AnonymousTuple((n, element) for n in names)
    type_signature = computation_types.NamedTupleType(
        (n, element_type) for n in names)
    value = federating_executor.FederatingExecutorValue(value, type_signature)

    result = self.run_sync(value.compute())

    expected_result = anonymous_tuple.AnonymousTuple((n, 10.0) for n in names)
    self.assertEqual(result, expected_result)

  def test_raises_type_error_with_unembedded_federated_type(self):
    value = [10.0, 11.0, 12.0]
    type_signature = type_factory.at_clients(tf.float32)
    value = federating_executor.FederatingExecutorValue(value, type_signature)

    with self.assertRaises(TypeError):
      self.run_sync(value.compute())

  def test_raises_runtime_error_with_unsupported_value_or_type(self):
    value = 10.0
    type_signature = computation_types.TensorType(tf.float32)
    value = federating_executor.FederatingExecutorValue(value, type_signature)

    with self.assertRaises(RuntimeError):
      self.run_sync(value.compute())


class FederatingExecutorCreateValueTest(executor_test_utils.AsyncTestCase,
                                        parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('placement_literal',
       *executor_test_utils.create_dummy_placement_literal()),
      ('computation_call',
       *executor_test_utils.create_dummy_computation_call()),
      ('computation_intrinsic',
       *executor_test_utils.create_dummy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_dummy_computation_lambda_empty()),
      ('computation_placement',
       *executor_test_utils.create_dummy_computation_placement()),
      ('computation_selection',
       *executor_test_utils.create_dummy_computation_selection()),
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_empty()),
      ('computation_tuple',
       *executor_test_utils.create_dummy_computation_tuple()),
      ('federated_type_at_clients',
       *executor_test_utils.create_dummy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_dummy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_dummy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_dummy_value_unplaced()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  def test_returns_value_with_value_and_type(self, value, type_signature):
    executor = create_test_executor()

    result = self.run_sync(executor.create_value(value, type_signature))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())

  # pyformat: disable
  @parameterized.named_parameters([
      ('placement_literal',
       *executor_test_utils.create_dummy_placement_literal()),
      ('computation_call',
       *executor_test_utils.create_dummy_computation_call()),
      ('computation_intrinsic',
       *executor_test_utils.create_dummy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_dummy_computation_lambda_empty()),
      ('computation_placement',
       *executor_test_utils.create_dummy_computation_placement()),
      ('computation_selection',
       *executor_test_utils.create_dummy_computation_selection()),
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_empty()),
      ('computation_tuple',
       *executor_test_utils.create_dummy_computation_tuple()),
  ])
  # pyformat: enable
  def test_returns_value_with_value_only(self, value, type_signature):
    executor = create_test_executor()

    result = self.run_sync(executor.create_value(value))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_intrinsic',
       *executor_test_utils.create_dummy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_dummy_computation_lambda_empty()),
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_empty()),
  ])
  # pyformat: enable
  def test_returns_value_with_computation_impl(self, proto, type_signature):
    executor = create_test_executor()
    value = computation_impl.ComputationImpl(proto,
                                             context_stack_impl.context_stack)

    result = self.run_sync(executor.create_value(value, type_signature))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())

  # pyformat: disable
  @parameterized.named_parameters([
      ('federated_type_at_clients',
       *executor_test_utils.create_dummy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_dummy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_dummy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_dummy_value_unplaced()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  def test_raises_type_error_with_value_only(self, value, type_signature):
    del type_signature  # Unused.
    executor = create_test_executor()

    with self.assertRaises(TypeError):
      self.run_sync(executor.create_value(value))

  # pyformat: disable
  @parameterized.named_parameters([
      ('placement_literal',
       *executor_test_utils.create_dummy_placement_literal()),
      ('computation_placement',
       *executor_test_utils.create_dummy_computation_placement()),
      ('federated_type_at_clients',
       *executor_test_utils.create_dummy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_dummy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_dummy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_dummy_value_unplaced()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  def test_raises_type_error_with_value_and_bad_type(self, value,
                                                     type_signature):
    del type_signature  # Unused.
    executor = create_test_executor()
    bad_type_signature = computation_types.TensorType(tf.string)

    with self.assertRaises(TypeError):
      self.run_sync(executor.create_value(value, bad_type_signature))

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_call',
       *executor_test_utils.create_dummy_computation_call()),
      ('computation_intrinsic',
       *executor_test_utils.create_dummy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_dummy_computation_lambda_empty()),
      ('computation_selection',
       *executor_test_utils.create_dummy_computation_selection()),
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_empty()),
      ('computation_tuple',
       *executor_test_utils.create_dummy_computation_tuple()),
  ])
  # pyformat: enable
  def test_raises_type_error_with_value_and_bad_type_skipped(
      self, value, type_signature):
    del type_signature  # Unused.
    self.skipTest(
        'TODO(b/152449402): `FederatingExecutor.create_value` method should '
        'fail if it is passed a computation and an incompatible type.')
    executor = create_test_executor()
    bad_type_signature = computation_types.TensorType(tf.string)

    with self.assertRaises(TypeError):
      self.run_sync(executor.create_value(value, bad_type_signature))

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_reference',
       *executor_test_utils.create_dummy_computation_reference()),
      ('function_type', lambda: 10, type_factory.unary_op(tf.int32)),
  ])
  # pyformat: enable
  def test_raises_value_error_with_value(self, value, type_signature):
    executor = create_test_executor()

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))

  def test_raises_value_error_with_unrecognized_computation_intrinsic(self):
    executor = create_test_executor()
    # A `ValueError` will be raised because `create_value` can not recognize the
    # following intrinsic, because it has not been added to the intrinsic
    # registry.
    value = pb.Computation(
        type=type_serialization.serialize_type(tf.int32),
        intrinsic=pb.Intrinsic(uri='unregistered_intrinsic'))
    type_signature = computation_types.TensorType(tf.int32)

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))

  def test_raises_value_error_with_unrecognized_computation_selection(self):
    executor = create_test_executor()
    source, _ = executor_test_utils.create_dummy_computation_tuple()
    type_signature = computation_types.NamedTupleType([])
    # A `ValueError` will be raised because `create_value` can not handle the
    # following `pb.Selection`, because does not set either a name or an index
    # field.
    value = pb.Computation(
        type=type_serialization.serialize_type(type_signature),
        selection=pb.Selection(source=source))

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))

  # pyformat: disable
  @parameterized.named_parameters([
      ('intrinsic_def_federated_broadcast',
       *executor_test_utils.create_dummy_intrinsic_def_federated_broadcast()),
      ('intrinsic_def_federated_eval_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_clients()),
      ('intrinsic_def_federated_map',
       *executor_test_utils.create_dummy_intrinsic_def_federated_map()),
      ('intrinsic_def_federated_map_all_equal',
       *executor_test_utils.create_dummy_intrinsic_def_federated_map_all_equal()),
      ('intrinsic_def_federated_value_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_dummy_value_at_clients_all_equal()),
      ('federated_type_at_clients',
       *executor_test_utils.create_dummy_value_at_clients())
  ])
  # pyformat: enable
  def test_raises_value_error_with_no_target_executor_clients(
      self, value, type_signature):
    executor = federating_executor.FederatingExecutor({
        placement_literals.SERVER: eager_tf_executor.EagerTFExecutor(),
        None: eager_tf_executor.EagerTFExecutor()
    })

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))

  # pyformat: disable
  @parameterized.named_parameters([
      ('intrinsic_def_federated_aggregate',
       *executor_test_utils.create_dummy_intrinsic_def_federated_aggregate()),
      ('intrinsic_def_federated_apply',
       *executor_test_utils.create_dummy_intrinsic_def_federated_apply()),
      ('intrinsic_def_federated_collect',
       *executor_test_utils.create_dummy_intrinsic_def_federated_collect()),
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_server()),
      ('intrinsic_def_federated_mean',
       *executor_test_utils.create_dummy_intrinsic_def_federated_mean()),
      ('intrinsic_def_federated_sum',
       *executor_test_utils.create_dummy_intrinsic_def_federated_sum()),
      ('intrinsic_def_federated_reduce',
       *executor_test_utils.create_dummy_intrinsic_def_federated_reduce()),
      ('intrinsic_def_federated_value_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_value_at_server()),
      ('intrinsic_def_federated_weighted_mean',
       *executor_test_utils.create_dummy_intrinsic_def_federated_weighted_mean()),
      ('intrinsic_def_federated_zip_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_zip_at_server()),
      ('federated_type_at_server',
       *executor_test_utils.create_dummy_value_at_server()),
  ])
  # pyformat: enable
  def test_raises_value_error_with_no_target_executor_server(
      self, value, type_signature):
    executor = federating_executor.FederatingExecutor({
        placement_literals.CLIENTS: eager_tf_executor.EagerTFExecutor(),
        None: eager_tf_executor.EagerTFExecutor()
    })
    value, type_signature = executor_test_utils.create_dummy_value_at_server()

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))

  def test_raises_value_error_with_no_target_executor_unplaced(self):
    executor = federating_executor.FederatingExecutor({
        placement_literals.SERVER: eager_tf_executor.EagerTFExecutor(),
        placement_literals.CLIENTS: eager_tf_executor.EagerTFExecutor(),
    })
    value, type_signature = executor_test_utils.create_dummy_value_unplaced()

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))

  def test_raises_value_error_with_unexpected_federated_type_at_clients(self):
    executor = create_test_executor()
    value = [10, 20]
    type_signature = type_factory.at_clients(tf.int32)

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))

  def test_raises_value_error_with_unexpected_federated_type_at_clients_all_equal(
      self):
    executor = create_test_executor()
    value = [10] * 3
    type_signature = type_factory.at_clients(tf.int32, all_equal=True)

    with self.assertRaises(ValueError):
      self.run_sync(executor.create_value(value, type_signature))


class FederatingExecutorCreateCallTest(executor_test_utils.AsyncTestCase,
                                       parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('intrinsic_def_federated_aggregate',
       *executor_test_utils.create_dummy_intrinsic_def_federated_aggregate(),
       [executor_test_utils.create_dummy_value_at_clients(),
        executor_test_utils.create_dummy_value_unplaced(),
        executor_test_utils.create_dummy_computation_tensorflow_add(),
        executor_test_utils.create_dummy_computation_tensorflow_add(),
        executor_test_utils.create_dummy_computation_tensorflow_identity()],
       43.0),
      ('intrinsic_def_federated_apply',
       *executor_test_utils.create_dummy_intrinsic_def_federated_apply(),
       [executor_test_utils.create_dummy_computation_tensorflow_identity(),
        executor_test_utils.create_dummy_value_at_server()],
       10.0),
      ('intrinsic_def_federated_broadcast',
       *executor_test_utils.create_dummy_intrinsic_def_federated_broadcast(),
       [executor_test_utils.create_dummy_value_at_server()],
       10.0),
      ('intrinsic_def_federated_collect',
       *executor_test_utils.create_dummy_intrinsic_def_federated_collect(),
       [executor_test_utils.create_dummy_value_at_clients()],
       tf.data.Dataset.from_tensor_slices([10.0, 11.0, 12.0])),
      ('intrinsic_def_federated_eval_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_clients(),
       [executor_test_utils.create_dummy_computation_tensorflow_constant()],
       [10.0] * 3),
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_server(),
       [executor_test_utils.create_dummy_computation_tensorflow_constant()],
       10.0),
      ('intrinsic_def_federated_map',
       *executor_test_utils.create_dummy_intrinsic_def_federated_map(),
       [executor_test_utils.create_dummy_computation_tensorflow_identity(),
        executor_test_utils.create_dummy_value_at_clients()],
       [10.0, 11.0, 12.0]),
      ('intrinsic_def_federated_map_all_equal',
       *executor_test_utils.create_dummy_intrinsic_def_federated_map_all_equal(),
       [executor_test_utils.create_dummy_computation_tensorflow_identity(),
        executor_test_utils.create_dummy_value_at_clients_all_equal()],
       10.0),
      ('intrinsic_def_federated_mean',
       *executor_test_utils.create_dummy_intrinsic_def_federated_mean(),
       [executor_test_utils.create_dummy_value_at_clients()],
       11.0),
      ('intrinsic_def_federated_sum',
       *executor_test_utils.create_dummy_intrinsic_def_federated_sum(),
       [executor_test_utils.create_dummy_value_at_clients()],
       33.0),
      ('intrinsic_def_federated_reduce',
       *executor_test_utils.create_dummy_intrinsic_def_federated_reduce(),
       [executor_test_utils.create_dummy_value_at_clients(),
        executor_test_utils.create_dummy_value_unplaced(),
        executor_test_utils.create_dummy_computation_tensorflow_add()],
       43.0),
      ('intrinsic_def_federated_value_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_value_at_clients(),
       [executor_test_utils.create_dummy_value_unplaced()],
       10.0),
      ('intrinsic_def_federated_value_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_value_at_server(),
       [executor_test_utils.create_dummy_value_unplaced()],
       10.0),
      ('intrinsic_def_federated_weighted_mean',
       *executor_test_utils.create_dummy_intrinsic_def_federated_weighted_mean(),
       [executor_test_utils.create_dummy_value_at_clients(),
        executor_test_utils.create_dummy_value_at_clients()],
       11.060606),
      ('intrinsic_def_federated_zip_at_clients',
       *executor_test_utils.create_dummy_intrinsic_def_federated_zip_at_clients(),
       [executor_test_utils.create_dummy_value_at_clients(),
        executor_test_utils.create_dummy_value_at_clients()],
       [anonymous_tuple.AnonymousTuple([(None, 10.0), (None, 10.0)]),
        anonymous_tuple.AnonymousTuple([(None, 11.0), (None, 11.0)]),
        anonymous_tuple.AnonymousTuple([(None, 12.0), (None, 12.0)])]),
      ('intrinsic_def_federated_zip_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_zip_at_server(),
       [executor_test_utils.create_dummy_value_at_server(),
        executor_test_utils.create_dummy_value_at_server()],
       anonymous_tuple.AnonymousTuple([(None, 10.0), (None, 10.0)])),
      ('computation_intrinsic',
       *executor_test_utils.create_dummy_computation_intrinsic(),
       [executor_test_utils.create_dummy_computation_tensorflow_constant()],
       10.0),
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_identity(),
       [executor_test_utils.create_dummy_value_unplaced()],
       10.0),
  ])
  # pyformat: enable
  def test_returns_value_with_comp_and_arg(self, comp, comp_type, args,
                                           expected_result):
    executor = create_test_executor()

    comp = self.run_sync(executor.create_value(comp, comp_type))
    elements = [self.run_sync(executor.create_value(*x)) for x in args]
    if len(elements) > 1:
      arg = self.run_sync(executor.create_tuple(elements))
    else:
      arg = elements[0]
    result = self.run_sync(executor.create_call(comp, arg))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     comp_type.result.compact_representation())
    actual_result = self.run_sync(result.compute())
    if (all_isinstance([actual_result, expected_result], list) or
        all_isinstance([actual_result, expected_result], tf.data.Dataset)):
      for actual_element, expected_element in zip(actual_result,
                                                  expected_result):
        self.assertEqual(actual_element, expected_element)
    else:
      self.assertEqual(actual_result, expected_result)

  def test_returns_value_with_intrinsic_def_federated_value_at_server_and_tuple(
      self):
    executor = create_test_executor(number_of_clients=3)
    arg, arg_type = executor_test_utils.create_dummy_computation_tuple()
    intrinsic_def = intrinsic_defs.FEDERATED_VALUE_AT_SERVER
    comp_type = computation_types.FunctionType(arg_type,
                                               type_factory.at_server(arg_type))
    comp = pb.Computation(
        type=type_serialization.serialize_type(comp_type),
        intrinsic=pb.Intrinsic(uri=intrinsic_def.uri))

    comp = self.run_sync(executor.create_value(comp, comp_type))
    arg = self.run_sync(executor.create_value(arg, arg_type))
    result = self.run_sync(executor.create_call(comp, arg))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     comp_type.result.compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = [10.0] * 2
    for actual_element, expected_element in zip(actual_result, expected_result):
      self.assertEqual(actual_element, expected_element)

  def test_returns_value_with_intrinsic_def_federated_eval_at_clients_and_random(
      self):
    executor = create_test_executor(number_of_clients=3)
    comp, comp_type = executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_clients(
    )
    arg, arg_type = executor_test_utils.create_dummy_computation_tensorflow_random(
    )

    comp = self.run_sync(executor.create_value(comp, comp_type))
    arg = self.run_sync(executor.create_value(arg, arg_type))
    result = self.run_sync(executor.create_call(comp, arg))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     comp_type.result.compact_representation())
    actual_result = self.run_sync(result.compute())
    unique_results = set([x.numpy() for x in actual_result])
    if len(actual_result) != len(unique_results):
      self.fail(
          'Expected the result to contain different random numbers, found {}.'
          .format(actual_result))

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_empty()),
  ])
  # pyformat: enable
  def test_returns_value_with_comp_only(self, comp, comp_type):
    executor = create_test_executor()

    comp = self.run_sync(executor.create_value(comp, comp_type))
    result = self.run_sync(executor.create_call(comp))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     comp_type.result.compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = []
    self.assertCountEqual(actual_result, expected_result)

  def test_raises_type_error_with_unembedded_comp(self):
    executor = create_test_executor()
    comp, _ = executor_test_utils.create_dummy_computation_tensorflow_identity()
    arg, arg_type = executor_test_utils.create_dummy_value_unplaced()

    arg = self.run_sync(executor.create_value(arg, arg_type))
    with self.assertRaises(TypeError):
      self.run_sync(executor.create_call(comp, arg))

  def test_raises_type_error_with_unembedded_arg(self):
    executor = create_test_executor()
    comp, comp_type = executor_test_utils.create_dummy_computation_tensorflow_identity(
    )
    arg, _ = executor_test_utils.create_dummy_value_unplaced()

    comp = self.run_sync(executor.create_value(comp, comp_type))
    with self.assertRaises(TypeError):
      self.run_sync(executor.create_call(comp, arg))

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_intrinsic',
       *executor_test_utils.create_dummy_computation_intrinsic()),
      ('computation_lambda',
       *executor_test_utils.create_dummy_computation_lambda_identity()),
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_identity()),
  ] + get_named_parameters_for_supported_intrinsics())
  # pyformat: enable
  def test_raises_type_error_with_comp_and_bad_arg(self, comp, comp_type):
    executor = create_test_executor()
    bad_arg = 'string'
    bad_arg_type = computation_types.TensorType(tf.string)

    comp = self.run_sync(executor.create_value(comp, comp_type))
    arg = self.run_sync(executor.create_value(bad_arg, bad_arg_type))
    with self.assertRaises(TypeError):
      self.run_sync(executor.create_call(comp, arg))

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_lambda',
       *executor_test_utils.create_dummy_computation_lambda_empty()),
      ('computation_placement',
       *executor_test_utils.create_dummy_computation_placement()),
      ('computation_tuple',
       *executor_test_utils.create_dummy_computation_tuple()),
      ('federated_type_at_clients',
       *executor_test_utils.create_dummy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_dummy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_dummy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_dummy_value_unplaced()),
  ])
  # pyformat: enable
  def test_raises_value_error_with_comp(self, comp, comp_type):
    executor = create_test_executor()

    comp = self.run_sync(executor.create_value(comp, comp_type))
    with self.assertRaises(ValueError):
      self.run_sync(executor.create_call(comp))

  def test_raises_not_implemented_error_with_intrinsic_def_federated_secure_sum(
      self):
    executor = create_test_executor()
    comp, comp_type = executor_test_utils.create_dummy_intrinsic_def_federated_secure_sum(
    )
    arg_1, arg_1_type = executor_test_utils.create_dummy_value_at_clients()
    arg_2, arg_2_type = executor_test_utils.create_dummy_value_unplaced()

    comp = self.run_sync(executor.create_value(comp, comp_type))
    arg_1 = self.run_sync(executor.create_value(arg_1, arg_1_type))
    arg_2 = self.run_sync(executor.create_value(arg_2, arg_2_type))
    args = self.run_sync(executor.create_tuple([arg_1, arg_2]))
    with self.assertRaises(NotImplementedError):
      self.run_sync(executor.create_call(comp, args))

  def test_raises_not_implemented_error_with_unimplemented_intrinsic(self):
    executor = create_test_executor()
    dummy_intrinsic = intrinsic_defs.IntrinsicDef(
        'DUMMY_INTRINSIC', 'dummy_intrinsic',
        computation_types.AbstractType('T'))
    comp = pb.Computation(
        intrinsic=pb.Intrinsic(uri='dummy_intrinsic'),
        type=type_serialization.serialize_type(tf.int32))

    comp = self.run_sync(executor.create_value(comp))
    with self.assertRaises(NotImplementedError):
      self.run_sync(executor.create_call(comp))


class FederatingExecutorCreateTupleTest(executor_test_utils.AsyncTestCase,
                                        parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('placement_literal',
       *executor_test_utils.create_dummy_placement_literal()),
      ('computation_call',
       *executor_test_utils.create_dummy_computation_call()),
      ('computation_placement',
       *executor_test_utils.create_dummy_computation_placement()),
      ('computation_selection',
       *executor_test_utils.create_dummy_computation_selection()),
      ('computation_tuple',
       *executor_test_utils.create_dummy_computation_tuple()),
      ('federated_type_at_clients',
       *executor_test_utils.create_dummy_value_at_clients()),
      ('federated_type_at_clients_all_equal',
       *executor_test_utils.create_dummy_value_at_clients_all_equal()),
      ('federated_type_at_server',
       *executor_test_utils.create_dummy_value_at_server()),
      ('unplaced_type',
       *executor_test_utils.create_dummy_value_unplaced()),
  ])
  # pyformat: enable
  def test_returns_value_with_elements_value(self, value, type_signature):
    executor = create_test_executor()

    element = self.run_sync(executor.create_value(value, type_signature))
    elements = [element] * 3
    type_signature = computation_types.NamedTupleType([type_signature] * 3)
    result = self.run_sync(executor.create_tuple(elements))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())
    # TODO(b/153578410): Some `FederatingExecutorValue` that can can be
    # constructed, can not be computed.

  # pyformat: disable
  @parameterized.named_parameters([
      ('intrinsic_def_federated_eval_at_server',
       *executor_test_utils.create_dummy_intrinsic_def_federated_eval_at_server(),
       *executor_test_utils.create_dummy_computation_tensorflow_constant()),
      ('computation_intrinsic',
       *executor_test_utils.create_dummy_computation_intrinsic(),
       *executor_test_utils.create_dummy_computation_tensorflow_constant()),
  ])
  # pyformat: enable
  def test_returns_value_with_elements_fn_and_arg(self, fn, fn_type, arg,
                                                  arg_type):
    executor = create_test_executor()

    fn = self.run_sync(executor.create_value(fn, fn_type))
    arg = self.run_sync(executor.create_value(arg, arg_type))
    element = self.run_sync(executor.create_call(fn, arg))
    elements = [element] * 3
    type_signature = computation_types.NamedTupleType([fn_type.result] * 3)
    result = self.run_sync(executor.create_tuple(elements))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = [self.run_sync(element.compute())] * 3
    self.assertCountEqual(actual_result, expected_result)

  # pyformat: disable
  @parameterized.named_parameters([
      ('computation_tensorflow',
       *executor_test_utils.create_dummy_computation_tensorflow_empty()),
  ])
  # pyformat: enable
  def test_returns_value_with_elements_fn_only(self, fn, fn_type):
    executor = create_test_executor()

    fn = self.run_sync(executor.create_value(fn, fn_type))
    element = self.run_sync(executor.create_call(fn))
    elements = [element] * 3
    type_signature = computation_types.NamedTupleType([fn_type.result] * 3)
    result = self.run_sync(executor.create_tuple(elements))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = [self.run_sync(element.compute())] * 3
    self.assertCountEqual(actual_result, expected_result)

  def test_raises_type_error_with_unembedded_elements(self):
    executor = create_test_executor()
    element, _ = executor_test_utils.create_dummy_value_unplaced()

    elements = [element] * 3
    with self.assertRaises(TypeError):
      self.run_sync(executor.create_tuple(elements))


class FederatingExecutorCreateSelectionTest(executor_test_utils.AsyncTestCase):

  def test_returns_value_with_source_and_index_computation_tensorflow(self):
    executor = create_test_executor()
    source, type_signature = executor_test_utils.create_dummy_computation_tensorflow_tuple(
    )

    source = self.run_sync(executor.create_value(source, type_signature))
    source = self.run_sync(executor.create_call(source))
    result = self.run_sync(executor.create_selection(source, index=0))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.result[0].compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = self.run_sync(source.compute())[0]
    self.assertEqual(actual_result, expected_result)

  def test_returns_value_with_source_and_index_anonymous_tuple(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_dummy_value_unplaced()

    element = self.run_sync(executor.create_value(element, element_type))
    elements = [element] * 3
    type_signature = computation_types.NamedTupleType([element_type] * 3)
    source = self.run_sync(executor.create_tuple(elements))
    result = self.run_sync(executor.create_selection(source, index=0))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature[0].compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = self.run_sync(source.compute())[0]
    self.assertEqual(actual_result, expected_result)

  def test_returns_value_with_source_and_name_computation_tensorflow(self):
    executor = create_test_executor()
    source, type_signature = executor_test_utils.create_dummy_computation_tensorflow_tuple(
    )

    source = self.run_sync(executor.create_value(source, type_signature))
    source = self.run_sync(executor.create_call(source))
    result = self.run_sync(executor.create_selection(source, name='a'))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature.result['a'].compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = self.run_sync(source.compute())['a']
    self.assertEqual(actual_result, expected_result)

  def test_returns_value_with_source_and_name_anonymous_tuple(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_dummy_value_unplaced()

    names = ['a', 'b', 'c']
    element = self.run_sync(executor.create_value(element, element_type))
    elements = anonymous_tuple.AnonymousTuple((n, element) for n in names)
    type_signature = computation_types.NamedTupleType(
        (n, element_type) for n in names)
    source = self.run_sync(executor.create_tuple(elements))
    result = self.run_sync(executor.create_selection(source, name='a'))

    self.assertIsInstance(result, federating_executor.FederatingExecutorValue)
    self.assertEqual(result.type_signature.compact_representation(),
                     type_signature['a'].compact_representation())
    actual_result = self.run_sync(result.compute())
    expected_result = self.run_sync(source.compute())['a']
    self.assertEqual(actual_result, expected_result)

  def test_raises_type_error_with_unembedded_source(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_dummy_value_unplaced()

    element = self.run_sync(executor.create_value(element, element_type))
    source = [element] * 3
    with self.assertRaises(TypeError):
      self.run_sync(executor.create_selection(source, index=0))

  def test_raises_type_error_with_not_tuple_type(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_dummy_value_unplaced()

    source = self.run_sync(executor.create_value(element, element_type))
    with self.assertRaises(TypeError):
      self.run_sync(executor.create_selection(source, index=0))

  def test_raises_value_error_with_no_index_or_name(self):
    executor = create_test_executor()
    element, element_type = executor_test_utils.create_dummy_value_unplaced()

    element = self.run_sync(executor.create_value(element, element_type))
    elements = [element] * 3
    source = self.run_sync(executor.create_tuple(elements))
    with self.assertRaises(ValueError):
      self.run_sync(executor.create_selection(source))

  def test_raises_value_error_with_unrecognized_generic_zero(self):
    executor = create_test_executor()

    value = intrinsic_defs.GENERIC_ZERO
    type_signature = computation_types.NamedTupleType(
        [computation_types.TensorType(tf.int32)] * 3)

    source = self.run_sync(executor.create_value(value, type_signature))
    with self.assertRaises(ValueError):
      self.run_sync(executor.create_selection(source, index=0))


if __name__ == '__main__':
  absltest.main()
