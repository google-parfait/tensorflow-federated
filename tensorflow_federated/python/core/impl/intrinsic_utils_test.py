# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for intrinsic_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_wrapper_instances
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import intrinsic_utils
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_constructors


class GenericConstantTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      intrinsic_utils.create_generic_constant(None, 0)

  def test_raises_non_scalar(self):
    with self.assertRaises(TypeError):
      intrinsic_utils.create_generic_constant([tf.int32], [0])

  def test_constructs_tensor_zero(self):
    tensor_type = computation_types.TensorType(tf.float32, [2, 2])
    tensor_zero = intrinsic_utils.create_generic_constant(tensor_type, 0)
    self.assertEqual(tensor_zero.type_signature, tensor_type)
    self.assertIsInstance(tensor_zero, computation_building_blocks.Call)
    executable_noarg_fn = computation_wrapper_instances.building_block_to_computation(
        tensor_zero.function)
    self.assertTrue(np.array_equal(executable_noarg_fn(), np.zeros([2, 2])))

  def test_create_unnamed_tuple_zero(self):
    tuple_type = [computation_types.TensorType(tf.float32, [2, 2])] * 2
    tuple_zero = intrinsic_utils.create_generic_constant(tuple_type, 0)
    self.assertEqual(tuple_zero.type_signature,
                     computation_types.to_type(tuple_type))
    self.assertIsInstance(tuple_zero, computation_building_blocks.Call)
    executable_noarg_fn = computation_wrapper_instances.building_block_to_computation(
        tuple_zero.function)
    self.assertLen(executable_noarg_fn(), 2)
    self.assertTrue(np.array_equal(executable_noarg_fn()[0], np.zeros([2, 2])))
    self.assertTrue(np.array_equal(executable_noarg_fn()[1], np.zeros([2, 2])))

  def test_create_named_tuple_one(self):
    tuple_type = [('a', computation_types.TensorType(tf.float32, [2, 2])),
                  ('b', computation_types.TensorType(tf.float32, [2, 2]))]
    tuple_zero = intrinsic_utils.create_generic_constant(tuple_type, 1)
    self.assertEqual(tuple_zero.type_signature,
                     computation_types.to_type(tuple_type))
    self.assertIsInstance(tuple_zero, computation_building_blocks.Call)
    executable_noarg_fn = computation_wrapper_instances.building_block_to_computation(
        tuple_zero.function)
    self.assertLen(executable_noarg_fn(), 2)
    self.assertTrue(np.array_equal(executable_noarg_fn().a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(executable_noarg_fn().b, np.ones([2, 2])))

  def test_create_federated_tensor_one(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(tf.float32, [2, 2]),
        placement_literals.CLIENTS)
    fed_zero = intrinsic_utils.create_generic_constant(fed_type, 1)
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, computation_building_blocks.Call)
    self.assertIsInstance(fed_zero.function,
                          computation_building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertIsInstance(fed_zero.argument, computation_building_blocks.Call)
    executable_unplaced_fn = computation_wrapper_instances.building_block_to_computation(
        fed_zero.argument.function)
    self.assertTrue(np.array_equal(executable_unplaced_fn(), np.ones([2, 2])))

  def test_create_federated_named_tuple_one(self):
    tuple_type = [('a', computation_types.TensorType(tf.float32, [2, 2])),
                  ('b', computation_types.TensorType(tf.float32, [2, 2]))]
    fed_type = computation_types.FederatedType(tuple_type,
                                               placement_literals.SERVER)
    fed_zero = intrinsic_utils.create_generic_constant(fed_type, 1)
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, computation_building_blocks.Call)
    self.assertIsInstance(fed_zero.function,
                          computation_building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri)
    self.assertIsInstance(fed_zero.argument, computation_building_blocks.Call)
    executable_unplaced_fn = computation_wrapper_instances.building_block_to_computation(
        fed_zero.argument.function)
    self.assertLen(executable_unplaced_fn(), 2)
    self.assertTrue(np.array_equal(executable_unplaced_fn().a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(executable_unplaced_fn().b, np.ones([2, 2])))

  def test_create_named_tuple_of_federated_tensors_zero(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(tf.float32, [2, 2]),
        placement_literals.CLIENTS,
        all_equal=True)
    tuple_type = [('a', fed_type), ('b', fed_type)]
    zero = intrinsic_utils.create_generic_constant(tuple_type, 0)
    fed_zero = zero.argument[0]

    self.assertEqual(zero.type_signature, computation_types.to_type(tuple_type))
    self.assertIsInstance(fed_zero.function,
                          computation_building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertIsInstance(fed_zero.argument, computation_building_blocks.Call)
    executable_unplaced_fn = computation_wrapper_instances.building_block_to_computation(
        fed_zero.argument.function)
    self.assertTrue(np.array_equal(executable_unplaced_fn(), np.zeros([2, 2])))


class BinaryOperatorBodyTest(absltest.TestCase):

  def test_apply_op_raises_on_none(self):
    with self.assertRaisesRegex(TypeError, 'ComputationBuildingBlock'):
      intrinsic_utils.apply_binary_operator_with_upcast(None, tf.multiply)

  def test_construct_op_raises_on_none_operator(self):
    with self.assertRaisesRegex(TypeError, 'found non-callable'):
      intrinsic_utils.create_binary_operator_with_upcast(tf.int32, None)

  def test_raises_incompatible_tuple_and_tensor(self):
    bad_type_ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType([[tf.int32, tf.int32], tf.float32],
                                        placement_literals.CLIENTS))
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      intrinsic_utils.apply_binary_operator_with_upcast(bad_type_ref,
                                                        tf.multiply)
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      intrinsic_utils.create_binary_operator_with_upcast(
          bad_type_ref.type_signature.member, tf.multiply)

  def test_raises_non_callable_op(self):
    bad_type_ref = computation_building_blocks.Reference(
        'x', [tf.float32, tf.float32])
    with self.assertRaisesRegex(TypeError, 'non-callable'):
      intrinsic_utils.apply_binary_operator_with_upcast(bad_type_ref,
                                                        tf.constant(0))
    with self.assertRaisesRegex(TypeError, 'non-callable'):
      intrinsic_utils.create_binary_operator_with_upcast(
          bad_type_ref, tf.constant(0))

  def test_raises_tuple_and_nonscalar_tensor(self):
    bad_type_ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[tf.int32, tf.int32],
             computation_types.TensorType(tf.float32, [2])],
            placement_literals.CLIENTS))
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      intrinsic_utils.apply_binary_operator_with_upcast(bad_type_ref,
                                                        tf.multiply)
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      intrinsic_utils.create_binary_operator_with_upcast(
          bad_type_ref.type_signature.member, tf.multiply)

  def test_raises_tuple_scalar_multiplied_by_nonscalar(self):
    bad_type_ref = computation_building_blocks.Reference(
        'x', [tf.int32, computation_types.TensorType(tf.float32, [2])])
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      intrinsic_utils.apply_binary_operator_with_upcast(bad_type_ref,
                                                        tf.multiply)
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      intrinsic_utils.create_binary_operator_with_upcast(
          bad_type_ref.type_signature, tf.multiply)

  def test_construct_generic_raises_federated_type(self):
    bad_type = computation_types.FederatedType(
        [[tf.int32, tf.int32],
         computation_types.TensorType(tf.float32, [2])],
        placement_literals.CLIENTS)
    with self.assertRaisesRegex(TypeError, 'argument that is not a two-tuple'):
      intrinsic_utils.create_binary_operator_with_upcast(bad_type, tf.multiply)

  def test_apply_integer_type_signature(self):
    ref = computation_building_blocks.Reference('x', [tf.int32, tf.int32])
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(multiplied.type_signature,
                     computation_types.to_type(tf.int32))

  def test_construct_integer_type_signature(self):
    ref = computation_building_blocks.Reference('x', [tf.int32, tf.int32])
    multiplier = intrinsic_utils.create_binary_operator_with_upcast(
        ref.type_signature, tf.multiply)
    self.assertEqual(
        multiplier.type_signature,
        type_constructors.binary_op(computation_types.to_type(tf.int32)))

  def test_multiply_federated_integer_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType([tf.int32, tf.int32],
                                        placement_literals.CLIENTS))
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))

  def test_divide_federated_float_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType([tf.float32, tf.float32],
                                        placement_literals.CLIENTS))
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))

  def test_multiply_federated_unnamed_tuple_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[tf.int32, tf.float32], [tf.int32, tf.float32]],
            placement_literals.CLIENTS))
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([tf.int32, tf.float32],
                                        placement_literals.CLIENTS))

  def test_multiply_federated_named_tuple_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[('a', tf.int32),
              ('b', tf.float32)], [('a', tf.int32), ('b', tf.float32)]],
            placement_literals.CLIENTS))
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.int32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))

  def test_divide_federated_named_tuple_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[('a', tf.int32),
              ('b', tf.float32)], [('a', tf.int32), ('b', tf.float32)]],
            placement_literals.CLIENTS))
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.divide)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.float64), ('b', tf.float32)],
                                        placement_literals.CLIENTS))

  def test_multiply_federated_named_tuple_with_scalar_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.float32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))

  def test_multiply_named_tuple_with_scalar_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x', [[('a', tf.float32), ('b', tf.float32)], tf.float32])
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.NamedTupleType([('a', tf.float32),
                                          ('b', tf.float32)]))

  def test_construct_multiply_op_named_tuple_with_scalar_type_signature(self):
    type_spec = computation_types.to_type([[('a', tf.float32),
                                            ('b', tf.float32)], tf.float32])
    multiplier = intrinsic_utils.create_binary_operator_with_upcast(
        type_spec, tf.multiply)
    expected_function_type = computation_types.FunctionType(
        type_spec, type_spec[0])
    self.assertEqual(multiplier.type_signature, expected_function_type)

  def test_construct_divide_op_named_tuple_with_scalar_type_signature(self):
    type_spec = computation_types.to_type([[('a', tf.float32),
                                            ('b', tf.float32)], tf.float32])
    multiplier = intrinsic_utils.create_binary_operator_with_upcast(
        type_spec, tf.divide)
    expected_function_type = computation_types.FunctionType(
        type_spec, type_spec[0])
    self.assertEqual(multiplier.type_signature, expected_function_type)

  def test_divide_federated_named_tuple_with_scalar_type_signature(self):
    ref = computation_building_blocks.Reference(
        'x',
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    multiplied = intrinsic_utils.apply_binary_operator_with_upcast(
        ref, tf.divide)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.float32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))


if __name__ == '__main__':
  absltest.main()
