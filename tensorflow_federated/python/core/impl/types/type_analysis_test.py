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

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory


class CountTypesTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('one',
       computation_types.TensorType(tf.int32),
       lambda t: t.is_tensor(),
       1),
      ('three',
       computation_types.NamedTupleType([tf.int32] * 3),
       lambda t: t.is_tensor(),
       3),
      ('nested',
       computation_types.NamedTupleType([[tf.int32] * 3] * 3),
       lambda t: t.is_tensor(),
       9),
  ])
  # pyformat: enable
  def test_returns_result(self, type_signature, predicate, expected_result):
    result = type_analysis.count(type_signature, predicate)
    self.assertEqual(result, expected_result)


class ContainsTypesTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('one_type',
       computation_types.TensorType(tf.int32),
       computation_types.TensorType),
      ('two_types',
       computation_types.NamedTupleType([tf.int32]),
       (computation_types.NamedTupleType, computation_types.TensorType)),
      ('less_types',
       computation_types.TensorType(tf.int32),
       (computation_types.NamedTupleType, computation_types.TensorType)),
      ('more_types',
       computation_types.NamedTupleType([tf.int32]),
       computation_types.TensorType),
  ])
  # pyformat: enable
  def test_returns_true(self, type_signature, types):
    result = type_analysis.contains(type_signature,
                                    lambda x: isinstance(x, types))
    self.assertTrue(result)

  @parameterized.named_parameters([
      ('one_type', computation_types.TensorType(tf.int32),
       computation_types.NamedTupleType),
  ])
  def test_returns_false(self, type_signature, types):
    result = type_analysis.contains(type_signature,
                                    lambda x: isinstance(x, types))
    self.assertFalse(result)


class ContainsOnlyTypesTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('one_type',
       computation_types.TensorType(tf.int32),
       computation_types.TensorType),
      ('two_types',
       computation_types.NamedTupleType([tf.int32]),
       (computation_types.NamedTupleType, computation_types.TensorType)),
      ('less_types',
       computation_types.TensorType(tf.int32),
       (computation_types.NamedTupleType, computation_types.TensorType)),
  ])
  # pyformat: enable
  def test_returns_true(self, type_signature, types):
    result = type_analysis.contains_only(type_signature,
                                         lambda x: isinstance(x, types))
    self.assertTrue(result)

  # pyformat: disable
  @parameterized.named_parameters([
      ('one_type',
       computation_types.TensorType(tf.int32),
       computation_types.NamedTupleType),
      ('more_types',
       computation_types.NamedTupleType([tf.int32]),
       computation_types.TensorType),
  ])
  # pyformat: enable
  def test_returns_false(self, type_signature, types):
    result = type_analysis.contains_only(type_signature,
                                         lambda x: isinstance(x, types))
    self.assertFalse(result)


class CheckWellFormedTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('abstract_type',
       computation_types.AbstractType('T')),
      ('federated_type',
       computation_types.FederatedType(tf.int32, placement_literals.CLIENTS)),
      ('function_type',
       computation_types.FunctionType(tf.int32, tf.int32)),
      ('named_tuple_type',
       computation_types.NamedTupleType([tf.int32] * 3)),
      ('placement_type',
       computation_types.PlacementType()),
      ('sequence_type',
       computation_types.SequenceType(tf.int32)),
      ('tensor_type',
       computation_types.TensorType(tf.int32)),
  ])
  # pyformat: enable
  def test_does_not_raise_type_error(self, type_signature):
    try:
      type_analysis.check_well_formed(type_signature)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters([
      ('federated_function_type',
       computation_types.FederatedType(
           computation_types.FunctionType(tf.int32, tf.int32),
           placement_literals.CLIENTS)),
      ('federated_federated_type',
       computation_types.FederatedType(
           computation_types.FederatedType(tf.int32,
                                           placement_literals.CLIENTS),
           placement_literals.CLIENTS)),
      ('sequence_sequence_type',
       computation_types.SequenceType(
           computation_types.SequenceType([tf.int32]))),
      ('sequence_federated_type',
       computation_types.SequenceType(
           computation_types.FederatedType(tf.int32,
                                           placement_literals.CLIENTS))),
      ('tuple_federated_function_type',
       computation_types.NamedTupleType([
           computation_types.FederatedType(
               computation_types.FunctionType(tf.int32, tf.int32),
               placement_literals.CLIENTS)
       ])),
      ('tuple_federated_federated_type',
       computation_types.NamedTupleType([
           computation_types.FederatedType(
               computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS),
               placement_literals.CLIENTS)
       ])),
      ('federated_tuple_function_type',
       computation_types.FederatedType(
           computation_types.NamedTupleType(
               [computation_types.FunctionType(tf.int32, tf.int32)]),
           placement_literals.CLIENTS)),
  ])
  # pyformat: enable
  def test_raises_type_error(self, type_signature):
    with self.assertRaises(TypeError):
      type_analysis.check_well_formed(type_signature)


class CheckAllAbstractTypesAreBoundTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('tensor_type', computation_types.TensorType(tf.int32)),
      ('function_type_with_no_arg',
       computation_types.FunctionType(None, tf.int32)),
      ('function_type_with_int_arg',
       computation_types.FunctionType(tf.int32, tf.int32)),
      ('function_type_with_abstract_arg',
       computation_types.FunctionType(
           computation_types.AbstractType('T'),
           computation_types.AbstractType('T'))),
      ('tuple_tuple_function_type_with_abstract_arg',
       computation_types.NamedTupleType([
           computation_types.NamedTupleType([
               computation_types.FunctionType(
                   computation_types.AbstractType('T'),
                   computation_types.AbstractType('T')),
           ])
       ])),
      ('function_type_with_unbound_function_arg',
       computation_types.FunctionType(
           computation_types.FunctionType(
               None, computation_types.AbstractType('T')),
           computation_types.AbstractType('T'))),
      ('function_type_with_sequence_arg',
       computation_types.FunctionType(
           computation_types.SequenceType(
               computation_types.AbstractType('T')),
           tf.int32)),
      ('function_type_with_two_abstract_args',
       computation_types.FunctionType(
           computation_types.NamedTupleType([
               computation_types.AbstractType('T'),
               computation_types.AbstractType('U'),
           ]),
           computation_types.NamedTupleType([
               computation_types.AbstractType('T'),
               computation_types.AbstractType('U'),
           ]))),
  ])
  # pyformat: enable
  def test_does_not_raise_type_error(self, type_spec):
    try:
      type_analysis.check_all_abstract_types_are_bound(type_spec)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  # pyformat: disable
  @parameterized.named_parameters([
      ('abstract_type', computation_types.AbstractType('T')),
      ('function_type_with_no_arg',
       computation_types.FunctionType(
           None, computation_types.AbstractType('T'))),
      ('function_type_with_int_arg',
       computation_types.FunctionType(
           tf.int32, computation_types.AbstractType('T'))),
      ('function_type_with_abstract_arg',
       computation_types.FunctionType(
           computation_types.AbstractType('T'),
           computation_types.AbstractType('U'))),
  ])
  # pyformat: enable
  def test_raises_type_error(self, type_spec):
    with self.assertRaises(TypeError):
      type_analysis.check_all_abstract_types_are_bound(type_spec)


class IsSumCompatibleTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('tensor_type', computation_types.TensorType(tf.int32)),
      ('tuple_type_int',
       computation_types.NamedTupleType([tf.int32, tf.int32],)),
      ('tuple_type_float',
       computation_types.NamedTupleType([tf.complex128, tf.float32,
                                         tf.float64])),
      ('federated_type',
       computation_types.FederatedType(tf.int32, placement_literals.CLIENTS)),
  ])
  def test_positive_examples(self, type_spec):
    self.assertTrue(type_analysis.is_sum_compatible(type_spec))

  @parameterized.named_parameters([
      ('tensor_type_bool', computation_types.TensorType(tf.bool)),
      ('tensor_type_string', computation_types.TensorType(tf.string)),
      ('tuple_type', computation_types.NamedTupleType([tf.int32, tf.bool])),
      ('sequence_type', computation_types.SequenceType(tf.int32)),
      ('placement_type', computation_types.PlacementType()),
      ('function_type', computation_types.FunctionType(tf.int32, tf.int32)),
      ('abstract_type', computation_types.AbstractType('T')),
  ])
  def test_negative_examples(self, type_spec):
    self.assertFalse(type_analysis.is_sum_compatible(type_spec))


class IsAverageCompatibleTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('tensor_type_float32', computation_types.TensorType(tf.float32)),
      ('tensor_type_float64', computation_types.TensorType(tf.float64)),
      ('tuple_type',
       computation_types.NamedTupleType([('x', tf.float32),
                                         ('y', tf.float64)])),
      ('federated_type',
       computation_types.FederatedType(tf.float32, placement_literals.CLIENTS)),
  ])
  def test_returns_true(self, type_spec):
    self.assertTrue(type_analysis.is_average_compatible(type_spec))

  @parameterized.named_parameters([
      ('tensor_type_int32', computation_types.TensorType(tf.int32)),
      ('tensor_type_int64', computation_types.TensorType(tf.int64)),
      ('sequence_type', computation_types.SequenceType(tf.float32)),
  ])
  def test_returns_false(self, type_spec):
    self.assertFalse(type_analysis.is_average_compatible(type_spec))


class CheckTypeTest(absltest.TestCase):

  def test_raises_type_error(self):
    type_analysis.check_type(10, computation_types.TensorType(tf.int32))
    self.assertRaises(TypeError, type_analysis.check_type, 10, tf.bool)


class CheckFederatedTypeTest(absltest.TestCase):

  def test_passes_or_raises_type_error(self):
    type_spec = computation_types.FederatedType(tf.int32,
                                                placement_literals.CLIENTS,
                                                False)
    type_analysis.check_federated_type(type_spec,
                                       computation_types.TensorType(tf.int32),
                                       placement_literals.CLIENTS, False)
    type_analysis.check_federated_type(type_spec,
                                       computation_types.TensorType(tf.int32),
                                       None, None)
    type_analysis.check_federated_type(type_spec, None,
                                       placement_literals.CLIENTS, None)
    type_analysis.check_federated_type(type_spec, None, None, False)
    self.assertRaises(TypeError, type_analysis.check_federated_type, type_spec,
                      tf.bool, None, None)
    self.assertRaises(TypeError, type_analysis.check_federated_type, type_spec,
                      None, placement_literals.SERVER, None)
    self.assertRaises(TypeError, type_analysis.check_federated_type, type_spec,
                      None, None, True)


class IsStructureOfIntegersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', computation_types.TensorType(tf.int32)),
      ('ints', computation_types.NamedTupleType([tf.int32, tf.int32])),
      ('federated_int_at_clients',
       computation_types.FederatedType(tf.int32, placement_literals.CLIENTS)),
  )
  def test_returns_true(self, type_spec):
    self.assertTrue(type_analysis.is_structure_of_integers(type_spec))

  @parameterized.named_parameters(
      ('bool', computation_types.TensorType(tf.bool)),
      ('string', computation_types.TensorType(tf.string)),
      ('int_and_bool', computation_types.NamedTupleType([tf.int32, tf.bool])),
      ('sequence_of_ints', computation_types.SequenceType(tf.int32)),
      ('placement', computation_types.PlacementType()),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('abstract', computation_types.AbstractType('T')),
  )
  def test_returns_false(self, type_spec):
    self.assertFalse(type_analysis.is_structure_of_integers(type_spec))


class IsValidBitwidthTypeForValueType(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('single int',
       computation_types.TensorType(tf.int32),
       computation_types.TensorType(tf.int32)),
      ('tuple',
       computation_types.NamedTupleType([tf.int32, tf.int32]),
       computation_types.NamedTupleType([tf.int32, tf.int32])),
      ('named tuple',
       computation_types.NamedTupleType([('x', tf.int32)]),
       computation_types.NamedTupleType([('x', tf.int32)])),
      ('single int for complex tensor',
       computation_types.TensorType(tf.int32),
       computation_types.TensorType(tf.int32, [5, 97, 204])),
      ('different kinds of ints',
       computation_types.TensorType(tf.int32),
       computation_types.TensorType(tf.int8)),
  )
  # pyformat: enable
  def test_returns_true(self, bitwidth_type, value_type):
    self.assertTrue(
        type_analysis.is_valid_bitwidth_type_for_value_type(
            bitwidth_type, value_type))

  # pyformat: disable
  @parameterized.named_parameters(
      ('single int_for_tuple',
       computation_types.TensorType(tf.int32),
       computation_types.NamedTupleType([tf.int32, tf.int32])),
      ('miscounted tuple',
       computation_types.NamedTupleType([tf.int32, tf.int32, tf.int32]),
       computation_types.NamedTupleType([tf.int32, tf.int32])),
      ('miscounted tuple 2',
       computation_types.NamedTupleType([tf.int32, tf.int32]),
       computation_types.NamedTupleType([tf.int32, tf.int32, tf.int32])),
      ('misnamed tuple',
       computation_types.NamedTupleType([('x', tf.int32)]),
       computation_types.NamedTupleType([('y', tf.int32)])),
  )
  # pyformat: enable
  def test_returns_false(self, bitwidth_type, value_type):
    self.assertFalse(
        type_analysis.is_valid_bitwidth_type_for_value_type(
            bitwidth_type, value_type))


class IsAnonTupleWithPyContainerTest(absltest.TestCase):

  def test_returns_true(self):
    value = anonymous_tuple.AnonymousTuple([('a', 0.0)])
    type_spec = computation_types.NamedTupleTypeWithPyContainerType(
        [('a', tf.float32)], dict)
    self.assertTrue(
        type_analysis.is_anon_tuple_with_py_container(value, type_spec))

  def test_returns_false_with_none_value(self):
    value = None
    type_spec = computation_types.NamedTupleTypeWithPyContainerType(
        [('a', tf.float32)], dict)
    self.assertFalse(
        type_analysis.is_anon_tuple_with_py_container(value, type_spec))

  def test_returns_false_with_named_tuple_type_spec(self):
    value = anonymous_tuple.AnonymousTuple([('a', 0.0)])
    type_spec = computation_types.NamedTupleType([('a', tf.float32)])
    self.assertFalse(
        type_analysis.is_anon_tuple_with_py_container(value, type_spec))


class IsConcreteInstanceOf(absltest.TestCase):

  def test_raises_with_int_first_argument(self):
    with self.assertRaises(TypeError):
      type_analysis.is_concrete_instance_of(
          1, computation_types.TensorType(tf.int32))

  def test_raises_with_int_second_argument(self):
    with self.assertRaises(TypeError):
      type_analysis.is_concrete_instance_of(
          computation_types.TensorType(tf.int32), 1)

  def test_raises_different_structures(self):
    with self.assertRaises(TypeError):
      type_analysis.is_concrete_instance_of(
          computation_types.TensorType(tf.int32),
          computation_types.NamedTupleType([tf.int32]))

  def test_raises_with_abstract_type_as_first_arg(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.TensorType(tf.int32)
    with self.assertRaises(TypeError):
      type_analysis.is_concrete_instance_of(t1, t2)

  def test_with_single_abstract_type_and_tensor_type(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.TensorType(tf.int32)
    self.assertTrue(type_analysis.is_concrete_instance_of(t2, t1))

  def test_raises_with_abstract_type_in_second_argument(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.AbstractType('T2')
    with self.assertRaises(TypeError):
      type_analysis.is_concrete_instance_of(t2, t1)

  def test_with_single_abstract_type_and_tuple_type(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.NamedTupleType([tf.int32])
    self.assertTrue(type_analysis.is_concrete_instance_of(t2, t1))

  def test_raises_with_conflicting_names(self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([('a', tf.int32), ('b', tf.int32)])
    with self.assertRaises(TypeError):
      type_analysis.is_concrete_instance_of(t2, t1)

  def test_raises_with_different_lengths(self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([tf.int32])
    with self.assertRaises(TypeError):
      type_analysis.is_concrete_instance_of(t2, t1)

  def test_succeeds_under_tuple(self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([
        computation_types.TensorType(tf.int32),
        computation_types.TensorType(tf.int32)
    ])
    self.assertTrue(type_analysis.is_concrete_instance_of(t2, t1))

  def test_fails_under_tuple_conflicting_concrete_types(self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([
        computation_types.TensorType(tf.int32),
        computation_types.TensorType(tf.float32)
    ])
    self.assertFalse(type_analysis.is_concrete_instance_of(t2, t1))

  def test_succeeds_abstract_type_under_sequence_type(self):
    t1 = computation_types.SequenceType(computation_types.AbstractType('T'))
    t2 = computation_types.SequenceType(tf.int32)
    self.assertTrue(type_analysis.is_concrete_instance_of(t2, t1))

  def test_fails_conflicting_concrete_types_under_sequence(self):
    t1 = computation_types.SequenceType([computation_types.AbstractType('T')] *
                                        2)
    t2 = computation_types.SequenceType([tf.int32, tf.float32])
    self.assertFalse(type_analysis.is_concrete_instance_of(t2, t1))

  def test_succeeds_single_function_type(self):
    t1 = computation_types.FunctionType(*[computation_types.AbstractType('T')] *
                                        2)
    t2 = computation_types.FunctionType(tf.int32, tf.int32)
    self.assertTrue(type_analysis.is_concrete_instance_of(t2, t1))

  def test_succeeds_function_different_parameter_and_return_types(self):
    t1 = computation_types.FunctionType(
        computation_types.AbstractType('T'),
        computation_types.AbstractType('U'))
    t2 = computation_types.FunctionType(tf.int32, tf.float32)
    self.assertTrue(type_analysis.is_concrete_instance_of(t2, t1))

  def test_fails_conflicting_binding_in_parameter_and_result(self):
    t1 = computation_types.FunctionType(*[computation_types.AbstractType('T')] *
                                        2)
    t2 = computation_types.FunctionType(tf.int32, tf.float32)
    self.assertFalse(type_analysis.is_concrete_instance_of(t2, t1))

  def test_abstract_federated_types_succeeds(self):
    t1 = computation_types.FederatedType(
        [computation_types.AbstractType('T1')] * 2,
        placement_literals.CLIENTS,
        all_equal=True)
    t2 = computation_types.FederatedType(
        [tf.int32] * 2, placement_literals.CLIENTS, all_equal=True)
    self.assertTrue(type_analysis.is_concrete_instance_of(t2, t1))

  def test_abstract_fails_on_different_federated_placements(self):
    t1 = computation_types.FederatedType(
        [computation_types.AbstractType('T1')] * 2,
        placement_literals.CLIENTS,
        all_equal=True)
    t2 = computation_types.FederatedType(
        [tf.int32] * 2, placement_literals.SERVER, all_equal=True)
    self.assertFalse(type_analysis.is_concrete_instance_of(t2, t1))

  def test_abstract_can_be_concretized_abstract_fails_on_different_federated_all_equal_bits(
      self):
    t1 = computation_types.FederatedType(
        [computation_types.AbstractType('T1')] * 2,
        placement_literals.CLIENTS,
        all_equal=True)
    t2 = computation_types.FederatedType(
        [tf.int32] * 2, placement_literals.SERVER, all_equal=True)
    self.assertFalse(type_analysis.is_concrete_instance_of(t2, t1))


def _convert_tensor_to_float(type_spec):
  if type_spec.is_tensor():
    return computation_types.TensorType(tf.float32, shape=type_spec.shape), True
  return type_spec, False


def _convert_abstract_type_to_tensor(type_spec):
  if type_spec.is_abstract():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_placement_type_to_tensor(type_spec):
  if type_spec.is_placement():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_function_to_tensor(type_spec):
  if type_spec.is_function():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_federated_to_tensor(type_spec):
  if type_spec.is_federated():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_sequence_to_tensor(type_spec):
  if type_spec.is_sequence():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_tuple_to_tensor(type_spec):
  if type_spec.is_tuple():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


class IsBinaryOpWithUpcastCompatibleTest(absltest.TestCase):

  def test_passes_on_none(self):
    self.assertTrue(
        type_analysis.is_binary_op_with_upcast_compatible_pair(None, None))

  def test_passes_empty_tuples(self):
    self.assertTrue(
        type_analysis.is_binary_op_with_upcast_compatible_pair(
            computation_types.NamedTupleType([]),
            computation_types.NamedTupleType([])))

  def test_fails_scalars_different_dtypes(self):
    self.assertFalse(
        type_analysis.is_binary_op_with_upcast_compatible_pair(
            computation_types.TensorType(tf.int32),
            computation_types.TensorType(tf.float32)))

  def test_passes_named_tuple_and_compatible_scalar(self):
    self.assertTrue(
        type_analysis.is_binary_op_with_upcast_compatible_pair(
            computation_types.NamedTupleType([
                ('a', computation_types.TensorType(tf.int32, [2, 2]))
            ]), computation_types.TensorType(tf.int32)))

  def test_fails_named_tuple_and_incompatible_scalar(self):
    self.assertFalse(
        type_analysis.is_binary_op_with_upcast_compatible_pair(
            computation_types.NamedTupleType([
                ('a', computation_types.TensorType(tf.int32, [2, 2]))
            ]), computation_types.TensorType(tf.float32)))

  def test_fails_compatible_scalar_and_named_tuple(self):
    self.assertFalse(
        type_analysis.is_binary_op_with_upcast_compatible_pair(
            computation_types.TensorType(tf.float32),
            computation_types.NamedTupleType([
                ('a', computation_types.TensorType(tf.int32, [2, 2]))
            ])))

  def test_fails_named_tuple_type_and_non_scalar_tensor(self):
    self.assertFalse(
        type_analysis.is_binary_op_with_upcast_compatible_pair(
            computation_types.NamedTupleType([
                ('a', computation_types.TensorType(tf.int32, [2, 2]))
            ]), computation_types.TensorType(tf.int32, [2])))


class TestCheckValidFederatedWeightedMeanArgumentTupleTypeTest(
    absltest.TestCase):

  def test_raises_type_error(self):
    type_analysis.check_valid_federated_weighted_mean_argument_tuple_type(
        computation_types.NamedTupleType([type_factory.at_clients(tf.float32)] *
                                         2))
    with self.assertRaises(TypeError):
      type_analysis.check_valid_federated_weighted_mean_argument_tuple_type(
          computation_types.NamedTupleType([type_factory.at_clients(tf.int32)] *
                                           2))


if __name__ == '__main__':
  absltest.main()
