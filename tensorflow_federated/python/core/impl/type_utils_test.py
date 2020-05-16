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

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory


class TypeUtilsTest(test.TestCase, parameterized.TestCase):

  def test_to_canonical_value_with_none(self):
    self.assertEqual(type_utils.to_canonical_value(None), None)

  def test_to_canonical_value_with_int(self):
    self.assertEqual(type_utils.to_canonical_value(1), 1)

  def test_to_canonical_value_with_float(self):
    self.assertEqual(type_utils.to_canonical_value(1.0), 1.0)

  def test_to_canonical_value_with_bool(self):
    self.assertEqual(type_utils.to_canonical_value(True), True)
    self.assertEqual(type_utils.to_canonical_value(False), False)

  def test_to_canonical_value_with_string(self):
    self.assertEqual(type_utils.to_canonical_value('a'), 'a')

  def test_to_canonical_value_with_list_of_ints(self):
    self.assertEqual(type_utils.to_canonical_value([1, 2, 3]), [1, 2, 3])

  def test_to_canonical_value_with_list_of_floats(self):
    self.assertEqual(
        type_utils.to_canonical_value([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

  def test_to_canonical_value_with_list_of_bools(self):
    self.assertEqual(
        type_utils.to_canonical_value([True, False]), [True, False])

  def test_to_canonical_value_with_list_of_strings(self):
    self.assertEqual(
        type_utils.to_canonical_value(['a', 'b', 'c']), ['a', 'b', 'c'])

  def test_to_canonical_value_with_list_of_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value([{
            'a': 1,
            'b': 0.1,
        }]), [anonymous_tuple.AnonymousTuple([
            ('a', 1),
            ('b', 0.1),
        ])])

  def test_to_canonical_value_with_list_of_ordered_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value(
            [collections.OrderedDict([
                ('a', 1),
                ('b', 0.1),
            ])]), [anonymous_tuple.AnonymousTuple([
                ('a', 1),
                ('b', 0.1),
            ])])

  def test_to_canonical_value_with_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value({
            'a': 1,
            'b': 0.1,
        }), anonymous_tuple.AnonymousTuple([
            ('a', 1),
            ('b', 0.1),
        ]))
    self.assertEqual(
        type_utils.to_canonical_value({
            'b': 0.1,
            'a': 1,
        }), anonymous_tuple.AnonymousTuple([
            ('a', 1),
            ('b', 0.1),
        ]))

  def test_to_canonical_value_with_ordered_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value(
            collections.OrderedDict([
                ('a', 1),
                ('b', 0.1),
            ])), anonymous_tuple.AnonymousTuple([
                ('a', 1),
                ('b', 0.1),
            ]))
    self.assertEqual(
        type_utils.to_canonical_value(
            collections.OrderedDict([
                ('b', 0.1),
                ('a', 1),
            ])), anonymous_tuple.AnonymousTuple([
                ('b', 0.1),
                ('a', 1),
            ]))

  def test_get_named_tuple_element_type(self):
    type_spec = [('a', tf.int32), ('b', tf.bool)]
    self.assertEqual(
        str(type_utils.get_named_tuple_element_type(type_spec, 'a')), 'int32')
    self.assertEqual(
        str(type_utils.get_named_tuple_element_type(type_spec, 'b')), 'bool')
    with self.assertRaises(ValueError):
      type_utils.get_named_tuple_element_type(type_spec, 'c')
    with self.assertRaises(TypeError):
      type_utils.get_named_tuple_element_type(tf.int32, 'a')
    with self.assertRaises(TypeError):
      type_utils.get_named_tuple_element_type(type_spec, 10)

  # pylint: disable=g-long-lambda,g-complex-comprehension
  @parameterized.parameters(*[
      computation_types.to_type(spec) for spec in ((
          lambda t, u: [
              # In constructing test cases, occurrences of 't' in all
              # expressions below are replaced with an abstract type 'T'.
              tf.int32,
              computation_types.FunctionType(tf.int32, tf.int32),
              computation_types.FunctionType(None, tf.int32),
              computation_types.FunctionType(t, t),
              [[computation_types.FunctionType(t, t), tf.bool]],
              computation_types.FunctionType(
                  computation_types.FunctionType(None, t), t),
              computation_types.FunctionType((computation_types.SequenceType(
                  t), computation_types.FunctionType((t, t), t)), t),
              computation_types.FunctionType(
                  computation_types.SequenceType(t), tf.int32),
              computation_types.FunctionType(
                  None, computation_types.FunctionType(t, t)),
              # In the test cases below, in addition to the 't' replacement
              # above, all occurrences of 'u' are replaced with an abstract type
              # 'U'.
              computation_types.FunctionType(
                  [t, computation_types.FunctionType(u, u), u], [t, u])
          ])(computation_types.AbstractType('T'),
             computation_types.AbstractType('U')))
  ])
  # pylint: enable=g-long-lambda,g-complex-comprehension
  def test_check_abstract_types_are_bound_valid_cases(self, type_spec):
    type_analysis.check_well_formed(type_spec)
    type_utils.check_all_abstract_types_are_bound(type_spec)

  # pylint: disable=g-long-lambda,g-complex-comprehension
  @parameterized.parameters(*[
      computation_types.to_type(spec) for spec in ((
          lambda t, u: [
              # In constructing test cases, similarly to the above, occurrences
              # of 't' and 'u' in all expressions below are replaced with
              # abstract types 'T' and 'U'.
              t,
              computation_types.FunctionType(tf.int32, t),
              computation_types.FunctionType(None, t),
              computation_types.FunctionType(t, u)
          ])(computation_types.AbstractType('T'),
             computation_types.AbstractType('U')))
  ])
  # pylint: enable=g-long-lambda,g-complex-comprehension
  def test_check_abstract_types_are_bound_invalid_cases(self, type_spec):
    self.assertRaises(TypeError, type_utils.check_all_abstract_types_are_bound,
                      type_spec)

  @parameterized.parameters(tf.int32, ([tf.int32, tf.int32],),
                            computation_types.FederatedType(
                                tf.int32, placements.CLIENTS),
                            ([tf.complex128, tf.float32, tf.float64],))
  def test_is_sum_compatible_positive_examples(self, type_spec):
    self.assertTrue(type_utils.is_sum_compatible(type_spec))

  @parameterized.parameters(tf.bool, tf.string, ([tf.int32, tf.bool],),
                            computation_types.SequenceType(tf.int32),
                            computation_types.PlacementType(),
                            computation_types.FunctionType(tf.int32, tf.int32),
                            computation_types.AbstractType('T'))
  def test_is_sum_compatible_negative_examples(self, type_spec):
    self.assertFalse(type_utils.is_sum_compatible(type_spec))

  @parameterized.parameters(tf.float32, tf.float64, ([('x', tf.float32),
                                                      ('y', tf.float64)],),
                            computation_types.FederatedType(
                                tf.float32, placements.CLIENTS))
  def test_is_average_compatible_true(self, type_spec):
    self.assertTrue(type_utils.is_average_compatible(type_spec))

  @parameterized.parameters(tf.int32, tf.int64,
                            computation_types.SequenceType(tf.float32))
  def test_is_average_compatible_false(self, type_spec):
    self.assertFalse(type_utils.is_average_compatible(type_spec))

  def test_is_assignable_from_with_tensor_type_and_invalid_type(self):
    t = computation_types.TensorType(tf.int32, [10])
    self.assertRaises(TypeError, type_utils.is_assignable_from, t, True)
    self.assertRaises(TypeError, type_utils.is_assignable_from, t, 10)

  def test_is_assignable_from_with_tensor_type_and_tensor_type(self):
    t = computation_types.TensorType(tf.int32, [10])
    self.assertFalse(
        type_utils.is_assignable_from(t,
                                      computation_types.TensorType(tf.int32)))
    self.assertFalse(
        type_utils.is_assignable_from(
            t, computation_types.TensorType(tf.int32, [5])))
    self.assertFalse(
        type_utils.is_assignable_from(
            t, computation_types.TensorType(tf.int32, [10, 10])))
    self.assertTrue(
        type_utils.is_assignable_from(
            t, computation_types.TensorType(tf.int32, 10)))

  def test_is_assignable_from_with_tensor_type_with_undefined_dims(self):
    t1 = computation_types.TensorType(tf.int32, [None])
    t2 = computation_types.TensorType(tf.int32, [10])
    self.assertTrue(type_utils.is_assignable_from(t1, t2))
    self.assertFalse(type_utils.is_assignable_from(t2, t1))

  def test_is_assignable_from_with_named_tuple_type(self):
    t1 = computation_types.NamedTupleType([tf.int32, ('a', tf.bool)])
    t2 = computation_types.NamedTupleType([tf.int32, ('a', tf.bool)])
    t3 = computation_types.NamedTupleType([tf.int32, ('b', tf.bool)])
    t4 = computation_types.NamedTupleType([tf.int32, ('a', tf.string)])
    t5 = computation_types.NamedTupleType([tf.int32])
    t6 = computation_types.NamedTupleType([tf.int32, tf.bool])
    self.assertTrue(type_utils.is_assignable_from(t1, t2))
    self.assertFalse(type_utils.is_assignable_from(t1, t3))
    self.assertFalse(type_utils.is_assignable_from(t1, t4))
    self.assertFalse(type_utils.is_assignable_from(t1, t5))
    self.assertTrue(type_utils.is_assignable_from(t1, t6))
    self.assertFalse(type_utils.is_assignable_from(t6, t1))

  def test_is_assignable_from_with_sequence_type(self):
    self.assertTrue(
        type_utils.is_assignable_from(
            computation_types.SequenceType(tf.int32),
            computation_types.SequenceType(tf.int32)))
    self.assertFalse(
        type_utils.is_assignable_from(
            computation_types.SequenceType(tf.int32),
            computation_types.SequenceType(tf.bool)))

  def test_is_assignable_from_with_function_type(self):
    t1 = computation_types.FunctionType(tf.int32, tf.bool)
    t2 = computation_types.FunctionType(tf.int32, tf.bool)
    t3 = computation_types.FunctionType(tf.int32, tf.int32)
    t4 = computation_types.TensorType(tf.int32)
    self.assertTrue(type_utils.is_assignable_from(t1, t1))
    self.assertTrue(type_utils.is_assignable_from(t1, t2))
    self.assertFalse(type_utils.is_assignable_from(t1, t3))
    self.assertFalse(type_utils.is_assignable_from(t1, t4))

  def test_is_assignable_from_with_abstract_type(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.AbstractType('T2')
    self.assertRaises(TypeError, type_utils.is_assignable_from, t1, t2)

  def test_is_assignable_from_with_placement_type(self):
    t1 = computation_types.PlacementType()
    t2 = computation_types.PlacementType()
    self.assertTrue(type_utils.is_assignable_from(t1, t1))
    self.assertTrue(type_utils.is_assignable_from(t1, t2))

  def test_is_assignable_from_with_federated_type(self):
    t1 = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    self.assertTrue(type_utils.is_assignable_from(t1, t1))
    t2 = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    self.assertTrue(type_utils.is_assignable_from(t1, t2))
    self.assertTrue(type_utils.is_assignable_from(t2, t2))
    self.assertFalse(type_utils.is_assignable_from(t2, t1))
    t3 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [10]), placements.CLIENTS)
    t4 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [None]), placements.CLIENTS)
    self.assertTrue(type_utils.is_assignable_from(t4, t3))
    self.assertFalse(type_utils.is_assignable_from(t3, t4))
    t5 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [10]), placements.SERVER)
    self.assertFalse(type_utils.is_assignable_from(t3, t5))
    self.assertFalse(type_utils.is_assignable_from(t5, t3))
    t6 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [10]),
        placements.CLIENTS,
        all_equal=True)
    self.assertTrue(type_utils.is_assignable_from(t3, t6))
    self.assertTrue(type_utils.is_assignable_from(t4, t6))
    self.assertFalse(type_utils.is_assignable_from(t6, t3))
    self.assertFalse(type_utils.is_assignable_from(t6, t4))

  def test_are_equivalent_types(self):
    t1 = computation_types.TensorType(tf.int32, [None])
    t2 = computation_types.TensorType(tf.int32, [10])
    t3 = computation_types.TensorType(tf.int32, [10])
    self.assertTrue(type_utils.are_equivalent_types(t1, t1))
    self.assertTrue(type_utils.are_equivalent_types(t2, t3))
    self.assertTrue(type_utils.are_equivalent_types(t3, t2))
    self.assertFalse(type_utils.are_equivalent_types(t1, t2))
    self.assertFalse(type_utils.are_equivalent_types(t2, t1))

  def test_check_type(self):
    type_utils.check_type(10, tf.int32)
    self.assertRaises(TypeError, type_utils.check_type, 10, tf.bool)

  def test_check_federated_type(self):
    type_spec = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                                False)
    type_utils.check_federated_type(type_spec, tf.int32, placements.CLIENTS,
                                    False)
    type_utils.check_federated_type(type_spec, tf.int32, None, None)
    type_utils.check_federated_type(type_spec, None, placements.CLIENTS, None)
    type_utils.check_federated_type(type_spec, None, None, False)
    self.assertRaises(TypeError, type_utils.check_federated_type, type_spec,
                      tf.bool, None, None)
    self.assertRaises(TypeError, type_utils.check_federated_type, type_spec,
                      None, placements.SERVER, None)
    self.assertRaises(TypeError, type_utils.check_federated_type, type_spec,
                      None, None, True)


class IsStructureOfIntegersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', tf.int32),
      ('ints', ([tf.int32, tf.int32],)),
      ('federated_int_at_clients',
       computation_types.FederatedType(tf.int32, placements.CLIENTS)),
  )
  def test_returns_true(self, type_spec):
    self.assertTrue(type_utils.is_structure_of_integers(type_spec))

  @parameterized.named_parameters(
      ('bool', tf.bool),
      ('string', tf.string),
      ('int_and_bool', ([tf.int32, tf.bool],)),
      ('sequence_of_ints', computation_types.SequenceType(tf.int32)),
      ('placement', computation_types.PlacementType()),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('abstract', computation_types.AbstractType('T')),
  )
  def test_returns_false(self, type_spec):
    self.assertFalse(type_utils.is_structure_of_integers(type_spec))


class IsValidBitwidthTypeForValueType(parameterized.TestCase):

  @parameterized.named_parameters(
      ('single int', tf.int32, tf.int32),
      ('tuple', (tf.int32, tf.int32), (tf.int32, tf.int32)),
      ('named tuple', computation_types.NamedTupleType(
          ['x', tf.int32]), computation_types.NamedTupleType(['x', tf.int32])),
      ('single int for complex tensor', tf.int32,
       tf.TensorSpec([5, 97, 204], dtype=tf.int32)),
      ('different kinds of ints', tf.int32, tf.int8),
  )
  def test_returns_true(self, bitwidth_type, value_type):
    self.assertTrue(
        type_utils.is_valid_bitwidth_type_for_value_type(
            bitwidth_type, value_type))

  @parameterized.named_parameters(
      ('single int_for_tuple', tf.int32, (tf.int32, tf.int32)),
      ('miscounted tuple', (tf.int32, tf.int32, tf.int32),
       (tf.int32, tf.int32)),
      ('miscounted tuple 2', (tf.int32, tf.int32),
       (tf.int32, tf.int32, tf.int32)),
      ('misnamed tuple', computation_types.NamedTupleType(
          ['x', tf.int32]), computation_types.NamedTupleType(['y', tf.int32])),
  )
  def test_returns_false(self, bitwidth_type, value_type):
    self.assertFalse(
        type_utils.is_valid_bitwidth_type_for_value_type(
            bitwidth_type, value_type))


class IsAnonTupleWithPyContainerTest(test.TestCase):

  def test_returns_true(self):
    value = anonymous_tuple.AnonymousTuple([('a', 0.0)])
    type_spec = computation_types.NamedTupleTypeWithPyContainerType(
        [('a', tf.float32)], dict)
    self.assertTrue(
        type_utils.is_anon_tuple_with_py_container(value, type_spec))

  def test_returns_false_with_none_value(self):
    value = None
    type_spec = computation_types.NamedTupleTypeWithPyContainerType(
        [('a', tf.float32)], dict)
    self.assertFalse(
        type_utils.is_anon_tuple_with_py_container(value, type_spec))

  def test_returns_false_with_named_tuple_type_spec(self):
    value = anonymous_tuple.AnonymousTuple([('a', 0.0)])
    type_spec = computation_types.NamedTupleType([('a', tf.float32)])
    self.assertFalse(
        type_utils.is_anon_tuple_with_py_container(value, type_spec))


class IsConcreteInstanceOf(test.TestCase):

  def test_is_concrete_instance_of_raises_with_int_first_argument(self):
    with self.assertRaises(TypeError):
      type_utils.is_concrete_instance_of(1, computation_types.to_type(tf.int32))

  def test_is_concrete_instance_of_raises_with_int_second_argument(self):
    with self.assertRaises(TypeError):
      type_utils.is_concrete_instance_of(computation_types.to_type(tf.int32), 1)

  def test_is_concrete_instance_of_raises_different_structures(self):
    with self.assertRaises(TypeError):
      type_utils.is_concrete_instance_of(
          computation_types.to_type(tf.int32),
          computation_types.to_type([tf.int32]))

  def test_is_concrete_instance_of_raises_with_abstract_type_as_first_arg(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.TensorType(tf.int32)
    with self.assertRaises(TypeError):
      type_utils.is_concrete_instance_of(t1, t2)

  def test_is_concrete_instance_of_with_single_abstract_type_and_tensor_type(
      self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.TensorType(tf.int32)
    self.assertTrue(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_raises_with_abstract_type_in_second_argument(
      self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.AbstractType('T2')
    with self.assertRaises(TypeError):
      type_utils.is_concrete_instance_of(t2, t1)

  def test_is_concrete_instance_of_with_single_abstract_type_and_tuple_type(
      self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.NamedTupleType([tf.int32])
    self.assertTrue(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_raises_with_conflicting_names(self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([('a', tf.int32), ('b', tf.int32)])
    with self.assertRaises(TypeError):
      type_utils.is_concrete_instance_of(t2, t1)

  def test_is_concrete_instance_of_raises_with_different_lengths(self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([tf.int32])
    with self.assertRaises(TypeError):
      type_utils.is_concrete_instance_of(t2, t1)

  def test_is_concrete_instance_of_succeeds_under_tuple(self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([
        computation_types.TensorType(tf.int32),
        computation_types.TensorType(tf.int32)
    ])
    self.assertTrue(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_fails_under_tuple_conflicting_concrete_types(
      self):
    t1 = computation_types.NamedTupleType(
        [computation_types.AbstractType('T1')] * 2)
    t2 = computation_types.NamedTupleType([
        computation_types.TensorType(tf.int32),
        computation_types.TensorType(tf.float32)
    ])
    self.assertFalse(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_succeeds_abstract_type_under_sequence_type(
      self):
    t1 = computation_types.SequenceType(computation_types.AbstractType('T'))
    t2 = computation_types.SequenceType(tf.int32)
    self.assertTrue(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_fails_conflicting_concrete_types_under_sequence(
      self):
    t1 = computation_types.SequenceType([computation_types.AbstractType('T')] *
                                        2)
    t2 = computation_types.SequenceType([tf.int32, tf.float32])
    self.assertFalse(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_succeeds_single_function_type(self):
    t1 = computation_types.FunctionType(*[computation_types.AbstractType('T')] *
                                        2)
    t2 = computation_types.FunctionType(tf.int32, tf.int32)
    self.assertTrue(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_succeeds_function_different_parameter_and_return_types(
      self):
    t1 = computation_types.FunctionType(
        computation_types.AbstractType('T'),
        computation_types.AbstractType('U'))
    t2 = computation_types.FunctionType(tf.int32, tf.float32)
    self.assertTrue(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_fails_conflicting_binding_in_parameter_and_result(
      self):
    t1 = computation_types.FunctionType(*[computation_types.AbstractType('T')] *
                                        2)
    t2 = computation_types.FunctionType(tf.int32, tf.float32)
    self.assertFalse(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_abstract_federated_types_succeeds(self):
    t1 = computation_types.FederatedType(
        [computation_types.AbstractType('T1')] * 2,
        placements.CLIENTS,
        all_equal=True)
    t2 = computation_types.FederatedType(
        [tf.int32] * 2, placements.CLIENTS, all_equal=True)
    self.assertTrue(type_utils.is_concrete_instance_of(t2, t1))

  def test_is_concrete_instance_of_abstract_fails_on_different_federated_placements(
      self):
    t1 = computation_types.FederatedType(
        [computation_types.AbstractType('T1')] * 2,
        placements.CLIENTS,
        all_equal=True)
    t2 = computation_types.FederatedType(
        [tf.int32] * 2, placements.SERVER, all_equal=True)
    self.assertFalse(type_utils.is_concrete_instance_of(t2, t1))

  def test_abstract_can_be_concretized_abstract_fails_on_different_federated_all_equal_bits(
      self):
    t1 = computation_types.FederatedType(
        [computation_types.AbstractType('T1')] * 2,
        placements.CLIENTS,
        all_equal=True)
    t2 = computation_types.FederatedType(
        [tf.int32] * 2, placements.SERVER, all_equal=True)
    self.assertFalse(type_utils.is_concrete_instance_of(t2, t1))


def _convert_tensor_to_float(type_spec):
  if isinstance(type_spec, computation_types.TensorType):
    return computation_types.TensorType(tf.float32, shape=type_spec.shape), True
  return type_spec, False


def _convert_abstract_type_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.AbstractType):
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_placement_type_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.PlacementType):
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_function_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.FunctionType):
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_federated_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.FederatedType):
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_sequence_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.SequenceType):
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_tuple_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.NamedTupleType):
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


class IsBinaryOpWithUpcastCompatibleTest(test.TestCase):

  def test_passes_on_none(self):
    self.assertTrue(
        type_utils.is_binary_op_with_upcast_compatible_pair(None, None))

  def test_passes_empty_tuples(self):
    self.assertTrue(type_utils.is_binary_op_with_upcast_compatible_pair([], []))

  def test_fails_scalars_different_dtypes(self):
    self.assertFalse(
        type_utils.is_binary_op_with_upcast_compatible_pair(
            tf.int32, tf.float32))

  def test_passes_named_tuple_and_compatible_scalar(self):
    self.assertTrue(
        type_utils.is_binary_op_with_upcast_compatible_pair(
            [('a', computation_types.TensorType(tf.int32, [2, 2]))], tf.int32))

  def test_fails_named_tuple_and_incompatible_scalar(self):
    self.assertFalse(
        type_utils.is_binary_op_with_upcast_compatible_pair(
            [('a', computation_types.TensorType(tf.int32, [2, 2]))],
            tf.float32))

  def test_fails_compatible_scalar_and_named_tuple(self):
    self.assertFalse(
        type_utils.is_binary_op_with_upcast_compatible_pair(
            tf.float32,
            [('a', computation_types.TensorType(tf.int32, [2, 2]))]))

  def test_fails_named_tuple_type_and_non_scalar_tensor(self):
    self.assertFalse(
        type_utils.is_binary_op_with_upcast_compatible_pair(
            [('a', computation_types.TensorType(tf.int32, [2, 2]))],
            computation_types.TensorType(tf.int32, [2])))

  def test_check_equivalent_types(self):
    type_utils.check_equivalent_types(tf.int32, tf.int32)
    with self.assertRaises(TypeError):
      type_utils.check_equivalent_types(tf.int32, tf.bool)

  def check_valid_federated_weighted_mean_argument_tuple_type(self):
    type_utils.check_valid_federated_weighted_mean_argument_tuple_type(
        computation_types.to_type([type_factory.at_clients(tf.float32)] * 2))
    with self.assertRaises(TypeError):
      type_utils.check_valid_federated_weighted_mean_argument_tuple_type(
          computation_types.to_type([type_factory.at_clients(tf.int32)] * 2))


if __name__ == '__main__':
  tf.test.main()
