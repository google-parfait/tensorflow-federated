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
"""Tests for type_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test_utils as common_test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import test_utils
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


class TypeUtilsTest(common_test_utils.TffTestCase, parameterized.TestCase):

  def test_infer_type_with_none(self):
    self.assertEqual(type_utils.infer_type(None), None)

  def test_infer_type_with_tff_value(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                value_impl.ValueImpl(
                    computation_building_blocks.Reference('foo', tf.bool),
                    context_stack_impl.context_stack))), 'bool')

  def test_infer_type_with_scalar_int_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.constant(1))), 'int32')

  def test_infer_type_with_scalar_bool_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.constant(False))), 'bool')

  def test_infer_type_with_int_array_tensor(self):
    self.assertEqual(
        str(type_utils.infer_type(tf.constant([10, 20]))), 'int32[2]')

  def test_infer_type_with_scalar_int_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable(10))), 'int32')

  def test_infer_type_with_scalar_bool_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable(True))), 'bool')

  def test_infer_type_with_scalar_float_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable(0.5))), 'float32')

  def test_infer_type_with_scalar_int_array_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable([10]))), 'int32[1]')

  def test_infer_type_with_int_dataset(self):
    self.assertEqual(
        str(type_utils.infer_type(tf.data.Dataset.from_tensors(10))), 'int32*')

  def test_infer_type_with_dict_dataset(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                tf.data.Dataset.from_tensors({
                    'a': 10,
                    'b': 20,
                }))), '<a=int32,b=int32>*')
    self.assertEqual(
        str(
            type_utils.infer_type(
                tf.data.Dataset.from_tensors({
                    'b': 20,
                    'a': 10,
                }))), '<a=int32,b=int32>*')

  def test_infer_type_with_ordered_dict_dataset(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                tf.data.Dataset.from_tensors(
                    collections.OrderedDict([
                        ('b', 20),
                        ('a', 10),
                    ])))), '<b=int32,a=int32>*')

  def test_infer_type_with_int(self):
    self.assertEqual(str(type_utils.infer_type(10)), 'int32')

  def test_infer_type_with_float(self):
    self.assertEqual(str(type_utils.infer_type(0.5)), 'float32')

  def test_infer_type_with_bool(self):
    self.assertEqual(str(type_utils.infer_type(True)), 'bool')

  def test_infer_type_with_string(self):
    self.assertEqual(str(type_utils.infer_type('abc')), 'string')

  def test_infer_type_with_unicode_string(self):
    self.assertEqual(str(type_utils.infer_type(u'abc')), 'string')

  def test_infer_type_with_numpy_int_array(self):
    self.assertEqual(str(type_utils.infer_type(np.array([10, 20]))), 'int64[2]')

  def test_infer_type_with_numpy_nested_int_array(self):
    self.assertEqual(
        str(type_utils.infer_type(np.array([[10], [20]]))), 'int64[2,1]')

  def test_infer_type_with_numpy_float64_scalar(self):
    self.assertEqual(str(type_utils.infer_type(np.float64(1))), 'float64')

  def test_infer_type_with_int_list(self):
    self.assertEqual(
        str(type_utils.infer_type([1, 2, 3])), '<int32,int32,int32>')

  def test_infer_type_with_nested_float_list(self):
    self.assertEqual(
        str(type_utils.infer_type([[0.1], [0.2], [0.3]])),
        '<<float32>,<float32>,<float32>>')

  def test_infer_type_with_anonymous_tuple(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                anonymous_tuple.AnonymousTuple([
                    ('a', 10),
                    (None, False),
                ]))), '<a=int32,bool>')

  def test_infer_type_with_nested_anonymous_tuple(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                anonymous_tuple.AnonymousTuple([
                    ('a', 10),
                    (None,
                     anonymous_tuple.AnonymousTuple([
                         (None, True),
                         (None, 0.5),
                     ])),
                ]))), '<a=int32,<bool,float32>>')

  def test_infer_type_with_namedtuple(self):
    self.assertEqual(
        str(type_utils.infer_type(collections.namedtuple('_', 'y x')(1, True))),
        '<y=int32,x=bool>')

  def test_infer_type_with_dict(self):
    self.assertEqual(
        str(type_utils.infer_type({
            'a': 1,
            'b': 2.0,
        })), '<a=int32,b=float32>')
    self.assertEqual(
        str(type_utils.infer_type({
            'b': 2.0,
            'a': 1,
        })), '<a=int32,b=float32>')

  def test_infer_type_with_ordered_dict(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                collections.OrderedDict([('b', 2.0), ('a', 1)]))),
        '<b=float32,a=int32>')

  def test_infer_type_with_dataset_list(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                [tf.data.Dataset.from_tensors(x) for x in [1, True, [0.5]]])),
        '<int32*,bool*,float32[1]*>')

  def test_infer_type_with_nested_dataset_list_tuple(self):
    self.assertEqual(
        str(
            type_utils.infer_type(
                tuple([(tf.data.Dataset.from_tensors(x),)
                       for x in [1, True, [0.5]]]))),
        '<<int32*>,<bool*>,<float32[1]*>>')

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

  def test_tf_dtypes_and_shapes_to_type_with_int(self):
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(tf.int32, tf.TensorShape(
                []))), 'int32')

  def test_tf_dtypes_and_shapes_to_type_with_int_vector(self):
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(tf.int32, tf.TensorShape(
                [2]))), 'int32[2]')

  def test_tf_dtypes_and_shapes_to_type_with_dict(self):
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                {
                    'a': tf.int32,
                    'b': tf.bool,
                },
                {
                    'a': tf.TensorShape([]),
                    'b': tf.TensorShape([5]),
                },
            )), '<a=int32,b=bool[5]>')
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                {
                    'b': tf.bool,
                    'a': tf.int32,
                },
                {
                    'a': tf.TensorShape([]),
                    'b': tf.TensorShape([5]),
                },
            )), '<a=int32,b=bool[5]>')

  def test_tf_dtypes_and_shapes_to_type_with_ordered_dict(self):
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                collections.OrderedDict([('b', tf.int32), ('a', tf.bool)]),
                collections.OrderedDict([
                    ('b', tf.TensorShape([1])),
                    ('a', tf.TensorShape([])),
                ]))), '<b=int32[1],a=bool>')

  def test_tf_dtypes_and_shapes_to_type_with_tuple(self):
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                (tf.int32, tf.bool),
                (tf.TensorShape([1]), tf.TensorShape([2])))),
        '<int32[1],bool[2]>')

  def test_tf_dtypes_and_shapes_to_type_with_list(self):
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                [tf.int32, tf.bool],
                [tf.TensorShape([1]), tf.TensorShape([2])])),
        '<int32[1],bool[2]>')

  def test_tf_dtypes_and_shapes_to_type_with_list_of_lists(self):
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                [[tf.int32, tf.int32], [tf.bool, tf.bool]], [
                    [tf.TensorShape([1]),
                     tf.TensorShape([2])],
                    [tf.TensorShape([]), tf.TensorShape([])],
                ])), '<<int32[1],int32[2]>,<bool,bool>>')

  def test_tf_dtypes_and_shapes_to_type_with_namedtuple(self):
    foo = collections.namedtuple('_', 'y x')
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                foo(x=tf.int32, y=tf.bool),
                foo(x=tf.TensorShape([1]), y=tf.TensorShape([2])))),
        '<y=bool[2],x=int32[1]>')

  def test_tf_dtypes_and_shapes_to_type_with_three_level_nesting(self):
    foo = collections.namedtuple('_', 'y x')
    self.assertEqual(
        str(
            type_utils.tf_dtypes_and_shapes_to_type(
                foo(x=[tf.int32, {
                    'bar': tf.float32
                }], y=tf.bool),
                foo(x=[tf.TensorShape([1]), {
                    'bar': tf.TensorShape([2])
                }],
                    y=tf.TensorShape([3])))),
        '<y=bool[3],x=<int32[1],<bar=float32[2]>>>')

  def test_type_to_tf_dtypes_and_shapes_with_int_scalar(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(tf.int32)
    test_utils.assert_nested_struct_eq(dtypes, tf.int32)
    test_utils.assert_nested_struct_eq(shapes, tf.TensorShape([]))

  def test_type_to_tf_dtypes_and_shapes_with_int_vector(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes((tf.int32, [10]))
    test_utils.assert_nested_struct_eq(dtypes, tf.int32)
    test_utils.assert_nested_struct_eq(shapes, tf.TensorShape([10]))

  def test_type_to_tf_dtypes_and_shapes_with_tensor_triple(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(
        [('a', (tf.int32, [5])), ('b', tf.bool), ('c', (tf.float32, [3]))])
    test_utils.assert_nested_struct_eq(dtypes, {
        'a': tf.int32,
        'b': tf.bool,
        'c': tf.float32
    })
    test_utils.assert_nested_struct_eq(shapes, {
        'a': tf.TensorShape([5]),
        'b': tf.TensorShape([]),
        'c': tf.TensorShape([3])
    })

  def test_type_to_tf_dtypes_and_shapes_with_two_level_tuple(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(
        [('a', tf.bool), ('b', [('c', tf.float32), ('d', (tf.int32, [20]))])])
    test_utils.assert_nested_struct_eq(dtypes, {
        'a': tf.bool,
        'b': {
            'c': tf.float32,
            'd': tf.int32
        }
    })
    test_utils.assert_nested_struct_eq(
        shapes, {
            'a': tf.TensorShape([]),
            'b': {
                'c': tf.TensorShape([]),
                'd': tf.TensorShape([20])
            }
        })

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

  # pylint: disable=g-long-lambda
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
  # pylint: enable=g-long-lambda
  def test_check_abstract_types_are_bound_valid_cases(self, type_spec):
    type_utils.check_well_formed(type_spec)
    type_utils.check_all_abstract_types_are_bound(type_spec)

  # pylint: disable=g-long-lambda
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
  # pylint: enable=g-long-lambda
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

  def test_check_federated_value_placement(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def _(x):
      type_utils.check_federated_value_placement(x, placements.CLIENTS)
      with self.assertRaises(TypeError):
        type_utils.check_federated_value_placement(x, placements.SERVER)
      return x

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

  def test_well_formed_check_fails_bad_types(self):
    nest_federated = computation_types.FederatedType(
        computation_types.FederatedType(tf.int32, placements.CLIENTS),
        placements.CLIENTS)
    with self.assertRaisesRegexp(TypeError,
                                 'A {int32}@CLIENTS has been encountered'):
      type_utils.check_well_formed(nest_federated)
    sequence_in_sequence = computation_types.SequenceType(
        computation_types.SequenceType([tf.int32]))
    with self.assertRaisesRegexp(TypeError,
                                 r'A <int32>\* has been encountered'):
      type_utils.check_well_formed(sequence_in_sequence)
    federated_function = computation_types.FederatedType(
        computation_types.FunctionType(tf.int32, tf.int32), placements.CLIENTS)
    with self.assertRaisesRegexp(TypeError,
                                 r'A \(int32 -> int32\) has been encountered'):
      type_utils.check_well_formed(federated_function)
    tuple_federated_function = computation_types.NamedTupleType(
        [federated_function])
    with self.assertRaisesRegexp(TypeError,
                                 r'A \(int32 -> int32\) has been encountered'):
      type_utils.check_well_formed(tuple_federated_function)

  def test_extra_well_formed_check_nested_types(self):
    nest_federated = computation_types.FederatedType(
        computation_types.FederatedType(tf.int32, placements.CLIENTS),
        placements.CLIENTS)
    tuple_federated_nest = computation_types.NamedTupleType([nest_federated])
    with self.assertRaisesRegexp(TypeError,
                                 r'A {int32}@CLIENTS has been encountered'):
      type_utils.check_well_formed(tuple_federated_nest)
    federated_inner = computation_types.FederatedType(tf.int32,
                                                      placements.CLIENTS)
    tuple_on_federated = computation_types.NamedTupleType([federated_inner])
    federated_outer = computation_types.FederatedType(tuple_on_federated,
                                                      placements.CLIENTS)
    with self.assertRaisesRegexp(TypeError,
                                 r'A {int32}@CLIENTS has been encountered'):
      type_utils.check_well_formed(federated_outer)
    multiple_nest = computation_types.NamedTupleType(
        [computation_types.NamedTupleType([federated_outer])])
    with self.assertRaisesRegexp(TypeError,
                                 r'A {int32}@CLIENTS has been encountered'):
      type_utils.check_well_formed(multiple_nest)
    sequence_of_federated = computation_types.SequenceType(federated_inner)
    with self.assertRaisesRegexp(TypeError,
                                 r'A {int32}@CLIENTS has been encountered'):
      type_utils.check_well_formed(sequence_of_federated)

  def test_preorder_call_count(self):

    class Counter(object):
      k = 0

    def _count_hits(given_type, arg):
      del given_type
      Counter.k += 1
      return arg

    sequence = computation_types.SequenceType(
        computation_types.SequenceType(
            computation_types.SequenceType(tf.int32)))
    type_utils.preorder_call(sequence, _count_hits, None)
    self.assertEqual(Counter.k, 4)
    federated = computation_types.FederatedType(
        computation_types.FederatedType(
            computation_types.FederatedType(tf.int32, placements.CLIENTS),
            placements.CLIENTS), placements.CLIENTS)
    type_utils.preorder_call(federated, _count_hits, None)
    self.assertEqual(Counter.k, 8)
    function = computation_types.FunctionType(
        computation_types.FunctionType(tf.int32, tf.int32), tf.int32)
    type_utils.preorder_call(function, _count_hits, None)
    self.assertEqual(Counter.k, 13)
    abstract = computation_types.AbstractType('T')
    type_utils.preorder_call(abstract, _count_hits, None)
    self.assertEqual(Counter.k, 14)
    placement = computation_types.PlacementType()
    type_utils.preorder_call(placement, _count_hits, None)
    self.assertEqual(Counter.k, 15)
    namedtuple = computation_types.NamedTupleType([
        tf.int32, tf.bool,
        computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ])
    type_utils.preorder_call(namedtuple, _count_hits, None)
    self.assertEqual(Counter.k, 20)
    nested_namedtuple = computation_types.NamedTupleType([namedtuple])
    type_utils.preorder_call(nested_namedtuple, _count_hits, None)
    self.assertEqual(Counter.k, 26)

  def test_well_formed_check_succeeds_good_types(self):
    federated = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    self.assertTrue(type_utils.check_well_formed(federated))
    tensor = computation_types.TensorType(tf.int32)
    self.assertTrue(type_utils.check_well_formed(tensor))
    namedtuple = computation_types.NamedTupleType(
        [tf.int32,
         computation_types.NamedTupleType([tf.int32, tf.int32])])
    self.assertTrue(type_utils.check_well_formed(namedtuple))
    sequence = computation_types.SequenceType(tf.int32)
    self.assertTrue(type_utils.check_well_formed(sequence))
    func = computation_types.FunctionType(tf.int32, tf.int32)
    self.assertTrue(type_utils.check_well_formed(func))
    abstract = computation_types.AbstractType('T')
    self.assertTrue(type_utils.check_well_formed(abstract))
    placement = computation_types.PlacementType()
    self.assertTrue(type_utils.check_well_formed(placement))


if __name__ == '__main__':
  tf.test.main()
