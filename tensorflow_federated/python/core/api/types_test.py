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
"""Tests for types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import types


class TypesTest(absltest.TestCase):

  def test_tensor_type_dtype_and_shape(self):
    t = types.TensorType(tf.int32, [10])
    self.assertEqual(t.dtype, tf.int32)
    self.assertEqual(t.shape, tf.TensorShape([10]))

  def test_tensor_type_is_assignable_from_invalid_type(self):
    t = types.TensorType(tf.int32, [10])
    self.assertRaises(TypeError, t.is_assignable_from, True)
    self.assertRaises(TypeError, t.is_assignable_from, 10)
    self.assertRaises(TypeError, t.is_assignable_from, tf.int32)

  def test_tensor_type_is_assignable_from_tensor_type(self):
    t = types.TensorType(tf.int32, [10])
    self.assertFalse(t.is_assignable_from(types.TensorType(tf.int32)))
    self.assertFalse(t.is_assignable_from(types.TensorType(tf.int32, [5])))
    self.assertFalse(t.is_assignable_from(types.TensorType(tf.int32, [10, 10])))
    self.assertTrue(t.is_assignable_from(types.TensorType(tf.int32, 10)))

  def test_tensor_type_is_assignable_with_undefined_dims(self):
    t1 = types.TensorType(tf.int32, [None])
    t2 = types.TensorType(tf.int32, [10])
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertFalse(t2.is_assignable_from(t1))

  def test_type_ordering(self):
    t1 = types.TensorType(tf.int32, [None])
    t2 = types.TensorType(tf.int32, [10])
    t3 = types.TensorType(tf.int32, [10])
    self.assertGreater(t1, t2)
    self.assertLess(t2, t1)
    self.assertNotEqual(t1, t2)
    self.assertNotEqual(t2, t1)
    self.assertEqual(t2, t2)
    self.assertEqual(t2, t3)
    self.assertEqual(t3, t2)
    self.assertLessEqual(t2, t1)
    self.assertLessEqual(t2, t3)
    self.assertLessEqual(t3, t2)
    self.assertGreaterEqual(t1, t2)
    self.assertGreaterEqual(t2, t3)
    self.assertGreaterEqual(t3, t2)

  def test_tensor_type_repr(self):
    self.assertEqual(repr(types.TensorType(tf.int32)), 'TensorType(tf.int32)')
    self.assertEqual(
        repr(types.TensorType(tf.int32, [10])), 'TensorType(tf.int32, [10])')
    self.assertEqual(
        repr(types.TensorType(tf.int32, [3, 5])),
        'TensorType(tf.int32, [3, 5])')

  def test_tensor_type_str(self):
    self.assertEqual(str(types.TensorType(tf.int32)), 'int32')
    self.assertEqual(str(types.TensorType(tf.int32, [10])), 'int32[10]')
    self.assertEqual(str(types.TensorType(tf.int32, [3, 5])), 'int32[3,5]')
    self.assertEqual(str(types.TensorType(tf.int32, [None])), 'int32[?]')
    self.assertEqual(
        str(types.TensorType(tf.int32, [None, None])), 'int32[?,?]')
    self.assertEqual(
        str(types.TensorType(tf.int32, [None, 10])), 'int32[?,10]')

  def test_named_tuple_type_repr(self):
    self.assertEqual(
        repr(types.NamedTupleType([tf.int32, ('a', tf.bool)])),
        'NamedTupleType([TensorType(tf.int32), (\'a\', TensorType(tf.bool))])')

  def test_named_tuple_type_str(self):
    self.assertEqual(str(types.NamedTupleType([tf.int32])), '<int32>')
    self.assertEqual(str(types.NamedTupleType([('a', tf.int32)])), '<a=int32>')
    self.assertEqual(str(types.NamedTupleType(('a', tf.int32))), '<a=int32>')
    self.assertEqual(
        str(types.NamedTupleType([tf.int32, tf.bool])),
        '<int32,bool>')
    self.assertEqual(
        str(types.NamedTupleType([('a', tf.int32), tf.float32])),
        '<a=int32,float32>')
    self.assertEqual(
        str(types.NamedTupleType([('a', tf.int32), ('b', tf.float32)])),
        '<a=int32,b=float32>')
    self.assertEqual(
        str(types.NamedTupleType([
            ('a', tf.int32),
            ('b', types.NamedTupleType([('x', tf.string), ('y', tf.bool)]))])),
        '<a=int32,b=<x=string,y=bool>>')

  def test_named_tuple_type_elements(self):
    self.assertEqual(
        repr(types.NamedTupleType([tf.int32, ('a', tf.bool)]).elements),
        '[(None, TensorType(tf.int32)), (\'a\', TensorType(tf.bool))]')

  def test_named_tuple_type_is_assignable_from(self):
    t1 = types.NamedTupleType([tf.int32, ('a', tf.bool)])
    t2 = types.NamedTupleType([tf.int32, ('a', tf.bool)])
    t3 = types.NamedTupleType([tf.int32, ('b', tf.bool)])
    t4 = types.NamedTupleType([tf.int32, ('a', tf.string)])
    t5 = types.NamedTupleType([tf.int32])
    t6 = types.NamedTupleType([tf.int32, tf.bool])
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertFalse(t1.is_assignable_from(t3))
    self.assertFalse(t1.is_assignable_from(t4))
    self.assertFalse(t1.is_assignable_from(t5))
    self.assertTrue(t6.is_assignable_from(t1))
    self.assertFalse(t1.is_assignable_from(t6))

  def test_sequence_type_repr(self):
    self.assertEqual(
        repr(types.SequenceType(tf.int32)),
        'SequenceType(TensorType(tf.int32))')
    self.assertEqual(
        repr(types.SequenceType((tf.int32, tf.bool))),
        'SequenceType(NamedTupleType([TensorType(tf.int32), '
        'TensorType(tf.bool)]))')

  def test_sequence_type_str(self):
    self.assertEqual(str(types.SequenceType(tf.int32)), 'int32*')
    self.assertEqual(
        str(types.SequenceType((tf.int32, tf.bool))), '<int32,bool>*')

  def test_sequence_type_element(self):
    self.assertEqual(str(types.SequenceType(tf.int32).element), 'int32')

  def test_sequence_type_is_assignable_from(self):
    self.assertTrue(
        types.SequenceType(tf.int32).is_assignable_from(
            types.SequenceType(tf.int32)))
    self.assertFalse(
        types.SequenceType(tf.int32).is_assignable_from(
            types.SequenceType(tf.bool)))

  def test_function_type_repr(self):
    self.assertEqual(
        repr(types.FunctionType(tf.int32, tf.bool)),
        'FunctionType(TensorType(tf.int32), TensorType(tf.bool))')
    self.assertEqual(
        repr(types.FunctionType(None, tf.bool)),
        'FunctionType(None, TensorType(tf.bool))')

  def test_function_type_str(self):
    self.assertEqual(
        str(types.FunctionType(tf.int32, tf.bool)), '(int32 -> bool)')
    self.assertEqual(
        str(types.FunctionType(None, tf.bool)), '( -> bool)')

  def test_function_type_parameter_and_result(self):
    t = types.FunctionType(tf.int32, tf.bool)
    self.assertEqual(str(t.parameter), 'int32')
    self.assertEqual(str(t.result), 'bool')

  def test_function_type_is_assignable_from(self):
    t1 = types.FunctionType(tf.int32, tf.bool)
    t2 = types.FunctionType(tf.int32, tf.bool)
    t3 = types.FunctionType(tf.int32, tf.int32)
    t4 = types.TensorType(tf.int32)
    self.assertTrue(t1.is_assignable_from(t1))
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertFalse(t1.is_assignable_from(t3))
    self.assertFalse(t1.is_assignable_from(t4))

  def test_abstract_type(self):
    t1 = types.AbstractType('T1')
    self.assertEqual(repr(t1), 'AbstractType(\'T1\')')
    self.assertEqual(str(t1), 'T1')
    self.assertEqual(t1.label, 'T1')
    self.assertRaises(TypeError, types.AbstractType, 10)
    t1_other = types.AbstractType('T1')
    t2 = types.AbstractType('T2')
    self.assertRaises(ValueError, t1.is_assignable_from, t1)
    self.assertRaises(ValueError, t1.is_assignable_from, t1_other)
    self.assertRaises(ValueError, t1.is_assignable_from, t2)

  def test_placement_type(self):
    t1 = types.PlacementType()
    self.assertEqual(repr(t1), 'PlacementType()')
    self.assertEqual(str(t1), 'placement')
    t2 = types.PlacementType()
    self.assertTrue(t1.is_assignable_from(t1))
    self.assertTrue(t1.is_assignable_from(t2))

  def test_federated_type(self):
    t1 = types.FederatedType(tf.int32, placements.CLIENTS)
    self.assertEqual(str(t1.member), 'int32')
    self.assertIs(t1.placement, placements.CLIENTS)
    self.assertFalse(t1.all_equal)
    self.assertEqual(
        repr(t1),
        'FederatedType(TensorType(tf.int32), '
        'PlacementLiteral(\'clients\'), False)')
    self.assertEqual(str(t1), '{int32}@CLIENTS')
    self.assertTrue(t1.is_assignable_from(t1))
    t2 = types.FederatedType(tf.int32, placements.CLIENTS, all_equal=True)
    self.assertEqual(str(t2), 'int32@CLIENTS')
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertTrue(t2.is_assignable_from(t2))
    self.assertFalse(t2.is_assignable_from(t1))
    t3 = types.FederatedType(
        types.TensorType(tf.int32, [10]), placements.CLIENTS)
    t4 = types.FederatedType(
        types.TensorType(tf.int32, [None]), placements.CLIENTS)
    self.assertTrue(t4.is_assignable_from(t3))
    self.assertFalse(t3.is_assignable_from(t4))
    t5 = types.FederatedType(
        types.TensorType(tf.int32, [10]), placements.SERVER)
    self.assertFalse(t3.is_assignable_from(t5))
    self.assertFalse(t5.is_assignable_from(t3))
    t6 = types.FederatedType(
        types.TensorType(tf.int32, [10]), placements.CLIENTS, all_equal=True)
    self.assertTrue(t3.is_assignable_from(t6))
    self.assertTrue(t4.is_assignable_from(t6))
    self.assertFalse(t6.is_assignable_from(t3))
    self.assertFalse(t6.is_assignable_from(t4))

  def test_to_type_with_tensor_type(self):
    s = types.TensorType(tf.int32)
    t = types.to_type(s)
    self.assertIsInstance(t, types.TensorType)
    self.assertEqual(str(t), 'int32')

  def test_to_type_with_tf_type(self):
    s = tf.int32
    t = types.to_type(s)
    self.assertIsInstance(t, types.TensorType)
    self.assertEqual(str(t), 'int32')

  def test_to_type_with_tf_type_and_shape(self):
    s = (tf.int32, [10])
    t = types.to_type(s)
    self.assertIsInstance(t, types.TensorType)
    self.assertEqual(str(t), 'int32[10]')

  def test_to_type_with_tf_type_and_shape_with_unknown_dimension(self):
    s = (tf.int32, [None])
    t = types.to_type(s)
    self.assertIsInstance(t, types.TensorType)
    self.assertEqual(str(t), 'int32[?]')

  def test_to_type_with_list_of_tf_types(self):
    s = [tf.int32, tf.bool]
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<int32,bool>')

  def test_to_type_with_tuple_of_tf_types(self):
    s = (tf.int32, tf.bool)
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<int32,bool>')

  def test_to_type_with_singleton_named_tf_type(self):
    s = ('a', tf.int32)
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<a=int32>')

  def test_to_type_with_list_of_named_tf_types(self):
    s = [('a', tf.int32), ('b', tf.bool)]
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_to_type_with_ordered_dict_of_tf_types(self):
    s = collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_to_type_with_nested_tuple_of_tf_types(self):
    s = (tf.int32, (tf.float32, tf.bool))
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<int32,<float32,bool>>')

  def test_to_type_with_nested_tuple_of_named_tf_types(self):
    s = (tf.int32, (('x', tf.float32), tf.bool))
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<int32,<x=float32,bool>>')

  def test_to_type_with_nested_tuple_of_named_nonscalar_tf_types(self):
    s = ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))
    t = types.to_type(s)
    self.assertIsInstance(t, types.NamedTupleType)
    self.assertEqual(str(t), '<int32[1],<x=float32[2],bool[3]>>')


if __name__ == '__main__':
  absltest.main()
