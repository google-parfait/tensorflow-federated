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
"""Tests for computation_types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements


class TensorTypeTest(absltest.TestCase):

  def test_dtype_and_shape(self):
    t = computation_types.TensorType(tf.int32, [10])
    self.assertEqual(t.dtype, tf.int32)
    self.assertEqual(t.shape, tf.TensorShape([10]))

  def test_repr(self):
    self.assertEqual(
        repr(computation_types.TensorType(tf.int32)), 'TensorType(tf.int32)')
    self.assertEqual(
        repr(computation_types.TensorType(tf.int32, [10])),
        'TensorType(tf.int32, [10])')
    self.assertEqual(
        repr(computation_types.TensorType(tf.int32, [3, 5])),
        'TensorType(tf.int32, [3, 5])')

  def test_str(self):
    self.assertEqual(str(computation_types.TensorType(tf.int32)), 'int32')
    self.assertEqual(
        str(computation_types.TensorType(tf.int32, [10])), 'int32[10]')
    self.assertEqual(
        str(computation_types.TensorType(tf.int32, [3, 5])), 'int32[3,5]')
    self.assertEqual(
        str(computation_types.TensorType(tf.int32, [None])), 'int32[?]')
    self.assertEqual(
        str(computation_types.TensorType(tf.int32, [None, None])), 'int32[?,?]')
    self.assertEqual(
        str(computation_types.TensorType(tf.int32, [None, 10])), 'int32[?,10]')

  def test_equality(self):
    foo = computation_types.TensorType(tf.int32, [10])
    foo_eq = computation_types.TensorType(tf.int32, [10])
    foo_ne_attr1 = computation_types.TensorType(tf.int64, [10])
    foo_ne_attr2 = computation_types.TensorType(tf.int32, [None])
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_attr1))
    self.assertFalse(foo_ne_attr1.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_attr2))
    self.assertFalse(foo_ne_attr2.__eq__(foo))
    self.assertTrue(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_attr1))
    self.assertTrue(foo_ne_attr1.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_attr2))
    self.assertTrue(foo_ne_attr2.__ne__(foo))


class NamedTupleTypeTest(absltest.TestCase):

  def test_repr(self):
    self.assertEqual(
        repr(computation_types.NamedTupleType([tf.int32, ('a', tf.bool)])),
        'NamedTupleType([TensorType(tf.int32), (\'a\', TensorType(tf.bool))])')

  def test_str(self):
    self.assertEqual(
        str(computation_types.NamedTupleType([tf.int32])), '<int32>')
    self.assertEqual(
        str(computation_types.NamedTupleType([('a', tf.int32)])), '<a=int32>')
    self.assertEqual(
        str(computation_types.NamedTupleType(('a', tf.int32))), '<a=int32>')
    self.assertEqual(
        str(computation_types.NamedTupleType([tf.int32, tf.bool])),
        '<int32,bool>')
    self.assertEqual(
        str(computation_types.NamedTupleType([('a', tf.int32), tf.float32])),
        '<a=int32,float32>')
    self.assertEqual(
        str(
            computation_types.NamedTupleType([('a', tf.int32),
                                              ('b', tf.float32)])),
        '<a=int32,b=float32>')
    self.assertEqual(
        str(
            computation_types.NamedTupleType([
                ('a', tf.int32),
                ('b',
                 computation_types.NamedTupleType([('x', tf.string),
                                                   ('y', tf.bool)]))
            ])), '<a=int32,b=<x=string,y=bool>>')

  def test_elements(self):
    self.assertEqual(
        repr(
            anonymous_tuple.to_elements(
                computation_types.NamedTupleType([tf.int32, ('a', tf.bool)]))),
        '[(None, TensorType(tf.int32)), (\'a\', TensorType(tf.bool))]')

  def test_with_none_keys(self):
    self.assertEqual(
        str(computation_types.NamedTupleType([(None, tf.int32)])), '<int32>')

  def test_equality_unnamed(self):
    foo = computation_types.NamedTupleType([tf.int32, tf.bool])
    foo_eq = computation_types.NamedTupleType([tf.int32, tf.bool])
    foo_ne_order = computation_types.NamedTupleType([tf.bool, tf.int32])
    foo_ne_value = computation_types.NamedTupleType([tf.float32, tf.float32])
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_order))
    self.assertFalse(foo_ne_order.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_value))
    self.assertFalse(foo_ne_value.__eq__(foo))
    self.assertFalse(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_order))
    self.assertTrue(foo_ne_order.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_value))
    self.assertTrue(foo_ne_value.__ne__(foo))

  def test_equality_named(self):
    foo = computation_types.NamedTupleType([('a', tf.int32), ('b', tf.bool)])
    foo_eq = computation_types.NamedTupleType([('a', tf.int32),
                                                ('b', tf.bool)])
    foo_ne_order = computation_types.NamedTupleType([('b', tf.bool),
                                                      ('a', tf.int32)])
    foo_ne_name = computation_types.NamedTupleType([('b', tf.int32),
                                                     ('a', tf.bool)])
    foo_ne_value = computation_types.NamedTupleType([('a', tf.float32),
                                                      ('b', tf.float32)])
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_order))
    self.assertFalse(foo_ne_order.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_name))
    self.assertFalse(foo_ne_name.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_value))
    self.assertFalse(foo_ne_value.__eq__(foo))
    self.assertFalse(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_order))
    self.assertTrue(foo_ne_order.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_name))
    self.assertTrue(foo_ne_name.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_value))
    self.assertTrue(foo_ne_value.__ne__(foo))


class NamedTupleTypeWithPyContainerTypeTest(absltest.TestCase):

  def test_dict(self):
    t = computation_types.NamedTupleTypeWithPyContainerType([('a', tf.int32)],
                                                            dict)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), dict)
    self.assertEqual(repr(t), 'NamedTupleType([(\'a\', TensorType(tf.int32))])')

  def test_ordered_dict(self):
    t = computation_types.NamedTupleTypeWithPyContainerType(
        [('a', tf.int32)], collections.OrderedDict)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), collections.OrderedDict)
    self.assertEqual(repr(t), 'NamedTupleType([(\'a\', TensorType(tf.int32))])')

  def test_tuple(self):
    t = computation_types.NamedTupleTypeWithPyContainerType([('a', tf.int32)],
                                                            tuple)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)
    self.assertEqual(repr(t), 'NamedTupleType([(\'a\', TensorType(tf.int32))])')

  def test_py_named_tuple(self):
    py_named_tuple_type = collections.namedtuple('test_tuple', ['a'])
    t = computation_types.NamedTupleTypeWithPyContainerType([('a', tf.int32)],
                                                            py_named_tuple_type)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), py_named_tuple_type)
    self.assertEqual(repr(t), 'NamedTupleType([(\'a\', TensorType(tf.int32))])')


class SequenceTypeTest(absltest.TestCase):

  def test_repr(self):
    self.assertEqual(
        repr(computation_types.SequenceType(tf.int32)),
        'SequenceType(TensorType(tf.int32))')
    self.assertEqual(
        repr(computation_types.SequenceType((tf.int32, tf.bool))),
        'SequenceType(NamedTupleType([TensorType(tf.int32), '
        'TensorType(tf.bool)]))')

  def test_str(self):
    self.assertEqual(str(computation_types.SequenceType(tf.int32)), 'int32*')
    self.assertEqual(
        str(computation_types.SequenceType((tf.int32, tf.bool))),
        '<int32,bool>*')

  def test_element(self):
    self.assertEqual(
        str(computation_types.SequenceType(tf.int32).element), 'int32')

  def test_equality(self):
    foo = computation_types.SequenceType(tf.int32)
    foo_eq = computation_types.SequenceType(tf.int32)
    foo_ne = computation_types.SequenceType(tf.bool)
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne))
    self.assertFalse(foo_ne.__eq__(foo))
    self.assertFalse(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne))
    self.assertTrue(foo_ne.__ne__(foo))


class FunctionTypeTest(absltest.TestCase):

  def test_repr(self):
    self.assertEqual(
        repr(computation_types.FunctionType(tf.int32, tf.bool)),
        'FunctionType(TensorType(tf.int32), TensorType(tf.bool))')
    self.assertEqual(
        repr(computation_types.FunctionType(None, tf.bool)),
        'FunctionType(None, TensorType(tf.bool))')

  def test_str(self):
    self.assertEqual(
        str(computation_types.FunctionType(tf.int32, tf.bool)),
        '(int32 -> bool)')
    self.assertEqual(
        str(computation_types.FunctionType(None, tf.bool)), '( -> bool)')

  def test_parameter_and_result(self):
    t = computation_types.FunctionType(tf.int32, tf.bool)
    self.assertEqual(str(t.parameter), 'int32')
    self.assertEqual(str(t.result), 'bool')

  def test_equality(self):
    foo = computation_types.FunctionType(tf.int32, tf.bool)
    foo_eq = computation_types.FunctionType(tf.int32, tf.bool)
    foo_ne_attr1 = computation_types.FunctionType(tf.bool, tf.bool)
    foo_ne_attr2 = computation_types.FunctionType(tf.int32, tf.int32)
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_attr1))
    self.assertFalse(foo_ne_attr1.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_attr2))
    self.assertFalse(foo_ne_attr2.__eq__(foo))
    self.assertFalse(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_attr1))
    self.assertTrue(foo_ne_attr1.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_attr2))
    self.assertTrue(foo_ne_attr2.__ne__(foo))


class AbstractTypeTest(absltest.TestCase):

  def test_construction(self):
    t1 = computation_types.AbstractType('T1')
    self.assertEqual(repr(t1), 'AbstractType(\'T1\')')
    self.assertEqual(str(t1), 'T1')
    self.assertEqual(t1.label, 'T1')
    self.assertRaises(TypeError, computation_types.AbstractType, 10)

  def test_equality(self):
    foo = computation_types.AbstractType('T')
    foo_eq = computation_types.AbstractType('T')
    foo_ne = computation_types.AbstractType('U')
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne))
    self.assertFalse(foo_ne.__eq__(foo))
    self.assertFalse(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne))
    self.assertTrue(foo_ne.__ne__(foo))


class PlacementTypeTest(absltest.TestCase):

  def test_construction(self):
    t1 = computation_types.PlacementType()
    self.assertEqual(repr(t1), 'PlacementType()')
    self.assertEqual(str(t1), 'placement')

  def test_equality(self):
    foo = computation_types.PlacementType()
    foo_eq = computation_types.PlacementType()
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))


class FederatedTypeTest(absltest.TestCase):

  def test_construction(self):
    t1 = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    self.assertEqual(str(t1.member), 'int32')
    self.assertIs(t1.placement, placements.CLIENTS)
    self.assertFalse(t1.all_equal)
    self.assertEqual(
        repr(t1), 'FederatedType(TensorType(tf.int32), '
        'PlacementLiteral(\'clients\'), False)')
    self.assertEqual(str(t1), '{int32}@CLIENTS')
    t2 = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    self.assertEqual(str(t2), 'int32@CLIENTS')

  def test_equality(self):
    foo = computation_types.FederatedType(tf.int32, placements.CLIENTS, False)
    foo_eq = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                             False)
    foo_ne_attr1 = computation_types.FederatedType(tf.bool, placements.CLIENTS,
                                                   False)
    foo_ne_attr2 = computation_types.FederatedType(tf.int32, placements.SERVER,
                                                   False)
    foo_ne_attr3 = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                                   True)
    self.assertTrue(foo.__eq__(foo))
    self.assertEqual(foo.__eq__(1), NotImplemented)
    self.assertTrue(foo.__eq__(foo_eq))
    self.assertTrue(foo_eq.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_attr1))
    self.assertFalse(foo_ne_attr1.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_attr2))
    self.assertFalse(foo_ne_attr2.__eq__(foo))
    self.assertFalse(foo.__eq__(foo_ne_attr3))
    self.assertFalse(foo_ne_attr3 == foo)
    self.assertFalse(foo.__ne__(foo))
    self.assertEqual(foo.__ne__(1), NotImplemented)
    self.assertFalse(foo.__ne__(foo_eq))
    self.assertFalse(foo_eq.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_attr1))
    self.assertTrue(foo_ne_attr1.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_attr2))
    self.assertTrue(foo_ne_attr2.__ne__(foo))
    self.assertTrue(foo.__ne__(foo_ne_attr3))
    self.assertTrue(foo_ne_attr3.__ne__(foo))


class ToTypeTest(absltest.TestCase):

  def test_tensor_type(self):
    s = computation_types.TensorType(tf.int32)
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32')

  def test_tf_type(self):
    s = tf.int32
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32')

  def test_tf_tensorspec(self):
    s = tf.TensorSpec([None, 3], dtype=tf.float32)
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'float32[?,3]')

  def test_tf_type_and_shape(self):
    s = (tf.int32, [10])
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32[10]')

  def test_tf_type_and_shape_with_unknown_dimension(self):
    s = (tf.int32, [None])
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32[?]')

  def test_list_of_tf_types(self):
    s = [tf.int32, tf.bool]
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertEqual(str(t), '<int32,bool>')

  def test_tuple_of_tf_types(self):
    s = (tf.int32, tf.bool)
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)
    self.assertEqual(str(t), '<int32,bool>')

  def test_singleton_named_tf_type(self):
    s = ('a', tf.int32)
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)
    self.assertEqual(str(t), '<a=int32>')

  def test_list_of_named_tf_types(self):
    s = [('a', tf.int32), ('b', tf.bool)]
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), list)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_ordered_dict_of_tf_types(self):
    s = collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), collections.OrderedDict)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_nested_tuple_of_tf_types(self):
    s = (tf.int32, (tf.float32, tf.bool))
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)
    self.assertEqual(str(t), '<int32,<float32,bool>>')

  def test_nested_tuple_of_named_tf_types(self):
    s = (tf.int32, (('x', tf.float32), tf.bool))
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t[1]), tuple)
    self.assertEqual(str(t), '<int32,<x=float32,bool>>')

  def test_nested_tuple_of_named_nonscalar_tf_types(self):
    s = ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t[1]), tuple)
    self.assertEqual(str(t), '<int32[1],<x=float32[2],bool[3]>>')

  def test_namedtuple_elements_two_tuples(self):
    elems = [tf.int32 for k in range(10)]
    t = computation_types.to_type(elems)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), list)
    for k in anonymous_tuple.to_elements(t):
      self.assertLen(k, 2)

  def test_namedtuples_addressable_by_name(self):
    elems = [('item' + str(k), tf.int32) for k in range(5)]
    t = computation_types.to_type(elems)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), list)
    self.assertIsInstance(t.item0, computation_types.TensorType)
    self.assertEqual(t.item0, t[0])

  def test_namedtuple_unpackable(self):
    elems = [('item' + str(k), tf.int32) for k in range(2)]
    t = computation_types.to_type(elems)
    a, b = t
    self.assertIsInstance(a, computation_types.TensorType)
    self.assertIsInstance(b, computation_types.TensorType)


if __name__ == '__main__':
  absltest.main()
