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
import attr
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
    t1 = computation_types.TensorType(tf.int32, [10])
    t2 = computation_types.TensorType(tf.int32, [10])
    t3 = computation_types.TensorType(tf.int32, [None])
    t4 = computation_types.TensorType(tf.int32, [None])
    self.assertEqual(t1, t2)
    self.assertEqual(t3, t4)
    self.assertNotEqual(t1, t3)


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

  def test_equality(self):
    t1 = computation_types.to_type([tf.int32, tf.bool])
    t2 = computation_types.to_type([tf.int32, tf.bool])
    t3 = computation_types.to_type([('a', tf.int32), ('b', tf.bool)])
    t4 = computation_types.to_type([('a', tf.int32), ('b', tf.bool)])
    t5 = computation_types.to_type([('b', tf.int32), ('a', tf.bool)])
    t6 = computation_types.to_type([('a', tf.bool), ('b', tf.int32)])
    self.assertEqual(t1, t2)
    self.assertEqual(t3, t4)
    self.assertNotEqual(t1, t3)
    self.assertNotEqual(t4, t5)
    self.assertNotEqual(t4, t6)


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

  def test_py_attr_class(self):

    @attr.s
    class TestFoo(object):
      A = attr.ib()

    t = computation_types.NamedTupleTypeWithPyContainerType([('A', tf.int32)],
                                                            TestFoo)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), TestFoo)
    self.assertEqual(repr(t), 'NamedTupleType([(\'A\', TensorType(tf.int32))])')


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
    t1 = computation_types.SequenceType(tf.int32)
    t2 = computation_types.SequenceType(tf.int32)
    t3 = computation_types.SequenceType(tf.bool)
    self.assertEqual(t1, t2)
    self.assertNotEqual(t1, t3)


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
    t1 = computation_types.FunctionType(tf.int32, tf.bool)
    t2 = computation_types.FunctionType(tf.int32, tf.bool)
    t3 = computation_types.FunctionType(tf.int32, tf.int32)
    t4 = computation_types.FunctionType(tf.bool, tf.bool)
    self.assertEqual(t1, t2)
    self.assertNotEqual(t1, t3)
    self.assertNotEqual(t1, t4)


class AbstractTypeTest(absltest.TestCase):

  def test_construction(self):
    t1 = computation_types.AbstractType('T1')
    self.assertEqual(repr(t1), 'AbstractType(\'T1\')')
    self.assertEqual(str(t1), 'T1')
    self.assertEqual(t1.label, 'T1')
    self.assertRaises(TypeError, computation_types.AbstractType, 10)

  def test_equality(self):
    t1 = computation_types.AbstractType('T')
    t2 = computation_types.AbstractType('T')
    t3 = computation_types.AbstractType('U')
    self.assertEqual(t1, t2)
    self.assertNotEqual(t1, t3)


class PlacementTypeTest(absltest.TestCase):

  def test_construction(self):
    t1 = computation_types.PlacementType()
    self.assertEqual(repr(t1), 'PlacementType()')
    self.assertEqual(str(t1), 'placement')

  def test_equality(self):
    t1 = computation_types.PlacementType()
    t2 = computation_types.PlacementType()
    self.assertEqual(t1, t2)


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
    t1 = computation_types.FederatedType(tf.int32, placements.CLIENTS, False)
    t2 = computation_types.FederatedType(tf.int32, placements.CLIENTS, False)
    t3 = computation_types.FederatedType(tf.bool, placements.CLIENTS, False)
    t4 = computation_types.FederatedType(tf.int32, placements.SERVER, False)
    t5 = computation_types.FederatedType(tf.int32, placements.CLIENTS, True)
    self.assertEqual(t1, t2)
    self.assertNotEqual(t1, t3)
    self.assertNotEqual(t1, t4)
    self.assertNotEqual(t1, t5)


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
    # Note: list of pairs should be interpreted as a plain NamedTupleType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(
        t, computation_types.NamedTupleTypeWithPyContainerType)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_list_of_partially_named_tf_types(self):
    s = [tf.bool, ('a', tf.int32)]
    t = computation_types.to_type(s)
    # Note: list of pairs should be interpreted as a plain NamedTupleType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(
        t, computation_types.NamedTupleTypeWithPyContainerType)
    self.assertEqual(str(t), '<bool,a=int32>')

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
    self.assertNotIsInstance(
        t[1], computation_types.NamedTupleTypeWithPyContainerType)
    self.assertEqual(str(t), '<int32,<x=float32,bool>>')

  def test_nested_tuple_of_named_nonscalar_tf_types(self):
    s = ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))
    t = computation_types.to_type(s)
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)
    self.assertNotIsInstance(
        t[1], computation_types.NamedTupleTypeWithPyContainerType)
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
    # Note: list of pairs should be interpreted as a plain NamedTupleType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(
        t, computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIsInstance(t.item0, computation_types.TensorType)
    self.assertEqual(t.item0, t[0])

  def test_namedtuple_unpackable(self):
    elems = [('item' + str(k), tf.int32) for k in range(2)]
    t = computation_types.to_type(elems)
    a, b = t
    self.assertIsInstance(a, computation_types.TensorType)
    self.assertIsInstance(b, computation_types.TensorType)

  def test_attrs_class(self):

    @attr.s
    class TestFoo(object):
      A = attr.ib()
      B = attr.ib()

    t = computation_types.to_type(TestFoo(A=tf.int32, B=(tf.float32, [2])))
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), TestFoo)
    self.assertEqual(str(t), '<A=int32,B=float32[2]>')

  def test_nested_attrs_class(self):

    @attr.s
    class TestFoo(object):
      A = attr.ib()
      B = attr.ib()

    @attr.s
    class TestFoo2(object):
      C = attr.ib()

    t = computation_types.to_type(
        TestFoo(A=[tf.int32, tf.bool], B=TestFoo2(C=(tf.float32, [2]))))
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), TestFoo)
    self.assertIsInstance(t.A,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t.A), list)
    self.assertIsInstance(t.B,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t.B), TestFoo2)
    self.assertEqual(str(t), '<A=<int32,bool>,B=<C=float32[2]>>')


class RepresentationTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      computation_types.compact_representation(None)
    with self.assertRaises(TypeError):
      computation_types.formatted_representation(None)

  def test_returns_string_for_abstract_type(self):
    type_spec = computation_types.AbstractType('T')
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, 'T')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, 'T')

  def test_returns_string_for_federated_type_clients(self):
    type_spec = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, '{int32}@CLIENTS')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, '{int32}@CLIENTS')

  def test_returns_string_for_federated_type_server(self):
    type_spec = computation_types.FederatedType(tf.int32, placements.SERVER)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, 'int32@SERVER')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, 'int32@SERVER')

  def test_returns_string_for_function_type(self):
    type_spec = computation_types.FunctionType(tf.int32, tf.float32)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, '(int32 -> float32)')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, '(int32 -> float32)')

  def test_returns_string_for_function_type_with_named_tuple_type_parameter(
      self):
    parameter = computation_types.NamedTupleType((tf.int32, tf.float32))
    type_spec = computation_types.FunctionType(parameter, tf.bool)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, '(<int32,float32> -> bool)')
    formatted_string = computation_types.formatted_representation(type_spec)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '(<\n'
        '  int32,\n'
        '  float32\n'
        '> -> bool)'
    )
    # pyformat: enable

  def test_returns_string_for_function_type_with_named_tuple_type_result(self):
    result = computation_types.NamedTupleType((tf.int32, tf.float32))
    type_spec = computation_types.FunctionType(tf.bool, result)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, '(bool -> <int32,float32>)')
    formatted_string = computation_types.formatted_representation(type_spec)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '(bool -> <\n'
        '  int32,\n'
        '  float32\n'
        '>)'
    )
    # pyformat: enable

  def test_returns_string_for_function_type_with_named_tuple_type_parameter_and_result(
      self):
    parameter = computation_types.NamedTupleType((tf.int32, tf.float32))
    result = computation_types.NamedTupleType((tf.bool, tf.string))
    type_spec = computation_types.FunctionType(parameter, result)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, '(<int32,float32> -> <bool,string>)')
    formatted_string = computation_types.formatted_representation(type_spec)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '(<\n'
        '  int32,\n'
        '  float32\n'
        '> -> <\n'
        '  bool,\n'
        '  string\n'
        '>)'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_unnamed(self):
    type_spec = computation_types.NamedTupleType((tf.int32, tf.float32))
    # compact_string = computation_types.compact_representation(type_spec)
    # self.assertEqual(compact_string, '<int32,float32>')
    formatted_string = computation_types.formatted_representation(type_spec)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '<\n'
        '  int32,\n'
        '  float32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_named(self):
    type_spec = computation_types.NamedTupleType(
        (('a', tf.int32), ('b', tf.float32)))
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, '<a=int32,b=float32>')
    formatted_string = computation_types.formatted_representation(type_spec)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '<\n'
        '  a=int32,\n'
        '  b=float32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_nested(self):
    type_spec_1 = computation_types.NamedTupleType((tf.int32, tf.float32))
    type_spec_2 = computation_types.NamedTupleType((type_spec_1, tf.bool))
    type_spec_3 = computation_types.NamedTupleType((type_spec_2, tf.string))
    compact_string = computation_types.compact_representation(type_spec_3)
    self.assertEqual(compact_string, '<<<int32,float32>,bool>,string>')
    formatted_string = computation_types.formatted_representation(type_spec_3)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '<\n'
        '  <\n'
        '    <\n'
        '      int32,\n'
        '      float32\n'
        '    >,\n'
        '    bool\n'
        '  >,\n'
        '  string\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_one_element(self):
    type_spec = computation_types.NamedTupleType((tf.int32,))
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, '<int32>')
    formatted_string = computation_types.formatted_representation(type_spec)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '<\n'
        '  int32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_placement_type(self):
    type_spec = computation_types.PlacementType()
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, 'placement')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, 'placement')

  def test_returns_string_for_sequence_type_int(self):
    type_spec = computation_types.SequenceType(tf.int32)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, 'int32*')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, 'int32*')

  def test_returns_string_for_sequence_type_float(self):
    type_spec = computation_types.SequenceType(tf.float32)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, 'float32*')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, 'float32*')

  def test_returns_string_for_sequence_type_named_tuple_type(self):
    element = computation_types.NamedTupleType((tf.int32, tf.float32))
    type_spec = computation_types.SequenceType(element)
    # compact_string = computation_types.compact_representation(type_spec)
    # self.assertEqual(compact_string, '<int32,float32>*')
    formatted_string = computation_types.formatted_representation(type_spec)
    # pyformat: disable
    self.assertEqual(
        formatted_string,
        '<\n'
        '  int32,\n'
        '  float32\n'
        '>*'
    )
    # pyformat: enable

  def test_returns_string_for_tensor_type_int(self):
    type_spec = computation_types.TensorType(tf.int32)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, 'int32')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, 'int32')

  def test_returns_string_for_tensor_type_float(self):
    type_spec = computation_types.TensorType(tf.float32)
    compact_string = computation_types.compact_representation(type_spec)
    self.assertEqual(compact_string, 'float32')
    formatted_string = computation_types.formatted_representation(type_spec)
    self.assertEqual(formatted_string, 'float32')


if __name__ == '__main__':
  absltest.main()
