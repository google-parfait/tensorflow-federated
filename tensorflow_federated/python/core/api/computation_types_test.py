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

from absl.testing import absltest
from absl.testing import parameterized
import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements


class TestCheckEquivalentTypesTest(absltest.TestCase):

  def test_raises_type_error(self):
    int_type = computation_types.TensorType(tf.int32)
    bool_type = computation_types.TensorType(tf.bool)
    int_type.check_equivalent_to(int_type)
    with self.assertRaises(TypeError):
      int_type.check_equivalent_to(bool_type)


class TensorTypeTest(absltest.TestCase):

  def test_unknown_tensorshape(self):
    t = computation_types.TensorType(tf.int32, tf.TensorShape(None))
    self.assertEqual(t.dtype, tf.int32)
    self.assertEqual(t.shape, tf.TensorShape(None))

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

  def test_identity(self):
    t1 = computation_types.TensorType(tf.int32, [10])
    t2 = computation_types.TensorType(tf.int32, [10])
    self.assertIs(t1, t2)

  def test_is_assignable_from(self):
    t = computation_types.TensorType(tf.int32, [10])
    self.assertFalse(
        t.is_assignable_from(computation_types.TensorType(tf.int32)))
    self.assertFalse(
        t.is_assignable_from(computation_types.TensorType(tf.int32, [5])))
    self.assertFalse(
        t.is_assignable_from(computation_types.TensorType(tf.int32, [10, 10])))
    self.assertTrue(
        t.is_assignable_from(computation_types.TensorType(tf.int32, 10)))

  def test_is_assignable_from_unknown_dims(self):
    t1 = computation_types.TensorType(tf.int32, [None])
    t2 = computation_types.TensorType(tf.int32, [10])
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertFalse(t2.is_assignable_from(t1))

  def test_is_equivalent_to(self):
    t1 = computation_types.TensorType(tf.int32, [None])
    t2 = computation_types.TensorType(tf.int32, [10])
    t3 = computation_types.TensorType(tf.int32, [10])
    self.assertTrue(t1.is_equivalent_to(t1))
    self.assertTrue(t2.is_equivalent_to(t3))
    self.assertTrue(t3.is_equivalent_to(t2))
    self.assertFalse(t1.is_equivalent_to(t2))
    self.assertFalse(t2.is_equivalent_to(t1))


class StructTypeTest(absltest.TestCase):

  def test_repr(self):
    self.assertEqual(
        repr(computation_types.StructType([tf.int32, ('a', tf.bool)])),
        'StructType([TensorType(tf.int32), (\'a\', TensorType(tf.bool))])')

  def test_str(self):
    self.assertEqual(str(computation_types.StructType([tf.int32])), '<int32>')
    self.assertEqual(
        str(computation_types.StructType([('a', tf.int32)])), '<a=int32>')
    self.assertEqual(
        str(computation_types.StructType(('a', tf.int32))), '<a=int32>')
    self.assertEqual(
        str(computation_types.StructType([tf.int32, tf.bool])), '<int32,bool>')
    self.assertEqual(
        str(computation_types.StructType([('a', tf.int32), tf.float32])),
        '<a=int32,float32>')
    self.assertEqual(
        str(computation_types.StructType([('a', tf.int32), ('b', tf.float32)])),
        '<a=int32,b=float32>')
    self.assertEqual(
        str(
            computation_types.StructType([('a', tf.int32),
                                          ('b',
                                           computation_types.StructType([
                                               ('x', tf.string), ('y', tf.bool)
                                           ]))])),
        '<a=int32,b=<x=string,y=bool>>')

  def test_elements(self):
    self.assertEqual(
        repr(
            structure.to_elements(
                computation_types.StructType([tf.int32, ('a', tf.bool)]))),
        '[(None, TensorType(tf.int32)), (\'a\', TensorType(tf.bool))]')

  def test_with_none_keys(self):
    self.assertEqual(
        str(computation_types.StructType([(None, tf.int32)])), '<int32>')

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

  def test_identity(self):
    shape = [('a', tf.int32), ('b', tf.bool)]
    t1 = computation_types.StructType(shape)
    t2 = computation_types.StructType(shape)
    self.assertIs(t1, t2)

  def test_is_assignable_from(self):
    t1 = computation_types.StructType([tf.int32, ('a', tf.bool)])
    t2 = computation_types.StructType([tf.int32, ('a', tf.bool)])
    t3 = computation_types.StructType([tf.int32, ('b', tf.bool)])
    t4 = computation_types.StructType([tf.int32, ('a', tf.string)])
    t5 = computation_types.StructType([tf.int32])
    t6 = computation_types.StructType([tf.int32, tf.bool])
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertFalse(t1.is_assignable_from(t3))
    self.assertFalse(t1.is_assignable_from(t4))
    self.assertFalse(t1.is_assignable_from(t5))
    self.assertTrue(t1.is_assignable_from(t6))
    self.assertFalse(t6.is_assignable_from(t1))


class StructWithPythonTypeTest(absltest.TestCase):

  def test_dict(self):
    t = computation_types.StructWithPythonType([('a', tf.int32)], dict)
    self.assertIs(t.python_container, dict)
    self.assertEqual(
        repr(t), 'StructType([(\'a\', TensorType(tf.int32))]) as dict')

  def test_ordered_dict(self):
    t = computation_types.StructWithPythonType([('a', tf.int32)],
                                               collections.OrderedDict)
    self.assertIs(t.python_container, collections.OrderedDict)
    self.assertEqual(
        repr(t), 'StructType([(\'a\', TensorType(tf.int32))]) as OrderedDict')

  def test_tuple(self):
    t = computation_types.StructWithPythonType([('a', tf.int32)], tuple)
    self.assertIs(t.python_container, tuple)
    self.assertEqual(
        repr(t), 'StructType([(\'a\', TensorType(tf.int32))]) as tuple')

  def test_py_named_tuple(self):
    py_named_tuple_type = collections.namedtuple('test_tuple', ['a'])
    t = computation_types.StructWithPythonType([('a', tf.int32)],
                                               py_named_tuple_type)
    self.assertIs(t.python_container, py_named_tuple_type)
    self.assertEqual(
        repr(t), 'StructType([(\'a\', TensorType(tf.int32))]) as test_tuple')

  def test_py_attr_class(self):

    @attr.s
    class TestFoo(object):
      a = attr.ib()

    t = computation_types.StructWithPythonType([('a', tf.int32)], TestFoo)
    self.assertIs(t.python_container, TestFoo)
    self.assertEqual(
        repr(t), 'StructType([(\'a\', TensorType(tf.int32))]) as TestFoo')

  def test_identity(self):
    t1 = computation_types.StructWithPythonType([('a', tf.int32)], dict)
    t2 = computation_types.StructWithPythonType([('a', tf.int32)], dict)
    self.assertIs(t1, t2)


class SequenceTypeTest(absltest.TestCase):

  def test_repr(self):
    self.assertEqual(
        repr(computation_types.SequenceType(tf.int32)),
        'SequenceType(TensorType(tf.int32))')
    self.assertEqual(
        repr(
            computation_types.SequenceType(
                computation_types.StructType((tf.int32, tf.bool)))),
        'SequenceType(StructType([TensorType(tf.int32), '
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

  def test_identity(self):
    t1 = computation_types.SequenceType(tf.int32)
    t2 = computation_types.SequenceType(tf.int32)
    self.assertIs(t1, t2)

  def test_is_assignable_from(self):
    self.assertTrue(
        computation_types.SequenceType(tf.int32).is_assignable_from(
            computation_types.SequenceType(tf.int32)))
    self.assertFalse(
        computation_types.SequenceType(tf.int32).is_assignable_from(
            computation_types.SequenceType(tf.bool)))


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

  def test_identity(self):
    t1 = computation_types.FunctionType(tf.int32, tf.bool)
    t2 = computation_types.FunctionType(tf.int32, tf.bool)
    self.assertIs(t1, t2)

  def test_is_assignable_from(self):
    t1 = computation_types.FunctionType(tf.int32, tf.bool)
    t2 = computation_types.FunctionType(tf.int32, tf.bool)
    t3 = computation_types.FunctionType(tf.int32, tf.int32)
    t4 = computation_types.TensorType(tf.int32)
    self.assertTrue(t1.is_assignable_from(t1))
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertFalse(t1.is_assignable_from(t3))
    self.assertFalse(t1.is_assignable_from(t4))


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

  def test_identity(self):
    t1 = computation_types.AbstractType('T')
    t2 = computation_types.AbstractType('T')
    self.assertIs(t1, t2)

  def test_is_assignable_from(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.AbstractType('T2')
    with self.assertRaises(TypeError):
      t1.is_assignable_from(t2)


class PlacementTypeTest(absltest.TestCase):

  def test_construction(self):
    t1 = computation_types.PlacementType()
    self.assertEqual(repr(t1), 'PlacementType()')
    self.assertEqual(str(t1), 'placement')

  def test_equality(self):
    t1 = computation_types.PlacementType()
    t2 = computation_types.PlacementType()
    self.assertEqual(t1, t2)

  def test_identity(self):
    t1 = computation_types.AbstractType('T')
    t2 = computation_types.AbstractType('T')
    self.assertIs(t1, t2)

  def test_is_assignable_from(self):
    t1 = computation_types.PlacementType()
    t2 = computation_types.PlacementType()
    self.assertTrue(t1.is_assignable_from(t1))
    self.assertTrue(t1.is_assignable_from(t2))


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

  def test_identity(self):
    t1 = computation_types.FederatedType(tf.int32, placements.CLIENTS, False)
    t2 = computation_types.FederatedType(tf.int32, placements.CLIENTS, False)
    self.assertIs(t1, t2)

  def test_with_federated_type(self):
    t1 = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    self.assertTrue(t1.is_assignable_from(t1))
    t2 = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    self.assertTrue(t1.is_assignable_from(t2))
    self.assertTrue(t2.is_assignable_from(t2))
    self.assertFalse(t2.is_assignable_from(t1))
    t3 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [10]), placements.CLIENTS)
    t4 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [None]), placements.CLIENTS)
    self.assertTrue(t4.is_assignable_from(t3))
    self.assertFalse(t3.is_assignable_from(t4))
    t5 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [10]), placements.SERVER)
    self.assertFalse(t3.is_assignable_from(t5))
    self.assertFalse(t5.is_assignable_from(t3))
    t6 = computation_types.FederatedType(
        computation_types.TensorType(tf.int32, [10]),
        placements.CLIENTS,
        all_equal=True)
    self.assertTrue(t3.is_assignable_from(t6))
    self.assertTrue(t4.is_assignable_from(t6))
    self.assertFalse(t6.is_assignable_from(t3))
    self.assertFalse(t6.is_assignable_from(t4))


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
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<int32,bool>')

  def test_tuple_of_tf_types(self):
    s = (tf.int32, tf.bool)
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertEqual(str(t), '<int32,bool>')

  def test_singleton_named_tf_type(self):
    s = ('a', tf.int32)
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertEqual(str(t), '<a=int32>')

  def test_list_of_named_tf_types(self):
    s = [('a', tf.int32), ('b', tf.bool)]
    t = computation_types.to_type(s)
    # Note: list of pairs should be interpreted as a plain StructType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_list_of_partially_named_tf_types(self):
    s = [tf.bool, ('a', tf.int32)]
    t = computation_types.to_type(s)
    # Note: list of pairs should be interpreted as a plain StructType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<bool,a=int32>')

  def test_ordered_dict_of_tf_types(self):
    s = collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, collections.OrderedDict)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_nested_tuple_of_tf_types(self):
    s = (tf.int32, (tf.float32, tf.bool))
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertEqual(str(t), '<int32,<float32,bool>>')

  def test_nested_tuple_of_named_tf_types(self):
    s = (tf.int32, (('x', tf.float32), tf.bool))
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertNotIsInstance(t[1], computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<int32,<x=float32,bool>>')

  def test_nested_tuple_of_named_nonscalar_tf_types(self):
    s = ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertNotIsInstance(t[1], computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<int32[1],<x=float32[2],bool[3]>>')

  def test_namedtuple_elements_two_tuples(self):
    elems = [tf.int32 for k in range(10)]
    t = computation_types.to_type(elems)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, list)
    for k in structure.iter_elements(t):
      self.assertLen(k, 2)

  def test_namedtuples_addressable_by_name(self):
    elems = [('item' + str(k), tf.int32) for k in range(5)]
    t = computation_types.to_type(elems)
    # Note: list of pairs should be interpreted as a plain StructType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)
    self.assertIsInstance(t.item0, computation_types.TensorType)
    self.assertEqual(t.item0, t[0])

  def test_namedtuple_unpackable(self):
    elems = [('item' + str(k), tf.int32) for k in range(2)]
    t = computation_types.to_type(elems)
    a, b = t
    self.assertIsInstance(a, computation_types.TensorType)
    self.assertIsInstance(b, computation_types.TensorType)

  def test_attrs_type(self):

    @attr.s
    class TestFoo(object):
      a = attr.ib(type=tf.int32)
      b = attr.ib(type=(tf.float32, [2]))

    t = computation_types.to_type(TestFoo)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, TestFoo)
    self.assertEqual(str(t), '<a=int32,b=float32[2]>')

  def test_attrs_class_missing_type_fails(self):

    @attr.s
    class TestFoo(object):
      a = attr.ib(type=tf.int32)
      b = attr.ib()  # no type parameter
      c = attr.ib()  # no type parameter

    expected_msg = (
        "Cannot infer tff.Type for attr.s class 'TestFoo' because some "
        "attributes were missing type specifications: ['b', 'c']")
    with self.assertRaisesWithLiteralMatch(TypeError, expected_msg):
      computation_types.to_type(TestFoo)

  def test_attrs_instance(self):

    @attr.s
    class TestFoo(object):
      a = attr.ib()
      b = attr.ib()

    t = computation_types.to_type(TestFoo(a=tf.int32, b=(tf.float32, [2])))
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, TestFoo)
    self.assertEqual(str(t), '<a=int32,b=float32[2]>')

  def test_nested_attrs_class(self):

    @attr.s
    class TestFoo(object):
      a = attr.ib()
      b = attr.ib()

    @attr.s
    class TestFoo2(object):
      c = attr.ib(type=(tf.float32, [2]))

    t = computation_types.to_type(TestFoo(a=[tf.int32, tf.bool], b=TestFoo2))
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, TestFoo)
    self.assertIsInstance(t.a, computation_types.StructWithPythonType)
    self.assertIs(t.a.python_container, list)
    self.assertIsInstance(t.b, computation_types.StructWithPythonType)
    self.assertIs(t.b.python_container, TestFoo2)
    self.assertEqual(str(t), '<a=<int32,bool>,b=<c=float32[2]>>')


class RepresentationTest(absltest.TestCase):

  def test_returns_string_for_abstract_type(self):
    type_spec = computation_types.AbstractType('T')

    self.assertEqual(type_spec.compact_representation(), 'T')
    self.assertEqual(type_spec.formatted_representation(), 'T')

  def test_returns_string_for_federated_type_clients(self):
    type_spec = computation_types.FederatedType(tf.int32, placements.CLIENTS)

    self.assertEqual(type_spec.compact_representation(), '{int32}@CLIENTS')
    self.assertEqual(type_spec.formatted_representation(), '{int32}@CLIENTS')

  def test_returns_string_for_federated_type_server(self):
    type_spec = computation_types.FederatedType(tf.int32, placements.SERVER)

    self.assertEqual(type_spec.compact_representation(), 'int32@SERVER')
    self.assertEqual(type_spec.formatted_representation(), 'int32@SERVER')

  def test_returns_string_for_function_type(self):
    type_spec = computation_types.FunctionType(tf.int32, tf.float32)

    self.assertEqual(type_spec.compact_representation(), '(int32 -> float32)')
    self.assertEqual(type_spec.formatted_representation(), '(int32 -> float32)')

  def test_returns_string_for_function_type_with_named_tuple_type_parameter(
      self):
    parameter = computation_types.StructType((tf.int32, tf.float32))
    type_spec = computation_types.FunctionType(parameter, tf.bool)

    self.assertEqual(type_spec.compact_representation(),
                     '(<int32,float32> -> bool)')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '(<\n'
        '  int32,\n'
        '  float32\n'
        '> -> bool)'
    )
    # pyformat: enable

  def test_returns_string_for_function_type_with_named_tuple_type_result(self):
    result = computation_types.StructType((tf.int32, tf.float32))
    type_spec = computation_types.FunctionType(tf.bool, result)

    self.assertEqual(type_spec.compact_representation(),
                     '(bool -> <int32,float32>)')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '(bool -> <\n'
        '  int32,\n'
        '  float32\n'
        '>)'
    )
    # pyformat: enable

  def test_returns_string_for_function_type_with_named_tuple_type_parameter_and_result(
      self):
    parameter = computation_types.StructType((tf.int32, tf.float32))
    result = computation_types.StructType((tf.bool, tf.string))
    type_spec = computation_types.FunctionType(parameter, result)

    self.assertEqual(type_spec.compact_representation(),
                     '(<int32,float32> -> <bool,string>)')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
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
    type_spec = computation_types.StructType((tf.int32, tf.float32))

    self.assertEqual(type_spec.compact_representation(), '<int32,float32>')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  int32,\n'
        '  float32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_named(self):
    type_spec = computation_types.StructType(
        (('a', tf.int32), ('b', tf.float32)))

    self.assertEqual(type_spec.compact_representation(), '<a=int32,b=float32>')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  a=int32,\n'
        '  b=float32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_nested(self):
    type_spec_1 = computation_types.StructType((tf.int32, tf.float32))
    type_spec_2 = computation_types.StructType((type_spec_1, tf.bool))
    type_spec_3 = computation_types.StructType((type_spec_2, tf.string))
    type_spec = type_spec_3

    self.assertEqual(type_spec.compact_representation(),
                     '<<<int32,float32>,bool>,string>')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
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

  def test_returns_string_for_named_tuple_type_with_one_element(self):
    type_spec = computation_types.StructType((tf.int32,))

    self.assertEqual(type_spec.compact_representation(), '<int32>')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  int32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_with_no_element(self):
    type_spec = computation_types.StructType([])

    self.assertEqual(type_spec.compact_representation(), '<>')
    self.assertEqual(type_spec.formatted_representation(), '<>')

  def test_returns_string_for_placement_type(self):
    type_spec = computation_types.PlacementType()

    self.assertEqual(type_spec.compact_representation(), 'placement')
    self.assertEqual(type_spec.formatted_representation(), 'placement')

  def test_returns_string_for_sequence_type_int(self):
    type_spec = computation_types.SequenceType(tf.int32)

    self.assertEqual(type_spec.compact_representation(), 'int32*')
    self.assertEqual(type_spec.formatted_representation(), 'int32*')

  def test_returns_string_for_sequence_type_float(self):
    type_spec = computation_types.SequenceType(tf.float32)

    self.assertEqual(type_spec.compact_representation(), 'float32*')
    self.assertEqual(type_spec.formatted_representation(), 'float32*')

  def test_returns_string_for_sequence_type_named_tuple_type(self):
    element = computation_types.StructType((tf.int32, tf.float32))
    type_spec = computation_types.SequenceType(element)

    self.assertEqual(type_spec.compact_representation(), '<int32,float32>*')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  int32,\n'
        '  float32\n'
        '>*'
    )
    # pyformat: enable

  def test_returns_string_for_tensor_type_int(self):
    type_spec = computation_types.TensorType(tf.int32)

    self.assertEqual(type_spec.compact_representation(), 'int32')
    self.assertEqual(type_spec.formatted_representation(), 'int32')

  def test_returns_string_for_tensor_type_float(self):
    type_spec = computation_types.TensorType(tf.float32)

    self.assertEqual(type_spec.compact_representation(), 'float32')
    self.assertEqual(type_spec.formatted_representation(), 'float32')


class CheckWellFormedTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('abstract_type',
       lambda: computation_types.AbstractType('T')),
      ('federated_type',
       lambda: computation_types.FederatedType(tf.int32, placements.CLIENTS)),
      ('function_type',
       lambda: computation_types.FunctionType(tf.int32, tf.int32)),
      ('named_tuple_type',
       lambda: computation_types.StructType([tf.int32] * 3)),
      ('placement_type',
       computation_types.PlacementType),
      ('sequence_type',
       lambda: computation_types.SequenceType(tf.int32)),
      ('tensor_type',
       lambda: computation_types.TensorType(tf.int32)),
  ])
  # pyformat: enable
  def test_does_not_raise_type_error(self, create_type_signature):
    try:
      create_type_signature()
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters([
      (
          'federated_function_type',
          lambda: computation_types.FederatedType(  # pylint: disable=g-long-lambda
              computation_types.FunctionType(tf.int32, tf.int32), placements.
              CLIENTS)),
      (
          'federated_federated_type',
          lambda: computation_types.FederatedType(  # pylint: disable=g-long-lambda
              computation_types.FederatedType(tf.int32, placements.CLIENTS),
              placements.CLIENTS)),
      (
          'sequence_sequence_type',
          lambda: computation_types.SequenceType(  # pylint: disable=g-long-lambda
              computation_types.SequenceType([tf.int32]))),
      (
          'sequence_federated_type',
          lambda: computation_types.SequenceType(  # pylint: disable=g-long-lambda
              computation_types.FederatedType(tf.int32, placements.CLIENTS))),
      (
          'tuple_federated_function_type',
          lambda: computation_types.StructType([  # pylint: disable=g-long-lambda
              computation_types.FederatedType(
                  computation_types.FunctionType(tf.int32, tf.int32), placements
                  .CLIENTS)
          ])),
      (
          'tuple_federated_federated_type',
          lambda: computation_types.StructType([  # pylint: disable=g-long-lambda
              computation_types.FederatedType(
                  computation_types.FederatedType(tf.int32, placements.CLIENTS),
                  placements.CLIENTS)
          ])),
      (
          'federated_tuple_function_type',
          lambda: computation_types.FederatedType(  # pylint: disable=g-long-lambda
              computation_types.StructType([
                  computation_types.FunctionType(tf.int32, tf.int32)
              ]), placements.CLIENTS)),
  ])
  # pyformat: enable
  def test_raises_type_error(self, create_type_signature):
    with self.assertRaises(TypeError):
      create_type_signature()


if __name__ == '__main__':
  absltest.main()
