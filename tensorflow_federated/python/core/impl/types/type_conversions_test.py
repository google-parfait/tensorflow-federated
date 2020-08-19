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
import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_conversions


class InferTypeTest(parameterized.TestCase):

  def test_with_none(self):
    self.assertIsNone(type_conversions.infer_type(None))

  def test_with_typed_object(self):

    class DummyTypedObject(typed_object.TypedObject):

      @property
      def type_signature(self):
        return computation_types.TensorType(tf.bool)

    dummy_type = type_conversions.infer_type(DummyTypedObject())
    self.assertEqual(dummy_type.compact_representation(), 'bool')

  def test_with_scalar_int_tensor(self):
    self.assertEqual(str(type_conversions.infer_type(tf.constant(1))), 'int32')

  def test_with_scalar_bool_tensor(self):
    self.assertEqual(
        str(type_conversions.infer_type(tf.constant(False))), 'bool')

  def test_with_int_array_tensor(self):
    self.assertEqual(
        str(type_conversions.infer_type(tf.constant([10, 20]))), 'int32[2]')

  def test_with_scalar_int_variable_tensor(self):
    self.assertEqual(str(type_conversions.infer_type(tf.Variable(10))), 'int32')

  def test_with_scalar_bool_variable_tensor(self):
    self.assertEqual(
        str(type_conversions.infer_type(tf.Variable(True))), 'bool')

  def test_with_scalar_float_variable_tensor(self):
    self.assertEqual(
        str(type_conversions.infer_type(tf.Variable(0.5))), 'float32')

  def test_with_scalar_int_array_variable_tensor(self):
    self.assertEqual(
        str(type_conversions.infer_type(tf.Variable([10]))), 'int32[1]')

  def test_with_int_dataset(self):
    self.assertEqual(
        str(type_conversions.infer_type(tf.data.Dataset.from_tensors(10))),
        'int32*')

  def test_with_ordered_dict_dataset(self):
    self.assertEqual(
        str(
            type_conversions.infer_type(
                tf.data.Dataset.from_tensors(
                    collections.OrderedDict([
                        ('b', 20),
                        ('a', 10),
                    ])))), '<b=int32,a=int32>*')

  def test_with_int(self):
    self.assertEqual(str(type_conversions.infer_type(10)), 'int32')

  def test_with_float(self):
    self.assertEqual(str(type_conversions.infer_type(0.5)), 'float32')

  def test_with_bool(self):
    self.assertEqual(str(type_conversions.infer_type(True)), 'bool')

  def test_with_string(self):
    self.assertEqual(str(type_conversions.infer_type('abc')), 'string')

  def test_with_np_int32(self):
    self.assertEqual(str(type_conversions.infer_type(np.int32(10))), 'int32')

  def test_with_np_int64(self):
    self.assertEqual(str(type_conversions.infer_type(np.int64(10))), 'int64')

  def test_with_np_float32(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.float32(10))), 'float32')

  def test_with_np_float64(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.float64(10))), 'float64')

  def test_with_np_bool(self):
    self.assertEqual(str(type_conversions.infer_type(np.bool(True))), 'bool')

  def test_with_unicode_string(self):
    self.assertEqual(str(type_conversions.infer_type(u'abc')), 'string')

  def test_with_numpy_int_array(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.array([10, 20]))), 'int64[2]')

  def test_with_numpy_nested_int_array(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.array([[10], [20]]))), 'int64[2,1]')

  def test_with_numpy_float64_scalar(self):
    self.assertEqual(str(type_conversions.infer_type(np.float64(1))), 'float64')

  def test_with_int_list(self):
    t = type_conversions.infer_type([1, 2, 3])
    self.assertEqual(str(t), '<int32,int32,int32>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, list)

  def test_with_nested_float_list(self):
    t = type_conversions.infer_type([[0.1], [0.2], [0.3]])
    self.assertEqual(str(t), '<<float32>,<float32>,<float32>>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, list)

  def test_with_structure(self):
    t = type_conversions.infer_type(
        structure.Struct([
            ('a', 10),
            (None, False),
        ]))
    self.assertEqual(str(t), '<a=int32,bool>')
    self.assertIsInstance(t, computation_types.StructType)
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)

  def test_with_nested_structure(self):
    t = type_conversions.infer_type(
        structure.Struct([
            ('a', 10),
            (None, structure.Struct([
                (None, True),
                (None, 0.5),
            ])),
        ]))
    self.assertEqual(str(t), '<a=int32,<bool,float32>>')
    self.assertIsInstance(t, computation_types.StructType)
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)

  def test_with_namedtuple(self):
    test_named_tuple = collections.namedtuple('TestNamedTuple', 'y x')
    t = type_conversions.infer_type(test_named_tuple(1, True))
    self.assertEqual(str(t), '<y=int32,x=bool>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, test_named_tuple)

  def test_with_dict(self):
    v1 = {
        'a': 1,
        'b': 2.0,
    }
    inferred_type = type_conversions.infer_type(v1)
    self.assertEqual(str(inferred_type), '<a=int32,b=float32>')
    self.assertIsInstance(inferred_type, computation_types.StructWithPythonType)
    self.assertIs(inferred_type.python_container, dict)

    v2 = {
        'b': 2.0,
        'a': 1,
    }
    inferred_type = type_conversions.infer_type(v2)
    self.assertEqual(str(inferred_type), '<a=int32,b=float32>')
    self.assertIsInstance(inferred_type, computation_types.StructWithPythonType)
    self.assertIs(inferred_type.python_container, dict)

  def test_with_ordered_dict(self):
    t = type_conversions.infer_type(
        collections.OrderedDict([('b', 2.0), ('a', 1)]))
    self.assertEqual(str(t), '<b=float32,a=int32>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, collections.OrderedDict)

  def test_with_nested_attrs_class(self):

    @attr.s
    class TestAttrClass(object):
      a = attr.ib()
      b = attr.ib()

    t = type_conversions.infer_type(TestAttrClass(a=0, b={'x': True, 'y': 0.0}))
    self.assertEqual(str(t), '<a=int32,b=<x=bool,y=float32>>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, TestAttrClass)
    self.assertIs(t.b.python_container, dict)

  def test_with_dataset_list(self):
    t = type_conversions.infer_type(
        [tf.data.Dataset.from_tensors(x) for x in [1, True, [0.5]]])
    self.assertEqual(str(t), '<int32*,bool*,float32[1]*>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, list)

  def test_with_nested_dataset_list_tuple(self):
    t = type_conversions.infer_type(
        tuple([(tf.data.Dataset.from_tensors(x),) for x in [1, True, [0.5]]]))
    self.assertEqual(str(t), '<<int32*>,<bool*>,<float32[1]*>>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)

  def test_with_dataset_of_named_tuple(self):
    test_named_tuple = collections.namedtuple('_', 'A B')
    t = type_conversions.infer_type(
        tf.data.Dataset.from_tensor_slices({
            'x': [0.0],
            'y': [1],
        }).map(lambda v: test_named_tuple(v['x'], v['y'])))
    self.assertEqual(str(t), '<A=float32,B=int32>*')
    self.assertIsInstance(t.element, computation_types.StructWithPythonType)
    self.assertIs(t.element.python_container, test_named_tuple)

  def test_with_empty_tuple(self):
    t = type_conversions.infer_type(())
    self.assertEqual(t, computation_types.StructWithPythonType([], tuple))


class TypeToTfDtypesAndShapesTest(test.TestCase):

  def test_with_int_scalar(self):
    type_signature = computation_types.TensorType(tf.int32)
    dtypes, shapes = type_conversions.type_to_tf_dtypes_and_shapes(
        type_signature)
    test.assert_nested_struct_eq(dtypes, tf.int32)
    test.assert_nested_struct_eq(shapes, tf.TensorShape([]))

  def test_with_int_vector(self):
    type_signature = computation_types.TensorType(tf.int32, [10])
    dtypes, shapes = type_conversions.type_to_tf_dtypes_and_shapes(
        type_signature)
    test.assert_nested_struct_eq(dtypes, tf.int32)
    test.assert_nested_struct_eq(shapes, tf.TensorShape([10]))

  def test_with_tensor_triple(self):
    type_signature = computation_types.StructWithPythonType([
        ('a', computation_types.TensorType(tf.int32, [5])),
        ('b', computation_types.TensorType(tf.bool)),
        ('c', computation_types.TensorType(tf.float32, [3])),
    ], collections.OrderedDict)
    dtypes, shapes = type_conversions.type_to_tf_dtypes_and_shapes(
        type_signature)
    test.assert_nested_struct_eq(dtypes, {
        'a': tf.int32,
        'b': tf.bool,
        'c': tf.float32
    })
    test.assert_nested_struct_eq(shapes, {
        'a': tf.TensorShape([5]),
        'b': tf.TensorShape([]),
        'c': tf.TensorShape([3])
    })

  def test_with_two_level_tuple(self):
    type_signature = computation_types.StructWithPythonType([
        ('a', tf.bool),
        ('b',
         computation_types.StructWithPythonType([
             ('c', computation_types.TensorType(tf.float32)),
             ('d', computation_types.TensorType(tf.int32, [20])),
         ], collections.OrderedDict)),
        ('e', computation_types.StructType([])),
    ], collections.OrderedDict)
    dtypes, shapes = type_conversions.type_to_tf_dtypes_and_shapes(
        type_signature)
    test.assert_nested_struct_eq(dtypes, {
        'a': tf.bool,
        'b': {
            'c': tf.float32,
            'd': tf.int32
        },
        'e': (),
    })
    test.assert_nested_struct_eq(
        shapes, {
            'a': tf.TensorShape([]),
            'b': {
                'c': tf.TensorShape([]),
                'd': tf.TensorShape([20])
            },
            'e': (),
        })


class TypeToTfTensorSpecsTest(test.TestCase):

  def test_with_int_scalar(self):
    type_signature = computation_types.TensorType(tf.int32)
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    test.assert_nested_struct_eq(tensor_specs, tf.TensorSpec([], tf.int32))

  def test_with_int_vector(self):
    type_signature = computation_types.TensorType(tf.int32, [10])
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    test.assert_nested_struct_eq(tensor_specs, tf.TensorSpec([10], tf.int32))

  def test_with_tensor_triple(self):
    type_signature = computation_types.StructWithPythonType([
        ('a', computation_types.TensorType(tf.int32, [5])),
        ('b', computation_types.TensorType(tf.bool)),
        ('c', computation_types.TensorType(tf.float32, [3])),
    ], collections.OrderedDict)
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    test.assert_nested_struct_eq(
        tensor_specs, {
            'a': tf.TensorSpec([5], tf.int32),
            'b': tf.TensorSpec([], tf.bool),
            'c': tf.TensorSpec([3], tf.float32)
        })

  def test_with_two_level_tuple(self):
    type_signature = computation_types.StructWithPythonType([
        ('a', tf.bool),
        ('b',
         computation_types.StructWithPythonType([
             ('c', computation_types.TensorType(tf.float32)),
             ('d', computation_types.TensorType(tf.int32, [20])),
         ], collections.OrderedDict)),
        ('e', computation_types.StructType([])),
    ], collections.OrderedDict)
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    test.assert_nested_struct_eq(
        tensor_specs, {
            'a': tf.TensorSpec([], tf.bool),
            'b': {
                'c': tf.TensorSpec([], tf.float32),
                'd': tf.TensorSpec([20], tf.int32)
            },
            'e': (),
        })

  def test_with_invalid_type(self):
    with self.assertRaises(TypeError):
      type_conversions.type_to_tf_tensor_specs(tf.constant([0.0]))

  def test_with_unnamed_element(self):
    type_signature = computation_types.StructType([tf.int32])
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    test.assert_nested_struct_eq(tensor_specs, (tf.TensorSpec([], tf.int32),))


class TypeToTfStructureTest(test.TestCase):

  def test_with_names(self):
    expected_structure = collections.OrderedDict([
        ('a', tf.TensorSpec(shape=(), dtype=tf.bool)),
        ('b',
         collections.OrderedDict([
             ('c', tf.TensorSpec(shape=(), dtype=tf.float32)),
             ('d', tf.TensorSpec(shape=(20,), dtype=tf.int32)),
         ])),
    ])
    type_spec = computation_types.StructWithPythonType(expected_structure,
                                                       collections.OrderedDict)
    tf_structure = type_conversions.type_to_tf_structure(type_spec)
    with tf.Graph().as_default():
      ds = tf.data.experimental.from_variant(
          tf.compat.v1.placeholder(tf.variant, shape=[]),
          structure=tf_structure)
      actual_structure = ds.element_spec
      self.assertEqual(expected_structure, actual_structure)

  def test_without_names(self):
    expected_structure = (
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    type_spec = computation_types.StructWithPythonType(expected_structure,
                                                       tuple)
    tf_structure = type_conversions.type_to_tf_structure(type_spec)
    with tf.Graph().as_default():
      ds = tf.data.experimental.from_variant(
          tf.compat.v1.placeholder(tf.variant, shape=[]),
          structure=tf_structure)
      actual_structure = ds.element_spec
      self.assertEqual(expected_structure, actual_structure)

  def test_with_none(self):
    with self.assertRaises(TypeError):
      type_conversions.type_to_tf_structure(None)

  def test_with_sequence_type(self):
    with self.assertRaises(ValueError):
      type_conversions.type_to_tf_structure(
          computation_types.SequenceType(tf.int32))

  def test_with_inconsistently_named_elements(self):
    with self.assertRaises(ValueError):
      type_conversions.type_to_tf_structure(
          computation_types.StructType([('a', tf.int32), tf.bool]))

  def test_with_no_elements(self):
    with self.assertRaises(ValueError):
      type_conversions.type_to_tf_structure(computation_types.StructType([]))


class TypeFromTensorsTest(test.TestCase):

  def test_with_single(self):
    v = tf.Variable(0.0, name='a', dtype=tf.float32, shape=[])
    result = type_conversions.type_from_tensors(v)
    self.assertEqual(str(result), 'float32')

  def test_with_non_convert_tensors(self):
    v1 = tf.Variable(0, name='foo', dtype=tf.int32, shape=[])
    v2 = {'bar'}
    d = collections.OrderedDict([('v1', v1), ('v2', v2)])
    # TODO(b/122081673): Change Exception back to ValueError once TFF moves to
    # be TF 2.0 only
    with self.assertRaisesRegex(Exception, 'supported type'):
      type_conversions.type_from_tensors(d)

  def test_with_nested_tensors(self):
    v1 = tf.Variable(0, name='foo', dtype=tf.int32, shape=[])
    v2 = tf.Variable(0, name='bar', dtype=tf.int32, shape=[])
    d = collections.OrderedDict([('v1', v1), ('v2', v2)])
    result = type_conversions.type_from_tensors(d)
    self.assertEqual(str(result), '<v1=int32,v2=int32>')

  def test_with_list_tensors(self):
    v1 = tf.Variable(0.0, name='a', dtype=tf.float32, shape=[])
    v2 = tf.Variable(0, name='b', dtype=tf.int32, shape=[])
    l = [v1, v2]
    result = type_conversions.type_from_tensors(l)
    self.assertEqual(str(result), '<float32,int32>')

  def test_with_named_tuple(self):
    test_type = collections.namedtuple('NestedTensors', ['x', 'y'])
    v1 = tf.Variable(0.0, name='a', dtype=tf.float32, shape=[])
    v2 = tf.Variable(0, name='b', dtype=tf.int32, shape=[])
    result = type_conversions.type_from_tensors(test_type(v1, v2))
    self.assertEqual(str(result), '<x=float32,y=int32>')


class TypeToPyContainerTest(test.TestCase):

  def test_not_anon_tuple_passthrough(self):
    value = (1, 2.0)
    result = type_conversions.type_to_py_container(
        (1, 2.0),
        computation_types.StructWithPythonType([tf.int32, tf.float32],
                                               container_type=list))
    self.assertEqual(result, value)

  def test_anon_tuple_return(self):
    anon_tuple = structure.Struct([(None, 1), (None, 2.0)])
    self.assertEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructType([tf.int32, tf.float32])),
        anon_tuple)

  def test_anon_tuple_without_names_to_container_without_names(self):
    anon_tuple = structure.Struct([(None, 1), (None, 2.0)])
    types = [tf.int32, tf.float32]
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, list)),
        [1, 2.0])
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, tuple)),
        (1, 2.0))

  def test_succeeds_with_federated_namedtupletype(self):
    anon_tuple = structure.Struct([(None, 1), (None, 2.0)])
    types = [tf.int32, tf.float32]
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.FederatedType(
                computation_types.StructWithPythonType(types, list),
                placement_literals.SERVER)), [1, 2.0])
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.FederatedType(
                computation_types.StructWithPythonType(types, tuple),
                placement_literals.SERVER)), (1, 2.0))

  def test_anon_tuple_with_names_to_container_without_names_fails(self):
    anon_tuple = structure.Struct([(None, 1), ('a', 2.0)])
    types = [tf.int32, tf.float32]
    with self.assertRaisesRegex(ValueError,
                                'contains a mix of named and unnamed elements'):
      type_conversions.type_to_py_container(
          anon_tuple, computation_types.StructWithPythonType(types, tuple))
    anon_tuple = structure.Struct([('a', 1), ('b', 2.0)])
    with self.assertRaisesRegex(ValueError, 'which does not support names'):
      type_conversions.type_to_py_container(
          anon_tuple, computation_types.StructWithPythonType(types, list))

  def test_anon_tuple_with_names_to_container_with_names(self):
    anon_tuple = structure.Struct([('a', 1), ('b', 2.0)])
    types = [('a', tf.int32), ('b', tf.float32)]
    self.assertDictEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, dict)), {
                'a': 1,
                'b': 2.0
            })
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.StructWithPythonType(types,
                                                   collections.OrderedDict)),
        collections.OrderedDict([('a', 1), ('b', 2.0)]))
    test_named_tuple = collections.namedtuple('TestNamedTuple', ['a', 'b'])
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.StructWithPythonType(types, test_named_tuple)),
        test_named_tuple(a=1, b=2.0))

    @attr.s
    class TestFoo(object):
      a = attr.ib()
      b = attr.ib()

    self.assertEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, TestFoo)),
        TestFoo(a=1, b=2.0))

  def test_anon_tuple_without_names_to_container_with_names_fails(self):
    anon_tuple = structure.Struct([(None, 1), (None, 2.0)])
    types = [('a', tf.int32), ('b', tf.float32)]
    with self.assertRaisesRegex(ValueError, 'value.*with unnamed elements'):
      type_conversions.type_to_py_container(
          anon_tuple, computation_types.StructWithPythonType(types, dict))

    with self.assertRaisesRegex(ValueError, 'value.*with unnamed elements'):
      type_conversions.type_to_py_container(
          anon_tuple,
          computation_types.StructWithPythonType(types,
                                                 collections.OrderedDict))

    test_named_tuple = collections.namedtuple('TestNamedTuple', ['a', 'b'])
    with self.assertRaisesRegex(ValueError, 'value.*with unnamed elements'):
      type_conversions.type_to_py_container(
          anon_tuple,
          computation_types.StructWithPythonType(types, test_named_tuple))

    @attr.s
    class TestFoo(object):
      a = attr.ib()
      b = attr.ib()

    with self.assertRaisesRegex(ValueError, 'value.*with unnamed elements'):
      type_conversions.type_to_py_container(
          anon_tuple, computation_types.StructWithPythonType(types, TestFoo))

  def test_nested_py_containers(self):
    anon_tuple = structure.Struct([
        (None, 1), (None, 2.0),
        ('dict_key',
         structure.Struct([('a', 3),
                           ('b', structure.Struct([(None, 4), (None, 5)]))]))
    ])

    dict_subtype = computation_types.StructWithPythonType(
        [('a', tf.int32),
         ('b',
          computation_types.StructWithPythonType([tf.int32, tf.int32], tuple))],
        dict)
    type_spec = computation_types.StructType([(None, tf.int32),
                                              (None, tf.float32),
                                              ('dict_key', dict_subtype)])

    expected_nested_structure = structure.Struct([
        (None, 1),
        (None, 2.0),
        ('dict_key', {
            'a': 3,
            'b': (4, 5)
        }),
    ])

    self.assertEqual(
        type_conversions.type_to_py_container(anon_tuple, type_spec),
        expected_nested_structure)

  def test_sequence_type_with_collections_sequence_elements(self):
    dataset_yielding_sequences = tf.data.Dataset.range(5).map(lambda t: (t, t))
    converted_dataset = type_conversions.type_to_py_container(
        dataset_yielding_sequences,
        computation_types.SequenceType((tf.int64, tf.int64)))
    actual_elements = list(converted_dataset)
    expected_elements = list(dataset_yielding_sequences)
    self.assertAllEqual(actual_elements, expected_elements)

  def test_sequence_type_with_collections_mapping_elements(self):
    dataset_yielding_mappings = tf.data.Dataset.range(5).map(
        lambda t: collections.OrderedDict(a=t, b=t))
    converted_dataset = type_conversions.type_to_py_container(
        dataset_yielding_mappings,
        computation_types.SequenceType(
            collections.OrderedDict(a=tf.int64, b=tf.int64)))
    actual_elements = list(converted_dataset)
    expected_elements = list(dataset_yielding_mappings)
    self.assertAllEqual(actual_elements, expected_elements)


class TypeToNonAllEqualTest(test.TestCase):

  def test_with_bool(self):
    for x in [True, False]:
      self.assertEqual(
          str(
              type_conversions.type_to_non_all_equal(
                  computation_types.FederatedType(
                      tf.int32, placement_literals.CLIENTS, all_equal=x))),
          '{int32}@CLIENTS')


if __name__ == '__main__':
  tf.test.main()
