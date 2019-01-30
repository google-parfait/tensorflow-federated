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
"""Tests for graph_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import six
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import type_utils


class GraphUtilsTest(test.TestCase):

  def _assert_binding_matches_type_and_value(self, binding, type_spec, val,
                                             graph):
    """Asserts that 'bindings' matches the given type, value, and graph."""
    self.assertIsInstance(binding, pb.TensorFlow.Binding)
    self.assertIsInstance(type_spec, computation_types.Type)
    binding_oneof = binding.WhichOneof('binding')
    if binding_oneof == 'tensor':
      self.assertTrue(tf.contrib.framework.is_tensor(val))
      if not isinstance(val, tf.Variable):
        # We insert a read_value() op for Variables, which produces
        # a name we don't control. Otherwise, names should match:
        self.assertEqual(binding.tensor.tensor_name, val.name)
      self.assertIsInstance(type_spec, computation_types.TensorType)
      self.assertEqual(type_spec.dtype, val.dtype.base_dtype)
      self.assertEqual(repr(type_spec.shape), repr(val.shape))
    elif binding_oneof == 'sequence':
      self.assertIsInstance(val, tf.data.Dataset)
      handle = graph.get_tensor_by_name(
          binding.sequence.iterator_string_handle_name)
      self.assertIn(
          str(handle.op.type), ['Placeholder', 'IteratorToStringHandle'])
      self.assertEqual(handle.dtype, tf.string)
      self.assertIsInstance(type_spec, computation_types.SequenceType)
      output_dtypes, output_shapes = (
          type_utils.type_to_tf_dtypes_and_shapes(type_spec.element))
      test.assert_nested_struct_eq(val.output_types, output_dtypes)
      test.assert_nested_struct_eq(val.output_shapes, output_shapes)
    elif binding_oneof == 'tuple':
      self.assertIsInstance(type_spec, computation_types.NamedTupleType)
      if not isinstance(val, (list, tuple, anonymous_tuple.AnonymousTuple)):
        self.assertIsInstance(val, dict)
        if isinstance(val, collections.OrderedDict):
          val = list(val.values())
        else:
          val = [v for _, v in sorted(six.iteritems(val))]
      for idx, e in enumerate(anonymous_tuple.to_elements(type_spec)):
        self._assert_binding_matches_type_and_value(binding.tuple.element[idx],
                                                    e[1], val[idx], graph)
    else:
      self.fail('Unknown binding.')

  def _assert_captured_result_eq_dtype(self, type_spec, binding, dtype):
    self.assertIsInstance(type_spec, computation_types.TensorType)
    self.assertEqual(str(type_spec), dtype)
    self.assertEqual(binding.WhichOneof('binding'), 'tensor')

  def _assert_is_placeholder(self, x, name, dtype, shape, graph):
    """Verifies that 'x' is a tf.placeholder with the given attributes."""
    self.assertEqual(x.name, name)
    self.assertEqual(x.dtype, dtype)
    self.assertEqual(x.shape.ndims, len(shape))
    for i, s in enumerate(shape):
      self.assertEqual(x.shape.dims[i].value, s)
    self.assertEqual(x.op.type, 'Placeholder')
    self.assertTrue(x.graph is graph)

  def _checked_capture_result(self, result):
    """Returns the captured result type after first verifying the binding."""
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(result, graph)
    self._assert_binding_matches_type_and_value(binding, type_spec, result,
                                                graph)
    return type_spec

  def _checked_stamp_parameter(self, name, spec, graph=None):
    """Returns object stamped in the graph after verifying its bindings."""
    if graph is None:
      graph = tf.get_default_graph()
    val, binding = graph_utils.stamp_parameter_in_graph(name, spec, graph)
    self._assert_binding_matches_type_and_value(binding,
                                                computation_types.to_type(spec),
                                                val, graph)
    return val

  def test_stamp_parameter_in_graph_with_scalar_int_explicit_graph(self):
    my_graph = tf.Graph()
    x = self._checked_stamp_parameter('foo', tf.int32, my_graph)
    self._assert_is_placeholder(x, 'foo:0', tf.int32, [], my_graph)

  def test_stamp_parameter_in_graph_with_int_vector_implicit_graph(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter('bar', (tf.int32, [5]))
    self._assert_is_placeholder(x, 'bar:0', tf.int32, [5], my_graph)

  def test_stamp_parameter_in_graph_with_int_vector_undefined_size(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter('bar', (tf.int32, [None]))
    self._assert_is_placeholder(x, 'bar:0', tf.int32, [None], my_graph)

  def test_stamp_parameter_in_graph_with_named_tuple(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter('foo', (('a', tf.int32),
                                                ('b', tf.bool)))
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_bool_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter('foo',
                                        computation_types.SequenceType(tf.bool))
    self.assertIsInstance(x, tf.data.Dataset)
    test.assert_nested_struct_eq(x.output_types, tf.bool)
    test.assert_nested_struct_eq(x.output_shapes, tf.TensorShape([]))

  def test_stamp_parameter_in_graph_with_int_vector_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo', computation_types.SequenceType((tf.int32, [50])))
    self.assertIsInstance(x, tf.data.Dataset)
    test.assert_nested_struct_eq(x.output_types, tf.int32)
    test.assert_nested_struct_eq(x.output_shapes, tf.TensorShape([50]))

  def test_stamp_parameter_in_graph_with_tensor_pair_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo',
          computation_types.SequenceType([('A', (tf.float32, [3, 4, 5])),
                                          ('B', (tf.int32, [1]))]))
    self.assertIsInstance(x, tf.data.Dataset)
    test.assert_nested_struct_eq(x.output_types, {
        'A': tf.float32,
        'B': tf.int32
    })
    test.assert_nested_struct_eq(x.output_shapes, {
        'A': tf.TensorShape([3, 4, 5]),
        'B': tf.TensorShape([1])
    })

  def test_capture_result_with_string(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph('a', graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'string')

  def test_capture_result_with_int(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(1, graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32')

  def test_capture_result_with_float(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(1.0, graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float32')

  def test_capture_result_with_bool(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(True, graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'bool')

  def test_capture_result_with_np_int32(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(
          np.int32(1), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32')

  def test_capture_result_with_np_int64(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(
          np.int64(1), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int64')

  def test_capture_result_with_np_float32(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(
          np.float32(1.0), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float32')

  def test_capture_result_with_np_float64(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(
          np.float64(1.0), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float64')

  def test_capture_result_with_np_bool(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(
          np.bool(True), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'bool')

  def test_capture_result_with_np_ndarray(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = graph_utils.capture_result_from_graph(
          np.ndarray(shape=(2, 0), dtype=np.int32), graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32[2,0]')

  def test_capture_result_with_int_placeholder(self):
    self.assertEqual(
        str(self._checked_capture_result(tf.placeholder(tf.int32, shape=[]))),
        'int32')

  def test_capture_result_with_int_variable(self):
    # Verifies that the variable dtype is not being captured as `int32_ref`,
    # since TFF has no concept of passing arguments by reference.
    self.assertEqual(
        str(
            self._checked_capture_result(
                tf.get_variable('foo', dtype=tf.int32, shape=[]))), 'int32')

  def test_capture_result_with_list_of_constants(self):
    self.assertEqual(
        str(self._checked_capture_result([tf.constant(1),
                                          tf.constant(True)])), '<int32,bool>')

  def test_capture_result_with_tuple_of_constants(self):
    self.assertEqual(
        str(self._checked_capture_result((tf.constant(1), tf.constant(True)))),
        '<int32,bool>')

  def test_capture_result_with_dict_of_constants(self):
    self.assertEqual(
        str(
            self._checked_capture_result({
                'a': tf.constant(1),
                'b': tf.constant(True),
            })), '<a=int32,b=bool>')
    self.assertEqual(
        str(
            self._checked_capture_result({
                'b': tf.constant(True),
                'a': tf.constant(1),
            })), '<a=int32,b=bool>')

  def test_capture_result_with_ordered_dict_of_constants(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                collections.OrderedDict([
                    ('b', tf.constant(True)),
                    ('a', tf.constant(1)),
                ]))), '<b=bool,a=int32>')

  def test_capture_result_with_namedtuple_of_constants(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                collections.namedtuple('_', 'x y')(tf.constant(1),
                                                   tf.constant(True)))),
        '<x=int32,y=bool>')

  def test_capture_result_with_anonymous_tuple_of_constants(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                anonymous_tuple.AnonymousTuple([
                    ('x', tf.constant(10)),
                    (None, tf.constant(True)),
                    ('y', tf.constant(0.66)),
                ]))), '<x=int32,bool,y=float32>')

  def test_capture_result_with_nested_lists_and_tuples(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                anonymous_tuple.AnonymousTuple([
                    ('x', collections.namedtuple('_', 'a b')({
                        'p': {
                            'q': tf.constant(True)
                        }
                    }, [tf.constant(False)])),
                    (None, [[tf.constant(10)]]),
                ]))), '<x=<a=<p=<q=bool>>,b=<bool>>,<<int32>>>')

  def test_capture_result_with_sequence_of_ints_using_from_tensors(self):
    ds = tf.data.Dataset.from_tensors(tf.constant(10))
    self.assertEqual(str(self._checked_capture_result(ds)), 'int32*')

  def test_capture_result_with_sequence_of_ints_using_range(self):
    ds = tf.data.Dataset.range(10)
    self.assertEqual(str(self._checked_capture_result(ds)), 'int64*')

  def test_compute_map_from_bindings_with_tuple_of_tensors(self):
    with tf.Graph().as_default() as graph:
      _, source = graph_utils.capture_result_from_graph(
          collections.OrderedDict([('foo', tf.constant(10, name='A')),
                                   ('bar', tf.constant(20, name='B'))]), graph)
      _, target = graph_utils.capture_result_from_graph(
          collections.OrderedDict([('foo', tf.constant(30, name='C')),
                                   ('bar', tf.constant(40, name='D'))]), graph)
    result = graph_utils.compute_map_from_bindings(source, target)
    self.assertEqual(
        str(result), 'OrderedDict([(\'A:0\', \'C:0\'), (\'B:0\', \'D:0\')])')

  def test_compute_map_from_bindings_with_sequence(self):
    source = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(
            iterator_string_handle_name='foo'))
    target = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(
            iterator_string_handle_name='bar'))
    result = graph_utils.compute_map_from_bindings(source, target)
    self.assertEqual(str(result), 'OrderedDict([(\'foo\', \'bar\')])')

  def test_extract_tensor_names_from_binding_with_tuple_of_tensors(self):
    with tf.Graph().as_default() as graph:
      _, binding = graph_utils.capture_result_from_graph(
          collections.OrderedDict([('foo', tf.constant(10, name='A')),
                                   ('bar', tf.constant(20, name='B'))]), graph)
    result = graph_utils.extract_tensor_names_from_binding(binding)
    self.assertEqual(str(sorted(result)), '[\'A:0\', \'B:0\']')

  def test_extract_tensor_names_from_binding_with_sequence(self):
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(
            iterator_string_handle_name='foo'))
    result = graph_utils.extract_tensor_names_from_binding(binding)
    self.assertEqual(str(sorted(result)), '[\'foo\']')

  def test_assemble_result_from_graph_with_named_tuple(self):
    type_spec = [('X', tf.int32), ('Y', tf.int32)]
    binding = pb.TensorFlow.Binding(
        tuple=pb.TensorFlow.NamedTupleBinding(element=[
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name='P')),
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name='Q'))
        ]))
    output_map = {'P': tf.constant(1, name='A'), 'Q': tf.constant(2, name='B')}
    result = graph_utils.assemble_result_from_graph(type_spec, binding,
                                                    output_map)
    self.assertEqual(
        str(result), '<X=Tensor("A:0", shape=(), dtype=int32),'
        'Y=Tensor("B:0", shape=(), dtype=int32)>')

  def test_assemble_result_from_graph_with_sequence(self):
    type_spec = computation_types.SequenceType([('X', tf.int32), ('Y',
                                                                  tf.int32)])
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(
            iterator_string_handle_name='foo'))
    data_set = tf.data.Dataset.from_tensors({
        'X': tf.constant(1),
        'Y': tf.constant(2)
    })
    it = data_set.make_one_shot_iterator()
    output_map = {'foo': it.string_handle()}
    result = graph_utils.assemble_result_from_graph(type_spec, binding,
                                                    output_map)
    self.assertIsInstance(result, tf.data.Dataset)
    self.assertEqual(
        str(result.output_types),
        'OrderedDict([(\'X\', tf.int32), (\'Y\', tf.int32)])')
    self.assertEqual(
        str(result.output_shapes),
        'OrderedDict([(\'X\', TensorShape([])), (\'Y\', TensorShape([]))])')

  def test_nested_structures_equal(self):
    self.assertTrue(graph_utils.nested_structures_equal([10, 20], [10, 20]))
    self.assertFalse(graph_utils.nested_structures_equal([10, 20], ['x']))

  def test_make_data_set_from_elements_with_empty_list(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [],
                                                 tf.float32)
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(tf.Session().run(ds.reduce(1.0, lambda x, y: x + y)), 1.0)

  def test_make_data_set_from_elements_with_list_of_ints(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(),
                                                 [1, 2, 3, 4], tf.int32)
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(tf.Session().run(ds.reduce(0, lambda x, y: x + y)), 10)

  def test_make_data_set_from_elements_with_list_of_dicts(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': 1,
        'b': 2,
    }, {
        'a': 3,
        'b': 4,
    }], [('a', tf.int32), ('b', tf.int32)])
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  def test_make_data_set_from_elements_with_list_of_ordered_dicts(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [
        collections.OrderedDict([
            ('a', 1),
            ('b', 2),
        ]),
        collections.OrderedDict([
            ('a', 3),
            ('b', 4),
        ]),
    ], [('a', tf.int32), ('b', tf.int32)])
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  def test_make_data_set_from_elements_with_list_of_lists(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [
        [[1], [2]],
        [[3], [4]],
    ], [[tf.int32], [tf.int32]])
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + tf.reduce_sum(y))), 10)

  def test_make_data_set_from_elements_with_list_of_anonymous_tuples(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [
        anonymous_tuple.AnonymousTuple([
            ('a', 1),
            ('b', 2),
        ]),
        anonymous_tuple.AnonymousTuple([
            ('a', 3),
            ('b', 4),
        ]),
    ], [('a', tf.int32), ('b', tf.int32)])
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  def test_make_data_set_from_elements_with_list_of_dicts_with_lists(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': [1],
        'b': [2],
    }, {
        'a': [3],
        'b': [4],
    }], [('a', [tf.int32]), ('b', [tf.int32])])

    self.assertIsInstance(ds, tf.data.Dataset)

    def reduce_func(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.Session().run(ds.reduce(0, reduce_func)), 10)

  def test_make_data_set_from_elements_with_list_of_dicts_with_tensors(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': 1,
        'b': 2,
    }, {
        'a': 3,
        'b': 4,
    }], [('a', tf.int32), ('b', tf.int32)])

    self.assertIsInstance(ds, tf.data.Dataset)

    def reduce_func(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.Session().run(ds.reduce(0, reduce_func)), 10)

  def test_make_data_set_from_elements_with_list_of_dicts_with_np_array(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': np.array([1], dtype=np.int32),
        'b': np.array([2], dtype=np.int32),
    }, {
        'a': np.array([3], dtype=np.int32),
        'b': np.array([4], dtype=np.int32),
    }], [('a', (tf.int32, [1])), ('b', (tf.int32, [1]))])
    self.assertIsInstance(ds, tf.data.Dataset)

    def reduce_func(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.Session().run(ds.reduce(0, reduce_func)), 10)

  def test_fetch_value_in_session_without_data_sets(self):
    x = anonymous_tuple.AnonymousTuple([
        ('a', anonymous_tuple.AnonymousTuple([
            ('b', tf.constant(10)),
        ])),
    ])
    with self.session() as sess:
      y = graph_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), '<a=<b=10>>')

  def test_make_empty_list_structure_for_element_type_spec_w_tuple_dict(self):
    type_spec = computation_types.to_type(
        [tf.int32, [('a', tf.bool), ('b', tf.float32)]])
    structure = graph_utils.make_empty_list_structure_for_element_type_spec(
        type_spec)
    self.assertEqual(
        str(structure), '([], OrderedDict([(\'a\', []), (\'b\', [])]))')

  def test_append_to_list_structure_for_element_type_spec_w_tuple_dict(self):
    type_spec = computation_types.to_type(
        [tf.int32, [('a', tf.bool), ('b', tf.float32)]])
    structure = tuple([[], collections.OrderedDict([('a', []), ('b', [])])])
    for value in [[10, {'a': 20, 'b': 30}], (40, [50, 60])]:
      graph_utils.append_to_list_structure_for_element_type_spec(
          structure, value, type_spec)
    self.assertEqual(
        str(structure),
        '([10, 40], OrderedDict([(\'a\', [20, 50]), (\'b\', [30, 60])]))')

  def test_append_to_list_structure_with_too_few_element_keys(self):
    type_spec = computation_types.to_type([('a', tf.int32), ('b', tf.int32)])
    structure = collections.OrderedDict([('a', []), ('b', [])])
    value = {'a': 10}
    with self.assertRaises(TypeError):
      graph_utils.append_to_list_structure_for_element_type_spec(
          structure, value, type_spec)

  def test_append_to_list_structure_with_too_many_element_keys(self):
    type_spec = computation_types.to_type([('a', tf.int32), ('b', tf.int32)])
    structure = collections.OrderedDict([('a', []), ('b', [])])
    value = {'a': 10, 'b': 20, 'c': 30}
    with self.assertRaises(TypeError):
      graph_utils.append_to_list_structure_for_element_type_spec(
          structure, value, type_spec)

  def test_append_to_list_structure_with_too_few_unnamed_elements(self):
    type_spec = computation_types.to_type([tf.int32, tf.int32])
    structure = tuple([[], []])
    value = [10]
    with self.assertRaises(TypeError):
      graph_utils.append_to_list_structure_for_element_type_spec(
          structure, value, type_spec)

  def test_append_to_list_structure_with_too_many_unnamed_elements(self):
    type_spec = computation_types.to_type([tf.int32, tf.int32])
    structure = tuple([[], []])
    value = [10, 20, 30]
    with self.assertRaises(TypeError):
      graph_utils.append_to_list_structure_for_element_type_spec(
          structure, value, type_spec)

  def test_to_tensor_slices_from_list_structure_for_element_type_spec(self):
    type_spec = computation_types.to_type(
        [tf.int32, [('a', tf.bool), ('b', tf.float32)]])
    structure = tuple([[10, 40],
                       collections.OrderedDict([('a', [20, 50]), ('b', [30,
                                                                        60])])])
    structure = (
        graph_utils.to_tensor_slices_from_list_structure_for_element_type_spec(
            structure, type_spec))

    expected_structure = tuple([
        np.array([10, 40], dtype=np.int32),
        collections.OrderedDict([('a', np.array([True, True], dtype=np.bool)),
                                 ('b', np.array([30.0, 60.0],
                                                dtype=np.float32))])
    ])

    self.assertEqual(
        str(structure).replace(' ', ''),
        str(expected_structure).replace(' ', ''))

  def _test_list_structure(self, type_spec, elements, expected_output_str):
    structure = graph_utils.make_empty_list_structure_for_element_type_spec(
        type_spec)
    for element_value in elements:
      graph_utils.append_to_list_structure_for_element_type_spec(
          structure, element_value, type_spec)
    structure = (
        graph_utils.to_tensor_slices_from_list_structure_for_element_type_spec(
            structure, type_spec))
    self.assertEqual(
        str(structure).replace(' ', ''), expected_output_str.replace(' ', ''))

  def test_list_structures_from_element_type_spec_with_none_value(self):
    self._test_list_structure([tf.int32, [('a', tf.bool), ('b', tf.float32)]],
                              [None],
                              str(
                                  tuple([
                                      np.array([], dtype=np.int32),
                                      collections.OrderedDict([
                                          ('a', np.array([], dtype=np.bool)),
                                          ('b', np.array([], dtype=np.float32)),
                                      ])
                                  ])))

  def test_list_structures_from_element_type_spec_with_int_value(self):
    self._test_list_structure(tf.int32, [1], '[1]')

  def test_list_structures_from_element_type_spec_with_empty_dict_value(self):
    self._test_list_structure(
        computation_types.NamedTupleType([]), [{}], 'OrderedDict()')

  def test_list_structures_from_element_type_spec_with_dict_value(self):
    self._test_list_structure([('a', tf.int32), ('b', tf.int32)], [{
        'a': 1,
        'b': 2
    }, {
        'a': 1,
        'b': 2
    }], 'OrderedDict([(\'a\',array([1,1],dtype=int32)),'
                              '(\'b\',array([2,2],dtype=int32))])')

  def test_list_structures_from_element_type_spec_with_no_values(self):
    self._test_list_structure(tf.int32, [], '[]')

  def test_list_structures_from_element_type_spec_with_int_values(self):
    self._test_list_structure(tf.int32, [1, 2, 3], '[1 2 3]')

  def test_list_structures_from_element_type_spec_with_empty_dict_values(self):
    self._test_list_structure(
        computation_types.NamedTupleType([]), [{}, {}, {}], 'OrderedDict()')

  def test_list_structures_from_element_type_spec_with_anonymous_tuples(self):
    self._test_list_structure(
        computation_types.NamedTupleType([('a', tf.int32)]), [
            anonymous_tuple.AnonymousTuple([('a', 1)]),
            anonymous_tuple.AnonymousTuple([('a', 2)])
        ], 'OrderedDict([(\'a\', array([1,2],dtype=int32))])')

  def test_list_structures_from_element_type_spec_with_empty_anon_tuples(self):
    self._test_list_structure(
        computation_types.NamedTupleType([]), [
            anonymous_tuple.AnonymousTuple([]),
            anonymous_tuple.AnonymousTuple([])
        ], 'OrderedDict()')

  def test_list_structures_from_element_type_spec_w_list_of_anon_tuples(self):
    self._test_list_structure(
        computation_types.NamedTupleType([
            computation_types.NamedTupleType([('a', tf.int32)])
        ]), [[anonymous_tuple.AnonymousTuple([('a', 1)])],
             [anonymous_tuple.AnonymousTuple([('a', 2)])]],
        '(OrderedDict([(\'a\', array([1,2],dtype=int32))]),)')

  def test_make_data_set_from_elements_with_wrong_elements(self):
    with self.assertRaises(TypeError):
      graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
          'a': 1
      }, {
          'a': 2
      }], tf.int32)


if __name__ == '__main__':
  test.main()
