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

# Dependency imports

import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test_utils as common_test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import test_utils
from tensorflow_federated.python.core.impl import type_utils


class GraphUtilsTest(common_test_utils.TffTestCase):

  def _assert_binding_matches_type_and_value(self, binding, type_spec, val,
                                             graph):
    """Asserts that 'bindings' matches the given type, value, and graph."""
    self.assertIsInstance(binding, pb.TensorFlow.Binding)
    self.assertIsInstance(type_spec, computation_types.Type)
    binding_oneof = binding.WhichOneof('binding')
    if binding_oneof == 'tensor':
      self.assertTrue(tf.contrib.framework.is_tensor(val))
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
      test_utils.assert_nested_struct_eq(val.output_types, output_dtypes)
      test_utils.assert_nested_struct_eq(val.output_shapes, output_shapes)
    else:
      self.assertEqual(binding_oneof, 'tuple')
      self.assertIsInstance(type_spec, computation_types.NamedTupleType)
      if not isinstance(val, (list, tuple, anonymous_tuple.AnonymousTuple)):
        self.assertIsInstance(val, dict)
        val = list(val.values())
      for idx, e in enumerate(anonymous_tuple.to_elements(type_spec)):
        self._assert_binding_matches_type_and_value(binding.tuple.element[idx],
                                                    e[1], val[idx], graph)

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
    type_spec, binding = graph_utils.capture_result_from_graph(result)
    self._assert_binding_matches_type_and_value(binding, type_spec, result,
                                                tf.get_default_graph())
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
    test_utils.assert_nested_struct_eq(x.output_types, tf.bool)
    test_utils.assert_nested_struct_eq(x.output_shapes, tf.TensorShape([]))

  def test_stamp_parameter_in_graph_with_int_vector_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo', computation_types.SequenceType((tf.int32, [50])))
    self.assertIsInstance(x, tf.data.Dataset)
    test_utils.assert_nested_struct_eq(x.output_types, tf.int32)
    test_utils.assert_nested_struct_eq(x.output_shapes, tf.TensorShape([50]))

  def test_stamp_parameter_in_graph_with_tensor_pair_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo',
          computation_types.SequenceType([('A', (tf.float32, [3, 4, 5])),
                                          ('B', (tf.int32, [1]))]))
    self.assertIsInstance(x, tf.data.Dataset)
    test_utils.assert_nested_struct_eq(x.output_types, {
        'A': tf.float32,
        'B': tf.int32
    })
    test_utils.assert_nested_struct_eq(x.output_shapes, {
        'A': tf.TensorShape([3, 4, 5]),
        'B': tf.TensorShape([1])
    })

  def test_capture_result_with_int_scalar(self):
    self.assertEqual(
        str(self._checked_capture_result(tf.placeholder(tf.int32, shape=[]))),
        'int32')

  def test_capture_result_with_int_var(self):
    # Verifies that the variable dtype is not being captured as `int32_ref`,
    # since TFF has no concept of passing arguments by reference.
    self.assertEqual(
        str(
            self._checked_capture_result(
                tf.get_variable('foo', dtype=tf.int32, shape=[]))), 'int32')

  def test_capture_result_with_scalar_list(self):
    self.assertEqual(
        str(self._checked_capture_result([tf.constant(1),
                                          tf.constant(True)])), '<int32,bool>')

  def test_capture_result_with_scalar_tuple(self):
    self.assertEqual(
        str(self._checked_capture_result((tf.constant(1), tf.constant(True)))),
        '<int32,bool>')

  def test_capture_result_with_scalar_ordered_dict(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                collections.OrderedDict([('a', tf.constant(1)),
                                         ('b', tf.constant(True))]))),
        '<a=int32,b=bool>')

  def test_capture_result_with_scalar_unordered_dict(self):
    self.assertIn(
        str(
            self._checked_capture_result({
                'a': tf.constant(1),
                'b': tf.constant(True)
            })), ['<a=int32,b=bool>', '<b=bool,a=int32>'])

  def test_capture_result_with_scalar_namedtuple(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                collections.namedtuple('_', 'x y')(tf.constant(1),
                                                   tf.constant(True)))),
        '<x=int32,y=bool>')

  def test_capture_result_with_scalar_anonymous_tuple(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                anonymous_tuple.AnonymousTuple([('x', tf.constant(10)),
                                                (None, tf.constant(True)),
                                                ('y', tf.constant(0.66))]))),
        '<x=int32,bool,y=float32>')

  def test_capture_result_with_nested_mixture_of_lists_and_tuples(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                anonymous_tuple.AnonymousTuple(
                    [('x', collections.namedtuple('_', 'a b')({
                        'p': {
                            'q': tf.constant(True)
                        }
                    }, [tf.constant(False)])), (None, [[tf.constant(10)]])]))),
        '<x=<a=<p=<q=bool>>,b=<bool>>,<<int32>>>')

  def test_capture_result_with_int_sequence_from_tensor(self):
    ds = tf.data.Dataset.from_tensors(tf.constant(10))
    self.assertEqual(str(self._checked_capture_result(ds)), 'int32*')

  def test_capture_result_with_int_sequence_from_range(self):
    ds = tf.data.Dataset.range(10)
    self.assertEqual(str(self._checked_capture_result(ds)), 'int64*')

  def test_compute_map_from_bindings_with_tuple_of_tensors(self):
    _, source = graph_utils.capture_result_from_graph(
        collections.OrderedDict([('foo', tf.constant(10, name='A')),
                                 ('bar', tf.constant(20, name='B'))]))
    _, target = graph_utils.capture_result_from_graph(
        collections.OrderedDict([('foo', tf.constant(30, name='C')),
                                 ('bar', tf.constant(40, name='D'))]))
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
    _, binding = graph_utils.capture_result_from_graph(
        collections.OrderedDict([('foo', tf.constant(10, name='A')),
                                 ('bar', tf.constant(20, name='B'))]))
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

  def test_make_data_set_from_elements_for_int_list(self):
    ds = graph_utils.make_data_set_from_elements(
        tf.get_default_graph(), [5, 7, 13, 9, 2, 50, 20], tf.int32)
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.Session().run(ds.reduce(np.int32(0), lambda x, y: x + y)), 106)

  def test_make_data_set_from_elements_for_int_pair_list(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'A': 2,
        'B': 3
    }, {
        'A': 4,
        'B': 5
    }, {
        'A': 6,
        'B': 7
    }], [('A', tf.int32), ('B', tf.int32)])
    self.assertIsInstance(ds, tf.data.Dataset)
    result = tf.Session().run(
        ds.reduce({
            'A': np.int32(0),
            'B': np.int32(1)
        }, lambda x, y: (x['A'] + y['A'], x['B'] * y['B'])))
    self.assertEqual(set(result.keys()), set(['A', 'B']))
    self.assertEqual(result['A'], 12)
    self.assertEqual(result['B'], 105)

  def test_fetch_value_in_session_without_data_sets(self):
    x = anonymous_tuple.AnonymousTuple([('A',
                                         anonymous_tuple.AnonymousTuple(
                                             [('B', tf.constant(10))]))])
    with self.session() as sess:
      y = graph_utils.fetch_value_in_session(x, sess)
    self.assertEqual(str(y), '<A=<B=10>>')


if __name__ == '__main__':
  tf.test.main()
