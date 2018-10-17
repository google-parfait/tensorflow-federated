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
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import graph_utils

from tensorflow_federated.python.core.impl.anonymous_tuple import AnonymousTuple


class GraphUtilsTest(tf.test.TestCase):

  def _assert_binding_matches_type_and_value(
      self, binding, type_spec, val, graph):
    """Asserts that 'bindings' matches the given type, value, and graph."""
    self.assertIsInstance(binding, pb.TensorFlow.Binding)
    self.assertIsInstance(type_spec, types.Type)
    binding_oneof = binding.WhichOneof('binding')
    if binding_oneof == 'tensor':
      self.assertTrue(tf.contrib.framework.is_tensor(val))
      self.assertEqual(binding.tensor.tensor_name, val.name)
      self.assertIsInstance(type_spec, types.TensorType)
      self.assertEqual(type_spec.dtype, val.dtype)
      self.assertEqual(repr(type_spec.shape), repr(val.shape))
    elif binding_oneof == 'sequence':
      self.assertIsInstance(val, tf.data.Dataset)
      handle = graph.get_tensor_by_name(
          binding.sequence.iterator_string_handle_name)
      self.assertEqual(str(handle.op.type), 'Placeholder')
      self.assertEqual(handle.dtype, tf.string)
      self.assertIsInstance(type_spec, types.SequenceType)
      output_dtypes, output_shapes = (
          graph_utils.get_nested_structure_dtypes_and_shapes(type_spec.element))
      self._assert_nest_struct_is(val.output_types, output_dtypes)
      self._assert_nest_struct_is(val.output_shapes, output_shapes)
    else:
      self.assertEqual(binding_oneof, 'tuple')
      self.assertIsInstance(type_spec, types.NamedTupleType)
      if not isinstance(val, (list, tuple, AnonymousTuple)):
        self.assertIsInstance(val, dict)
        val = val.values()
      for idx, e in enumerate(type_spec.elements):
        self._assert_binding_matches_type_and_value(
            binding.tuple.element[idx], e[1], val[idx], graph)

  def _assert_is_placeholder(self, x, name, dtype, shape, graph):
    """Verifies that 'x' is a tf.placeholder with the given attributes."""
    self.assertEqual(x.name, name)
    self.assertEqual(x.dtype, dtype)
    self.assertEqual(x.shape.ndims, len(shape))
    for i, s in enumerate(shape):
      self.assertEqual(x.shape.dims[i].value, s)
    self.assertEqual(x.op.type, 'Placeholder')
    self.assertTrue(x.graph is graph)

  def _assert_nest_struct_is(self, x, y):
    """Verifies that nested structures 'x' and 'y' are the same."""
    tf.contrib.framework.nest.assert_same_structure(x, y)
    self.assertEqual(
        tf.contrib.framework.nest.flatten(x),
        tf.contrib.framework.nest.flatten(y))

  def _checked_capture_result(self, result):
    """Returns the captured result type after first verifying the binding."""
    type_spec, binding = graph_utils.capture_result_from_graph(result)
    self._assert_binding_matches_type_and_value(
        binding, type_spec, result, tf.get_default_graph())
    return type_spec

  def _checked_stamp_parameter(self, name, spec, graph=None):
    """Returns object stamped in the graph after verifying its bindings."""
    val, binding = graph_utils.stamp_parameter_in_graph(name, spec, graph)
    self._assert_binding_matches_type_and_value(
        binding,
        types.to_type(spec),
        val,
        graph if graph else tf.get_default_graph())
    return val

  def _do_test_nested_structure_dtypes_and_shapes(self, spec, dtypes, shapes):
    """Tests 'get_nested_structure_dtypes_and_shapes' against the given args."""
    actual_dtypes, actual_shapes = (
        graph_utils.get_nested_structure_dtypes_and_shapes(spec))
    self._assert_nest_struct_is(actual_dtypes, dtypes)
    self._assert_nest_struct_is(actual_shapes, shapes)

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
      x = self._checked_stamp_parameter(
          'foo', (('a', tf.int32), ('b', tf.bool)))
    self.assertIsInstance(x, AnonymousTuple)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_bool_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter('foo', types.SequenceType(tf.bool))
    self.assertIsInstance(x, tf.data.Dataset)
    self._assert_nest_struct_is(x.output_types, tf.bool)
    self._assert_nest_struct_is(x.output_shapes, tf.TensorShape([]))

  def test_stamp_parameter_in_graph_with_int_vector_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo', types.SequenceType((tf.int32, [50])))
    self.assertIsInstance(x, tf.data.Dataset)
    self._assert_nest_struct_is(x.output_types, tf.int32)
    self._assert_nest_struct_is(x.output_shapes, tf.TensorShape([50]))

  def test_stamp_parameter_in_graph_with_tensor_pair_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter('foo', types.SequenceType(
          [('A', (tf.float32, [3, 4, 5])), ('B', (tf.int32, [1]))]))
    self.assertIsInstance(x, tf.data.Dataset)
    self._assert_nest_struct_is(
        x.output_types, {'A': tf.float32, 'B': tf.int32})
    self._assert_nest_struct_is(
        x.output_shapes,
        {'A': tf.TensorShape([3, 4, 5]), 'B': tf.TensorShape([1])})

  def test_get_nested_structure_dtypes_and_shapes_with_int_scalar(self):
    self._do_test_nested_structure_dtypes_and_shapes(
        tf.int32, tf.int32, tf.TensorShape([]))

  def test_get_nested_structure_dtypes_and_shapes_with_int_vector(self):
    self._do_test_nested_structure_dtypes_and_shapes(
        (tf.int32, [10]), tf.int32, tf.TensorShape([10]))

  def test_get_nested_structure_dtypes_and_shapes_with_tensor_triple(self):
    self._do_test_nested_structure_dtypes_and_shapes(
        [('a', (tf.int32, [5])), ('b', tf.bool), ('c', (tf.float32, [3]))],
        {'a': tf.int32, 'b': tf.bool, 'c': tf.float32},
        {'a': tf.TensorShape([5]),
         'b': tf.TensorShape([]),
         'c': tf.TensorShape([3])})

  def test_get_nested_structure_dtypes_and_shapes_with_two_level_tuple(self):
    self._do_test_nested_structure_dtypes_and_shapes(
        [('a', tf.bool), ('b', [('c', tf.float32), ('d', (tf.int32, [20]))])],
        {'a': tf.bool, 'b': {'c': tf.float32, 'd': tf.int32}},
        {'a': tf.TensorShape([]), 'b': {
            'c': tf.TensorShape([]), 'd': tf.TensorShape([20])}})

  def test_capture_result_with_int_scalar(self):
    self.assertEqual(
        str(self._checked_capture_result(tf.placeholder(tf.int32, shape=[]))),
        'int32')

  def test_capture_result_with_scalar_list(self):
    self.assertEqual(
        str(self._checked_capture_result([tf.constant(1), tf.constant(True)])),
        '<int32,bool>')

  def test_capture_result_with_scalar_tuple(self):
    self.assertEqual(
        str(self._checked_capture_result((tf.constant(1), tf.constant(True)))),
        '<int32,bool>')

  def test_capture_result_with_scalar_ordered_dict(self):
    self.assertEqual(
        str(self._checked_capture_result(collections.OrderedDict(
            [('a', tf.constant(1)), ('b', tf.constant(True))]))),
        '<a=int32,b=bool>')

  def test_capture_result_with_scalar_unordered_dict(self):
    self.assertIn(
        str(self._checked_capture_result(
            {'a': tf.constant(1), 'b': tf.constant(True)})),
        ['<a=int32,b=bool>', '<b=bool,a=int32>'])

  def test_capture_result_with_scalar_namedtuple(self):
    self.assertEqual(
        str(self._checked_capture_result(
            collections.namedtuple('_', 'x y')(
                tf.constant(1), tf.constant(True)))),
        '<x=int32,y=bool>')

  def test_capture_result_with_scalar_anonymous_tuple(self):
    self.assertEqual(
        str(self._checked_capture_result(
            AnonymousTuple([
                ('x', tf.constant(10)),
                (None, tf.constant(True)),
                ('y', tf.constant(0.66))]))),
        '<x=int32,bool,y=float32>')

  def test_capture_result_with_nested_mixture_of_lists_and_tuples(self):
    self.assertEqual(
        str(self._checked_capture_result(
            AnonymousTuple([
                ('x', collections.namedtuple('_', 'a b')(
                    {'p': {'q': tf.constant(True)}}, [tf.constant(False)])),
                (None, [[tf.constant(10)]])]))),
        '<x=<a=<p=<q=bool>>,b=<bool>>,<<int32>>>')


if __name__ == '__main__':
  tf.test.main()
