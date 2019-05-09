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
"""Tests for graph_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import attr
import numpy as np
import six
from six.moves import range
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
      self.assertIsInstance(val, graph_utils.DATASET_REPRESENTATION_TYPES)
      sequence_oneof = binding.sequence.WhichOneof('binding')
      if sequence_oneof == 'iterator_string_handle_name':
        # TODO(b/129956296): Eventually delete this deprecated code path.
        handle = graph.get_tensor_by_name(
            binding.sequence.iterator_string_handle_name)
        self.assertIn(
            str(handle.op.type), ['Placeholder', 'IteratorToStringHandle'])
        self.assertEqual(handle.dtype, tf.string)
      else:
        self.assertEqual(sequence_oneof, 'variant_tensor_name')
        variant_tensor = graph.get_tensor_by_name(
            binding.sequence.variant_tensor_name)
        op = str(variant_tensor.op.type)
        self.assertTrue((op == 'Placeholder') or ('Dataset' in op))
        self.assertEqual(variant_tensor.dtype, tf.variant)
      self.assertIsInstance(type_spec, computation_types.SequenceType)
      output_dtypes, output_shapes = (
          type_utils.type_to_tf_dtypes_and_shapes(type_spec.element))
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_types(val), output_dtypes)
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_shapes(val), output_shapes)
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
    graph = tf.get_default_graph()
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
      x = self._checked_stamp_parameter(
          'foo',
          computation_types.NamedTupleType([('a', tf.int32), ('b', tf.bool)]))
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_py_container_named_tuple(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter(
          'foo',
          computation_types.NamedTupleTypeWithPyContainerType(
              [('a', tf.int32), ('b', tf.bool)], collections.OrderedDict))
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_bool_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter('foo',
                                        computation_types.SequenceType(tf.bool))
      self.assertIsInstance(x, graph_utils.DATASET_REPRESENTATION_TYPES)
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_types(x), tf.bool)
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_shapes(x), tf.TensorShape([]))

  def test_stamp_parameter_in_graph_with_int_vector_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo', computation_types.SequenceType((tf.int32, [50])))
      self.assertIsInstance(x, graph_utils.DATASET_REPRESENTATION_TYPES)
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_types(x), tf.int32)
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_shapes(x), tf.TensorShape([50]))

  def test_stamp_parameter_in_graph_with_tensor_ordered_dict_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo',
          computation_types.SequenceType(
              collections.OrderedDict([('A', (tf.float32, [3, 4, 5])),
                                       ('B', (tf.int32, [1]))])))
      self.assertIsInstance(x, graph_utils.DATASET_REPRESENTATION_TYPES)
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_types(x), {
              'A': tf.float32,
              'B': tf.int32
          })
      test.assert_nested_struct_eq(
          tf.compat.v1.data.get_output_shapes(x), {
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

  @test.graph_mode_test
  def test_capture_result_with_int_placeholder(self):
    self.assertEqual(
        str(self._checked_capture_result(tf.placeholder(tf.int32, shape=[]))),
        'int32')

  @test.graph_mode_test
  def test_capture_result_with_int_variable(self):
    # Verifies that the variable dtype is not being captured as `int32_ref`,
    # since TFF has no concept of passing arguments by reference.
    self.assertEqual(
        str(
            self._checked_capture_result(
                tf.get_variable('foo', dtype=tf.int32, shape=[]))), 'int32')

  @test.graph_mode_test
  def test_capture_result_with_list_of_constants(self):
    t = self._checked_capture_result([tf.constant(1), tf.constant(True)])
    self.assertEqual(str(t), '<int32,bool>')
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), list)

  @test.graph_mode_test
  def test_capture_result_with_tuple_of_constants(self):
    t = self._checked_capture_result((tf.constant(1), tf.constant(True)))
    self.assertEqual(str(t), '<int32,bool>')
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), tuple)

  @test.graph_mode_test
  def test_capture_result_with_dict_of_constants(self):
    t1 = self._checked_capture_result({
        'a': tf.constant(1),
        'b': tf.constant(True),
    })
    self.assertEqual(str(t1), '<a=int32,b=bool>')
    self.assertIsInstance(t1,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t1), dict)

    t2 = self._checked_capture_result({
        'b': tf.constant(True),
        'a': tf.constant(1),
    })
    self.assertEqual(str(t2), '<a=int32,b=bool>')
    self.assertIsInstance(t2,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t2), dict)

  @test.graph_mode_test
  def test_capture_result_with_ordered_dict_of_constants(self):
    t = self._checked_capture_result(
        collections.OrderedDict([
            ('b', tf.constant(True)),
            ('a', tf.constant(1)),
        ]))
    self.assertEqual(str(t), '<b=bool,a=int32>')
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), collections.OrderedDict)

  @test.graph_mode_test
  def test_capture_result_with_namedtuple_of_constants(self):
    test_named_tuple = collections.namedtuple('_', 'x y')
    t = self._checked_capture_result(
        test_named_tuple(tf.constant(1), tf.constant(True)))
    self.assertEqual(str(t), '<x=int32,y=bool>')
    self.assertIsInstance(t,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t), test_named_tuple)

  @test.graph_mode_test
  def test_capture_result_with_attrs_of_constants(self):

    @attr.s
    class TestFoo(object):
      x = attr.ib()
      y = attr.ib()

    graph = tf.get_default_graph()
    type_spec, _ = graph_utils.capture_result_from_graph(
        TestFoo(tf.constant(1), tf.constant(True)), graph)
    self.assertEqual(str(type_spec), '<x=int32,y=bool>')
    self.assertIsInstance(type_spec,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            type_spec), TestFoo)

  @test.graph_mode_test
  def test_capture_result_with_anonymous_tuple_of_constants(self):
    t = self._checked_capture_result(
        anonymous_tuple.AnonymousTuple([
            ('x', tf.constant(10)),
            (None, tf.constant(True)),
            ('y', tf.constant(0.66)),
        ]))
    self.assertEqual(str(t), '<x=int32,bool,y=float32>')
    self.assertIsInstance(t, computation_types.NamedTupleType)
    self.assertNotIsInstance(
        t, computation_types.NamedTupleTypeWithPyContainerType)

  @test.graph_mode_test
  def test_capture_result_with_nested_lists_and_tuples(self):
    named_tuple_type = collections.namedtuple('_', 'a b')
    t = self._checked_capture_result(
        anonymous_tuple.AnonymousTuple([
            ('x',
             named_tuple_type({'p': {
                 'q': tf.constant(True)
             }}, [tf.constant(False)])),
            (None, [[tf.constant(10)]]),
        ]))
    self.assertEqual(str(t), '<x=<a=<p=<q=bool>>,b=<bool>>,<<int32>>>')
    self.assertIsInstance(t, computation_types.NamedTupleType)
    self.assertNotIsInstance(
        t, computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIsInstance(t.x,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t.x), named_tuple_type)
    self.assertIsInstance(t[1],
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            t[1]), list)

  @test.graph_mode_test
  def test_capture_result_with_sequence_of_ints_using_from_tensors(self):
    ds = tf.data.Dataset.from_tensors(tf.constant(10))
    self.assertEqual(str(self._checked_capture_result(ds)), 'int32*')

  @test.graph_mode_test
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

  @test.graph_mode_test
  def test_assemble_result_from_graph_with_named_tuple(self):
    test_named_tuple = collections.namedtuple('_', 'X Y')
    type_spec = test_named_tuple(tf.int32, tf.int32)
    binding = pb.TensorFlow.Binding(
        tuple=pb.TensorFlow.NamedTupleBinding(element=[
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name='P')),
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name='Q'))
        ]))
    tensor_a = tf.constant(1, name='A')
    tensor_b = tf.constant(2, name='B')
    output_map = {'P': tensor_a, 'Q': tensor_b}
    result = graph_utils.assemble_result_from_graph(type_spec, binding,
                                                    output_map)
    self.assertIsInstance(result, test_named_tuple)
    self.assertEqual(result.X, tensor_a)
    self.assertEqual(result.Y, tensor_b)

  @test.graph_mode_test
  def test_assemble_result_from_graph_with_sequence_of_odicts(self):
    type_spec = computation_types.SequenceType(
        collections.OrderedDict([('X', tf.int32), ('Y', tf.int32)]))
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
    self.assertIsInstance(result, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        str(result.output_types),
        'OrderedDict([(\'X\', tf.int32), (\'Y\', tf.int32)])')
    self.assertEqual(
        str(result.output_shapes),
        'OrderedDict([(\'X\', TensorShape([])), (\'Y\', TensorShape([]))])')

  @test.graph_mode_test
  def test_assemble_result_from_graph_with_sequence_of_namedtuples(self):
    named_tuple_type = collections.namedtuple('TestNamedTuple', 'X Y')
    type_spec = computation_types.SequenceType(
        named_tuple_type(tf.int32, tf.int32))
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
    self.assertIsInstance(result, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        str(result.output_types), 'TestNamedTuple(X=tf.int32, Y=tf.int32)')
    self.assertEqual(
        str(result.output_shapes),
        'TestNamedTuple(X=TensorShape([]), Y=TensorShape([]))')

  def test_make_dummy_element_TensorType(self):
    type_spec = computation_types.TensorType(tf.float32,
                                             [None, 10, None, 10, 10])
    elem = graph_utils._make_dummy_element_for_type_spec(type_spec)
    correct_elem = np.zeros([1, 10, 1, 10, 10], np.float32)
    self.assertEqual(elem.tolist(), correct_elem.tolist())

  def test_make_dummy_element_NamedTupleType(self):
    tensor1 = computation_types.TensorType(tf.float32, [None, 10, None, 10, 10])
    tensor2 = computation_types.TensorType(tf.int32, [10, None, 10])
    namedtuple = computation_types.NamedTupleType([('x', tensor1),
                                                   ('y', tensor2)])
    unnamedtuple = computation_types.NamedTupleType([('x', tensor1),
                                                     ('y', tensor2)])
    elem = graph_utils._make_dummy_element_for_type_spec(namedtuple)
    correct_list = [
        np.zeros([1, 10, 1, 10, 10], np.float32),
        np.zeros([10, 1, 10], np.int32)
    ]
    self.assertEqual(len(elem), len(correct_list))
    for k in range(len(elem)):
      self.assertEqual(elem[k].tolist(), correct_list[k].tolist())
    unnamed_elem = graph_utils._make_dummy_element_for_type_spec(unnamedtuple)
    self.assertEqual(len(unnamed_elem), len(correct_list))
    for k in range(len(unnamed_elem)):
      self.assertEqual(unnamed_elem[k].tolist(), correct_list[k].tolist())

  def test_nested_structures_equal(self):
    self.assertTrue(graph_utils.nested_structures_equal([10, 20], [10, 20]))
    self.assertFalse(graph_utils.nested_structures_equal([10, 20], ['x']))

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [],
                                                 tf.float32)
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(tf.Session().run(ds.reduce(1.0, lambda x, y: x + y)), 1.0)

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list_definite_tensor(self):
    ds = graph_utils.make_data_set_from_elements(
        tf.get_default_graph(), [],
        computation_types.TensorType(tf.float32, [None, 10]))
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(ds.output_shapes.as_list(),
                     tf.TensorShape([1, 10]).as_list())
    self.assertEqual(tf.Session().run(ds.reduce(1.0, lambda x, y: x + y)), 1.0)

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list_definite_tuple(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [], [
        computation_types.TensorType(tf.float32, [None, 10]),
        computation_types.TensorType(tf.float32, [None, 5])
    ])
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(ds.output_shapes, ([1, 10], [1, 5]))

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_ints(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(),
                                                 [1, 2, 3, 4], tf.int32)
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(tf.Session().run(ds.reduce(0, lambda x, y: x + y)), 10)

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': 1,
        'b': 2,
    }, {
        'a': 3,
        'b': 4,
    }], [('a', tf.int32), ('b', tf.int32)])
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  @test.graph_mode_test
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
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_lists(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [
        [[1], [2]],
        [[3], [4]],
    ], [[tf.int32], [tf.int32]])
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + tf.reduce_sum(y))), 10)

  @test.graph_mode_test
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
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)
    self.assertEqual(
        tf.Session().run(ds.reduce(0, lambda x, y: x + y['a'] + y['b'])), 10)

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_lists(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': [1],
        'b': [2],
    }, {
        'a': [3],
        'b': [4],
    }], [('a', [tf.int32]), ('b', [tf.int32])])

    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.Session().run(ds.reduce(0, reduce_fn)), 10)

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_tensors(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': 1,
        'b': 2,
    }, {
        'a': 3,
        'b': 4,
    }], [('a', tf.int32), ('b', tf.int32)])

    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.Session().run(ds.reduce(0, reduce_fn)), 10)

  @test.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_np_array(self):
    ds = graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'a': np.array([1], dtype=np.int32),
        'b': np.array([2], dtype=np.int32),
    }, {
        'a': np.array([3], dtype=np.int32),
        'b': np.array([4], dtype=np.int32),
    }], [('a', (tf.int32, [1])), ('b', (tf.int32, [1]))])
    self.assertIsInstance(ds, graph_utils.DATASET_REPRESENTATION_TYPES)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.Session().run(ds.reduce(0, reduce_fn)), 10)

  @test.graph_mode_test
  def test_fetch_value_in_session_with_string(self):
    x = tf.constant('abc')
    with tf.Session() as sess:
      y = graph_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), 'abc')

  @test.graph_mode_test
  def test_fetch_value_in_session_without_data_sets(self):
    x = anonymous_tuple.AnonymousTuple([
        ('a', anonymous_tuple.AnonymousTuple([
            ('b', tf.constant(10)),
        ])),
    ])
    with tf.Session() as sess:
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

  def test_make_data_set_from_elements_with_odd_last_batch(self):
    graph_utils.make_data_set_from_elements(
        tf.get_default_graph(),
        [np.array([1, 2]), np.array([3])],
        computation_types.TensorType(tf.int32, tf.TensorShape([None])))
    graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'x': np.array([1, 2])
    }, {
        'x': np.array([3])
    }], [('x', computation_types.TensorType(tf.int32, tf.TensorShape([None])))])

  def test_make_data_set_from_elements_with_odd_all_batches(self):
    graph_utils.make_data_set_from_elements(
        tf.get_default_graph(), [
            np.array([1, 2]),
            np.array([3]),
            np.array([4, 5, 6]),
            np.array([7, 8])
        ], computation_types.TensorType(tf.int32, tf.TensorShape([None])))
    graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'x': np.array([1, 2])
    }, {
        'x': np.array([3])
    }, {
        'x': np.array([4, 5, 6])
    }, {
        'x': np.array([7, 8])
    }], [('x', computation_types.TensorType(tf.int32, tf.TensorShape([None])))])

  def test_make_data_set_from_elements_with_just_one_batch(self):
    graph_utils.make_data_set_from_elements(
        tf.get_default_graph(), [np.array([1])],
        computation_types.TensorType(tf.int32, tf.TensorShape([None])))
    graph_utils.make_data_set_from_elements(tf.get_default_graph(), [{
        'x': np.array([1])
    }], [('x', computation_types.TensorType(tf.int32, tf.TensorShape([None])))])

  def test_one_shot_dataset_with_defuns(self):
    with tf.Graph().as_default() as graph:
      ds1 = tf.data.Dataset.from_tensor_slices([1, 1])
      it1 = ds1.make_one_shot_iterator()
      sh1 = it1.string_handle()

      dtype = tf.int32
      shape = tf.TensorShape([])

      def make():
        it2 = tf.data.Iterator.from_string_handle(sh1, dtype, shape)
        return tf.data.Dataset.range(1).repeat().map(lambda _: it2.get_next())

      ds2 = graph_utils.OneShotDataset(
          make, computation_types.TensorType(dtype, shape))

      @tf.function
      def foo():
        return ds2.reduce(np.int32(0), lambda x, y: x + y)

      result = foo()

    with tf.Session(graph=graph) as sess:
      self.assertEqual(sess.run(result), 2)

  def test_make_dataset_from_variant_tensor_constructs_dataset(self):
    with tf.Graph().as_default():
      ds = graph_utils.make_dataset_from_variant_tensor(
          tf.data.experimental.to_variant(tf.data.Dataset.range(5)), tf.int64)
      self.assertIsInstance(ds, tf.compat.v2.data.Dataset)
      result = ds.reduce(np.int64(0), lambda x, y: x + y)
      with tf.Session() as sess:
        self.assertEqual(sess.run(result), 10)

  def test_make_dataset_from_variant_tensor_fails_with_bad_tensor(self):
    with self.assertRaises(TypeError):
      with tf.Graph().as_default():
        graph_utils.make_dataset_from_variant_tensor(tf.constant(10), tf.int32)

  def test_make_dataset_from_variant_tensor_fails_with_bad_type(self):
    with self.assertRaises(TypeError):
      with tf.Graph().as_default():
        graph_utils.make_dataset_from_variant_tensor(
            tf.data.experimental.to_variant(tf.data.Dataset.range(5)), 'a')


if __name__ == '__main__':
  test.main()
