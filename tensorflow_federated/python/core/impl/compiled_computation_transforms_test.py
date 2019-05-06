# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for compiled_computation_transforms.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import compiled_computation_transforms
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import tensorflow_serialization


def _create_compiled_computation(py_fn, arg_type):
  proto, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      py_fn, arg_type, context_stack_impl.context_stack)
  return computation_building_blocks.CompiledComputation(proto)


class CompiledComputationUtilsTest(test.TestCase):

  def test_select_graph_output_with_none_comp_raises_type_error(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.select_graph_output(None, index=0)

  def test_select_graph_output_with_no_selection_raises_value_error(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32)])

    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.select_graph_output(foo)

  def test_select_graph_output_with_wrong_return_type_raises_type_error(self):
    computation_arg_type = computation_types.to_type(tf.int32)

    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(TypeError):
      compiled_computation_transforms.select_graph_output(foo, index=0)

  def test_select_graph_output_by_name_bad_name_raises_value_error(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32)])

    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.select_graph_output(foo, name='x')

  def test_select_graph_output_by_index_single_level_of_nesting(self):
    computation_arg_type = computation_types.NamedTupleType(
        [tf.int32, tf.float32])

    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=0)
    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=1)

    self.assertEqual(first_element_selected.type_signature.result,
                     foo.type_signature.result[0])
    self.assertEqual(foo.proto.tensorflow.graph_def,
                     first_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[0].tensor,
                     first_element_selected.proto.tensorflow.result.tensor)

    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.graph_def,
                     second_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[1].tensor,
                     second_element_selected.proto.tensorflow.result.tensor)

  def test_select_graph_output_by_name_single_level_of_nesting(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32)])

    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='a')
    self.assertEqual(first_element_selected.type_signature.result,
                     computation_types.to_type(tf.int32))

    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='b')
    self.assertEqual(second_element_selected.type_signature.result,
                     computation_types.to_type(tf.float32))

    self.assertEqual(foo.proto.tensorflow.graph_def,
                     first_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[0].tensor,
                     first_element_selected.proto.tensorflow.result.tensor)

    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.graph_def,
                     second_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[1].tensor,
                     second_element_selected.proto.tensorflow.result.tensor)

  def test_select_graph_output_by_index_two_nested_levels_keeps_nested_type(
      self):
    nested_type1 = computation_types.NamedTupleType([('a', tf.int32),
                                                     ('b', tf.float32)])
    nested_type2 = computation_types.NamedTupleType([('c', tf.int32),
                                                     ('d', tf.float32)])

    computation_arg_type = computation_types.NamedTupleType([
        ('x', nested_type1), ('y', nested_type2)
    ])

    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=0)
    self.assertEqual(first_element_selected.type_signature.result, nested_type1)

    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=1)
    self.assertEqual(second_element_selected.type_signature.result,
                     nested_type2)

    self.assertEqual(foo.proto.tensorflow.graph_def,
                     first_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[0].tuple,
                     first_element_selected.proto.tensorflow.result.tuple)

    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.graph_def,
                     second_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[1].tuple,
                     second_element_selected.proto.tensorflow.result.tuple)

  def test_select_graph_output_by_name_two_nested_levels_keeps_nested_type(
      self):
    nested_type1 = computation_types.NamedTupleType([('a', tf.int32),
                                                     ('b', tf.float32)])
    nested_type2 = computation_types.NamedTupleType([('c', tf.int32),
                                                     ('d', tf.float32)])

    computation_arg_type = computation_types.NamedTupleType([
        ('x', nested_type1), ('y', nested_type2)
    ])

    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='x')
    self.assertEqual(first_element_selected.type_signature.result, nested_type1)

    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='y')
    self.assertEqual(second_element_selected.type_signature.result,
                     nested_type2)

    self.assertEqual(foo.proto.tensorflow.graph_def,
                     first_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[0].tuple,
                     first_element_selected.proto.tensorflow.result.tuple)

    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.graph_def,
                     second_element_selected.proto.tensorflow.graph_def)
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.tuple.element[1].tuple,
                     second_element_selected.proto.tensorflow.result.tuple)


if __name__ == '__main__':
  test.main()
