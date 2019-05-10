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

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import compiled_computation_transforms
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import tensorflow_serialization


def _create_compiled_computation(py_fn, arg_type):
  proto, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      py_fn, arg_type, context_stack_impl.context_stack)
  return computation_building_blocks.CompiledComputation(proto)


def _to_computation_impl(building_block):
  return computation_impl.ComputationImpl(building_block.proto,
                                          context_stack_impl.context_stack)


class CompiledComputationTransformsTest(test.TestCase):

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

  def test_permute_graph_inputs_with_none_comp_raises_type_error(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.permute_graph_inputs(None, [0])

  def test_permute_graph_inputs_with_integer_map_raises_type_error(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32)])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(TypeError):
      compiled_computation_transforms.permute_graph_inputs(foo, 0)

  def test_permute_graph_inputs_with_list_of_strings_raises_type_error(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32)])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(TypeError):
      compiled_computation_transforms.permute_graph_inputs(foo, ['a'])

  def test_permute_graph_inputs_wrong_permutation_length_raises_value_error(
      self):
    computation_arg_type = computation_types.NamedTupleType(
        [tf.int32, tf.float32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0])

  def test_permute_graph_inputs_repeated_indices_raises_value_error(self):
    computation_arg_type = computation_types.NamedTupleType(
        [tf.int32, tf.float32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0, 0])

  def test_permute_graph_inputs_large_index_raises_value_error(self):
    computation_arg_type = computation_types.NamedTupleType(
        [tf.int32, tf.float32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0, 2])

  def test_permute_graph_inputs_negative_index_raises_value_error(self):
    computation_arg_type = computation_types.NamedTupleType(
        [tf.int32, tf.float32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0, -1])

  def test_permute_graph_inputs_identity_permutation_noops(self):
    computation_arg_type = computation_types.NamedTupleType(
        [tf.int32, tf.float32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    mapped_to_identity = compiled_computation_transforms.permute_graph_inputs(
        foo, [0, 1])

    self.assertEqual(mapped_to_identity.proto, foo.proto)

  def test_permute_graph_inputs_identity_permutation_leaves_names_alone(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32)])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    mapped_to_identity = compiled_computation_transforms.permute_graph_inputs(
        foo, [0, 1])

    self.assertEqual(mapped_to_identity.proto, foo.proto)
    self.assertEqual(mapped_to_identity.type_signature, foo.type_signature)

  def test_permute_graph_inputs_flip_input_order_changes_only_parameters(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32),
                                                             ('c', tf.bool)])
    permuted_arg_type = computation_types.NamedTupleType([('c', tf.bool),
                                                          ('a', tf.int32),
                                                          ('b', tf.float32)])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    permuted_inputs = compiled_computation_transforms.permute_graph_inputs(
        foo, [2, 0, 1])

    self.assertEqual(permuted_inputs.type_signature.parameter,
                     permuted_arg_type)
    self.assertEqual(permuted_inputs.type_signature.result,
                     foo.type_signature.result)
    self.assertEqual(permuted_inputs.proto.tensorflow.graph_def,
                     foo.proto.tensorflow.graph_def)
    self.assertEqual(permuted_inputs.proto.tensorflow.initialize_op,
                     foo.proto.tensorflow.initialize_op)
    self.assertEqual(permuted_inputs.proto.tensorflow.result,
                     foo.proto.tensorflow.result)

  def test_permute_graph_inputs_flip_input_order_executes_correctly(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32),
                                                             ('c', tf.bool)])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    flipped_inputs = compiled_computation_transforms.permute_graph_inputs(
        foo, [1, 0, 2])
    executable_flipped_inputs = _to_computation_impl(flipped_inputs)
    expected_result = anonymous_tuple.AnonymousTuple([('a', 0), ('b', 1.0),
                                                      ('c', True)])
    anonymous_tuple_input = anonymous_tuple.AnonymousTuple([('b', 1.0),
                                                            ('a', 0),
                                                            ('c', True)])

    self.assertEqual(executable_flipped_inputs([1., 0, True]), expected_result)
    self.assertEqual(
        executable_flipped_inputs(anonymous_tuple_input), expected_result)
    with self.assertRaises(TypeError):
      executable_flipped_inputs([0, 1., True])
    with self.assertRaises(TypeError):
      executable_flipped_inputs(expected_result)


class WrapParameterAsTupleTest(test.TestCase):

  def test_wrap_graph_parameter_as_tuple_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.wraph_graph_parameter_as_tuple(None)

  def test_wrap_graph_parameter_as_tuple_wraps_tuple(self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.wraph_graph_parameter_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        [foo.type_signature.parameter], foo.type_signature.result)
    executable_wrapped_inputs = _to_computation_impl(wrapped_inputs)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_inputs([[1]]), executable_foo([1]))

  def test_wrap_graph_parameter_as_tuple_wraps_sequence(self):
    computation_arg_type = computation_types.SequenceType(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.wraph_graph_parameter_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        [foo.type_signature.parameter], foo.type_signature.result)
    executable_wrapped_inputs = _to_computation_impl(wrapped_inputs)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_inputs([[1]]), executable_foo([1]))

  def test_wrap_graph_parameter_as_tuple_wraps_tensor(self):
    computation_arg_type = computation_types.to_type(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.wraph_graph_parameter_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        [foo.type_signature.parameter], foo.type_signature.result)
    executable_wrapped_inputs = _to_computation_impl(wrapped_inputs)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_inputs([1]), executable_foo(1))


class GraphInputPaddingTest(test.TestCase):

  def test_pad_graph_inputs_to_match_type_raises_on_none(self):
    with self.assertRaisesRegexp(TypeError, r'Expected.*CompiledComputation'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          None, computation_types.to_type([tf.int32]))

  def test_pad_graph_inputs_to_match_type_raises_on_wrong_requested_type(self):
    comp = _create_compiled_computation(lambda x: x,
                                        computation_types.to_type([tf.int32]))
    tensor_type = computation_types.to_type(tf.int32)
    with self.assertRaisesRegexp(TypeError, r'Expected.*NamedTupleType'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, tensor_type)

  def test_pad_graph_inputs_to_match_type_raises_on_wrong_graph_parameter_type(
      self):
    comp = _create_compiled_computation(lambda x: x,
                                        computation_types.to_type(tf.int32))
    with self.assertRaisesRegexp(
        TypeError,
        r'Can only pad inputs of a CompiledComputation with parameter type tuple'
    ):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.to_type([tf.int32]))

  def test_pad_graph_inputs_to_match_type_raises_on_requested_type_too_short(
      self):
    comp = _create_compiled_computation(
        lambda x: x, computation_types.to_type([tf.int32] * 3))
    with self.assertRaisesRegexp(ValueError, r'must have more elements'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.to_type([tf.int32] * 2))

  def test_pad_graph_inputs_to_match_type_raises_on_mismatched_graph_type_and_requested_type(
      self):
    comp = _create_compiled_computation(lambda x: x,
                                        computation_types.to_type([tf.float32]))
    with self.assertRaisesRegexp(TypeError, r'must match the beginning'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.to_type([tf.int32] * 2))

  def test_pad_graph_inputs_to_match_type_preserves_named_type_signature(self):
    computation_arg_type = computation_types.to_type([('a', tf.int32)])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo,
        computation_types.NamedTupleType([('a', tf.int32), ('b', tf.float32)]))
    expetected_type_signature = computation_types.FunctionType(
        [('a', tf.int32), ('b', tf.float32)], [('a', tf.int32)])

    self.assertEqual(padded_inputs.type_signature, expetected_type_signature)

  def test_pad_graph_inputs_to_match_type_preserves_unnamed_type_signature(
      self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo, computation_types.NamedTupleType([tf.int32, tf.float32]))
    expetected_type_signature = computation_types.FunctionType(
        [tf.int32, tf.float32], [tf.int32])

    self.assertEqual(padded_inputs.type_signature, expetected_type_signature)

  def test_pad_graph_inputs_to_match_type_add_single_int_executes_correctly(
      self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo, computation_types.NamedTupleType([tf.int32, tf.float32]))
    executable_padded_inputs = _to_computation_impl(padded_inputs)

    expected_result = anonymous_tuple.AnonymousTuple([(None, 1)])

    self.assertEqual(executable_padded_inputs([1, 0.]), expected_result)
    self.assertEqual(executable_padded_inputs([1, 10.]), expected_result)


if __name__ == '__main__':
  test.main()
