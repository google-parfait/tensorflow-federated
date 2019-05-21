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

from six.moves import range
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


class ConcatenateTFBlocksTest(test.TestCase):

  def test_concatenenate_tensorflow_blocks_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(None)

  def test_concatenenate_tensorflow_blocks_raises_no_iterable(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(foo)

  def test_concatenenate_tensorflow_blocks_raises_bad_comp_in_list(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bad_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(
          [foo, bad_comp])

  def test_concatenate_tensorflow_blocks_raises_list_of_one(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    with self.assertRaises(ValueError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([foo])

  def test_concatenate_tensorflow_blocks_no_arg(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda: tf.constant(1.0), None)
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar])
    self.assertIsInstance(merged_comp,
                          computation_building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(None,
                                                       [tf.float32, tf.float32])
    self.assertEqual(merged_comp.type_signature, concatenated_type)

    executable = _to_computation_impl(merged_comp)
    expected_result = anonymous_tuple.AnonymousTuple([(None, 0.0), (None, 1.0)])
    self.assertAlmostEqual(executable(), expected_result)

  def test_concatenate_tensorflow_blocks_mix_of_arg_and_no_arg(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda x: x + tf.constant(1.0),
                                       tf.float32)
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar])
    self.assertIsInstance(merged_comp,
                          computation_building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(tf.float32,
                                                       [tf.float32, tf.float32])
    self.assertEqual(merged_comp.type_signature, concatenated_type)

    executable = _to_computation_impl(merged_comp)
    expected_result = anonymous_tuple.AnonymousTuple([(None, 0.0), (None, 1.0)])
    self.assertAlmostEqual(executable(0.), expected_result)

  def test_concatenate_tensorflow_blocks_tensor_args(self):
    foo = _create_compiled_computation(lambda x: x + tf.constant(0.0),
                                       tf.float32)
    bar = _create_compiled_computation(lambda x: x + tf.constant(1.0),
                                       tf.float32)
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar])
    self.assertIsInstance(merged_comp,
                          computation_building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType([tf.float32, tf.float32],
                                                       [tf.float32, tf.float32])
    self.assertEqual(merged_comp.type_signature, concatenated_type)

    executable = _to_computation_impl(merged_comp)
    expected_result = anonymous_tuple.AnonymousTuple([(None, 1.0), (None, 1.0)])
    self.assertAlmostEqual(executable([1., 0.]), expected_result)
    expected_result = anonymous_tuple.AnonymousTuple([(None, 2.0), (None, 3.0)])
    self.assertAlmostEqual(executable([2., 2.]), expected_result)

  def test_concatenate_tensorflow_blocks_unnamed_tuple_args(self):
    foo = _create_compiled_computation(
        lambda x: [x[0] + tf.constant(0.0), x[1] + tf.constant(1.0)],
        [tf.float32, tf.float32])
    bar = _create_compiled_computation(
        lambda x: [x[0] + tf.constant(1.0), x[1] + tf.constant(1.0)],
        [tf.float32, tf.float32])
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar])
    self.assertIsInstance(merged_comp,
                          computation_building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(
        [[tf.float32, tf.float32], [tf.float32, tf.float32]],
        [[tf.float32, tf.float32], [tf.float32, tf.float32]])
    self.assertEqual(str(merged_comp.type_signature), str(concatenated_type))

    executable = _to_computation_impl(merged_comp)
    expected_1 = anonymous_tuple.AnonymousTuple([(None, 1.), (None, 1.)])
    expected_2 = anonymous_tuple.AnonymousTuple([(None, 1.), (None, 2.)])
    expected_result = anonymous_tuple.AnonymousTuple([(None, expected_1),
                                                      (None, expected_2)])

    self.assertEqual(executable([[1., 0.], [0., 1.]])[0], expected_result[0])
    self.assertEqual(executable([[1., 0.], [0., 1.]])[1], expected_result[1])

  def test_concatenate_tensorflow_blocks_named_tuple_args(self):
    foo = _create_compiled_computation(lambda x: x, [('a', tf.float32),
                                                     ('b', tf.float32)])
    bar = _create_compiled_computation(lambda x: x, [('c', tf.float32),
                                                     ('d', tf.float32)])
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar])
    self.assertIsInstance(merged_comp,
                          computation_building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(
        [[('a', tf.float32),
          ('b', tf.float32)], [('c', tf.float32), ('d', tf.float32)]],
        [[('a', tf.float32),
          ('b', tf.float32)], [('c', tf.float32), ('d', tf.float32)]])
    self.assertEqual(str(merged_comp.type_signature), str(concatenated_type))

    executable = _to_computation_impl(merged_comp)
    expected_1 = anonymous_tuple.AnonymousTuple([('a', 1.), ('b', 0.)])
    expected_2 = anonymous_tuple.AnonymousTuple([('c', 0.), ('d', 1.)])
    expected_result = anonymous_tuple.AnonymousTuple([(None, expected_1),
                                                      (None, expected_2)])

    self.assertEqual(executable([[1., 0.], [0., 1.]])[0], expected_result[0])
    self.assertEqual(executable([[1., 0.], [0., 1.]])[1], expected_result[1])

  def test_concatenate_tensorflow_blocks_sequence_parameters_and_results(self):
    foo = _create_compiled_computation(
        lambda ds: ds.reduce(tf.constant(0, tf.int64), lambda x, y: x + y),
        computation_types.SequenceType(tf.int64))

    bar = _create_compiled_computation(lambda: tf.data.Dataset.range(5), None)

    merged_reduce_comps = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, foo])
    merged_input_comps = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [bar, bar])

    concat_input_type_signature = computation_types.FunctionType(
        None, [
            computation_types.SequenceType(tf.int64),
            computation_types.SequenceType(tf.int64)
        ])
    concat_reduce_type_signature = computation_types.FunctionType(
        concat_input_type_signature.result, [tf.int64, tf.int64])

    executable_reduce = _to_computation_impl(merged_reduce_comps)
    executable_input = _to_computation_impl(merged_input_comps)

    self.assertEqual(concat_input_type_signature,
                     merged_input_comps.type_signature)
    self.assertEqual(concat_reduce_type_signature,
                     merged_reduce_comps.type_signature)
    self.assertEqual(executable_reduce(executable_input())[0], 10)
    self.assertEqual(executable_reduce(executable_input())[1], 10)


def _create_simple_selection_from_called_graph():
  noarg_tuple = _create_compiled_computation(
      lambda: [tf.constant(0.), tf.constant(1.)], None)
  called_noarg_tuple = computation_building_blocks.Call(noarg_tuple, None)
  selected_result = computation_building_blocks.Selection(
      called_noarg_tuple, index=0)
  return selected_result


class SelectionFromCalledTensorFlowBlockTest(test.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_selection_from_called_graph()
    logic = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock()
    self.assertTrue(logic.should_transform(pattern))

  def test_output_selection_should_not_transform_unselected_call(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.), tf.constant(1.)], None)
    called_noarg_tuple = computation_building_blocks.Call(noarg_tuple, None)
    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    self.assertFalse(output_selector.should_transform(called_noarg_tuple))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_selection_from_called_graph()
    logic = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock()
    parsed_selection = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, computation_building_blocks.Call)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_selection_from_called_graph()
    logic = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock()
    parsed = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)

  def test_output_selection_executes_zeroth_element(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.), tf.constant(1.)], None)
    called_noarg_tuple = computation_building_blocks.Call(noarg_tuple, None)
    selected_zero = computation_building_blocks.Selection(
        called_noarg_tuple, index=0)
    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_zero = output_selector.transform(selected_zero)
    executable_zero = _to_computation_impl(parsed_zero.function)
    self.assertEqual(executable_zero(), 0.0)

  def test_output_selection_executes_first_element(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.), tf.constant(1.)], None)
    called_noarg_tuple = computation_building_blocks.Call(noarg_tuple, None)
    selected_one = computation_building_blocks.Selection(
        called_noarg_tuple, index=1)
    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_one = output_selector.transform(selected_one)
    executable_one = _to_computation_impl(parsed_one.function)
    self.assertEqual(executable_one(), 1.0)

  def test_output_selection_executes_when_selecting_by_name(self):
    # pyformat: disable
    noarg_tuple = _create_compiled_computation(
        lambda: {'a': tf.constant(0.), 'b': tf.constant(1.)}, None)
    # pyformat: enable
    called_noarg_tuple = computation_building_blocks.Call(noarg_tuple, None)
    selected_a = computation_building_blocks.Selection(
        called_noarg_tuple, name='a')
    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_a = output_selector.transform(selected_a)
    executable_a = _to_computation_impl(parsed_a.function)
    self.assertEqual(executable_a(), 0.0)


def _create_simple_lambda_wrapping_graph():
  integer_identity = _create_compiled_computation(lambda x: x, tf.int32)
  x_ref = computation_building_blocks.Reference('x', tf.int32)
  called_integer_identity = computation_building_blocks.Call(
      integer_identity, x_ref)
  lambda_wrap = computation_building_blocks.Lambda('x', tf.int32,
                                                   called_integer_identity)
  return lambda_wrap


class LambdaWrappingGraphTest(test.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    self.assertFalse(logic.should_transform(integer_square))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    parsed_selection = logic.transform(pattern)
    self.assertIsInstance(parsed_selection,
                          computation_building_blocks.CompiledComputation)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    parsed = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)

  def test_unwraps_identity(self):
    integer_identity = _create_simple_lambda_wrapping_graph()
    lambda_unwrapper = compiled_computation_transforms.LambdaWrappingGraph()
    unwrapped_identity_function = lambda_unwrapper.transform(integer_identity)
    executable_identity = _to_computation_impl(unwrapped_identity_function)
    for k in range(5):
      self.assertEqual(executable_identity(k), k)

  def test_unwraps_square(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    x_ref = computation_building_blocks.Reference('x', tf.int32)
    called_integer_square = computation_building_blocks.Call(
        integer_square, x_ref)
    lambda_wrap = computation_building_blocks.Lambda('x', tf.int32,
                                                     called_integer_square)
    lambda_unwrapper = compiled_computation_transforms.LambdaWrappingGraph()
    unwrapped_square = lambda_unwrapper.transform(lambda_wrap)
    executable_square = _to_computation_impl(unwrapped_square)
    for k in range(5):
      self.assertEqual(executable_square(k), k * k)


def _create_simple_tuple_of_called_graphs():
  noarg_const = _create_compiled_computation(lambda: tf.constant(1.), None)
  called_const = computation_building_blocks.Call(noarg_const, None)
  tuple_of_called_graphs = computation_building_blocks.Tuple([called_const] * 2)
  return tuple_of_called_graphs


class TupleCalledGraphsTest(test.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_tuple_of_called_graphs()
    logic = compiled_computation_transforms.TupleCalledGraphs()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    self.assertFalse(tuple_parser.should_transform(integer_square))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_tuple_of_called_graphs()
    logic = compiled_computation_transforms.TupleCalledGraphs()
    parsed_selection = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, computation_building_blocks.Call)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_tuple_of_called_graphs()
    logic = compiled_computation_transforms.TupleCalledGraphs()
    parsed = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)

  def test_no_arg_functions_execute(self):
    noarg_const_0 = _create_compiled_computation(lambda: tf.constant(0.), None)
    noarg_const_1 = _create_compiled_computation(lambda: tf.constant(1), None)
    called_noarg_const_0 = computation_building_blocks.Call(noarg_const_0, None)
    called_noarg_const_1 = computation_building_blocks.Call(noarg_const_1, None)
    tuple_of_called_graphs = computation_building_blocks.Tuple(
        [called_noarg_const_0, called_noarg_const_1])
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    parsed_tuple = tuple_parser.transform(tuple_of_called_graphs)
    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    lambda_wrap = computation_building_blocks.Lambda('x', tf.int32,
                                                     parsed_tuple)
    executable = _to_computation_impl(lambda_wrap)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertEqual(executable(10)[0], 0.)
    self.assertEqual(executable(0)[1], 1)

  def test_single_function_which_takes_a_parameter_executes(self):
    noarg_const_0 = _create_compiled_computation(lambda: tf.constant(0.), None)
    integer_square = _create_compiled_computation(lambda x: x**2, tf.int32)
    called_noarg_const_0 = computation_building_blocks.Call(noarg_const_0, None)
    square_arg = computation_building_blocks.Reference('x', tf.int32)
    called_square = computation_building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = computation_building_blocks.Tuple(
        [called_noarg_const_0, called_square])
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    parsed_tuple = tuple_parser.transform(tuple_of_called_graphs)
    lambda_wrap = computation_building_blocks.Lambda('x', tf.int32,
                                                     parsed_tuple)
    executable = _to_computation_impl(lambda_wrap)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    for k in range(5):
      self.assertEqual(executable(k)[0], 0.)
      self.assertEqual(executable(k)[1], k**2)

  def test_two_functions_which_takes_tensor_parameters_executes(self):
    float_cube = _create_compiled_computation(lambda x: x**3, tf.float32)
    integer_square = _create_compiled_computation(lambda x: x**2, tf.int32)
    cube_arg = computation_building_blocks.Reference('y', tf.float32)
    called_cube = computation_building_blocks.Call(float_cube, cube_arg)
    square_arg = computation_building_blocks.Reference('x', tf.int32)
    called_square = computation_building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = computation_building_blocks.Tuple(
        [called_cube, called_square])
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    parsed_tuple = tuple_parser.transform(tuple_of_called_graphs)
    lambda_arg = computation_building_blocks.Reference('lambda_arg',
                                                       [tf.float32, tf.int32])
    block_to_result = computation_building_blocks.Block(
        [('y', computation_building_blocks.Selection(lambda_arg, index=0)),
         ('x', computation_building_blocks.Selection(lambda_arg, index=1))],
        parsed_tuple)
    lambda_wrap = computation_building_blocks.Lambda('lambda_arg',
                                                     [tf.float32, tf.int32],
                                                     block_to_result)
    executable = _to_computation_impl(lambda_wrap)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertRegexMatch(parsed_tuple.tff_repr,
                          [r'comp#[a-zA-Z0-9]{8}\(<y,x>\)'])
    for k in range(5):
      self.assertEqual(executable([k * 1., k])[0], (k * 1.)**3)
      self.assertEqual(executable([k * 1., k])[1], k**2)

  def test_tensor_plus_tuple_parameter_executes(self):
    select_from_tuple = _create_compiled_computation(lambda x: x[0],
                                                     [tf.float32, tf.float32])
    integer_square = _create_compiled_computation(lambda x: x**2, tf.int32)
    selection_arg = computation_building_blocks.Reference(
        'y', [tf.float32, tf.float32])
    called_selection = computation_building_blocks.Call(select_from_tuple,
                                                        selection_arg)
    square_arg = computation_building_blocks.Reference('x', tf.int32)
    called_square = computation_building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = computation_building_blocks.Tuple(
        [called_selection, called_square])
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    parsed_tuple = tuple_parser.transform(tuple_of_called_graphs)
    lambda_arg = computation_building_blocks.Reference(
        'lambda_arg', [[tf.float32, tf.float32], tf.int32])
    block_to_result = computation_building_blocks.Block(
        [('y', computation_building_blocks.Selection(lambda_arg, index=0)),
         ('x', computation_building_blocks.Selection(lambda_arg, index=1))],
        parsed_tuple)
    lambda_wrap = computation_building_blocks.Lambda(
        'lambda_arg', [[tf.float32, tf.float32], tf.int32], block_to_result)
    executable = _to_computation_impl(lambda_wrap)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertRegexMatch(parsed_tuple.tff_repr,
                          [r'comp#[a-zA-Z0-9]{8}\(<y,x>\)'])
    for k in range(5):
      self.assertEqual(executable([[k * 1., k * 2.], k])[0], k * 1.)
      self.assertEqual(executable([[k * 1., k * 2.], k])[1], k**2)

  def test_tensor_plus_named_tuple_parameter_executes(self):
    select_from_tuple = _create_compiled_computation(lambda x: x.a,
                                                     [('a', tf.float32),
                                                      ('b', tf.float32)])
    integer_square = _create_compiled_computation(lambda x: x**2, tf.int32)
    selection_arg = computation_building_blocks.Reference(
        'y', [('a', tf.float32), ('b', tf.float32)])
    called_selection = computation_building_blocks.Call(select_from_tuple,
                                                        selection_arg)
    square_arg = computation_building_blocks.Reference('x', tf.int32)
    called_square = computation_building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = computation_building_blocks.Tuple(
        [called_selection, called_square])
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    parsed_tuple = tuple_parser.transform(tuple_of_called_graphs)
    lambda_arg = computation_building_blocks.Reference(
        'lambda_arg', [[('a', tf.float32), ('b', tf.float32)], tf.int32])
    block_to_result = computation_building_blocks.Block(
        [('y', computation_building_blocks.Selection(lambda_arg, index=0)),
         ('x', computation_building_blocks.Selection(lambda_arg, index=1))],
        parsed_tuple)
    lambda_wrap = computation_building_blocks.Lambda(
        'lambda_arg', [[('a', tf.float32),
                        ('b', tf.float32)], tf.int32], block_to_result)
    executable = _to_computation_impl(lambda_wrap)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertRegexMatch(parsed_tuple.tff_repr,
                          [r'comp#[a-zA-Z0-9]{8}\(<y,x>\)'])
    for k in range(5):
      self.assertEqual(executable([[k * 1., k * 2.], k])[0], k * 1.)
      self.assertEqual(executable([[k * 1., k * 2.], k])[1], k**2)


if __name__ == '__main__':
  test.main()
