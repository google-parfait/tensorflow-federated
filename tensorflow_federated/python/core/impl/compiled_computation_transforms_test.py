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

import collections

from absl.testing import absltest
from absl.testing import parameterized
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import compiled_computation_transforms
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import transformation_utils


def _create_compiled_computation(py_fn, arg_type):
  proto, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      py_fn, arg_type, context_stack_impl.context_stack)
  return computation_building_blocks.CompiledComputation(proto)


def _to_computation_impl(building_block):
  return computation_impl.ComputationImpl(building_block.proto,
                                          context_stack_impl.context_stack)


class CompiledComputationTransformsTest(parameterized.TestCase):

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


class WrapParameterAsTupleTest(parameterized.TestCase):

  def test_bind_graph_parameter_as_tuple_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_parameter_as_tuple(None)

  def test_bind_graph_parameter_as_tuple_raises_on_non_string_name(self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_parameter_as_tuple(foo, name=1)

  def test_bind_graph_parameter_as_tuple_wraps_tuple(self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        [foo.type_signature.parameter], foo.type_signature.result)
    executable_wrapped_inputs = _to_computation_impl(wrapped_inputs)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_inputs([[1]]), executable_foo([1]))

  def test_bind_graph_parameter_as_tuple_wraps_sequence(self):
    computation_arg_type = computation_types.SequenceType(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        [foo.type_signature.parameter], foo.type_signature.result)
    executable_wrapped_inputs = _to_computation_impl(wrapped_inputs)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_inputs([[1]]), executable_foo([1]))

  def test_bind_graph_parameter_as_tuple_wraps_tensor(self):
    computation_arg_type = computation_types.to_type(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        [foo.type_signature.parameter], foo.type_signature.result)
    executable_wrapped_inputs = _to_computation_impl(wrapped_inputs)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_inputs([1]), executable_foo(1))

  def test_bind_graph_parameter_as_tuple_adds_name(self):
    computation_arg_type = computation_types.to_type(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo, name='a')
    expected_type_signature = computation_types.FunctionType(
        [('a', foo.type_signature.parameter)], foo.type_signature.result)
    executable_wrapped_inputs = _to_computation_impl(wrapped_inputs)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_inputs([1]), executable_foo(1))


class WrapResultAsTupleTest(parameterized.TestCase):

  def test_bind_graph_result_as_tuple_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_result_as_tuple(None)

  def test_bind_graph_result_as_tuple_raises_on_non_string_name(self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_result_as_tuple(foo, name=1)

  def test_bind_graph_result_as_tuple_wraps_tuple(self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter, [foo.type_signature.result])
    executable_wrapped_output = _to_computation_impl(wrapped_output)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_output([1])[0], executable_foo([1]))

  def test_bind_graph_result_as_tuple_wraps_sequence(self):
    computation_arg_type = computation_types.SequenceType(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter, [foo.type_signature.result])
    executable_wrapped_output = _to_computation_impl(wrapped_output)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_output([1])[0], executable_foo([1]))

  def test_bind_graph_result_as_tuple_wraps_tensor(self):
    computation_arg_type = computation_types.to_type(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo)
    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter, [foo.type_signature.result])
    executable_wrapped_output = _to_computation_impl(wrapped_output)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    self.assertEqual(executable_wrapped_output(1)[0], executable_foo(1))

  def test_bind_graph_result_as_tuple_adds_name(self):
    computation_arg_type = computation_types.to_type(tf.int32)
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo, name='a')
    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter, [('a', foo.type_signature.result)])
    executable_wrapped_output = _to_computation_impl(wrapped_output)
    executable_foo = _to_computation_impl(foo)

    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    self.assertEqual(
        executable_wrapped_output(1),
        anonymous_tuple.AnonymousTuple([('a', executable_foo(1))]))


class GraphInputPaddingTest(parameterized.TestCase):

  def test_pad_graph_inputs_to_match_type_raises_on_none(self):
    with self.assertRaisesRegex(TypeError, r'Expected.*CompiledComputation'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          None, computation_types.to_type([tf.int32]))

  def test_pad_graph_inputs_to_match_type_raises_on_wrong_requested_type(self):
    comp = _create_compiled_computation(lambda x: x,
                                        computation_types.to_type([tf.int32]))
    tensor_type = computation_types.to_type(tf.int32)
    with self.assertRaisesRegex(TypeError, r'Expected.*NamedTupleType'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, tensor_type)

  def test_pad_graph_inputs_to_match_type_raises_on_wrong_graph_parameter_type(
      self):
    comp = _create_compiled_computation(lambda x: x,
                                        computation_types.to_type(tf.int32))
    with self.assertRaisesRegex(
        TypeError,
        r'Can only pad inputs of a CompiledComputation with parameter type tuple'
    ):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.to_type([tf.int32]))

  def test_pad_graph_inputs_to_match_type_raises_on_requested_type_too_short(
      self):
    comp = _create_compiled_computation(
        lambda x: x, computation_types.to_type([tf.int32] * 3))
    with self.assertRaisesRegex(ValueError, r'must have more elements'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.to_type([tf.int32] * 2))

  def test_pad_graph_inputs_to_match_type_raises_on_mismatched_graph_type_and_requested_type(
      self):
    comp = _create_compiled_computation(lambda x: x,
                                        computation_types.to_type([tf.float32]))
    with self.assertRaisesRegex(TypeError, r'must match the beginning'):
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

  def test_pad_graph_inputs_to_match_type_adds_names_to_unnamed_tuple(self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo,
        computation_types.NamedTupleType([('a', tf.int32), ('b', tf.float32)]))
    expected_type_signature = computation_types.FunctionType(
        [('a', tf.int32), ('b', tf.float32)], [tf.int32])

    self.assertEqual(padded_inputs.type_signature, expected_type_signature)

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

  def test_pad_graph_inputs_to_match_type_adds_names_to_unnamed_tuple_and_executes(
      self):
    computation_arg_type = computation_types.to_type([tf.int32])
    foo = _create_compiled_computation(lambda x: x, computation_arg_type)
    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo,
        computation_types.NamedTupleType([('a', tf.int32), ('b', tf.float32)]))
    executable_padded_inputs = _to_computation_impl(padded_inputs)
    expected_result = anonymous_tuple.AnonymousTuple([(None, 1)])

    self.assertEqual(
        executable_padded_inputs({
            'a': 1,
            'b': 0.
        }), expected_result)
    self.assertEqual(
        executable_padded_inputs({
            'a': 1,
            'b': 10.
        }), expected_result)


class ConcatenateTFBlocksTest(parameterized.TestCase):

  def test_concatenenate_tensorflow_blocks_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(
          None, [None])

  def test_concatenenate_tensorflow_blocks_raises_no_iterable(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(foo, [None])

  def test_concatenenate_tensorflow_blocks_raises_bad_comp_in_list(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bad_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(
          [foo, bad_comp], [None, None])

  def test_concatenate_tensorflow_blocks_fails_empty_list(self):
    with self.assertRaises(ValueError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([], [None])

  def test_concatenate_tensorflow_blocks_raises_bad_names_list_length(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda: tf.constant(1.0), None)
    with self.assertRaises(ValueError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([foo, bar],
                                                                    [None])

  def test_concatenate_tensorflow_blocks_raises_bad_names_list_type(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda: tf.constant(1.0), None)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([foo, bar],
                                                                    'x')

  def test_concatenate_tensorflow_blocks_raises_bad_names_list_element_type(
      self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda: tf.constant(1.0), None)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([foo, bar],
                                                                    ['x', 1])

  def test_concatenate_tensorflow_blocks_no_arg(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda: tf.constant(1.0), None)
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar], [None, None])
    self.assertIsInstance(merged_comp,
                          computation_building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(None,
                                                       [tf.float32, tf.float32])
    self.assertEqual(merged_comp.type_signature, concatenated_type)

    executable = _to_computation_impl(merged_comp)
    expected_result = anonymous_tuple.AnonymousTuple([(None, 0.0), (None, 1.0)])
    self.assertAlmostEqual(executable(), expected_result)

  def test_concatenate_tensorflow_blocks_named_outputs_type_preserved(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda: tf.constant(1.0), None)
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar], ['a', 'b'])
    self.assertIsInstance(merged_comp,
                          computation_building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(None,
                                                       [('a', tf.float32),
                                                        ('b', tf.float32)])
    self.assertEqual(merged_comp.type_signature, concatenated_type)

  def test_concatenate_tensorflow_blocks_mix_of_arg_and_no_arg(self):
    foo = _create_compiled_computation(lambda: tf.constant(0.0), None)
    bar = _create_compiled_computation(lambda x: x + tf.constant(1.0),
                                       tf.float32)
    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar], [None, None])
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
        [foo, bar], [None, None])
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
        [foo, bar], [None, None])
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
        [foo, bar], [None, None])
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
        [foo, foo], [None, None])
    merged_input_comps = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [bar, bar], [None, None])

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


class SelectionFromCalledTensorFlowBlockTest(parameterized.TestCase):

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
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, computation_building_blocks.Call)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_selection_from_called_graph()
    logic = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_output_selection_executes_zeroth_element(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.), tf.constant(1.)], None)
    called_noarg_tuple = computation_building_blocks.Call(noarg_tuple, None)
    selected_zero = computation_building_blocks.Selection(
        called_noarg_tuple, index=0)
    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_zero, mutated = output_selector.transform(selected_zero)
    executable_zero = _to_computation_impl(parsed_zero.function)
    self.assertEqual(executable_zero(), 0.0)
    self.assertTrue(mutated)

  def test_output_selection_executes_first_element(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.), tf.constant(1.)], None)
    called_noarg_tuple = computation_building_blocks.Call(noarg_tuple, None)
    selected_one = computation_building_blocks.Selection(
        called_noarg_tuple, index=1)
    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_one, mutated = output_selector.transform(selected_one)
    executable_one = _to_computation_impl(parsed_one.function)
    self.assertEqual(executable_one(), 1.0)
    self.assertTrue(mutated)

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
    parsed_a, mutated = output_selector.transform(selected_a)
    executable_a = _to_computation_impl(parsed_a.function)
    self.assertEqual(executable_a(), 0.0)
    self.assertTrue(mutated)


def _create_simple_lambda_wrapping_graph():
  integer_identity = _create_compiled_computation(lambda x: x, tf.int32)
  x_ref = computation_building_blocks.Reference('x', tf.int32)
  called_integer_identity = computation_building_blocks.Call(
      integer_identity, x_ref)
  lambda_wrap = computation_building_blocks.Lambda('x', tf.int32,
                                                   called_integer_identity)
  return lambda_wrap


class LambdaWrappingGraphTest(parameterized.TestCase):

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
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection,
                          computation_building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_unwraps_identity(self):
    integer_identity = _create_simple_lambda_wrapping_graph()
    lambda_unwrapper = compiled_computation_transforms.LambdaWrappingGraph()
    unwrapped_identity_function, mutated = lambda_unwrapper.transform(
        integer_identity)
    executable_identity = _to_computation_impl(unwrapped_identity_function)
    self.assertTrue(mutated)
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
    unwrapped_square, mutated = lambda_unwrapper.transform(lambda_wrap)
    executable_square = _to_computation_impl(unwrapped_square)
    self.assertTrue(mutated)
    for k in range(5):
      self.assertEqual(executable_square(k), k * k)


def _create_simple_tuple_of_called_graphs():
  noarg_const = _create_compiled_computation(lambda: tf.constant(1.), None)
  called_const = computation_building_blocks.Call(noarg_const, None)
  tuple_of_called_graphs = computation_building_blocks.Tuple([called_const] * 2)
  return tuple_of_called_graphs


class TupleCalledGraphsTest(parameterized.TestCase):

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
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, computation_building_blocks.Call)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_tuple_of_called_graphs()
    logic = compiled_computation_transforms.TupleCalledGraphs()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_named_tuple_of_graphs_preserves_type(self):
    noarg_const_0 = _create_compiled_computation(lambda: tf.constant(0.), None)
    noarg_const_1 = _create_compiled_computation(lambda: tf.constant(1), None)
    called_noarg_const_0 = computation_building_blocks.Call(noarg_const_0, None)
    called_noarg_const_1 = computation_building_blocks.Call(noarg_const_1, None)
    tuple_of_called_graphs = computation_building_blocks.Tuple([
        ('a', called_noarg_const_0), ('b', called_noarg_const_1)
    ])
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)
    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertTrue(mutated)

  def test_no_arg_functions_execute(self):
    noarg_const_0 = _create_compiled_computation(lambda: tf.constant(0.), None)
    noarg_const_1 = _create_compiled_computation(lambda: tf.constant(1), None)
    called_noarg_const_0 = computation_building_blocks.Call(noarg_const_0, None)
    called_noarg_const_1 = computation_building_blocks.Call(noarg_const_1, None)
    tuple_of_called_graphs = computation_building_blocks.Tuple(
        [called_noarg_const_0, called_noarg_const_1])
    tuple_parser = compiled_computation_transforms.TupleCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)
    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    lambda_wrap = computation_building_blocks.Lambda('x', tf.int32,
                                                     parsed_tuple)
    executable = _to_computation_impl(lambda_wrap)

    self.assertTrue(mutated)
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
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)
    lambda_wrap = computation_building_blocks.Lambda('x', tf.int32,
                                                     parsed_tuple)
    executable = _to_computation_impl(lambda_wrap)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertTrue(mutated)
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
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)
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

    self.assertTrue(mutated)
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
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)
    lambda_arg = computation_building_blocks.Reference(
        'lambda_arg', [[tf.float32, tf.float32], tf.int32])
    block_to_result = computation_building_blocks.Block(
        [('y', computation_building_blocks.Selection(lambda_arg, index=0)),
         ('x', computation_building_blocks.Selection(lambda_arg, index=1))],
        parsed_tuple)
    lambda_wrap = computation_building_blocks.Lambda(
        'lambda_arg', [[tf.float32, tf.float32], tf.int32], block_to_result)
    executable = _to_computation_impl(lambda_wrap)

    self.assertTrue(mutated)
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
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)
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

    self.assertTrue(mutated)
    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertRegexMatch(parsed_tuple.tff_repr,
                          [r'comp#[a-zA-Z0-9]{8}\(<y,x>\)'])
    for k in range(5):
      self.assertEqual(executable([[k * 1., k * 2.], k])[0], k * 1.)
      self.assertEqual(executable([[k * 1., k * 2.], k])[1], k**2)


def _simulate_permutation_behavior(tuple_type, permutation):
  type_elements = anonymous_tuple.to_elements(tuple_type)
  constructed_type_elements = []
  for k in permutation:
    constructed_type_elements.append(type_elements[k])
  return computation_types.NamedTupleType(constructed_type_elements)


def _construct_permutation_tuple(n, m, offset):
  assert offset + m < n
  tuple_type_elements = [(str(k),
                          computation_types.AbstractType('T{}'.format(k)))
                         for k in range(n)]
  initial_type = computation_types.NamedTupleType(tuple_type_elements)
  selected_indices = [j + offset for j in range(m)]
  return ('tuple_type_{}_select_{}_indices_offset_{}'.format(n, m, offset),
          initial_type, selected_indices)


def _construct_permutation_tuple_collection(max_length):
  permutation_tuples = []
  for n in range(max_length):
    for m in range(n):
      for offset in range(n - m):
        permutation_tuples.append(_construct_permutation_tuple(n, m, offset))
  return permutation_tuples


class RemapGraphInputsTest(parameterized.TestCase):

  def test_raises_on_bad_computation(self):
    tuple_type = computation_types.to_type([tf.int32])
    bad_comp = computation_building_blocks.Data(
        'x', computation_types.AbstractType('T'))
    with self.assertRaises(TypeError):
      compiled_computation_transforms._remap_graph_inputs(
          bad_comp, [0], tuple_type)

  def test_raises_on_bad_type(self):
    tensor_type = computation_types.to_type(tf.int32)
    tuple_identity = _create_compiled_computation(lambda x: x, [tf.int32])
    with self.assertRaises(TypeError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, [0], tensor_type)

  def test_raises_on_non_list_of_indices(self):
    tuple_type = computation_types.to_type([tf.int32])
    tuple_identity = _create_compiled_computation(lambda x: x, [tf.int32])
    with self.assertRaises(TypeError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, 0, tuple_type)

  def test_raises_on_repeated_indices(self):
    tuple_type = computation_types.to_type([tf.int32, tf.int32])
    tuple_identity = _create_compiled_computation(lambda x: x,
                                                  [tf.int32, tf.int32])
    with self.assertRaises(ValueError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, [0, 0], tuple_type)

  def test_raises_on_bad_index(self):
    tuple_type = computation_types.to_type([tf.int32, tf.int32])
    tuple_identity = _create_compiled_computation(lambda x: x,
                                                  [tf.int32, tf.int32])
    with self.assertRaises(ValueError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, [-1, 0], tuple_type)

  def test_permute_and_pad_index_0_of_two_tuple(self):
    index_list = [0]
    tuple_type = computation_types.NamedTupleType([tf.float32, tf.int32])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(to_pad, tuple_type)
    self.assertEqual(to_permute, [0, 1])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_index_1_of_two_tuple(self):
    index_list = [1]
    tuple_type = computation_types.NamedTupleType([tf.float32, tf.int32])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(to_pad,
                     computation_types.NamedTupleType([tf.int32, tf.float32]))
    self.assertEqual(to_permute, [1, 0])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_identity_on_two_tuple(self):
    index_list = [0, 1]
    tuple_type = computation_types.NamedTupleType([tf.float32, tf.int32])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(to_pad, tuple_type)
    self.assertEqual(to_permute, [0, 1])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_inversion_of_two_tuple(self):
    index_list = [1, 0]
    tuple_type = computation_types.NamedTupleType([tf.float32, tf.int32])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(to_pad,
                     computation_types.NamedTupleType([tf.int32, tf.float32]))
    self.assertEqual(to_permute, [1, 0])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_inversion_of_named_two_tuple(self):
    index_list = [1, 0]
    tuple_type = computation_types.NamedTupleType([('a', tf.float32),
                                                   ('b', tf.int32)])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(
        to_pad,
        computation_types.NamedTupleType([('b', tf.int32), ('a', tf.float32)]))
    self.assertEqual(to_permute, [1, 0])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_single_index_deep_in_tuple(self):
    index_list = [5]
    tuple_type_list = [tf.float32, tf.int32] * 5
    tuple_type = computation_types.NamedTupleType(tuple_type_list)
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    to_pad_first_type = tuple_type_list.pop(5)
    tuple_type_list.insert(0, to_pad_first_type)
    self.assertEqual(to_pad, computation_types.NamedTupleType(tuple_type_list))
    self.assertEqual(to_permute, [1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(result_of_applying_permutation, tuple_type)

  @parameterized.named_parameters(*_construct_permutation_tuple_collection(5))
  def test_permute_and_pad_round_trip(self, initial_type, selected_indices):
    to_pad = compiled_computation_transforms._construct_padding(
        selected_indices, initial_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        selected_indices, initial_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(result_of_applying_permutation, initial_type)


def _create_simple_lambda_call_selection_from_arg():
  integer_identity = _create_compiled_computation(lambda x: x, tf.int32)
  tuple_reference = computation_building_blocks.Reference(
      'x', [tf.float32, tf.int32, tf.bool])
  selection_1 = computation_building_blocks.Selection(tuple_reference, index=1)
  called_identity = computation_building_blocks.Call(integer_identity,
                                                     selection_1)
  lambda_wrapping_call = computation_building_blocks.Lambda(
      'x', tuple_reference.type_signature, called_identity)
  return lambda_wrapping_call


class LambdaCallSelectionFromArgTest(parameterized.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_lambda_call_selection_from_arg()
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    lambda_parser = compiled_computation_transforms.LambdaCallSelectionFromArg()
    self.assertFalse(lambda_parser.should_transform(integer_square))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_lambda_call_selection_from_arg()
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection,
                          computation_building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_lambda_call_selection_from_arg()
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed_lambda, mutated = logic.transform(pattern)
    self.assertEqual(parsed_lambda.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_constructs_appropriate_type_selection_by_index(self):
    integer_identity = _create_compiled_computation(lambda x: x, tf.int32)
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.float32, tf.int32, tf.bool])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    called_identity = computation_building_blocks.Call(integer_identity,
                                                       selection_1)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed_lambda, mutated = logic.transform(lambda_wrapping_call)
    self.assertEqual(parsed_lambda.type_signature,
                     lambda_wrapping_call.type_signature)
    self.assertIsInstance(parsed_lambda,
                          computation_building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_executes_correctly_selection_by_index(self):
    integer_identity = _create_compiled_computation(lambda x: x, tf.int32)
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.float32, tf.int32, tf.bool])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    called_identity = computation_building_blocks.Call(integer_identity,
                                                       selection_1)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed, mutated = logic.transform(lambda_wrapping_call)
    executable_parsed = _to_computation_impl(parsed)
    self.assertTrue(mutated)
    for k in range(5):
      self.assertEqual(executable_parsed([k * 1., k, True]), k)

  def test_constructs_appropriate_type_selection_by_name(self):
    integer_square = _create_compiled_computation(lambda x: x**2, tf.int32)
    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.float32), ('b', tf.int32), ('c', tf.bool)])
    selection_b = computation_building_blocks.Selection(
        tuple_reference, name='b')
    called_square = computation_building_blocks.Call(integer_square,
                                                     selection_b)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_square)
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed, mutated = logic.transform(lambda_wrapping_call)
    self.assertTrue(mutated)
    self.assertEqual(parsed.type_signature, lambda_wrapping_call.type_signature)
    self.assertIsInstance(parsed,
                          computation_building_blocks.CompiledComputation)

  def test_executes_correctly_selection_by_name(self):
    integer_square = _create_compiled_computation(lambda x: x**2, tf.int32)
    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.float32), ('b', tf.int32), ('c', tf.bool)])
    selection_b = computation_building_blocks.Selection(
        tuple_reference, name='b')
    called_square = computation_building_blocks.Call(integer_square,
                                                     selection_b)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_square)
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed, mutated = logic.transform(lambda_wrapping_call)
    executable_parsed = _to_computation_impl(parsed)
    self.assertTrue(mutated)
    for k in range(5):
      self.assertEqual(
          executable_parsed({
              'a': k * 1.,
              'b': k,
              'c': True
          }), k**2)


def _create_simple_lambda_call_tuple_of_selections_from_arg():
  identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
  tuple_reference = computation_building_blocks.Reference(
      'x', [tf.float32, tf.int32, tf.bool])
  selection_1 = computation_building_blocks.Selection(tuple_reference, index=1)
  selection_0 = computation_building_blocks.Selection(tuple_reference, index=0)
  tuple_of_selections = computation_building_blocks.Tuple(
      [selection_1, selection_0])
  called_identity = computation_building_blocks.Call(identity,
                                                     tuple_of_selections)
  lambda_wrapping_call = computation_building_blocks.Lambda(
      'x', tuple_reference.type_signature, called_identity)
  return lambda_wrapping_call


class LambdaToCalledTupleOfSelectionsFromArgTest(parameterized.TestCase):

  def test_transform_raises_on_wrong_lengths(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32] * 3)
    tuple_reference = computation_building_blocks.Reference('x', [tf.int32] * 2)
    selection = computation_building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = computation_building_blocks.Tuple([selection] * 3)
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    with self.assertRaisesRegex(ValueError,
                                'Inputs to TF computations cannot be masked'):
      logic.transform(lambda_wrapping_call)

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_lambda_call_tuple_of_selections_from_arg()
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    self.assertFalse(logic.should_transform(integer_square))

  def test_does_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(integer_square)
    self.assertEqual(parsed, integer_square)
    self.assertFalse(mutated)

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_lambda_call_tuple_of_selections_from_arg()
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed,
                          computation_building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_lambda_call_tuple_of_selections_from_arg()
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_unnamed_tuple_pad_and_permute(
      self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.float32, tf.int32, tf.bool])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, index=0)
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    self.assertEqual(parsed.type_signature, lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_unnamed_tuple_pad_and_permute(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.float32, tf.int32, tf.bool])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, index=0)
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    executable_parsed = _to_computation_impl(parsed)
    result = executable_parsed([0., 1, True])
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 0.)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_unnamed_tuple_permute_only(self):
    identity = _create_compiled_computation(lambda x: x,
                                            [tf.bool, tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.float32, tf.int32, tf.bool])
    selection_2 = computation_building_blocks.Selection(
        tuple_reference, index=2)
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, index=0)
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_2, selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    self.assertEqual(parsed.type_signature, lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_unnamed_tuple_permute_only(self):
    identity = _create_compiled_computation(lambda x: x,
                                            [tf.bool, tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.float32, tf.int32, tf.bool])
    selection_2 = computation_building_blocks.Selection(
        tuple_reference, index=2)
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, index=0)
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_2, selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    executable_parsed = _to_computation_impl(parsed)
    result = executable_parsed([0., 1, True])
    self.assertEqual(result[0], True)
    self.assertEqual(result[1], 1)
    self.assertEqual(result[2], 0.)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_named_tuple_name_selection_pad_and_permute(
      self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.float32), ('b', tf.int32), ('c', tf.bool)])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, name='b')
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, name='a')
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    self.assertEqual(parsed.type_signature, lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_named_tuple_name_selection_pad_and_permute(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.float32), ('b', tf.int32), ('c', tf.bool)])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, name='b')
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, name='a')
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    executable_parsed = _to_computation_impl(parsed)
    result = executable_parsed({'a': 0., 'b': 1, 'c': False})
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 0.)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_named_tuple_index_selection(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.float32), ('b', tf.int32), ('c', tf.bool)])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, index=0)
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    self.assertEqual(parsed.type_signature, lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_named_tuple_index_selection(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.float32), ('b', tf.int32), ('c', tf.bool)])
    selection_1 = computation_building_blocks.Selection(
        tuple_reference, index=1)
    selection_0 = computation_building_blocks.Selection(
        tuple_reference, index=0)
    tuple_of_selections = computation_building_blocks.Tuple(
        [selection_1, selection_0])
    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_of_selections)
    lambda_wrapping_call = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    executable_parsed = _to_computation_impl(parsed)
    result = executable_parsed({'a': 0., 'b': 1, 'c': False})
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 0.)
    self.assertTrue(mutated)


class ComposeTensorFlowBlocksTest(parameterized.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks(None)

  def test_raises_on_single_computation(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks(identity)

  def test_raises_bad_arg_in_list(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks([identity, 0])

  def test_raises_mismatched_parameter_and_result_types(self):
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])
    bad_type_identity = _create_compiled_computation(lambda x: x,
                                                     [tf.float32, tf.int32])
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks(
          [identity, bad_type_identity])

  def test_composes_no_arg_fn_with_add_one_types_correctly(self):
    noarg_fn = _create_compiled_computation(lambda: 0, None)
    add_one_fn = _create_compiled_computation(lambda x: x + 1, tf.int32)
    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [add_one_fn, noarg_fn])
    expected_type = computation_types.FunctionType(None, tf.int32)
    self.assertEqual(composed_fn.type_signature, expected_type)

  def test_composes_no_arg_fn_with_add_one_executes_correctly(self):
    noarg_fn = _create_compiled_computation(lambda: 0, None)
    add_one_fn = _create_compiled_computation(lambda x: x + 1, tf.int32)
    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [add_one_fn, noarg_fn])
    executable_constant_one = _to_computation_impl(composed_fn)
    self.assertEqual(executable_constant_one(), 1)

  def test_composes_tensor_functions_types_correctly(self):
    int_to_float_fn = _create_compiled_computation(
        lambda x: tf.cast(x, tf.float32) * 2.0, tf.int32)
    float_to_float_fn = _create_compiled_computation(lambda x: x * 2.0,
                                                     tf.float32)
    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_to_float_fn, int_to_float_fn])
    expected_type = computation_types.FunctionType(tf.int32, tf.float32)
    self.assertEqual(composed_fn.type_signature, expected_type)

  def test_composes_tensor_function_executes_correctly(self):
    int_to_float_fn = _create_compiled_computation(
        lambda x: tf.cast(x, tf.float32) * 2.0, tf.int32)
    float_to_float_fn = _create_compiled_computation(lambda x: x * 2.0,
                                                     tf.float32)
    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_to_float_fn, int_to_float_fn])
    executable_mult_by_four = _to_computation_impl(composed_fn)
    for k in range(5):
      self.assertEqual(executable_mult_by_four(k), k * 4.)

  def test_compose_integer_identities_executes_correctly(self):
    identity = _create_compiled_computation(lambda x: x, tf.int32)
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [identity, identity])
    executable = _to_computation_impl(composed)
    self.assertEqual(executable(0), 0)

  def test_composes_unnamed_tuple_functions_types_correctly(self):
    int_float_flip = _create_compiled_computation(lambda x: [x[1], x[0]],
                                                  [tf.int32, tf.float32])
    float_int_flip = _create_compiled_computation(lambda x: [x[1], x[0]],
                                                  [tf.float32, tf.int32])
    composed_fn_float_int = compiled_computation_transforms.compose_tensorflow_blocks(
        [int_float_flip, float_int_flip])
    composed_fn_int_float = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_int_flip, int_float_flip])
    expected_type_int_float = computation_types.FunctionType(
        [tf.int32, tf.float32], [tf.int32, tf.float32])
    expected_type_float_int = computation_types.FunctionType(
        [tf.float32, tf.int32], [tf.float32, tf.int32])
    self.assertEqual(composed_fn_float_int.type_signature,
                     expected_type_float_int)
    self.assertEqual(composed_fn_int_float.type_signature,
                     expected_type_int_float)

  def test_composes_unnamed_tuple_functions_executes_correctly(self):
    int_float_flip = _create_compiled_computation(lambda x: [x[1], x[0]],
                                                  [tf.int32, tf.float32])
    float_int_flip = _create_compiled_computation(lambda x: [x[1], x[0]],
                                                  [tf.float32, tf.int32])
    composed_fn_float_int = compiled_computation_transforms.compose_tensorflow_blocks(
        [int_float_flip, float_int_flip])
    composed_fn_int_float = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_int_flip, int_float_flip])
    executable_float_int = _to_computation_impl(composed_fn_float_int)
    executable_int_float = _to_computation_impl(composed_fn_int_float)
    self.assertEqual(executable_float_int([10., 0])[0], 10.)
    self.assertEqual(executable_float_int([10., 0])[1], 0)
    self.assertLen(executable_float_int([10., 0]), 2)
    self.assertEqual(executable_int_float(10, 0.)[0], 10)
    self.assertEqual(executable_int_float(10, 0.)[1], 0.)
    self.assertLen(executable_int_float([10, 0.]), 2)

  def test_composes_named_tuple_function_with_unnamed_tuple_function_types_correctly(
      self):
    drop_names = _create_compiled_computation(lambda x: [x[0], x[1]],
                                              [('a', tf.int32),
                                               ('b', tf.float32)])
    unnamed_identity = _create_compiled_computation(lambda x: x,
                                                    [tf.int32, tf.float32])
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [unnamed_identity, drop_names])
    expected_type = computation_types.FunctionType([('a', tf.int32),
                                                    ('b', tf.float32)],
                                                   [tf.int32, tf.float32])
    self.assertEqual(composed.type_signature, expected_type)

  def test_composes_named_tuple_function_with_unnamed_tuple_function_executes_correctly(
      self):
    drop_names = _create_compiled_computation(lambda x: [x[0], x[1]],
                                              [('a', tf.int32),
                                               ('b', tf.float32)])
    unnamed_identity = _create_compiled_computation(lambda x: x,
                                                    [tf.int32, tf.float32])
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [unnamed_identity, drop_names])
    executable_drop_names = _to_computation_impl(composed)
    self.assertEqual(executable_drop_names({'a': 0, 'b': 1.})[0], 0)
    self.assertEqual(executable_drop_names({'a': 0, 'b': 1.})[1], 1.)
    self.assertLen(executable_drop_names({'a': 0, 'b': 1.}), 2)

  def test_composes_named_tuple_functions_types_correctly(self):
    flip_order = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        [('a', tf.int32), ('b', tf.float32)])
    identity = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        [('b', tf.float32), ('a', tf.int32)])
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [identity, flip_order])
    expected_type = computation_types.FunctionType([('a', tf.int32),
                                                    ('b', tf.float32)],
                                                   [('b', tf.float32),
                                                    ('a', tf.int32)])
    self.assertEqual(str(composed.type_signature), str(expected_type))

  def test_composes_named_tuple_functions_executes_correctly(self):
    flip_order = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        [('a', tf.int32), ('b', tf.float32)])
    identity = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        [('b', tf.float32), ('a', tf.int32)])
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [identity, flip_order])
    executable = _to_computation_impl(composed)
    self.assertEqual(
        executable(collections.OrderedDict({
            'a': 0,
            'b': 1.
        }))[0], 1.)
    self.assertEqual(
        executable(collections.OrderedDict({
            'a': 0,
            'b': 1.
        }))[1], 0)
    self.assertLen(executable(collections.OrderedDict({'a': 0, 'b': 1.})), 2)

  def test_composes_sequence_functions_types_correctly(self):
    reduce_ds = _create_compiled_computation(
        lambda ds: ds.reduce(tf.constant(0, tf.int64), lambda x, y: x + y),
        computation_types.SequenceType(tf.int64))

    produce_ds = _create_compiled_computation(lambda: tf.data.Dataset.range(5),
                                              None)
    integer_result = compiled_computation_transforms.compose_tensorflow_blocks(
        [reduce_ds, produce_ds])

    self.assertEqual(integer_result.type_signature,
                     computation_types.FunctionType(None, tf.int64))

  def test_composes_sequence_functions_executes_correctly(self):
    reduce_ds = _create_compiled_computation(
        lambda ds: ds.reduce(tf.constant(0, tf.int64), lambda x, y: x + y),
        computation_types.SequenceType(tf.int64))

    produce_ds = _create_compiled_computation(lambda: tf.data.Dataset.range(5),
                                              None)
    integer_result = compiled_computation_transforms.compose_tensorflow_blocks(
        [reduce_ds, produce_ds])

    executable_reduce = _to_computation_impl(integer_result)
    self.assertEqual(executable_reduce(), 10)


def _create_simple_called_composition_of_tf_blocks():
  zero_fn = _create_compiled_computation(lambda: 0, None)
  zero = computation_building_blocks.Call(zero_fn, None)
  add_one = _create_compiled_computation(lambda x: x + 1, tf.int32)
  one = computation_building_blocks.Call(add_one, zero)
  return one


def _count_compiled_computations_under(comp):
  count = [0]

  def _count(comp):
    if isinstance(comp, computation_building_blocks.CompiledComputation):
      count[0] += 1
    return comp, False

  transformation_utils.transform_postorder(comp, _count)

  return count[0]


class CalledCompositionOfTensorFlowBlocksTest(parameterized.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_called_composition_of_tf_blocks()
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertFalse(logic.should_transform(integer_square))

  def test_should_not_transform_single_called_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    int_ref = computation_building_blocks.Reference('x', tf.int32)
    called_square = computation_building_blocks.Call(integer_square, int_ref)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertFalse(logic.should_transform(called_square))

  def test_should_not_transform_called_lambda_on_called_compiled_computation(
      self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    int_ref = computation_building_blocks.Reference('x', tf.int32)
    called_square = computation_building_blocks.Call(integer_square, int_ref)
    lambda_wrapper = computation_building_blocks.Lambda('x', tf.int32,
                                                        called_square)
    outer_int_ref = computation_building_blocks.Reference('y', tf.int32)
    called_lambda = computation_building_blocks.Call(lambda_wrapper,
                                                     outer_int_ref)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertFalse(logic.should_transform(called_lambda))

  def test_does_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, mutated = logic.transform(integer_square)
    self.assertEqual(parsed, integer_square)
    self.assertFalse(mutated)

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_called_composition_of_tf_blocks()
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed, computation_building_blocks.Call)
    self.assertIsInstance(parsed.function,
                          computation_building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_transform_reduces_number_of_compiled_computations(self):
    pattern = _create_simple_called_composition_of_tf_blocks()
    original_count = _count_compiled_computations_under(pattern)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, _ = logic.transform(pattern)
    new_count = _count_compiled_computations_under(parsed)
    self.assertLess(new_count, original_count)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_called_composition_of_tf_blocks()
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly(self):
    pattern = _create_simple_called_composition_of_tf_blocks()
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, _ = logic.transform(pattern)
    lambda_wrapping_parsed = computation_building_blocks.Lambda(
        'x', tf.int32, parsed)
    executable = _to_computation_impl(lambda_wrapping_parsed)
    self.assertEqual(executable(0), 1)
    self.assertEqual(executable(1), 1)
    self.assertEqual(executable(2), 1)

  def test_constructs_correct_type_signature_named_tuple_argument(self):
    identity = _create_compiled_computation(lambda x: x, [('a', tf.int32),
                                                          ('b', tf.float32)])
    sel_int = _create_compiled_computation(lambda x: x.a, [('a', tf.int32),
                                                           ('b', tf.float32)])

    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.int32), ('b', tf.float32)])

    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_reference)
    called_integer_selection = computation_building_blocks.Call(
        sel_int, called_identity)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, mutated = logic.transform(called_integer_selection)
    self.assertEqual(parsed.type_signature,
                     called_integer_selection.type_signature)
    self.assertEqual(parsed.argument.type_signature,
                     tuple_reference.type_signature)
    self.assertTrue(mutated)

  def test_executes_named_tuple_argument(self):
    identity = _create_compiled_computation(lambda x: x, [('a', tf.int32),
                                                          ('b', tf.float32)])
    sel_int = _create_compiled_computation(lambda x: x.a, [('a', tf.int32),
                                                           ('b', tf.float32)])

    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.int32), ('b', tf.float32)])

    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_reference)
    called_integer_selection = computation_building_blocks.Call(
        sel_int, called_identity)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, _ = logic.transform(called_integer_selection)
    lambda_wrapping_parsed = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, parsed)
    executable = _to_computation_impl(lambda_wrapping_parsed)
    self.assertEqual(executable({'a': 1, 'b': 0.}), 1)
    self.assertEqual(executable({'a': 0, 'b': 1.}), 0)

  def test_constructs_correct_type_signature_named_tuple_result(self):
    namer = _create_compiled_computation(
        lambda x: collections.OrderedDict([('a', x[0]), ('b', x[1])]),
        [tf.int32, tf.float32])
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])

    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.int32, tf.float32])

    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_reference)
    called_namer = computation_building_blocks.Call(namer, called_identity)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, mutated = logic.transform(called_namer)
    self.assertEqual(parsed.type_signature, called_namer.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_named_tuple_result(self):
    namer = _create_compiled_computation(
        lambda x: collections.OrderedDict([('a', x[0]), ('b', x[1])]),
        [tf.int32, tf.float32])
    identity = _create_compiled_computation(lambda x: x, [tf.int32, tf.float32])

    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.int32, tf.float32])

    called_identity = computation_building_blocks.Call(identity,
                                                       tuple_reference)
    called_namer = computation_building_blocks.Call(namer, called_identity)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, _ = logic.transform(called_namer)
    lambda_wrapping_parsed = computation_building_blocks.Lambda(
        'x', tuple_reference.type_signature, parsed)
    executable = _to_computation_impl(lambda_wrapping_parsed)
    self.assertEqual(executable([1, 0.])[0], 1)
    self.assertEqual(executable([1, 0.]).a, 1)
    self.assertEqual(executable([1, 0.])[1], 0.)
    self.assertEqual(executable([1, 0.]).b, 0.)
    self.assertEqual(executable([0, 1.])[0], 0)
    self.assertEqual(executable([0, 1.]).a, 0)
    self.assertEqual(executable([0, 1.])[1], 1.)
    self.assertEqual(executable([0, 1.]).b, 1.)


def _create_simple_called_graph_on_replicated_arg(n_replicates=2):
  tuple_identity = _create_compiled_computation(lambda x: x,
                                                [tf.int32] * n_replicates)
  ref_to_int = computation_building_blocks.Reference('x', tf.int32)
  called_tuple_id = computation_building_blocks.Call(
      tuple_identity,
      computation_building_blocks.Tuple([ref_to_int] * n_replicates))
  return called_tuple_id


class CalledGraphOnReplicatedArgTest(absltest.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_called_graph_on_replicated_arg()
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_transform_identifies_longer_pattern(self):
    pattern = _create_simple_called_graph_on_replicated_arg(n_replicates=5)
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertFalse(logic.should_transform(integer_square))

  def test_should_not_transform_non_tuple_wrapped_lambda_to_called_graph(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    int_ref = computation_building_blocks.Reference('x', tf.int32)
    called_square = computation_building_blocks.Call(integer_square, int_ref)
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertFalse(logic.should_transform(called_square))

  def test_does_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(lambda x: x * x, tf.int32)
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(integer_square)
    self.assertEqual(parsed, integer_square)
    self.assertFalse(mutated)

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_called_graph_on_replicated_arg()
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed, computation_building_blocks.Call)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_called_graph_on_replicated_arg()
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_simple_case(self):
    pattern = _create_simple_called_graph_on_replicated_arg()
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, _ = logic.transform(pattern)
    lambda_wrapped_parsed = computation_building_blocks.Lambda(
        'x', tf.int32, parsed)
    executable = _to_computation_impl(lambda_wrapped_parsed)
    self.assertEqual(
        executable(0), anonymous_tuple.AnonymousTuple([(None, 0), (None, 0)]))
    self.assertEqual(
        executable(1), anonymous_tuple.AnonymousTuple([(None, 1), (None, 1)]))
    self.assertEqual(
        executable(2), anonymous_tuple.AnonymousTuple([(None, 2), (None, 2)]))

  def test_executes_correctly_several_replicates(self):
    pattern = _create_simple_called_graph_on_replicated_arg(n_replicates=5)
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, _ = logic.transform(pattern)
    lambda_wrapped_parsed = computation_building_blocks.Lambda(
        'x', tf.int32, parsed)
    executable = _to_computation_impl(lambda_wrapped_parsed)
    result_on_0 = executable(0)

    for k in range(5):
      self.assertEqual(result_on_0[k], 0)
    result_on_1 = executable(1)
    for k in range(5):
      self.assertEqual(result_on_1[k], 1)
    self.assertLen(result_on_0, 5)
    self.assertLen(result_on_1, 5)

  def test_constructs_correct_type_signature_nested_tuple_argument(self):
    slicer = _create_compiled_computation(
        lambda x: [x[0][0], x[1][1]],
        [[tf.int32, tf.float32], [tf.int32, tf.float32]])
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.int32, tf.float32])

    called_slicer = computation_building_blocks.Call(
        slicer,
        computation_building_blocks.Tuple([tuple_reference, tuple_reference]))
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(called_slicer)
    self.assertEqual(parsed.type_signature, called_slicer.type_signature)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_nested_named_tuple_argument(self):
    slicer = _create_compiled_computation(
        lambda x: [x[0][0], x[1][1]],
        [[('a', tf.int32),
          ('b', tf.float32)], [('a', tf.int32), ('b', tf.float32)]])
    tuple_reference = computation_building_blocks.Reference(
        'x', [('a', tf.int32), ('b', tf.float32)])

    called_slicer = computation_building_blocks.Call(
        slicer,
        computation_building_blocks.Tuple([tuple_reference, tuple_reference]))
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(called_slicer)
    self.assertEqual(parsed.type_signature, called_slicer.type_signature)
    self.assertTrue(mutated)

  def test_execution_nested_tuple_argument(self):
    slicer = _create_compiled_computation(
        lambda x: [x[0][0], x[1][1]],
        [[tf.int32, tf.float32], [tf.int32, tf.float32]])
    tuple_reference = computation_building_blocks.Reference(
        'x', [tf.int32, tf.float32])

    called_slicer = computation_building_blocks.Call(
        slicer,
        computation_building_blocks.Tuple([tuple_reference, tuple_reference]))
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, _ = logic.transform(called_slicer)
    lambda_wrapper = computation_building_blocks.Lambda('x',
                                                        [tf.int32, tf.float32],
                                                        parsed)
    executable = _to_computation_impl(lambda_wrapper)
    self.assertEqual(executable([0, 1.])[0], 0)
    self.assertEqual(executable([0, 1.])[1], 1.)
    self.assertEqual(executable([1, 0.])[0], 1)
    self.assertEqual(executable([1, 0.])[1], 0.)


if __name__ == '__main__':
  absltest.main()
