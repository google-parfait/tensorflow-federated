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

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import compiled_computation_transforms
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_transformations
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def _create_compiled_computation(py_fn, parameter_type):
  proto, type_signature = tensorflow_computation_factory.create_computation_for_py_fn(
      py_fn, parameter_type)
  return building_blocks.CompiledComputation(
      proto, type_signature=type_signature)


class CompiledComputationTransformsTest(test.TestCase, parameterized.TestCase):

  def test_select_graph_output_with_none_comp_raises_type_error(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.select_graph_output(None, index=0)

  def test_select_graph_output_with_no_selection_raises_value_error(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32),
                                                         ('b', tf.float32)])

    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.select_graph_output(foo)

  def test_select_graph_output_with_wrong_return_type_raises_type_error(self):
    computation_arg_type = computation_types.TensorType(tf.int32)

    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(TypeError):
      compiled_computation_transforms.select_graph_output(foo, index=0)

  def test_select_graph_output_by_name_bad_name_raises_value_error(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32),
                                                         ('b', tf.float32)])

    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.select_graph_output(foo, name='x')

  def test_select_graph_output_by_index_single_level_of_nesting(self):
    computation_arg_type = computation_types.StructType([tf.int32, tf.float32])

    foo = building_block_factory.create_compiled_identity(computation_arg_type)
    foo_pruned_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        foo.proto)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=0)
    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=1)

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            first_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(first_element_selected.type_signature.result,
                     foo.type_signature.result[0])
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[0].tensor,
                     first_element_selected.proto.tensorflow.result.tensor)

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            second_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[1].tensor,
                     second_element_selected.proto.tensorflow.result.tensor)

  def test_select_graph_output_by_name_single_level_of_nesting(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32),
                                                         ('b', tf.float32)])

    foo = building_block_factory.create_compiled_identity(computation_arg_type)
    foo_pruned_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        foo.proto)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='a')
    self.assertEqual(first_element_selected.type_signature.result,
                     computation_types.TensorType(tf.int32))

    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='b')
    self.assertEqual(second_element_selected.type_signature.result,
                     computation_types.TensorType(tf.float32))

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            first_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[0].tensor,
                     first_element_selected.proto.tensorflow.result.tensor)

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            second_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[1].tensor,
                     second_element_selected.proto.tensorflow.result.tensor)

  def test_select_graph_output_by_index_two_nested_levels_keeps_nested_type(
      self):
    nested_type1 = computation_types.StructType([('a', tf.int32),
                                                 ('b', tf.float32)])
    nested_type2 = computation_types.StructType([('c', tf.int32),
                                                 ('d', tf.float32)])

    computation_arg_type = computation_types.StructType([('x', nested_type1),
                                                         ('y', nested_type2)])

    foo = building_block_factory.create_compiled_identity(computation_arg_type)
    foo_pruned_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        foo.proto)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=0)
    self.assertEqual(first_element_selected.type_signature.result, nested_type1)

    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, index=1)
    self.assertEqual(second_element_selected.type_signature.result,
                     nested_type2)

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            first_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[0].struct,
                     first_element_selected.proto.tensorflow.result.struct)

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            second_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[1].struct,
                     second_element_selected.proto.tensorflow.result.struct)

  def test_select_graph_output_by_name_two_nested_levels_keeps_nested_type(
      self):
    nested_type1 = computation_types.StructType([('a', tf.int32),
                                                 ('b', tf.float32)])
    nested_type2 = computation_types.StructType([('c', tf.int32),
                                                 ('d', tf.float32)])
    computation_arg_type = computation_types.StructType([('x', nested_type1),
                                                         ('y', nested_type2)])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)
    foo_pruned_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        foo.proto)

    first_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='x')
    self.assertEqual(first_element_selected.type_signature.result, nested_type1)

    second_element_selected = compiled_computation_transforms.select_graph_output(
        foo, name='y')
    self.assertEqual(second_element_selected.type_signature.result,
                     nested_type2)

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            first_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(foo.proto.tensorflow.parameter,
                     first_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     first_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[0].struct,
                     first_element_selected.proto.tensorflow.result.struct)

    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            second_element_selected.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(second_element_selected.type_signature.result,
                     foo.type_signature.result[1])
    self.assertEqual(foo.proto.tensorflow.parameter,
                     second_element_selected.proto.tensorflow.parameter)
    self.assertEqual(foo.proto.tensorflow.initialize_op,
                     second_element_selected.proto.tensorflow.initialize_op)
    self.assertEqual(foo.proto.tensorflow.result.struct.element[1].struct,
                     second_element_selected.proto.tensorflow.result.struct)

  def test_permute_graph_inputs_with_none_comp_raises_type_error(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.permute_graph_inputs(None, [0])

  def test_permute_graph_inputs_with_integer_map_raises_type_error(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32)])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(TypeError):
      compiled_computation_transforms.permute_graph_inputs(foo, 0)

  def test_permute_graph_inputs_with_list_of_strings_raises_type_error(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32)])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(TypeError):
      compiled_computation_transforms.permute_graph_inputs(foo, ['a'])

  def test_permute_graph_inputs_wrong_permutation_length_raises_value_error(
      self):
    computation_arg_type = computation_types.StructType([tf.int32, tf.float32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0])

  def test_permute_graph_inputs_repeated_indices_raises_value_error(self):
    computation_arg_type = computation_types.StructType([tf.int32, tf.float32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0, 0])

  def test_permute_graph_inputs_large_index_raises_value_error(self):
    computation_arg_type = computation_types.StructType([tf.int32, tf.float32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0, 2])

  def test_permute_graph_inputs_negative_index_raises_value_error(self):
    computation_arg_type = computation_types.StructType([tf.int32, tf.float32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    with self.assertRaises(ValueError):
      compiled_computation_transforms.permute_graph_inputs(foo, [0, -1])

  def test_permute_graph_inputs_identity_permutation_noops(self):
    computation_arg_type = computation_types.StructType([tf.int32, tf.float32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    mapped_to_identity = compiled_computation_transforms.permute_graph_inputs(
        foo, [0, 1])

    self.assertEqual(mapped_to_identity.proto.tensorflow.parameter,
                     foo.proto.tensorflow.parameter)
    self.assertEqual(mapped_to_identity.proto.tensorflow.result,
                     foo.proto.tensorflow.result)
    self.assertEqual(mapped_to_identity.proto.tensorflow.initialize_op,
                     foo.proto.tensorflow.initialize_op)
    foo_pruned_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        foo.proto)
    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            mapped_to_identity.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(mapped_to_identity.type_signature, foo.type_signature)

  def test_permute_graph_inputs_identity_permutation_leaves_names_alone(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32),
                                                         ('b', tf.float32)])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)
    foo_pruned_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        foo.proto)

    mapped_to_identity = compiled_computation_transforms.permute_graph_inputs(
        foo, [0, 1])

    self.assertEqual(mapped_to_identity.proto.tensorflow.parameter,
                     foo.proto.tensorflow.parameter)
    self.assertEqual(mapped_to_identity.proto.tensorflow.result,
                     foo.proto.tensorflow.result)
    self.assertEqual(mapped_to_identity.proto.tensorflow.initialize_op,
                     foo.proto.tensorflow.initialize_op)
    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            mapped_to_identity.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            foo_pruned_proto.tensorflow.graph_def))
    self.assertEqual(mapped_to_identity.type_signature, foo.type_signature)

  def test_permute_graph_inputs_flip_input_order_changes_only_parameters(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32),
                                                         ('b', tf.float32),
                                                         ('c', tf.bool)])
    permuted_arg_type = computation_types.StructType([('c', tf.bool),
                                                      ('a', tf.int32),
                                                      ('b', tf.float32)])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    permuted_inputs = compiled_computation_transforms.permute_graph_inputs(
        foo, [2, 0, 1])

    self.assertEqual(permuted_inputs.type_signature.parameter,
                     permuted_arg_type)
    self.assertEqual(permuted_inputs.type_signature.result,
                     foo.type_signature.result)
    pruned_foo_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        foo.proto)
    self.assertProtoEquals(
        serialization_utils.unpack_graph_def(
            permuted_inputs.proto.tensorflow.graph_def),
        serialization_utils.unpack_graph_def(
            pruned_foo_proto.tensorflow.graph_def))
    self.assertEqual(permuted_inputs.proto.tensorflow.initialize_op,
                     foo.proto.tensorflow.initialize_op)
    self.assertEqual(permuted_inputs.proto.tensorflow.result,
                     foo.proto.tensorflow.result)

  def test_permute_graph_inputs_flip_input_order_executes_correctly(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32),
                                                         ('b', tf.float32),
                                                         ('c', tf.bool)])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    flipped_inputs = compiled_computation_transforms.permute_graph_inputs(
        foo, [1, 0, 2])

    expected_result = structure.Struct([
        ('a', 0),
        ('b', 1.0),
        ('c', True),
    ])
    structure_input = structure.Struct([
        ('b', 1.0),
        ('a', 0),
        ('c', True),
    ])
    result = test_utils.run_tensorflow(flipped_inputs.proto, [1.0, 0, True])
    self.assertEqual(result, expected_result)
    result = test_utils.run_tensorflow(flipped_inputs.proto, structure_input)
    self.assertEqual(result, expected_result)
    with self.assertRaises(TypeError):
      test_utils.run_tensorflow(flipped_inputs.proto, [0, 1.0, True])
    with self.assertRaises(TypeError):
      test_utils.run_tensorflow(flipped_inputs.proto, expected_result)


class WrapParameterAsTupleTest(test.TestCase, parameterized.TestCase):

  def test_bind_graph_parameter_as_tuple_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_parameter_as_tuple(None)

  def test_bind_graph_parameter_as_tuple_raises_on_non_string_name(self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_parameter_as_tuple(foo, name=1)

  def test_bind_graph_parameter_as_tuple_wraps_tuple(self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo)

    parameter_type = computation_types.StructType(
        [foo.type_signature.parameter])
    expected_type_signature = computation_types.FunctionType(
        parameter_type, foo.type_signature.result)
    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_inputs.proto, [[1]])
    expected_result = test_utils.run_tensorflow(foo.proto, [1])
    self.assertEqual(actual_result, expected_result)

  def assertSequenceEqual(self, a, b):
    """Assert two tff.SequenceType values are the same."""
    if (isinstance(a, collections.Sequence) and
        isinstance(b, collections.Sequence)):
      sequence = zip(a, b)
    elif isinstance(a, tf.data.Dataset) and isinstance(b, tf.data.Dataset):
      sequence = tf.data.Dataset.zip(a, b)
    else:
      self.fail('Value is not a sequence, got types a={!s}, b={!s}'.format(
          type(a), type(b)))
    for element in sequence:
      self.assertEqual(element[0], element[1])

  def test_bind_graph_parameter_as_tuple_wraps_sequence(self):
    computation_arg_type = computation_types.SequenceType(tf.int32)
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo)

    parameter_type = computation_types.StructType(
        [foo.type_signature.parameter])
    expected_type_signature = computation_types.FunctionType(
        parameter_type, foo.type_signature.result)
    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_inputs.proto, [[1]])
    expected_result = test_utils.run_tensorflow(foo.proto, [1])
    self.assertSequenceEqual(actual_result, expected_result)

  def test_bind_graph_parameter_as_tuple_wraps_tensor(self):
    computation_arg_type = computation_types.TensorType(tf.int32)
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo)

    parameter_type = computation_types.StructType(
        [foo.type_signature.parameter])
    expected_type_signature = computation_types.FunctionType(
        parameter_type, foo.type_signature.result)
    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_inputs.proto, [1])
    expected_result = test_utils.run_tensorflow(foo.proto, 1)
    self.assertEqual(actual_result, expected_result)

  def test_bind_graph_parameter_as_tuple_adds_name(self):
    computation_arg_type = computation_types.TensorType(tf.int32)
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_inputs = compiled_computation_transforms.bind_graph_parameter_as_tuple(
        foo, name='a')

    expected_type_signature = computation_types.FunctionType(
        computation_types.StructType((
            'a',
            foo.type_signature.parameter,
        )), foo.type_signature.result)
    self.assertEqual(wrapped_inputs.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_inputs.proto, [1])
    expected_result = test_utils.run_tensorflow(foo.proto, 1)
    self.assertEqual(actual_result, expected_result)


class WrapResultAsTupleTest(test.TestCase, parameterized.TestCase):

  def test_bind_graph_result_as_tuple_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_result_as_tuple(None)

  def test_bind_graph_result_as_tuple_raises_on_non_string_name(self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.bind_graph_result_as_tuple(foo, name=1)

  def test_bind_graph_result_as_tuple_wraps_tuple(self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo)

    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter,
        computation_types.StructType([foo.type_signature.result]))
    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_output.proto, [1])
    expected_result = test_utils.run_tensorflow(foo.proto, [1])
    self.assertEqual(actual_result[0], expected_result)

  def test_bind_graph_result_as_tuple_wraps_sequence(self):
    computation_arg_type = computation_types.SequenceType(tf.int32)
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo)

    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter,
        computation_types.StructType([foo.type_signature.result]))
    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_output.proto, [1])
    expected_result = test_utils.run_tensorflow(foo.proto, [1])
    self.assertSequenceEqual(actual_result[0], expected_result)

  def test_bind_graph_result_as_tuple_wraps_tensor(self):
    computation_arg_type = computation_types.TensorType(tf.int32)
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo)

    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter,
        computation_types.StructType([foo.type_signature.result]))
    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_output.proto, [1])
    expected_result = test_utils.run_tensorflow(foo.proto, [1])
    self.assertEqual(actual_result[0], expected_result)

  def test_bind_graph_result_as_tuple_adds_name(self):
    computation_arg_type = computation_types.TensorType(tf.int32)
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    wrapped_output = compiled_computation_transforms.bind_graph_result_as_tuple(
        foo, name='a')

    expected_type_signature = computation_types.FunctionType(
        foo.type_signature.parameter,
        computation_types.StructType((
            'a',
            foo.type_signature.result,
        )))
    self.assertEqual(wrapped_output.type_signature, expected_type_signature)
    actual_result = test_utils.run_tensorflow(wrapped_output.proto, 1)
    expected_result = test_utils.run_tensorflow(foo.proto, 1)
    self.assertEqual(actual_result[0], expected_result)


class GraphInputPaddingTest(test.TestCase, parameterized.TestCase):

  def test_pad_graph_inputs_to_match_type_raises_on_none(self):
    with self.assertRaisesRegex(TypeError, r'Expected.*CompiledComputation'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          None, computation_types.StructType([tf.int32]))

  def test_pad_graph_inputs_to_match_type_raises_on_wrong_requested_type(self):
    comp = building_block_factory.create_compiled_identity(
        computation_types.StructType([tf.int32]))
    tensor_type = computation_types.TensorType(tf.int32)
    with self.assertRaisesRegex(TypeError, r'Expected.*StructType'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, tensor_type)

  def test_pad_graph_inputs_to_match_type_raises_on_wrong_graph_parameter_type(
      self):
    comp = building_block_factory.create_compiled_identity(
        computation_types.TensorType(tf.int32))
    with self.assertRaisesRegex(
        TypeError,
        r'Can only pad inputs of a CompiledComputation with parameter type struct'
    ):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.StructType([tf.int32]))

  def test_pad_graph_inputs_to_match_type_raises_on_requested_type_too_short(
      self):
    comp = building_block_factory.create_compiled_identity(
        computation_types.StructType([tf.int32] * 3))
    with self.assertRaisesRegex(ValueError, r'must have more elements'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.StructType([tf.int32] * 2))

  def test_pad_graph_inputs_to_match_type_raises_on_mismatched_graph_type_and_requested_type(
      self):
    comp = building_block_factory.create_compiled_identity(
        computation_types.StructType([tf.float32]))
    with self.assertRaisesRegex(TypeError, r'must match the beginning'):
      compiled_computation_transforms.pad_graph_inputs_to_match_type(
          comp, computation_types.StructType([tf.int32] * 2))

  def test_pad_graph_inputs_to_match_type_preserves_named_type_signature(self):
    computation_arg_type = computation_types.StructType([('a', tf.int32)])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo, computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    expected_type_signature = computation_types.FunctionType(
        [('a', tf.int32), ('b', tf.float32)], [('a', tf.int32)])

    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    padded_inputs.type_signature.check_equivalent_to(expected_type_signature)

  def test_pad_graph_inputs_to_match_type_adds_names_to_unnamed_tuple(self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo, computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    expected_type_signature = computation_types.FunctionType(
        [('a', tf.int32), ('b', tf.float32)], [tf.int32])

    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    padded_inputs.type_signature.check_equivalent_to(expected_type_signature)

  def test_pad_graph_inputs_to_match_type_preserves_unnamed_type_signature(
      self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo, computation_types.StructType([tf.int32, tf.float32]))
    expected_type_signature = computation_types.FunctionType(
        [tf.int32, tf.float32], [tf.int32])

    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    padded_inputs.type_signature.check_equivalent_to(expected_type_signature)

  def test_pad_graph_inputs_to_match_type_add_single_int_executes_correctly(
      self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo, computation_types.StructType([tf.int32, tf.float32]))

    expected_result = structure.Struct([(None, 1)])
    actual_result = test_utils.run_tensorflow(padded_inputs.proto, [1, 0.0])
    self.assertEqual(actual_result, expected_result)
    actual_result = test_utils.run_tensorflow(padded_inputs.proto, [1, 10.0])
    self.assertEqual(actual_result, expected_result)

  def test_pad_graph_inputs_to_match_type_adds_names_to_unnamed_tuple_and_executes(
      self):
    computation_arg_type = computation_types.StructType([tf.int32])
    foo = building_block_factory.create_compiled_identity(computation_arg_type)

    padded_inputs = compiled_computation_transforms.pad_graph_inputs_to_match_type(
        foo, computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))

    expected_result = structure.Struct([(None, 1)])
    actual_result = test_utils.run_tensorflow(padded_inputs.proto, {
        'a': 1,
        'b': 0.0,
    })
    self.assertEqual(actual_result, expected_result)
    actual_result = test_utils.run_tensorflow(padded_inputs.proto, {
        'a': 1,
        'b': 10.0,
    })
    self.assertEqual(actual_result, expected_result)


class ConcatenateTFBlocksTest(test.TestCase, parameterized.TestCase):

  def test_concatenenate_tensorflow_blocks_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(
          None, [None])

  def test_concatenenate_tensorflow_blocks_raises_no_iterable(self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(foo, [None])

  def test_concatenenate_tensorflow_blocks_raises_bad_comp_in_list(self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    bad_comp = building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks(
          [foo, bad_comp], [None, None])

  def test_concatenate_tensorflow_blocks_fails_empty_list(self):
    with self.assertRaises(ValueError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([], [None])

  def test_concatenate_tensorflow_blocks_raises_bad_names_list_length(self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    bar_type = computation_types.TensorType(tf.float32)
    bar = building_block_factory.create_tensorflow_constant(bar_type, 1.0)
    with self.assertRaises(ValueError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([foo, bar],
                                                                    [None])

  def test_concatenate_tensorflow_blocks_raises_bad_names_list_type(self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    bar_type = computation_types.TensorType(tf.float32)
    bar = building_block_factory.create_tensorflow_constant(bar_type, 1.0)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([foo, bar],
                                                                    'x')

  def test_concatenate_tensorflow_blocks_raises_bad_names_list_element_type(
      self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    bar_type = computation_types.TensorType(tf.float32)
    bar = building_block_factory.create_tensorflow_constant(bar_type, 1.0)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.concatenate_tensorflow_blocks([foo, bar],
                                                                    ['x', 1])

  def test_concatenate_tensorflow_blocks_no_arg(self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    bar_type = computation_types.TensorType(tf.float32)
    bar = building_block_factory.create_tensorflow_constant(bar_type, 1.0)

    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo.function, bar.function], [None, None])

    self.assertIsInstance(merged_comp, building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(None,
                                                       [tf.float32, tf.float32])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    merged_comp.type_signature.check_equivalent_to(concatenated_type)
    actual_result = test_utils.run_tensorflow(merged_comp.proto, None)
    expected_result = structure.Struct([(None, 0.0), (None, 1.0)])
    self.assertAlmostEqual(actual_result, expected_result)

  def test_concatenate_tensorflow_blocks_named_outputs_type_preserved(self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    bar_type = computation_types.TensorType(tf.float32)
    bar = building_block_factory.create_tensorflow_constant(bar_type, 1.0)

    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo.function, bar.function], ['a', 'b'])

    self.assertIsInstance(merged_comp, building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(None,
                                                       [('a', tf.float32),
                                                        ('b', tf.float32)])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    merged_comp.type_signature.check_equivalent_to(concatenated_type)

  def test_concatenate_tensorflow_blocks_mix_of_arg_and_no_arg(self):
    foo_type = computation_types.TensorType(tf.float32)
    foo = building_block_factory.create_tensorflow_constant(foo_type, 0.0)
    bar = _create_compiled_computation(lambda x: x + tf.constant(1.0),
                                       computation_types.TensorType(tf.float32))

    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo.function, bar], [None, None])

    self.assertIsInstance(merged_comp, building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(tf.float32,
                                                       [tf.float32, tf.float32])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    merged_comp.type_signature.check_equivalent_to(concatenated_type)
    actual_result = test_utils.run_tensorflow(merged_comp.proto, 0.0)
    expected_result = structure.Struct([(None, 0.0), (None, 1.0)])
    self.assertAlmostEqual(actual_result, expected_result)

  def test_concatenate_tensorflow_blocks_tensor_args(self):
    foo = _create_compiled_computation(lambda x: x + tf.constant(0.0),
                                       computation_types.TensorType(tf.float32))
    bar = _create_compiled_computation(lambda x: x + tf.constant(1.0),
                                       computation_types.TensorType(tf.float32))

    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar], [None, None])

    self.assertIsInstance(merged_comp, building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType([tf.float32, tf.float32],
                                                       [tf.float32, tf.float32])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    merged_comp.type_signature.check_equivalent_to(concatenated_type)
    actual_result = test_utils.run_tensorflow(merged_comp.proto, [1.0, 0.0])
    expected_result = structure.Struct([(None, 1.0), (None, 1.0)])
    self.assertAlmostEqual(actual_result, expected_result)
    actual_result = test_utils.run_tensorflow(merged_comp.proto, [2.0, 2.0])
    expected_result = structure.Struct([(None, 2.0), (None, 3.0)])
    self.assertAlmostEqual(actual_result, expected_result)

  def test_concatenate_tensorflow_blocks_unnamed_tuple_args(self):
    foo = _create_compiled_computation(
        lambda x: [x[0] + tf.constant(0.0), x[1] + tf.constant(1.0)],
        computation_types.StructType([tf.float32, tf.float32]))
    bar = _create_compiled_computation(
        lambda x: [x[0] + tf.constant(1.0), x[1] + tf.constant(1.0)],
        computation_types.StructType([tf.float32, tf.float32]))

    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar], [None, None])

    self.assertIsInstance(merged_comp, building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(
        [[tf.float32, tf.float32], [tf.float32, tf.float32]],
        [[tf.float32, tf.float32], [tf.float32, tf.float32]])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    merged_comp.type_signature.check_equivalent_to(concatenated_type)
    actual_result = test_utils.run_tensorflow(merged_comp.proto,
                                              [[1.0, 0.0], [0.0, 1.0]])
    expected_result = structure.Struct([(None, 1.0), (None, 1.0)])
    self.assertEqual(actual_result[0], expected_result)
    actual_result = test_utils.run_tensorflow(merged_comp.proto,
                                              [[1.0, 0.0], [0.0, 1.0]])
    expected_result = structure.Struct([(None, 1.0), (None, 2.0)])
    self.assertEqual(actual_result[1], expected_result)

  def test_concatenate_tensorflow_blocks_named_tuple_args(self):
    foo_type = computation_types.StructType([('a', tf.float32),
                                             ('b', tf.float32)])
    foo = building_block_factory.create_compiled_identity(foo_type)
    bar_type = computation_types.StructType([('c', tf.float32),
                                             ('d', tf.float32)])
    bar = building_block_factory.create_compiled_identity(bar_type)

    merged_comp = compiled_computation_transforms.concatenate_tensorflow_blocks(
        [foo, bar], [None, None])

    self.assertIsInstance(merged_comp, building_blocks.CompiledComputation)
    concatenated_type = computation_types.FunctionType(
        [[('a', tf.float32),
          ('b', tf.float32)], [('c', tf.float32), ('d', tf.float32)]],
        [[('a', tf.float32),
          ('b', tf.float32)], [('c', tf.float32), ('d', tf.float32)]])
    self.assertEqual(str(merged_comp.type_signature), str(concatenated_type))
    actual_result = test_utils.run_tensorflow(merged_comp.proto,
                                              [[1.0, 0.0], [0.0, 1.0]])
    expected_result = structure.Struct([('a', 1.), ('b', 0.)])
    self.assertEqual(actual_result[0], expected_result)
    expected_result = structure.Struct([('c', 0.), ('d', 1.)])
    self.assertEqual(actual_result[1], expected_result)

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
            computation_types.SequenceType(tf.int64),
        ])
    concat_reduce_type_signature = computation_types.FunctionType(
        concat_input_type_signature.result, [tf.int64, tf.int64])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    concat_input_type_signature.check_equivalent_to(
        merged_input_comps.type_signature)
    concat_reduce_type_signature.check_equivalent_to(
        merged_reduce_comps.type_signature)
    input_result = test_utils.run_tensorflow(merged_input_comps.proto)
    actual_result = test_utils.run_tensorflow(merged_reduce_comps.proto,
                                              input_result)
    self.assertEqual(actual_result[0], 10)
    self.assertEqual(actual_result[1], 10)


def _create_simple_selection_from_called_graph():
  noarg_tuple = _create_compiled_computation(
      lambda: [tf.constant(0.), tf.constant(1.)], None)
  called_noarg_tuple = building_blocks.Call(noarg_tuple, None)
  selected_result = building_blocks.Selection(called_noarg_tuple, index=0)
  return selected_result


class SelectionFromCalledTensorFlowBlockTest(test.TestCase,
                                             parameterized.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_selection_from_called_graph()
    logic = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock()
    self.assertTrue(logic.should_transform(pattern))

  def test_output_selection_should_not_transform_unselected_call(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.), tf.constant(1.)], None)
    called_noarg_tuple = building_blocks.Call(noarg_tuple, None)
    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    self.assertFalse(output_selector.should_transform(called_noarg_tuple))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_selection_from_called_graph()
    logic = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock()
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, building_blocks.Call)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_selection_from_called_graph()
    logic = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_output_selection_executes_zeroth_element(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.0), tf.constant(1.0)], None)
    called_noarg_tuple = building_blocks.Call(noarg_tuple, None)
    selected_zero = building_blocks.Selection(called_noarg_tuple, index=0)

    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_zero, mutated = output_selector.transform(selected_zero)

    result = test_utils.run_tensorflow(parsed_zero.function.proto)
    self.assertEqual(result, 0.0)
    self.assertTrue(mutated)

  def test_output_selection_executes_first_element(self):
    noarg_tuple = _create_compiled_computation(
        lambda: [tf.constant(0.0), tf.constant(1.0)], None)
    called_noarg_tuple = building_blocks.Call(noarg_tuple, None)
    selected_one = building_blocks.Selection(called_noarg_tuple, index=1)

    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_one, mutated = output_selector.transform(selected_one)

    result = test_utils.run_tensorflow(parsed_one.function.proto)
    self.assertEqual(result, 1.0)
    self.assertTrue(mutated)

  def test_output_selection_executes_when_selecting_by_name(self):
    fn = lambda: {'a': tf.constant(0.0), 'b': tf.constant(1.0)}
    noarg_tuple = _create_compiled_computation(fn, None)
    called_noarg_tuple = building_blocks.Call(noarg_tuple, None)
    selected_a = building_blocks.Selection(called_noarg_tuple, name='a')

    output_selector = compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(
    )
    parsed_a, mutated = output_selector.transform(selected_a)

    result = test_utils.run_tensorflow(parsed_a.function.proto)
    self.assertEqual(result, 0.0)
    self.assertTrue(mutated)


def _create_simple_lambda_wrapping_graph():
  tensor_type = computation_types.TensorType(tf.int32)
  integer_identity = building_block_factory.create_compiled_identity(
      tensor_type)
  x_ref = building_blocks.Reference('x', tf.int32)
  called_integer_identity = building_blocks.Call(integer_identity, x_ref)
  lambda_wrap = building_blocks.Lambda('x', tf.int32, called_integer_identity)
  return lambda_wrap


def _create_simple_lambda_calling_graph_with_arg_thrown_on_floor():
  tensor_type = computation_types.TensorType(tf.int32)
  integer_identity = building_block_factory.create_compiled_identity(
      tensor_type)
  x_data = building_blocks.Data('x', tf.int32)
  called_integer_identity = building_blocks.Call(integer_identity, x_data)
  lambda_wrap = building_blocks.Lambda('y', tf.int32, called_integer_identity)
  return lambda_wrap


class LambdaWrappingGraphTest(test.TestCase, parameterized.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    self.assertFalse(logic.should_transform(integer_square))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_should_transform_arg_thrown_on_floor(self):
    lambda_throwing_arg_on_floor = _create_simple_lambda_calling_graph_with_arg_thrown_on_floor(
    )
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    self.assertTrue(logic.should_transform(lambda_throwing_arg_on_floor))

  def test_transform_with_arg_thrown_on_floow_constructs_correct_root_node(
      self):
    pattern = _create_simple_lambda_calling_graph_with_arg_thrown_on_floor()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone_arg_thrown_on_floor(self):
    pattern = _create_simple_lambda_calling_graph_with_arg_thrown_on_floor()
    logic = compiled_computation_transforms.LambdaWrappingGraph()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_unwraps_identity(self):
    integer_identity = _create_simple_lambda_wrapping_graph()

    lambda_unwrapper = compiled_computation_transforms.LambdaWrappingGraph()
    unwrapped_function, mutated = lambda_unwrapper.transform(integer_identity)

    for k in range(5):
      result = test_utils.run_tensorflow(unwrapped_function.proto, k)
      self.assertEqual(result, k)
    self.assertTrue(mutated)

  def test_unwraps_square(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    x_ref = building_blocks.Reference('x', tf.int32)
    called_integer_square = building_blocks.Call(integer_square, x_ref)
    lambda_wrap = building_blocks.Lambda('x', tf.int32, called_integer_square)

    lambda_unwrapper = compiled_computation_transforms.LambdaWrappingGraph()
    unwrapped_function, mutated = lambda_unwrapper.transform(lambda_wrap)

    for k in range(5):
      result = test_utils.run_tensorflow(unwrapped_function.proto, k)
      self.assertEqual(result, k * k)
    self.assertTrue(mutated)


def _create_simple_tuple_of_called_graphs():
  tensor_type = computation_types.TensorType(tf.float32)
  called_const = building_block_factory.create_tensorflow_constant(
      tensor_type, 1.0)
  tuple_of_called_graphs = building_blocks.Struct([called_const, called_const])
  return tuple_of_called_graphs


class StructCalledGraphsTest(test.TestCase, parameterized.TestCase):

  def test_empty_tuple(self):
    pattern = building_blocks.Struct([])
    logic = compiled_computation_transforms.StructCalledGraphs()
    transformed, _ = logic.transform(pattern)
    self.assertEqual(transformed.type_signature, pattern.type_signature)
    self.assertIsInstance(transformed, building_blocks.Call)
    self.assertIsInstance(transformed.function,
                          building_blocks.CompiledComputation)
    self.assertIsNone(transformed.argument)

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_tuple_of_called_graphs()
    logic = compiled_computation_transforms.StructCalledGraphs()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    self.assertFalse(tuple_parser.should_transform(integer_square))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_tuple_of_called_graphs()
    logic = compiled_computation_transforms.StructCalledGraphs()
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, building_blocks.Call)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_tuple_of_called_graphs()
    logic = compiled_computation_transforms.StructCalledGraphs()
    parsed, mutated = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)
    self.assertTrue(mutated)

  def test_named_tuple_of_graphs_preserves_type(self):
    called_noarg_const_0_type = computation_types.TensorType(tf.float32)
    called_noarg_const_0 = building_block_factory.create_tensorflow_constant(
        called_noarg_const_0_type, 0.0)
    called_noarg_const_1_type = computation_types.TensorType(tf.int32)
    called_noarg_const_1 = building_block_factory.create_tensorflow_constant(
        called_noarg_const_1_type, 1)
    tuple_of_called_graphs = building_blocks.Struct([
        ('a', called_noarg_const_0), ('b', called_noarg_const_1)
    ])
    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)
    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertTrue(mutated)

  def test_no_arg_functions_execute(self):
    called_noarg_const_0_type = computation_types.TensorType(tf.float32)
    called_noarg_const_0 = building_block_factory.create_tensorflow_constant(
        called_noarg_const_0_type, 0.0)
    called_noarg_const_1_type = computation_types.TensorType(tf.int32)
    called_noarg_const_1 = building_block_factory.create_tensorflow_constant(
        called_noarg_const_1_type, 1)
    tuple_of_called_graphs = building_blocks.Struct(
        [called_noarg_const_0, called_noarg_const_1])

    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    result = test_utils.run_tensorflow(parsed_tuple.function.proto, 10)
    self.assertEqual(result[0], 0.0)
    result = test_utils.run_tensorflow(parsed_tuple.function.proto, 0)
    self.assertEqual(result[1], 1)
    self.assertTrue(mutated)

  def test_single_function_which_takes_a_parameter_executes(self):
    called_noarg_const_0_type = computation_types.TensorType(tf.float32)
    called_noarg_const_0 = building_block_factory.create_tensorflow_constant(
        called_noarg_const_0_type, 0.0)
    integer_square = _create_compiled_computation(
        lambda x: x**2, computation_types.TensorType(tf.int32))
    square_arg = building_blocks.Reference('x', tf.int32)
    called_square = building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = building_blocks.Struct(
        [called_noarg_const_0, called_square])

    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    for k in range(5):
      result = test_utils.run_tensorflow(parsed_tuple.function.proto, k)
      self.assertEqual(result[0], 0.0)
      self.assertEqual(result[1], k**2)
    self.assertTrue(mutated)

  def test_two_functions_which_takes_tensor_parameters_executes(self):
    float_cube = _create_compiled_computation(
        lambda x: x**3, computation_types.TensorType(tf.float32))
    integer_square = _create_compiled_computation(
        lambda x: x**2, computation_types.TensorType(tf.int32))
    cube_arg = building_blocks.Reference('y', tf.float32)
    called_cube = building_blocks.Call(float_cube, cube_arg)
    square_arg = building_blocks.Reference('x', tf.int32)
    called_square = building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = building_blocks.Struct(
        [called_cube, called_square])

    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertRegexMatch(parsed_tuple.compact_representation(),
                          [r'comp#[a-zA-Z0-9]*\(<y,x>\)'])
    for k in range(5):
      result = test_utils.run_tensorflow(parsed_tuple.function.proto,
                                         [k * 1.0, k])
      self.assertEqual(result[0], (k * 1.0)**3)
      self.assertEqual(result[1], k**2)
    self.assertTrue(mutated)

  def test_tensor_plus_tuple_parameter_executes(self):
    select_from_tuple = _create_compiled_computation(
        lambda x: x[0], computation_types.StructType([tf.float32, tf.float32]))
    integer_square = _create_compiled_computation(
        lambda x: x**2, computation_types.TensorType(tf.int32))
    selection_arg = building_blocks.Reference(
        'y', computation_types.StructType([tf.float32, tf.float32]))
    called_selection = building_blocks.Call(select_from_tuple, selection_arg)
    square_arg = building_blocks.Reference('x', tf.int32)
    called_square = building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = building_blocks.Struct(
        [called_selection, called_square])

    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertRegexMatch(parsed_tuple.compact_representation(),
                          [r'comp#[a-zA-Z0-9]*\(<y,x>\)'])
    for k in range(5):
      result = test_utils.run_tensorflow(parsed_tuple.function.proto,
                                         [[k * 1.0, k * 2.0], k])
      self.assertEqual(result[0], k * 1.0)
      self.assertEqual(result[1], k**2)
    self.assertTrue(mutated)

  def test_tensor_plus_named_tuple_parameter_executes(self):
    select_from_tuple = _create_compiled_computation(
        lambda x: x.a,
        computation_types.StructType([('a', tf.float32), ('b', tf.float32)]))
    integer_square = _create_compiled_computation(
        lambda x: x**2, computation_types.TensorType(tf.int32))
    selection_arg = building_blocks.Reference('y', [('a', tf.float32),
                                                    ('b', tf.float32)])
    called_selection = building_blocks.Call(select_from_tuple, selection_arg)
    square_arg = building_blocks.Reference('x', tf.int32)
    called_square = building_blocks.Call(integer_square, square_arg)
    tuple_of_called_graphs = building_blocks.Struct(
        [called_selection, called_square])

    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    parsed_tuple, mutated = tuple_parser.transform(tuple_of_called_graphs)

    self.assertEqual(parsed_tuple.type_signature,
                     tuple_of_called_graphs.type_signature)
    self.assertRegexMatch(parsed_tuple.compact_representation(),
                          [r'comp#[a-zA-Z0-9]*\(<y,x>\)'])
    for k in range(5):
      result = test_utils.run_tensorflow(parsed_tuple.function.proto,
                                         [[k * 1.0, k * 2.0], k])
      self.assertEqual(result[0], k * 1.0)
      self.assertEqual(result[1], k**2)
    self.assertTrue(mutated)

  def test_transform_results_in_fewer_ops_with_identical_args(self):
    called_const_type = computation_types.TensorType(tf.float32)
    called_const = building_block_factory.create_tensorflow_constant(
        called_const_type, 1.0)
    id_applied_const_type = computation_types.TensorType(tf.float32)
    id_applied_const = building_blocks.Call(
        building_block_factory.create_compiled_identity(id_applied_const_type),
        called_const)
    tuple_with_identical_args = building_blocks.Struct(
        [id_applied_const, id_applied_const])

    called_float_type = computation_types.TensorType(tf.float32)
    called_float = building_block_factory.create_tensorflow_constant(
        called_float_type, 1.0)
    called_int_type = computation_types.TensorType(tf.int32)
    called_int = building_block_factory.create_tensorflow_constant(
        called_int_type, 1)
    id_applied_float_type = computation_types.TensorType(tf.float32)
    id_applied_float = building_blocks.Call(
        building_block_factory.create_compiled_identity(id_applied_float_type),
        called_float)
    id_applied_int_type = computation_types.TensorType(tf.int32)
    id_applied_int = building_blocks.Call(
        building_block_factory.create_compiled_identity(id_applied_int_type),
        called_int)
    tuple_with_distinct_args = building_blocks.Struct(
        [id_applied_float, id_applied_int])

    tuple_parser = compiled_computation_transforms.StructCalledGraphs()
    identical_tuple_parsed, _ = tuple_parser.transform(
        tuple_with_identical_args)
    distinct_tuple_parsed, _ = tuple_parser.transform(tuple_with_distinct_args)
    ops_under_identical_tuple = tree_analysis.count_tensorflow_ops_under(
        identical_tuple_parsed)
    ops_under_distinct_tuple = tree_analysis.count_tensorflow_ops_under(
        distinct_tuple_parsed)

    self.assertLess(ops_under_identical_tuple, ops_under_distinct_tuple)


def _simulate_permutation_behavior(tuple_type, permutation):
  type_elements = structure.to_elements(tuple_type)
  constructed_type_elements = []
  for k in permutation:
    constructed_type_elements.append(type_elements[k])
  return computation_types.StructType(constructed_type_elements)


def _construct_permutation_tuple(n, m, offset):
  assert offset + m < n
  tuple_type_elements = [(str(k),
                          computation_types.AbstractType('T{}'.format(k)))
                         for k in range(n)]
  initial_type = computation_types.StructType(tuple_type_elements)
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


class RemapGraphInputsTest(test.TestCase, parameterized.TestCase):

  def test_raises_on_bad_computation(self):
    tuple_type = computation_types.StructType([tf.int32])
    bad_comp = building_blocks.Data('x', computation_types.AbstractType('T'))
    with self.assertRaises(TypeError):
      compiled_computation_transforms._remap_graph_inputs(
          bad_comp, [0], tuple_type)

  def test_raises_on_bad_type(self):
    tuple_type = computation_types.StructType([tf.int32])
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    tensor_type = computation_types.TensorType(tf.int32)
    with self.assertRaises(TypeError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, [0], tensor_type)

  def test_raises_on_non_list_of_indices(self):
    tuple_type = computation_types.StructType([tf.int32])
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, 0, tuple_type)

  def test_raises_on_repeated_indices(self):
    tuple_type = computation_types.StructType([tf.int32, tf.int32])
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    with self.assertRaises(ValueError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, [0, 0], tuple_type)

  def test_raises_on_bad_index(self):
    tuple_type = computation_types.StructType([tf.int32, tf.int32])
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    with self.assertRaises(ValueError):
      compiled_computation_transforms._remap_graph_inputs(
          tuple_identity, [-1, 0], tuple_type)

  def test_permute_and_pad_index_0_of_two_tuple(self):
    index_list = [0]
    tuple_type = computation_types.StructType([tf.float32, tf.int32])
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
    tuple_type = computation_types.StructType([tf.float32, tf.int32])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(to_pad,
                     computation_types.StructType([tf.int32, tf.float32]))
    self.assertEqual(to_permute, [1, 0])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_identity_on_two_tuple(self):
    index_list = [0, 1]
    tuple_type = computation_types.StructType([tf.float32, tf.int32])
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
    tuple_type = computation_types.StructType([tf.float32, tf.int32])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(to_pad,
                     computation_types.StructType([tf.int32, tf.float32]))
    self.assertEqual(to_permute, [1, 0])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_inversion_of_named_two_tuple(self):
    index_list = [1, 0]
    tuple_type = computation_types.StructType([('a', tf.float32),
                                               ('b', tf.int32)])
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    result_of_applying_permutation = _simulate_permutation_behavior(
        to_pad, to_permute)
    self.assertEqual(
        to_pad,
        computation_types.StructType([('b', tf.int32), ('a', tf.float32)]))
    self.assertEqual(to_permute, [1, 0])
    self.assertEqual(result_of_applying_permutation, tuple_type)

  def test_permute_and_pad_single_index_deep_in_tuple(self):
    index_list = [5]
    tuple_type_list = [tf.float32, tf.int32] * 5
    tuple_type = computation_types.StructType(tuple_type_list)
    to_pad = compiled_computation_transforms._construct_padding(
        index_list, tuple_type)
    to_permute = compiled_computation_transforms._construct_permutation(
        index_list, tuple_type)
    to_pad_first_type = tuple_type_list.pop(5)
    tuple_type_list.insert(0, to_pad_first_type)
    self.assertEqual(to_pad, computation_types.StructType(tuple_type_list))
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
  tensor_type = computation_types.TensorType(tf.int32)
  integer_identity = building_block_factory.create_compiled_identity(
      tensor_type)
  tuple_reference = building_blocks.Reference('x',
                                              [tf.float32, tf.int32, tf.bool])
  selection_1 = building_blocks.Selection(tuple_reference, index=1)
  called_identity = building_blocks.Call(integer_identity, selection_1)
  lambda_wrapping_call = building_blocks.Lambda('x',
                                                tuple_reference.type_signature,
                                                called_identity)
  return lambda_wrapping_call


class LambdaCallSelectionFromArgTest(test.TestCase, parameterized.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_lambda_call_selection_from_arg()
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    lambda_parser = compiled_computation_transforms.LambdaCallSelectionFromArg()
    self.assertFalse(lambda_parser.should_transform(integer_square))

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_lambda_call_selection_from_arg()
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed_selection, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed_selection, building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_lambda_call_selection_from_arg()
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed_lambda, mutated = logic.transform(pattern)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed_lambda.type_signature.check_equivalent_to(pattern.type_signature)
    self.assertTrue(mutated)

  def test_constructs_appropriate_type_selection_by_index(self):
    tensor_type = computation_types.TensorType(tf.int32)
    integer_identity = building_block_factory.create_compiled_identity(
        tensor_type)
    tuple_reference = building_blocks.Reference('x',
                                                [tf.float32, tf.int32, tf.bool])
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    called_identity = building_blocks.Call(integer_identity, selection_1)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed_lambda, mutated = logic.transform(lambda_wrapping_call)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed_lambda.type_signature.check_equivalent_to(
        lambda_wrapping_call.type_signature)
    self.assertIsInstance(parsed_lambda, building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_executes_correctly_selection_by_index(self):
    tensor_type = computation_types.TensorType(tf.int32)
    integer_identity = building_block_factory.create_compiled_identity(
        tensor_type)
    tuple_reference = building_blocks.Reference('x',
                                                [tf.float32, tf.int32, tf.bool])
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    called_identity = building_blocks.Call(integer_identity, selection_1)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)

    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed, mutated = logic.transform(lambda_wrapping_call)

    for k in range(5):
      result = test_utils.run_tensorflow(parsed.proto, [k * 1.0, k, True])
      self.assertEqual(result, k)
    self.assertTrue(mutated)

  def test_constructs_appropriate_type_selection_by_name(self):
    integer_square = _create_compiled_computation(
        lambda x: x**2, computation_types.TensorType(tf.int32))
    tuple_reference = building_blocks.Reference('x', [('a', tf.float32),
                                                      ('b', tf.int32),
                                                      ('c', tf.bool)])
    selection_b = building_blocks.Selection(tuple_reference, name='b')
    called_square = building_blocks.Call(integer_square, selection_b)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_square)
    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed, mutated = logic.transform(lambda_wrapping_call)
    self.assertTrue(mutated)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(
        lambda_wrapping_call.type_signature)
    self.assertIsInstance(parsed, building_blocks.CompiledComputation)

  def test_executes_correctly_selection_by_name(self):
    integer_square = _create_compiled_computation(
        lambda x: x**2, computation_types.TensorType(tf.int32))
    tuple_reference = building_blocks.Reference('x', [('a', tf.float32),
                                                      ('b', tf.int32),
                                                      ('c', tf.bool)])
    selection_b = building_blocks.Selection(tuple_reference, name='b')
    called_square = building_blocks.Call(integer_square, selection_b)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_square)

    logic = compiled_computation_transforms.LambdaCallSelectionFromArg()
    parsed, mutated = logic.transform(lambda_wrapping_call)

    for k in range(5):
      result = test_utils.run_tensorflow(parsed.proto, {
          'a': k * 1.0,
          'b': k,
          'c': True,
      })
      self.assertEqual(result, k**2)
    self.assertTrue(mutated)


def _create_simple_lambda_call_tuple_of_selections_from_arg():
  tuple_type = computation_types.StructType([tf.int32, tf.float32])
  identity = building_block_factory.create_compiled_identity(tuple_type)
  tuple_reference = building_blocks.Reference('x',
                                              [tf.float32, tf.int32, tf.bool])
  selection_1 = building_blocks.Selection(tuple_reference, index=1)
  selection_0 = building_blocks.Selection(tuple_reference, index=0)
  tuple_of_selections = building_blocks.Struct([selection_1, selection_0])
  called_identity = building_blocks.Call(identity, tuple_of_selections)
  lambda_wrapping_call = building_blocks.Lambda('x',
                                                tuple_reference.type_signature,
                                                called_identity)
  return lambda_wrapping_call


class LambdaToCalledTupleOfSelectionsFromArgTest(test.TestCase,
                                                 parameterized.TestCase):

  def test_transform_raises_on_wrong_lengths(self):
    tuple_type = computation_types.StructType([tf.int32] * 3)
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x', [tf.int32] * 2)
    selection = building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = building_blocks.Struct([selection] * 3)
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
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
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    self.assertFalse(logic.should_transform(integer_square))

  def test_does_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
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
    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_leaves_type_signature_alone(self):
    pattern = _create_simple_lambda_call_tuple_of_selections_from_arg()
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(pattern)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(pattern.type_signature)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_unnamed_tuple_pad_and_permute(
      self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x',
                                                [tf.float32, tf.int32, tf.bool])
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    selection_0 = building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = building_blocks.Struct([selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(
        lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_unnamed_tuple_pad_and_permute(self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x',
                                                [tf.float32, tf.int32, tf.bool])
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    selection_0 = building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = building_blocks.Struct([selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)

    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)

    result = test_utils.run_tensorflow(parsed.proto, [0.0, 1, True])
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 0.0)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_unnamed_tuple_permute_only(self):
    tuple_type = computation_types.StructType([tf.bool, tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x',
                                                [tf.float32, tf.int32, tf.bool])
    selection_2 = building_blocks.Selection(tuple_reference, index=2)
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    selection_0 = building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = building_blocks.Struct(
        [selection_2, selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(
        lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_unnamed_tuple_permute_only(self):
    tuple_type = computation_types.StructType([tf.bool, tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x',
                                                [tf.float32, tf.int32, tf.bool])
    selection_2 = building_blocks.Selection(tuple_reference, index=2)
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    selection_0 = building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = building_blocks.Struct(
        [selection_2, selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)

    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)

    result = test_utils.run_tensorflow(parsed.proto, [0., 1, True])
    self.assertEqual(result[0], True)
    self.assertEqual(result[1], 1)
    self.assertEqual(result[2], 0.)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_named_tuple_name_selection_pad_and_permute(
      self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x', [('a', tf.float32),
                                                      ('b', tf.int32),
                                                      ('c', tf.bool)])
    selection_1 = building_blocks.Selection(tuple_reference, name='b')
    selection_0 = building_blocks.Selection(tuple_reference, name='a')
    tuple_of_selections = building_blocks.Struct([selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(
        lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_named_tuple_name_selection_pad_and_permute(self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x', [('a', tf.float32),
                                                      ('b', tf.int32),
                                                      ('c', tf.bool)])
    selection_1 = building_blocks.Selection(tuple_reference, name='b')
    selection_0 = building_blocks.Selection(tuple_reference, name='a')
    tuple_of_selections = building_blocks.Struct([selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)

    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)

    result = test_utils.run_tensorflow(parsed.proto, {
        'a': 0.0,
        'b': 1,
        'c': False,
    })
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 0.0)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_named_tuple_index_selection(self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x', [('a', tf.float32),
                                                      ('b', tf.int32),
                                                      ('c', tf.bool)])
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    selection_0 = building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = building_blocks.Struct([selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)
    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(
        lambda_wrapping_call.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_named_tuple_index_selection(self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    tuple_reference = building_blocks.Reference('x', [('a', tf.float32),
                                                      ('b', tf.int32),
                                                      ('c', tf.bool)])
    selection_1 = building_blocks.Selection(tuple_reference, index=1)
    selection_0 = building_blocks.Selection(tuple_reference, index=0)
    tuple_of_selections = building_blocks.Struct([selection_1, selection_0])
    called_identity = building_blocks.Call(identity, tuple_of_selections)
    lambda_wrapping_call = building_blocks.Lambda(
        'x', tuple_reference.type_signature, called_identity)

    logic = compiled_computation_transforms.LambdaToCalledTupleOfSelectionsFromArg(
    )
    parsed, mutated = logic.transform(lambda_wrapping_call)

    result = test_utils.run_tensorflow(parsed.proto, {
        'a': 0.,
        'b': 1,
        'c': False,
    })
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 0.)
    self.assertTrue(mutated)


class ComposeTensorFlowBlocksTest(test.TestCase, parameterized.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks(None)

  def test_raises_on_single_computation(self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks(identity)

  def test_raises_bad_arg_in_list(self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks([identity, 0])

  def test_raises_mismatched_parameter_and_result_types(self):
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    bad_tuple_type = computation_types.StructType([tf.float32, tf.int32])
    bad_identity = building_block_factory.create_compiled_identity(
        bad_tuple_type)
    with self.assertRaises(TypeError):
      compiled_computation_transforms.compose_tensorflow_blocks(
          [identity, bad_identity])

  def test_composes_no_arg_fn_with_add_one_types_correctly(self):
    tensor_type = computation_types.TensorType(tf.int32)
    noarg_fn = building_block_factory.create_tensorflow_constant(tensor_type, 0)
    add_one_fn = _create_compiled_computation(
        lambda x: x + 1, computation_types.TensorType(tf.int32))
    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [add_one_fn, noarg_fn.function])
    expected_type = computation_types.FunctionType(None, tf.int32)
    self.assertEqual(composed_fn.type_signature, expected_type)

  def test_composes_no_arg_fn_with_add_one_executes_correctly(self):
    tensor_type = computation_types.TensorType(tf.int32)
    noarg_fn = building_block_factory.create_tensorflow_constant(tensor_type, 0)
    add_one_fn = _create_compiled_computation(
        lambda x: x + 1, computation_types.TensorType(tf.int32))

    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [add_one_fn, noarg_fn.function])

    result = test_utils.run_tensorflow(composed_fn.proto)
    self.assertEqual(result, 1)

  def test_composes_tensor_functions_types_correctly(self):
    int_to_float_fn = _create_compiled_computation(
        lambda x: tf.cast(x, tf.float32) * 2.0,
        computation_types.TensorType(tf.int32))
    float_to_float_fn = _create_compiled_computation(
        lambda x: x * 2.0, computation_types.TensorType(tf.float32))
    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_to_float_fn, int_to_float_fn])
    expected_type = computation_types.FunctionType(tf.int32, tf.float32)
    self.assertEqual(composed_fn.type_signature, expected_type)

  def test_composes_tensor_function_executes_correctly(self):
    int_to_float_fn = _create_compiled_computation(
        lambda x: tf.cast(x, tf.float32) * 2.0,
        computation_types.TensorType(tf.int32))
    float_to_float_fn = _create_compiled_computation(
        lambda x: x * 2.0, computation_types.TensorType(tf.float32))

    composed_fn = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_to_float_fn, int_to_float_fn])

    for k in range(5):
      result = test_utils.run_tensorflow(composed_fn.proto, k)
      self.assertEqual(result, k * 4.0)

  def test_compose_integer_identities_executes_correctly(self):
    tensor_type = computation_types.TensorType(tf.int32)
    identity = building_block_factory.create_compiled_identity(tensor_type)

    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [identity, identity])

    result = test_utils.run_tensorflow(composed.proto, 0)
    self.assertEqual(result, 0)

  def test_composes_unnamed_tuple_functions_types_correctly(self):
    int_float_flip = _create_compiled_computation(
        lambda x: [x[1], x[0]],
        computation_types.StructType([tf.int32, tf.float32]))
    float_int_flip = _create_compiled_computation(
        lambda x: [x[1], x[0]],
        computation_types.StructType([tf.float32, tf.int32]))
    composed_fn_float_int = compiled_computation_transforms.compose_tensorflow_blocks(
        [int_float_flip, float_int_flip])
    composed_fn_int_float = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_int_flip, int_float_flip])
    expected_type_int_float = computation_types.FunctionType(
        [tf.int32, tf.float32], [tf.int32, tf.float32])
    expected_type_float_int = computation_types.FunctionType(
        [tf.float32, tf.int32], [tf.float32, tf.int32])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    composed_fn_float_int.type_signature.check_equivalent_to(
        expected_type_float_int)
    composed_fn_int_float.type_signature.check_equivalent_to(
        expected_type_int_float)

  def test_composes_unnamed_tuple_functions_executes_correctly(self):
    int_float_flip = _create_compiled_computation(
        lambda x: [x[1], x[0]],
        computation_types.StructType([tf.int32, tf.float32]))
    float_int_flip = _create_compiled_computation(
        lambda x: [x[1], x[0]],
        computation_types.StructType([tf.float32, tf.int32]))

    composed_fn_float_int = compiled_computation_transforms.compose_tensorflow_blocks(
        [int_float_flip, float_int_flip])

    result = test_utils.run_tensorflow(composed_fn_float_int.proto, [10.0, 0])
    self.assertEqual(result[0], 10.0)
    self.assertEqual(result[1], 0)
    self.assertLen(result, 2)

    composed_fn_int_float = compiled_computation_transforms.compose_tensorflow_blocks(
        [float_int_flip, int_float_flip])

    result = test_utils.run_tensorflow(composed_fn_int_float.proto, [10, 0.0])
    self.assertEqual(result[0], 10)
    self.assertEqual(result[1], 0.0)
    self.assertLen(result, 2)

  def test_composes_named_tuple_function_with_unnamed_tuple_function_types_correctly(
      self):
    drop_names = _create_compiled_computation(
        lambda x: [x[0], x[1]],
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    unamed_types = computation_types.StructType([tf.int32, tf.float32])
    unnamed_identity = building_block_factory.create_compiled_identity(
        unamed_types)
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [unnamed_identity, drop_names])
    expected_type = computation_types.FunctionType([('a', tf.int32),
                                                    ('b', tf.float32)],
                                                   [tf.int32, tf.float32])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    composed.type_signature.check_equivalent_to(expected_type)

  def test_composes_named_tuple_function_with_unnamed_tuple_function_executes_correctly(
      self):
    drop_names = _create_compiled_computation(
        lambda x: [x[0], x[1]],
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    unamed_types = computation_types.StructType([tf.int32, tf.float32])
    unnamed_identity = building_block_factory.create_compiled_identity(
        unamed_types)
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [unnamed_identity, drop_names])
    result = test_utils.run_tensorflow(composed.proto, {'a': 0, 'b': 1.0})
    self.assertEqual(result[0], 0)
    self.assertEqual(result[1], 1.0)
    self.assertLen(result, 2)

  def test_composes_named_tuple_functions_types_correctly(self):
    flip_order = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    identity = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        computation_types.StructType([('b', tf.float32), ('a', tf.int32)]))
    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [identity, flip_order])
    expected_type = computation_types.FunctionType([('a', tf.int32),
                                                    ('b', tf.float32)],
                                                   [('b', tf.float32),
                                                    ('a', tf.int32)])
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    composed.type_signature.check_equivalent_to(expected_type)

  def test_composes_named_tuple_functions_executes_correctly(self):
    flip_order = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    identity = _create_compiled_computation(
        lambda x: collections.OrderedDict([('b', x.b), ('a', x.a)]),
        computation_types.StructType([('b', tf.float32), ('a', tf.int32)]))

    composed = compiled_computation_transforms.compose_tensorflow_blocks(
        [identity, flip_order])

    result = test_utils.run_tensorflow(
        composed.proto, collections.OrderedDict({
            'a': 0,
            'b': 1.0,
        }))
    self.assertEqual(result[0], 1.0)
    self.assertEqual(result[1], 0)
    self.assertLen(result, 2)

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

    result = test_utils.run_tensorflow(integer_result.proto)
    self.assertEqual(result, 10)


def _create_simple_called_composition_of_tf_blocks():
  tensor_type = computation_types.TensorType(tf.int32)
  zero = building_block_factory.create_tensorflow_constant(tensor_type, 0)
  add_one = _create_compiled_computation(lambda x: x + 1,
                                         computation_types.TensorType(tf.int32))
  one = building_blocks.Call(add_one, zero)
  return one


class CalledCompositionOfTensorFlowBlocksTest(test.TestCase,
                                              parameterized.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_called_composition_of_tf_blocks()
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertFalse(logic.should_transform(integer_square))

  def test_should_not_transform_single_called_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    int_ref = building_blocks.Reference('x', tf.int32)
    called_square = building_blocks.Call(integer_square, int_ref)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertFalse(logic.should_transform(called_square))

  def test_should_not_transform_called_lambda_on_called_compiled_computation(
      self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    int_ref = building_blocks.Reference('x', tf.int32)
    called_square = building_blocks.Call(integer_square, int_ref)
    lambda_wrapper = building_blocks.Lambda('x', tf.int32, called_square)
    outer_int_ref = building_blocks.Reference('y', tf.int32)
    called_lambda = building_blocks.Call(lambda_wrapper, outer_int_ref)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    self.assertFalse(logic.should_transform(called_lambda))

  def test_does_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
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
    self.assertIsInstance(parsed, building_blocks.Call)
    self.assertIsInstance(parsed.function, building_blocks.CompiledComputation)
    self.assertTrue(mutated)

  def test_transform_reduces_number_of_compiled_computations(self):
    pattern = _create_simple_called_composition_of_tf_blocks()
    original_count = tree_analysis.count_types(
        pattern, building_blocks.CompiledComputation)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, _ = logic.transform(pattern)
    new_count = tree_analysis.count_types(parsed,
                                          building_blocks.CompiledComputation)
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

    result = test_utils.run_tensorflow(parsed.function.proto, 0)
    self.assertEqual(result, 1)
    result = test_utils.run_tensorflow(parsed.function.proto, 1)
    self.assertEqual(result, 1)
    result = test_utils.run_tensorflow(parsed.function.proto, 2)
    self.assertEqual(result, 1)

  def test_constructs_correct_type_signature_named_tuple_argument(self):
    tuple_type = computation_types.StructType([('a', tf.int32),
                                               ('b', tf.float32)])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    sel_int = _create_compiled_computation(
        lambda x: x.a,
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))

    tuple_reference = building_blocks.Reference('x', [('a', tf.int32),
                                                      ('b', tf.float32)])

    called_identity = building_blocks.Call(identity, tuple_reference)
    called_integer_selection = building_blocks.Call(sel_int, called_identity)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, mutated = logic.transform(called_integer_selection)
    self.assertEqual(parsed.type_signature,
                     called_integer_selection.type_signature)
    self.assertEqual(parsed.argument.type_signature,
                     tuple_reference.type_signature)
    self.assertTrue(mutated)

  def test_executes_named_tuple_argument(self):
    tuple_type = computation_types.StructType([('a', tf.int32),
                                               ('b', tf.float32)])
    identity = building_block_factory.create_compiled_identity(tuple_type)
    sel_int = _create_compiled_computation(
        lambda x: x.a,
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))

    tuple_reference = building_blocks.Reference('x', [('a', tf.int32),
                                                      ('b', tf.float32)])
    called_identity = building_blocks.Call(identity, tuple_reference)
    called_integer_selection = building_blocks.Call(sel_int, called_identity)

    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, _ = logic.transform(called_integer_selection)

    result = test_utils.run_tensorflow(parsed.function.proto, {
        'a': 1,
        'b': 0.0
    })
    self.assertEqual(result, 1)
    result = test_utils.run_tensorflow(parsed.function.proto, {
        'a': 0,
        'b': 1.0
    })
    self.assertEqual(result, 0)

  def test_constructs_correct_type_signature_named_tuple_result(self):
    namer = _create_compiled_computation(
        lambda x: collections.OrderedDict([('a', x[0]), ('b', x[1])]),
        computation_types.StructType([tf.int32, tf.float32]))
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)

    tuple_reference = building_blocks.Reference('x', [tf.int32, tf.float32])

    called_identity = building_blocks.Call(identity, tuple_reference)
    called_namer = building_blocks.Call(namer, called_identity)
    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, mutated = logic.transform(called_namer)
    self.assertEqual(parsed.type_signature, called_namer.type_signature)
    self.assertTrue(mutated)

  def test_executes_correctly_named_tuple_result(self):
    namer = _create_compiled_computation(
        lambda x: collections.OrderedDict([('a', x[0]), ('b', x[1])]),
        computation_types.StructType([tf.int32, tf.float32]))
    tuple_type = computation_types.StructType([tf.int32, tf.float32])
    identity = building_block_factory.create_compiled_identity(tuple_type)

    tuple_reference = building_blocks.Reference('x', [tf.int32, tf.float32])

    called_identity = building_blocks.Call(identity, tuple_reference)
    called_namer = building_blocks.Call(namer, called_identity)

    logic = compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(
    )
    parsed, _ = logic.transform(called_namer)

    result = test_utils.run_tensorflow(parsed.function.proto, [1, 0.0])
    self.assertEqual(result[0], 1)
    self.assertEqual(result.a, 1)
    self.assertEqual(result[1], 0.)
    self.assertEqual(result.b, 0.)
    result = test_utils.run_tensorflow(parsed.function.proto, [0, 1.0])
    self.assertEqual(result[0], 0)
    self.assertEqual(result.a, 0)
    self.assertEqual(result[1], 1.0)
    self.assertEqual(result.b, 1.0)


def _create_simple_called_graph_on_replicated_arg(n_replicates=2):
  tuple_type = computation_types.StructType([tf.int32] * n_replicates)
  tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
  ref_to_int = building_blocks.Reference('x', tf.int32)
  called_tuple_id = building_blocks.Call(
      tuple_identity, building_blocks.Struct([ref_to_int] * n_replicates))
  return called_tuple_id


class CalledGraphOnReplicatedArgTest(test.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_called_graph_on_replicated_arg()
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_transform_identifies_longer_pattern(self):
    pattern = _create_simple_called_graph_on_replicated_arg(n_replicates=5)
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertFalse(logic.should_transform(integer_square))

  def test_should_not_transform_non_tuple_wrapped_lambda_to_called_graph(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    int_ref = building_blocks.Reference('x', tf.int32)
    called_square = building_blocks.Call(integer_square, int_ref)
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    self.assertFalse(logic.should_transform(called_square))

  def test_does_not_transform_compiled_computation(self):
    integer_square = _create_compiled_computation(
        lambda x: x * x, computation_types.TensorType(tf.int32))
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(integer_square)
    self.assertEqual(parsed, integer_square)
    self.assertFalse(mutated)

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_called_graph_on_replicated_arg()
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(pattern)
    self.assertIsInstance(parsed, building_blocks.Call)
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

    result = test_utils.run_tensorflow(parsed.function.proto, 0)
    self.assertEqual(result, structure.Struct([(None, 0), (None, 0)]))
    result = test_utils.run_tensorflow(parsed.function.proto, 1)
    self.assertEqual(result, structure.Struct([(None, 1), (None, 1)]))
    result = test_utils.run_tensorflow(parsed.function.proto, 2)
    self.assertEqual(result, structure.Struct([(None, 2), (None, 2)]))

  def test_executes_correctly_several_replicates(self):
    pattern = _create_simple_called_graph_on_replicated_arg(n_replicates=5)

    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, _ = logic.transform(pattern)

    result = test_utils.run_tensorflow(parsed.function.proto, 0)
    for k in range(5):
      self.assertEqual(result[k], 0)
    self.assertLen(result, 5)
    result = test_utils.run_tensorflow(parsed.function.proto, 1)
    for k in range(5):
      self.assertEqual(result[k], 1)
    self.assertLen(result, 5)

  def test_constructs_correct_type_signature_nested_tuple_argument(self):
    slicer = _create_compiled_computation(
        lambda x: [x[0][0], x[1][1]],
        computation_types.StructType([[tf.int32, tf.float32],
                                      [tf.int32, tf.float32]]))
    tuple_reference = building_blocks.Reference('x', [tf.int32, tf.float32])

    called_slicer = building_blocks.Call(
        slicer, building_blocks.Struct([tuple_reference, tuple_reference]))
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(called_slicer)
    self.assertEqual(parsed.type_signature, called_slicer.type_signature)
    self.assertTrue(mutated)

  def test_constructs_correct_type_signature_nested_named_tuple_argument(self):
    slicer = _create_compiled_computation(
        lambda x: [x[0][0], x[1][1]],
        computation_types.StructType([[('a', tf.int32), ('b', tf.float32)],
                                      [('a', tf.int32), ('b', tf.float32)]]))
    tuple_reference = building_blocks.Reference('x', [('a', tf.int32),
                                                      ('b', tf.float32)])

    called_slicer = building_blocks.Call(
        slicer, building_blocks.Struct([tuple_reference, tuple_reference]))
    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, mutated = logic.transform(called_slicer)
    self.assertEqual(parsed.type_signature, called_slicer.type_signature)
    self.assertTrue(mutated)

  def test_execution_nested_tuple_argument(self):
    slicer = _create_compiled_computation(
        lambda x: [x[0][0], x[1][1]],
        computation_types.StructType([[tf.int32, tf.float32],
                                      [tf.int32, tf.float32]]))
    tuple_reference = building_blocks.Reference('x', [tf.int32, tf.float32])

    called_slicer = building_blocks.Call(
        slicer, building_blocks.Struct([tuple_reference, tuple_reference]))

    logic = compiled_computation_transforms.CalledGraphOnReplicatedArg()
    parsed, _ = logic.transform(called_slicer)

    result = test_utils.run_tensorflow(parsed.function.proto, [0, 1.0])
    self.assertEqual(result[0], 0)
    self.assertEqual(result[1], 1.0)
    result = test_utils.run_tensorflow(parsed.function.proto, [1, 0.0])
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 0.)


def _create_simple_lambda_wrapping_noarg_graph():
  embedded_type = computation_types.TensorType(tf.int32)
  embedded_constant = building_block_factory.create_tensorflow_constant(
      embedded_type, 0)
  return building_blocks.Lambda('x', tf.float32, embedded_constant)


class LambdaWrappingNoArgGraphTest(test.TestCase, parameterized.TestCase):

  def test_should_transform_identifies_correct_pattern(self):
    pattern = _create_simple_lambda_wrapping_noarg_graph()
    logic = compiled_computation_transforms.LambdaWrappingNoArgGraph()
    self.assertTrue(logic.should_transform(pattern))

  def test_should_transform_does_not_identify_lambda_to_graph_with_arg(self):
    pattern = _create_simple_lambda_wrapping_graph()
    logic = compiled_computation_transforms.LambdaWrappingNoArgGraph()
    self.assertFalse(logic.should_transform(pattern))

  def test_transform_leaves_type_signature_untouched(self):
    pattern = _create_simple_lambda_wrapping_noarg_graph()
    logic = compiled_computation_transforms.LambdaWrappingNoArgGraph()
    parsed, _ = logic.transform(pattern)
    self.assertEqual(parsed.type_signature, pattern.type_signature)

  def test_transform_constructs_correct_root_node(self):
    pattern = _create_simple_lambda_wrapping_noarg_graph()
    logic = compiled_computation_transforms.LambdaWrappingNoArgGraph()
    parsed, _ = logic.transform(pattern)
    self.assertIsInstance(parsed, building_blocks.CompiledComputation)

  def test_updates_init_op(self):

    with tf.Graph().as_default() as graph:
      var = tf.Variable(initial_value=0.0, name='var1', import_scope='')
      assign_op = var.assign_add(tf.constant(1.0))
      out = tf.add(1.0, assign_op)
      init_op_name = tf.compat.v1.global_variables_initializer().name

    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        out, graph)
    type_spec = computation_types.FunctionType(None, result_type)
    serialized_type_spec = type_serialization.serialize_type(type_spec)

    proto_with_init_op = pb.TensorFlow(
        graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
        initialize_op=init_op_name,
        result=result_binding)

    constant_with_init_op = building_blocks.Call(
        building_blocks.CompiledComputation(
            pb.Computation(
                type=serialized_type_spec, tensorflow=proto_with_init_op)),
        None)
    lambda_wrapping_constant = building_blocks.Lambda('x', tf.float32,
                                                      constant_with_init_op)
    logic = compiled_computation_transforms.LambdaWrappingNoArgGraph()
    parsed, transformed = logic.transform(lambda_wrapping_constant)
    self.assertTrue(transformed)
    split_init_op_name = parsed.proto.tensorflow.initialize_op.split('/')
    self.assertNotEmpty(split_init_op_name[0])
    self.assertEqual(split_init_op_name[1], init_op_name)

  @parameterized.named_parameters([(str(n), n * 1.0) for n in range(10)])
  def test_function_returned_independent_of_argument(self, arg):
    pattern = _create_simple_lambda_wrapping_noarg_graph()

    logic = compiled_computation_transforms.LambdaWrappingNoArgGraph()
    parsed, _ = logic.transform(pattern)

    result = test_utils.run_tensorflow(parsed.proto, arg)
    self.assertEqual(result, 0)


class TensorFlowOptimizerTest(test.TestCase):

  def test_should_transform_compiled_computation(self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    self.assertTrue(tf_optimizer.should_transform(compiled_computation))

  def test_should_not_transform_reference(self):
    reference = building_blocks.Reference('x', tf.int32)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    self.assertFalse(tf_optimizer.should_transform(reference))

  def test_transform_compiled_computation_returns_compiled_computation(self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)

  def test_transform_compiled_computation_semantic_equivalence(self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)
    zero_before_transform = test_utils.run_tensorflow(
        compiled_computation.proto, 0)
    zero_after_transform = test_utils.run_tensorflow(transformed_comp.proto, 0)
    self.assertEqual(zero_before_transform, zero_after_transform)


if __name__ == '__main__':
  test.main()
