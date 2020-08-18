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

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import tree_to_cc_transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances


def _create_compiled_computation(py_fn, parameter_type):
  proto, type_signature = tensorflow_computation_factory.create_computation_for_py_fn(
      py_fn, parameter_type)
  return building_blocks.CompiledComputation(
      proto, type_signature=type_signature)


def parse_tff_to_tf(comp):
  comp, _ = tree_transformations.insert_called_tf_identity_at_leaves(comp)
  parser_callable = tree_to_cc_transformations.TFParser()
  comp, _ = tree_transformations.replace_called_lambda_with_block(comp)
  comp, _ = tree_transformations.inline_block_locals(comp)
  comp, _ = tree_transformations.replace_selection_from_tuple_with_element(comp)
  new_comp, transformed = transformation_utils.transform_postorder(
      comp, parser_callable)
  return new_comp, transformed


class ParseTFFToTFTest(test.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      parse_tff_to_tf(None)

  def test_does_not_transform_standalone_intrinsic(self):
    type_signature = computation_types.TensorType(tf.int32)
    standalone_intrinsic = building_blocks.Intrinsic('dummy', type_signature)
    non_transformed, _ = parse_tff_to_tf(standalone_intrinsic)
    self.assertEqual(standalone_intrinsic.compact_representation(),
                     non_transformed.compact_representation())

  def test_replaces_lambda_to_selection_from_called_graph_with_tf_of_same_type(
      self):
    identity_tf_block_type = computation_types.StructType(
        [tf.int32, tf.float32])
    identity_tf_block = building_block_factory.create_compiled_identity(
        identity_tf_block_type)
    tuple_ref = building_blocks.Reference('x', [tf.int32, tf.float32])
    called_tf_block = building_blocks.Call(identity_tf_block, tuple_ref)
    selection_from_call = building_blocks.Selection(called_tf_block, index=1)
    lambda_wrapper = building_blocks.Lambda('x', [tf.int32, tf.float32],
                                            selection_from_call)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)
    self.assertEqual(exec_lambda([0, 1.]), exec_tf([0, 1.]))

  def test_replaces_lambda_to_called_graph_with_tf_of_same_type(self):
    identity_tf_block_type = computation_types.TensorType(tf.int32)
    identity_tf_block = building_block_factory.create_compiled_identity(
        identity_tf_block_type)
    int_ref = building_blocks.Reference('x', tf.int32)
    called_tf_block = building_blocks.Call(identity_tf_block, int_ref)
    lambda_wrapper = building_blocks.Lambda('x', tf.int32, called_tf_block)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)
    self.assertEqual(exec_lambda(2), exec_tf(2))

  def test_replaces_lambda_to_called_graph_on_selection_from_arg_with_tf_of_same_type(
      self):
    identity_tf_block_type = computation_types.TensorType(tf.int32)
    identity_tf_block = building_block_factory.create_compiled_identity(
        identity_tf_block_type)
    tuple_ref = building_blocks.Reference('x', [tf.int32, tf.float32])
    selected_int = building_blocks.Selection(tuple_ref, index=0)
    called_tf_block = building_blocks.Call(identity_tf_block, selected_int)
    lambda_wrapper = building_blocks.Lambda('x', [tf.int32, tf.float32],
                                            called_tf_block)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)

    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)
    self.assertEqual(exec_lambda([3, 4.]), exec_tf([3, 4.]))

  def test_replaces_lambda_to_called_graph_on_selection_from_arg_with_tf_of_same_type_with_names(
      self):
    identity_tf_block_type = computation_types.TensorType(tf.int32)
    identity_tf_block = building_block_factory.create_compiled_identity(
        identity_tf_block_type)
    tuple_ref = building_blocks.Reference('x', [('a', tf.int32),
                                                ('b', tf.float32)])
    selected_int = building_blocks.Selection(tuple_ref, index=0)
    called_tf_block = building_blocks.Call(identity_tf_block, selected_int)
    lambda_wrapper = building_blocks.Lambda('x', [('a', tf.int32),
                                                  ('b', tf.float32)],
                                            called_tf_block)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    self.assertEqual(parsed.type_signature, lambda_wrapper.type_signature)
    self.assertEqual(exec_lambda({'a': 5, 'b': 6.}), exec_tf({'a': 5, 'b': 6.}))

  def test_replaces_lambda_to_called_graph_on_tuple_of_selections_from_arg_with_tf_of_same_type(
      self):
    identity_tf_block_type = computation_types.StructType([tf.int32, tf.bool])
    identity_tf_block = building_block_factory.create_compiled_identity(
        identity_tf_block_type)
    tuple_ref = building_blocks.Reference('x', [tf.int32, tf.float32, tf.bool])
    selected_int = building_blocks.Selection(tuple_ref, index=0)
    selected_bool = building_blocks.Selection(tuple_ref, index=2)
    created_tuple = building_blocks.Struct([selected_int, selected_bool])
    called_tf_block = building_blocks.Call(identity_tf_block, created_tuple)
    lambda_wrapper = building_blocks.Lambda('x',
                                            [tf.int32, tf.float32, tf.bool],
                                            called_tf_block)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)
    self.assertEqual(exec_lambda([7, 8., True]), exec_tf([7, 8., True]))

  def test_replaces_lambda_to_called_graph_on_tuple_of_selections_from_arg_with_tf_of_same_type_with_names(
      self):
    identity_tf_block_type = computation_types.StructType([tf.int32, tf.bool])
    identity_tf_block = building_block_factory.create_compiled_identity(
        identity_tf_block_type)
    tuple_ref = building_blocks.Reference('x', [('a', tf.int32),
                                                ('b', tf.float32),
                                                ('c', tf.bool)])
    selected_int = building_blocks.Selection(tuple_ref, index=0)
    selected_bool = building_blocks.Selection(tuple_ref, index=2)
    created_tuple = building_blocks.Struct([selected_int, selected_bool])
    called_tf_block = building_blocks.Call(identity_tf_block, created_tuple)
    lambda_wrapper = building_blocks.Lambda('x', [('a', tf.int32),
                                                  ('b', tf.float32),
                                                  ('c', tf.bool)],
                                            called_tf_block)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    self.assertEqual(parsed.type_signature, lambda_wrapper.type_signature)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)
    self.assertEqual(
        exec_lambda({
            'a': 9,
            'b': 10.,
            'c': False
        }), exec_tf({
            'a': 9,
            'b': 10.,
            'c': False
        }))

  def test_replaces_lambda_to_unnamed_tuple_of_called_graphs_with_tf_of_same_type(
      self):
    int_tensor_type = computation_types.TensorType(tf.int32)
    int_identity_tf_block = building_block_factory.create_compiled_identity(
        int_tensor_type)
    float_tensor_type = computation_types.TensorType(tf.float32)
    float_identity_tf_block = building_block_factory.create_compiled_identity(
        float_tensor_type)
    tuple_ref = building_blocks.Reference('x', [tf.int32, tf.float32])
    selected_int = building_blocks.Selection(tuple_ref, index=0)
    selected_float = building_blocks.Selection(tuple_ref, index=1)

    called_int_tf_block = building_blocks.Call(int_identity_tf_block,
                                               selected_int)
    called_float_tf_block = building_blocks.Call(float_identity_tf_block,
                                                 selected_float)
    tuple_of_called_graphs = building_blocks.Struct(
        [called_int_tf_block, called_float_tf_block])
    lambda_wrapper = building_blocks.Lambda('x', [tf.int32, tf.float32],
                                            tuple_of_called_graphs)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)

    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)
    self.assertEqual(exec_lambda([11, 12.]), exec_tf([11, 12.]))

  def test_replaces_lambda_to_named_tuple_of_called_graphs_with_tf_of_same_type(
      self):
    int_tensor_type = computation_types.TensorType(tf.int32)
    int_identity_tf_block = building_block_factory.create_compiled_identity(
        int_tensor_type)
    float_tensor_type = computation_types.TensorType(tf.float32)
    float_identity_tf_block = building_block_factory.create_compiled_identity(
        float_tensor_type)
    tuple_ref = building_blocks.Reference('x', [tf.int32, tf.float32])
    selected_int = building_blocks.Selection(tuple_ref, index=0)
    selected_float = building_blocks.Selection(tuple_ref, index=1)

    called_int_tf_block = building_blocks.Call(int_identity_tf_block,
                                               selected_int)
    called_float_tf_block = building_blocks.Call(float_identity_tf_block,
                                                 selected_float)
    tuple_of_called_graphs = building_blocks.Struct([('a', called_int_tf_block),
                                                     ('b',
                                                      called_float_tf_block)])
    lambda_wrapper = building_blocks.Lambda('x', [tf.int32, tf.float32],
                                            tuple_of_called_graphs)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)
    self.assertEqual(exec_lambda([13, 14.]), exec_tf([13, 14.]))

  def test_replaces_lambda_to_called_composition_of_tf_blocks_with_tf_of_same_type_named_param(
      self):
    selection_type = computation_types.StructType([('a', tf.int32),
                                                   ('b', tf.float32)])
    selection_tf_block = _create_compiled_computation(lambda x: x[0],
                                                      selection_type)
    add_one_int_type = computation_types.TensorType(tf.int32)
    add_one_int_tf_block = _create_compiled_computation(lambda x: x + 1,
                                                        add_one_int_type)
    int_ref = building_blocks.Reference('x', [('a', tf.int32),
                                              ('b', tf.float32)])
    called_selection = building_blocks.Call(selection_tf_block, int_ref)
    one_added = building_blocks.Call(add_one_int_tf_block, called_selection)
    lambda_wrapper = building_blocks.Lambda('x', [('a', tf.int32),
                                                  ('b', tf.float32)], one_added)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)

    self.assertEqual(
        exec_lambda({
            'a': 15,
            'b': 16.
        }), exec_tf({
            'a': 15,
            'b': 16.
        }))

  def test_replaces_lambda_to_called_tf_block_with_replicated_lambda_arg_with_tf_block_of_same_type(
      self):
    sum_and_add_one_type = computation_types.StructType([tf.int32, tf.int32])
    sum_and_add_one = _create_compiled_computation(lambda x: x[0] + x[1] + 1,
                                                   sum_and_add_one_type)
    int_ref = building_blocks.Reference('x', tf.int32)
    tuple_of_ints = building_blocks.Struct((int_ref, int_ref))
    summed = building_blocks.Call(sum_and_add_one, tuple_of_ints)
    lambda_wrapper = building_blocks.Lambda('x', tf.int32, summed)

    parsed, modified = parse_tff_to_tf(lambda_wrapper)
    exec_lambda = computation_wrapper_instances.building_block_to_computation(
        lambda_wrapper)
    exec_tf = computation_wrapper_instances.building_block_to_computation(
        parsed)

    self.assertIsInstance(parsed, building_blocks.CompiledComputation)
    self.assertTrue(modified)
    # TODO(b/157172423): change to assertEqual when Py container is preserved.
    parsed.type_signature.check_equivalent_to(lambda_wrapper.type_signature)

    self.assertEqual(exec_lambda(17), exec_tf(17))


if __name__ == '__main__':
  factory = executor_stacks.local_executor_factory()
  context = execution_context.ExecutionContext(executor_fn=factory)
  set_default_context.set_default_context(context)
  test.main()
