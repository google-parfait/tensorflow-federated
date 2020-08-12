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

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import tree_to_cc_transformations
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import placement_literals


class RemoveLambdasAndBlocksTest(test.TestCase):

  def assertNoLambdasOrBlocks(self, comp):

    def _transform(comp):
      if (comp.is_call() and comp.function.is_lambda()) or comp.is_block():
        raise AssertionError('Encountered disallowed computation: {}'.format(
            comp.compact_representation()))
      return comp, True

    transformation_utils.transform_postorder(comp, _transform)

  def test_with_simple_called_lambda(self):
    identity_lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    called_lambda = building_blocks.Call(identity_lam,
                                         building_blocks.Data('a', tf.int32))
    lambdas_and_blocks_removed, modified = transformations.remove_called_lambdas_and_blocks(
        called_lambda)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)
    self.assertEqual(lambdas_and_blocks_removed.compact_representation(), 'a')

  def test_with_simple_block(self):
    data = building_blocks.Data('a', tf.int32)
    simple_block = building_blocks.Block([('x', data)],
                                         building_blocks.Reference(
                                             'x', tf.int32))
    lambdas_and_blocks_removed, modified = transformations.remove_called_lambdas_and_blocks(
        simple_block)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)
    self.assertEqual(lambdas_and_blocks_removed.compact_representation(), 'a')

  def test_with_structure_replacing_federated_map(self):
    function_type = computation_types.FunctionType(tf.int32, tf.int32)
    tuple_ref = building_blocks.Reference('arg', [
        function_type,
        tf.int32,
    ])
    fn = building_blocks.Selection(tuple_ref, index=0)
    arg = building_blocks.Selection(tuple_ref, index=1)
    called_fn = building_blocks.Call(fn, arg)
    concrete_fn = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    concrete_arg = building_blocks.Data('a', tf.int32)
    arg_tuple = building_blocks.Struct([concrete_fn, concrete_arg])
    generated_structure = building_blocks.Block([('arg', arg_tuple)], called_fn)
    lambdas_and_blocks_removed, modified = transformations.remove_called_lambdas_and_blocks(
        generated_structure)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)

  def test_with_structure_replacing_federated_zip(self):
    fed_tuple = building_blocks.Reference(
        'tup',
        computation_types.FederatedType([tf.int32] * 3,
                                        placement_literals.CLIENTS))
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    zipped = building_block_factory.create_federated_zip(unzipped)
    placement_unwrapped, _ = tree_transformations.unwrap_placement(zipped)
    placement_gone = placement_unwrapped.argument
    lambdas_and_blocks_removed, modified = transformations.remove_called_lambdas_and_blocks(
        placement_gone)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)

  def test_with_nested_called_lambdas(self):
    identity_lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    ref_to_fn = building_blocks.Reference('fn', identity_lam.type_signature)
    data = building_blocks.Data('a', tf.int32)
    called_inner_lambda = building_blocks.Call(ref_to_fn, data)
    higher_level_lambda = building_blocks.Lambda('fn',
                                                 identity_lam.type_signature,
                                                 called_inner_lambda)
    lambdas_and_blocks_removed, modified = transformations.remove_called_lambdas_and_blocks(
        higher_level_lambda)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)

  def test_with_multiple_reference_indirection(self):
    identity_lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    tuple_wrapping_ref = building_blocks.Struct(
        [building_blocks.Reference('a', identity_lam.type_signature)])
    selection_from_ref = building_blocks.Selection(
        building_blocks.Reference('b', tuple_wrapping_ref.type_signature),
        index=0)
    data = building_blocks.Data('a', tf.int32)
    called_lambda_with_indirection = building_blocks.Call(
        building_blocks.Reference('c', selection_from_ref.type_signature), data)
    blk = building_blocks.Block([
        ('a', identity_lam),
        ('b', tuple_wrapping_ref),
        ('c', selection_from_ref),
    ], called_lambda_with_indirection)
    lambdas_and_blocks_removed, modified = transformations.remove_called_lambdas_and_blocks(
        blk)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)

  def test_with_higher_level_lambdas(self):
    data = building_blocks.Data('a', tf.int32)
    dummy = building_blocks.Reference('z', tf.int32)
    lowest_lambda = building_blocks.Lambda(
        'z', tf.int32,
        building_blocks.Struct(
            [dummy, building_blocks.Reference('x', tf.int32)]))
    middle_lambda = building_blocks.Lambda('x', tf.int32, lowest_lambda)
    lam_arg = building_blocks.Reference('x', middle_lambda.type_signature)
    rez = building_blocks.Call(lam_arg, data)
    left_lambda = building_blocks.Lambda('x', middle_lambda.type_signature, rez)
    higher_call = building_blocks.Call(left_lambda, middle_lambda)
    high_call = building_blocks.Call(higher_call, data)
    lambdas_and_blocks_removed, modified = transformations.remove_called_lambdas_and_blocks(
        high_call)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)


class TensorFlowCallingLambdaOnConcreteArgTest(test.TestCase):

  def test_raises_wrong_arguments(self):
    good_param = building_blocks.Reference('x', tf.int32)
    good_body = building_blocks.Struct(
        [building_blocks.Reference('x', tf.int32)])
    good_arg = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_body, good_body, good_arg)
    with self.assertRaises(TypeError):
      transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_param, [good_param], good_arg)
    with self.assertRaises(TypeError):
      transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_param, good_body, [good_arg])

  def test_raises_arg_does_not_match_param(self):
    good_param = building_blocks.Reference('x', tf.int32)
    good_body = building_blocks.Struct(
        [building_blocks.Reference('x', tf.int32)])
    bad_arg_type = building_blocks.Data('y', tf.float32)
    with self.assertRaises(TypeError):
      transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_param, good_body, bad_arg_type)

  def test_constructs_called_tf_block_of_correct_type_signature(self):
    param = building_blocks.Reference('x', tf.int32)
    body = building_blocks.Struct([building_blocks.Reference('x', tf.int32)])
    arg = building_blocks.Reference('y', tf.int32)
    tf_block = transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, arg)
    self.assertIsInstance(tf_block, building_blocks.Call)
    self.assertIsInstance(tf_block.function,
                          building_blocks.CompiledComputation)
    self.assertEqual(tf_block.type_signature, body.type_signature)

  def test_preserves_named_type(self):
    param = building_blocks.Reference('x', tf.int32)
    body = building_blocks.Struct([('a',
                                    building_blocks.Reference('x', tf.int32))])
    arg = building_blocks.Reference('y', tf.int32)
    tf_block = transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, arg)
    self.assertIsInstance(tf_block, building_blocks.Call)
    self.assertIsInstance(tf_block.function,
                          building_blocks.CompiledComputation)
    self.assertEqual(tf_block.type_signature, body.type_signature)

  def test_generated_tensorflow_executes_correctly_int_parameter(self):
    param = building_blocks.Reference('x', tf.int32)
    body = building_blocks.Struct([
        building_blocks.Reference('x', tf.int32),
        building_blocks.Reference('x', tf.int32)
    ])
    int_constant_type = computation_types.TensorType(tf.int32)
    int_constant = building_block_factory.create_tensorflow_constant(
        int_constant_type, 0)
    tf_block = transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, int_constant)
    result = test_utils.run_tensorflow(tf_block.function.proto)
    self.assertLen(result, 2)
    self.assertEqual(result[0], 0)
    self.assertEqual(result[1], 0)

  def test_generated_tensorflow_executes_correctly_tuple_parameter(self):
    param = building_blocks.Reference('x', [tf.int32, tf.float32])
    body = building_blocks.Struct([
        building_blocks.Selection(param, index=1),
        building_blocks.Selection(param, index=0)
    ])
    int_constant_type = computation_types.StructType([tf.int32, tf.float32])
    int_constant = building_block_factory.create_tensorflow_constant(
        int_constant_type, 1)
    tf_block = transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, int_constant)
    result = test_utils.run_tensorflow(tf_block.function.proto)
    self.assertLen(result, 2)
    self.assertEqual(result[0], 1.)
    self.assertEqual(result[1], 1)

  def test_generated_tensorflow_executes_correctly_sequence_parameter(self):
    param = building_blocks.Reference('x',
                                      computation_types.SequenceType(tf.int32))
    body = building_blocks.Struct([param])
    sequence_ref = building_blocks.Reference(
        'y', computation_types.SequenceType(tf.int32))
    tf_block = transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, sequence_ref)
    result = test_utils.run_tensorflow(tf_block.function.proto, list(range(5)))
    self.assertLen(result, 1)
    self.assertAllEqual(result[0], list(range(5)))


class BlockLocalsTFGraphTest(test.TestCase):

  def test_raises_with_naked_graph_as_block_local(self):
    tensor_type = computation_types.TensorType(tf.int32)
    graph = building_block_factory.create_compiled_identity(tensor_type)
    block_locals = [('graph', graph)]
    ref_to_graph = building_blocks.Reference('graph', graph.type_signature)
    block = building_blocks.Block(block_locals, ref_to_graph)
    with self.assertRaises(ValueError):
      transformations.create_tensorflow_representing_block(block)

  def test_raises_with_lambda_in_result(self):
    ref_to_int = building_blocks.Reference('var', tf.int32)
    first_tf_id_type = computation_types.TensorType(tf.int32)
    first_tf_id = building_block_factory.create_compiled_identity(
        first_tf_id_type)
    called_tf_id = building_blocks.Call(first_tf_id, ref_to_int)
    block_locals = [('call', called_tf_id)]
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    lam = building_blocks.Lambda(ref_to_call.name, ref_to_call.type_signature,
                                 ref_to_call)
    block = building_blocks.Block(block_locals, lam)
    with self.assertRaises(ValueError):
      transformations.create_tensorflow_representing_block(block)

  def test_returns_correct_structure_with_tuple_in_result(self):
    ref_to_int = building_blocks.Reference('var', tf.int32)
    first_tf_id_type = computation_types.TensorType(tf.int32)
    first_tf_id = building_block_factory.create_compiled_identity(
        first_tf_id_type)
    called_tf_id = building_blocks.Call(first_tf_id, ref_to_int)
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    second_tf_id_type = computation_types.TensorType(tf.int32)
    second_tf_id = building_block_factory.create_compiled_identity(
        second_tf_id_type)
    second_called = building_blocks.Call(second_tf_id, ref_to_call)
    ref_to_second_call = building_blocks.Reference('second_call',
                                                   called_tf_id.type_signature)
    block_locals = [('call', called_tf_id), ('second_call', second_called)]
    block = building_blocks.Block(
        block_locals,
        building_blocks.Struct([ref_to_second_call, ref_to_second_call]))
    tf_representing_block, _ = transformations.create_tensorflow_representing_block(
        block)
    self.assertEqual(tf_representing_block.type_signature, block.type_signature)
    self.assertIsInstance(tf_representing_block, building_blocks.Call)
    self.assertIsInstance(tf_representing_block.function,
                          building_blocks.CompiledComputation)
    self.assertIsInstance(tf_representing_block.argument,
                          building_blocks.Reference)
    self.assertEqual(tf_representing_block.argument.name, 'var')

  def test_executes_correctly_with_tuple_in_result(self):
    ref_to_int = building_blocks.Reference('var', tf.int32)
    first_tf_id_type = computation_types.TensorType(tf.int32)
    first_tf_id = building_block_factory.create_compiled_identity(
        first_tf_id_type)
    called_tf_id = building_blocks.Call(first_tf_id, ref_to_int)
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    second_tf_id_type = computation_types.TensorType(tf.int32)
    second_tf_id = building_block_factory.create_compiled_identity(
        second_tf_id_type)
    second_called = building_blocks.Call(second_tf_id, ref_to_call)
    ref_to_second_call = building_blocks.Reference('second_call',
                                                   called_tf_id.type_signature)
    block_locals = [('call', called_tf_id), ('second_call', second_called)]
    block = building_blocks.Block(
        block_locals,
        building_blocks.Struct([ref_to_second_call, ref_to_second_call]))
    tf_representing_block, _ = transformations.create_tensorflow_representing_block(
        block)
    result_ones = test_utils.run_tensorflow(
        tf_representing_block.function.proto, 1)
    self.assertAllEqual(result_ones, [1, 1])
    result_zeros = test_utils.run_tensorflow(
        tf_representing_block.function.proto, 0)
    self.assertAllEqual(result_zeros, [0, 0])

  def test_returns_correct_structure_with_no_unbound_references(self):
    concrete_int_type = computation_types.TensorType(tf.int32)
    concrete_int = building_block_factory.create_tensorflow_constant(
        concrete_int_type, 1)
    first_tf_id_type = computation_types.TensorType(tf.int32)
    first_tf_id = building_block_factory.create_compiled_identity(
        first_tf_id_type)
    called_tf_id = building_blocks.Call(first_tf_id, concrete_int)
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    second_tf_id_type = computation_types.TensorType(tf.int32)
    second_tf_id = building_block_factory.create_compiled_identity(
        second_tf_id_type)
    second_called = building_blocks.Call(second_tf_id, ref_to_call)
    ref_to_second_call = building_blocks.Reference('second_call',
                                                   called_tf_id.type_signature)
    block_locals = [('call', called_tf_id), ('second_call', second_called)]
    block = building_blocks.Block(
        block_locals,
        building_blocks.Struct([ref_to_second_call, ref_to_second_call]))
    tf_representing_block, _ = transformations.create_tensorflow_representing_block(
        block)
    self.assertEqual(tf_representing_block.type_signature, block.type_signature)
    self.assertIsInstance(tf_representing_block, building_blocks.Call)
    self.assertIsInstance(tf_representing_block.function,
                          building_blocks.CompiledComputation)
    self.assertIsNone(tf_representing_block.argument)

  def test_returned_tensorflow_executes_correctly_with_no_unbound_refs(self):
    concrete_int_type = computation_types.TensorType(tf.int32)
    concrete_int = building_block_factory.create_tensorflow_constant(
        concrete_int_type, 1)
    first_tf_id_type = computation_types.TensorType(tf.int32)
    first_tf_id = building_block_factory.create_compiled_identity(
        first_tf_id_type)
    called_tf_id = building_blocks.Call(first_tf_id, concrete_int)
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    second_tf_id_type = computation_types.TensorType(tf.int32)
    second_tf_id = building_block_factory.create_compiled_identity(
        second_tf_id_type)
    second_called = building_blocks.Call(second_tf_id, ref_to_call)
    ref_to_second_call = building_blocks.Reference('second_call',
                                                   called_tf_id.type_signature)
    block_locals = [('call', called_tf_id), ('second_call', second_called)]
    block = building_blocks.Block(
        block_locals,
        building_blocks.Struct([ref_to_second_call, ref_to_second_call]))
    tf_representing_block, _ = transformations.create_tensorflow_representing_block(
        block)
    result = test_utils.run_tensorflow(tf_representing_block.function.proto)
    self.assertAllEqual(result, [1, 1])

  def test_returns_single_called_graph_with_selection_in_result(self):
    ref_to_tuple = building_blocks.Reference('var', [tf.int32, tf.int32])
    first_tf_id = building_block_factory.create_compiled_identity(
        ref_to_tuple.type_signature)
    called_tf_id = building_blocks.Call(first_tf_id, ref_to_tuple)
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    block_locals = [('call', called_tf_id)]
    block = building_blocks.Block(
        block_locals, building_blocks.Selection(ref_to_call, index=0))
    tf_representing_block, _ = transformations.create_tensorflow_representing_block(
        block)
    self.assertEqual(tf_representing_block.type_signature, block.type_signature)
    self.assertIsInstance(tf_representing_block, building_blocks.Call)
    self.assertIsInstance(tf_representing_block.function,
                          building_blocks.CompiledComputation)
    self.assertIsInstance(tf_representing_block.argument,
                          building_blocks.Reference)
    self.assertEqual(tf_representing_block.argument.name, 'var')

  def test_returns_single_called_graph_after_resolving_multiple_variables(self):
    ref_to_int = building_blocks.Reference('var', tf.int32)
    first_tf_id_type = computation_types.TensorType(tf.int32)
    first_tf_id = building_block_factory.create_compiled_identity(
        first_tf_id_type)
    called_tf_id = building_blocks.Call(first_tf_id, ref_to_int)
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    second_tf_id_type = computation_types.TensorType(tf.int32)
    second_tf_id = building_block_factory.create_compiled_identity(
        second_tf_id_type)
    second_called = building_blocks.Call(second_tf_id, ref_to_call)
    ref_to_second_call = building_blocks.Reference('second_call',
                                                   called_tf_id.type_signature)
    block_locals = [('call', called_tf_id), ('second_call', second_called)]
    block = building_blocks.Block(block_locals, ref_to_second_call)
    tf_representing_block, _ = transformations.create_tensorflow_representing_block(
        block)
    self.assertEqual(tf_representing_block.type_signature, block.type_signature)
    self.assertIsInstance(tf_representing_block, building_blocks.Call)
    self.assertIsInstance(tf_representing_block.function,
                          building_blocks.CompiledComputation)
    self.assertIsInstance(tf_representing_block.argument,
                          building_blocks.Reference)
    self.assertEqual(tf_representing_block.argument.name, 'var')

  def test_executes_correctly_after_resolving_multiple_variables(self):
    ref_to_int = building_blocks.Reference('var', tf.int32)
    first_tf_id_type = computation_types.TensorType(tf.int32)
    first_tf_id = building_block_factory.create_compiled_identity(
        first_tf_id_type)
    called_tf_id = building_blocks.Call(first_tf_id, ref_to_int)
    ref_to_call = building_blocks.Reference('call', called_tf_id.type_signature)
    second_tf_id_type = computation_types.TensorType(tf.int32)
    second_tf_id = building_block_factory.create_compiled_identity(
        second_tf_id_type)
    second_called = building_blocks.Call(second_tf_id, ref_to_call)
    ref_to_second_call = building_blocks.Reference('second_call',
                                                   called_tf_id.type_signature)
    block_locals = [('call', called_tf_id), ('second_call', second_called)]
    block = building_blocks.Block(block_locals, ref_to_second_call)
    tf_representing_block, _ = transformations.create_tensorflow_representing_block(
        block)
    result_one = test_utils.run_tensorflow(tf_representing_block.function.proto,
                                           1)
    self.assertEqual(result_one, 1)
    result_zero = test_utils.run_tensorflow(
        tf_representing_block.function.proto, 0)
    self.assertEqual(result_zero, 0)

  def test_ops_not_duplicated_in_resulting_tensorflow(self):

    def _construct_block_and_inlined_tuple(k):
      concrete_int_type = computation_types.TensorType(tf.int32)
      concrete_int = building_block_factory.create_tensorflow_constant(
          concrete_int_type, 1)
      first_tf_id_type = computation_types.TensorType(tf.int32)
      first_tf_id = building_block_factory.create_compiled_identity(
          first_tf_id_type)
      called_tf_id = building_blocks.Call(first_tf_id, concrete_int)
      for _ in range(k):
        # Simulating large TF computation
        called_tf_id = building_blocks.Call(first_tf_id, called_tf_id)
      ref_to_call = building_blocks.Reference('call',
                                              called_tf_id.type_signature)
      block_locals = [('call', called_tf_id)]
      block = building_blocks.Block(
          block_locals, building_blocks.Struct([ref_to_call, ref_to_call]))
      inlined_tuple = building_blocks.Struct([called_tf_id, called_tf_id])
      return block, inlined_tuple

    block_with_5_ids, inlined_tuple_with_5_ids = _construct_block_and_inlined_tuple(
        5)
    block_with_10_ids, inlined_tuple_with_10_ids = _construct_block_and_inlined_tuple(
        10)
    tf_representing_block_with_5_ids, _ = transformations.create_tensorflow_representing_block(
        block_with_5_ids)
    tf_representing_block_with_10_ids, _ = transformations.create_tensorflow_representing_block(
        block_with_10_ids)
    block_ops_with_5_ids = tree_analysis.count_tensorflow_ops_under(
        tf_representing_block_with_5_ids)
    block_ops_with_10_ids = tree_analysis.count_tensorflow_ops_under(
        tf_representing_block_with_10_ids)

    parser_callable = tree_to_cc_transformations.TFParser()
    naively_generated_tf_with_5_ids, _ = transformation_utils.transform_postorder(
        inlined_tuple_with_5_ids, parser_callable)
    naively_generated_tf_with_10_ids, _ = transformation_utils.transform_postorder(
        inlined_tuple_with_10_ids, parser_callable)
    tuple_ops_with_5_ids = tree_analysis.count_tensorflow_ops_under(
        naively_generated_tf_with_5_ids)
    tuple_ops_with_10_ids = tree_analysis.count_tensorflow_ops_under(
        naively_generated_tf_with_10_ids)

    # asserting that block ops are linear in k with slope 1.
    self.assertEqual((block_ops_with_10_ids - block_ops_with_5_ids) / 5, 1)
    # asserting that tuple ops are linear in k with slope 2.
    self.assertEqual((tuple_ops_with_10_ids - tuple_ops_with_5_ids) / 5, 2)


class DeduplicateCalledGraphsTest(test.TestCase):

  def test_raises_bad_type(self):
    with self.assertRaises(TypeError):
      transformations.remove_duplicate_called_graphs(1)

  def test_raises_non_unique_names(self):
    data = building_blocks.Data('a', tf.int32)
    block = building_blocks.Block([('x', data), ('x', data)], data)
    with self.assertRaises(ValueError):
      transformations.remove_duplicate_called_graphs(block)

  def test_returns_higher_level_lambda_untransformed(self):
    lower_level_lambda = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    higher_level_lambda = building_blocks.Lambda('y', tf.int32,
                                                 lower_level_lambda)
    untransformed, modified_indicator = transformations.remove_duplicate_called_graphs(
        higher_level_lambda)
    self.assertEqual(untransformed, higher_level_lambda)
    self.assertFalse(modified_indicator)

  def test_returns_comp_with_block_untransformed(self):
    data = building_blocks.Data('a', tf.int32)
    block = building_blocks.Block([('x', data), ('y', data)], data)
    untransformed, modified_indicator = transformations.remove_duplicate_called_graphs(
        block)
    self.assertEqual(untransformed, block)
    self.assertFalse(modified_indicator)

  def test_returns_tf_computation_with_functional_type(self):
    param = building_blocks.Reference('x', [('a', tf.int32), ('b', tf.float32)])
    sel = building_blocks.Selection(source=param, index=0)
    tup = building_blocks.Struct([sel, sel, sel])
    lam = building_blocks.Lambda(param.name, param.type_signature, tup)
    transformed, modified_indicator = transformations.remove_duplicate_called_graphs(
        lam)
    self.assertTrue(modified_indicator)
    self.assertIsInstance(transformed, building_blocks.CompiledComputation)
    self.assertEqual(transformed.type_signature, lam.type_signature)

  def test_returns_called_tf_computation_with_non_functional_type(self):
    constant_tuple_type = computation_types.StructType([tf.int32, tf.float32])
    constant_tuple = building_block_factory.create_tensorflow_constant(
        constant_tuple_type, 1)
    sel = building_blocks.Selection(source=constant_tuple, index=0)
    tup = building_blocks.Struct([sel, sel, sel])
    transformed, modified_indicator = transformations.remove_duplicate_called_graphs(
        tup)
    self.assertTrue(modified_indicator)
    self.assertEqual(transformed.type_signature, tup.type_signature)
    self.assertIsInstance(transformed, building_blocks.Call)
    self.assertIsInstance(transformed.function,
                          building_blocks.CompiledComputation)
    self.assertIsNone(transformed.argument)

  def test_deduplicates_by_counting_ops(self):

    def _construct_inlined_tuple(k):
      constant_tuple_type = computation_types.TensorType(tf.int32)
      concrete_int = building_block_factory.create_tensorflow_constant(
          constant_tuple_type, 1)
      first_tf_fn = building_block_factory.create_tensorflow_binary_operator(
          concrete_int.type_signature, tf.add)
      call = building_blocks.Call(
          first_tf_fn, building_blocks.Struct([concrete_int, concrete_int]))
      for _ in range(k):
        # Simulating large TF computation
        call = building_blocks.Call(first_tf_fn,
                                    building_blocks.Struct([call, call]))
      return building_blocks.Struct([call, call])

    def _count_ops_parameterized_by_layers(k):
      inlined_tuple_with_k_layers = _construct_inlined_tuple(k)
      tf_representing_block_with_k_layers, _ = transformations.remove_duplicate_called_graphs(
          inlined_tuple_with_k_layers)
      block_ops_with_k_layers = tree_analysis.count_tensorflow_ops_under(
          tf_representing_block_with_k_layers)
      parser_callable = tree_to_cc_transformations.TFParser()
      naively_generated_tf_with_k_layers, _ = transformation_utils.transform_postorder(
          inlined_tuple_with_k_layers, parser_callable)
      naive_ops_with_k_layers = tree_analysis.count_tensorflow_ops_under(
          naively_generated_tf_with_k_layers)
      return block_ops_with_k_layers, naive_ops_with_k_layers

    block_ops_with_0_layers, tuple_ops_with_0_layers = _count_ops_parameterized_by_layers(
        0)
    block_ops_with_1_layers, tuple_ops_with_1_layers = _count_ops_parameterized_by_layers(
        1)
    block_ops_with_2_layers, tuple_ops_with_2_layers = _count_ops_parameterized_by_layers(
        2)
    block_ops_with_3_layers, tuple_ops_with_3_layers = _count_ops_parameterized_by_layers(
        3)

    # asserting that block ops are linear in k.
    self.assertEqual(block_ops_with_1_layers - block_ops_with_0_layers,
                     block_ops_with_2_layers - block_ops_with_1_layers)
    self.assertEqual(block_ops_with_3_layers - block_ops_with_2_layers,
                     block_ops_with_2_layers - block_ops_with_1_layers)

    # asserting that tuple ops are exponential in k.
    first_factor = (tuple_ops_with_2_layers - tuple_ops_with_1_layers) / (
        tuple_ops_with_1_layers - tuple_ops_with_0_layers)
    second_factor = (tuple_ops_with_3_layers - tuple_ops_with_2_layers) / (
        tuple_ops_with_2_layers - tuple_ops_with_1_layers)
    self.assertEqual(first_factor, second_factor)


class DedupeAndMergeTupleIntrinsicsTest(test.TestCase):

  def test_noops_in_case_of_distinct_maps(self):
    called_intrinsic1 = test_utils.create_dummy_called_federated_map(
        parameter_name='a', parameter_type=tf.int32)
    called_intrinsic2 = test_utils.create_dummy_called_federated_map(
        parameter_name='a', parameter_type=tf.float32)
    calls = building_blocks.Struct((called_intrinsic1, called_intrinsic2))
    comp = calls

    deduped_and_merged_comp, deduped_modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)
    directly_merged_comp, directly_modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertTrue(deduped_modified)
    self.assertTrue(directly_modified)
    self.assertEqual(deduped_and_merged_comp.compact_representation(),
                     directly_merged_comp.compact_representation())

  def test_noops_in_case_of_distinct_applies(self):
    called_intrinsic1 = test_utils.create_dummy_called_federated_apply(
        parameter_name='a', parameter_type=tf.int32)
    called_intrinsic2 = test_utils.create_dummy_called_federated_apply(
        parameter_name='a', parameter_type=tf.float32)
    calls = building_blocks.Struct((called_intrinsic1, called_intrinsic2))
    comp = calls

    deduped_and_merged_comp, deduped_modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_APPLY.uri)
    directly_merged_comp, directly_modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_APPLY.uri)

    self.assertTrue(deduped_modified)
    self.assertTrue(directly_modified)
    self.assertEqual(deduped_and_merged_comp.compact_representation(),
                     directly_merged_comp.compact_representation())

  def test_constructs_broadcast_of_tuple_with_one_element(self):
    called_intrinsic = test_utils.create_dummy_called_federated_broadcast()
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_BROADCAST.uri)

    federated_broadcast = []

    def _find_federated_broadcast(comp):
      if building_block_analysis.is_called_intrinsic(
          comp, intrinsic_defs.FEDERATED_BROADCAST.uri):
        federated_broadcast.append(comp)
      return comp, False

    transformation_utils.transform_postorder(transformed_comp,
                                             _find_federated_broadcast)

    self.assertTrue(modified)
    self.assertEqual(comp.compact_representation(),
                     '<federated_broadcast(data),federated_broadcast(data)>')

    self.assertLen(federated_broadcast, 1)
    self.assertLen(federated_broadcast[0].type_signature.member, 1)
    self.assertEqual(
        transformed_comp.formatted_representation(), '(_var1 -> <\n'
        '  _var1[0],\n'
        '  _var1[0]\n'
        '>)((x -> <\n'
        '  x[0]\n'
        '>)((let\n'
        '  value=federated_broadcast(federated_apply(<\n'
        '    (arg -> <\n'
        '      arg\n'
        '    >),\n'
        '    <\n'
        '      data\n'
        '    >[0]\n'
        '  >))\n'
        ' in <\n'
        '  federated_map_all_equal(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >)\n'
        '>)))')

  def test_dedupe_noops_in_case_of_distinct_broadcasts(self):
    called_intrinsic1 = test_utils.create_dummy_called_federated_broadcast(
        tf.int32)
    called_intrinsic2 = test_utils.create_dummy_called_federated_broadcast(
        tf.float32)
    calls = building_blocks.Struct((called_intrinsic1, called_intrinsic2))
    comp = calls

    deduped_and_merged_comp, deduped_modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_BROADCAST.uri)

    directly_merged_comp, directly_modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_BROADCAST.uri)

    self.assertTrue(deduped_modified)
    self.assertTrue(directly_modified)
    self.assertEqual(deduped_and_merged_comp.compact_representation(),
                     directly_merged_comp.compact_representation())

  def test_constructs_aggregate_of_tuple_with_one_element(self):
    called_intrinsic = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    federated_agg = []

    def _find_federated_aggregate(comp):
      if building_block_analysis.is_called_intrinsic(
          comp, intrinsic_defs.FEDERATED_AGGREGATE.uri):
        federated_agg.append(comp)
      return comp, False

    transformation_utils.transform_postorder(transformed_comp,
                                             _find_federated_aggregate)
    self.assertTrue(modified)
    self.assertLen(federated_agg, 1)
    self.assertLen(federated_agg[0].type_signature.member, 1)
    self.assertEqual(
        transformed_comp.formatted_representation(), '(_var1 -> <\n'
        '  _var1[0],\n'
        '  _var1[0]\n'
        '>)((x -> <\n'
        '  x[0]\n'
        '>)((let\n'
        '  value=federated_aggregate(<\n'
        '    federated_map(<\n'
        '      (arg -> <\n'
        '        arg\n'
        '      >),\n'
        '      <\n'
        '        data\n'
        '      >[0]\n'
        '    >),\n'
        '    <\n'
        '      data\n'
        '    >,\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> data)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](<\n'
        '        <\n'
        '          _var2[0][0],\n'
        '          _var2[1][0]\n'
        '        >\n'
        '      >[0])\n'
        '    >)),\n'
        '    (let\n'
        '      _var3=<\n'
        '        (b -> data)\n'
        '      >\n'
        '     in (_var4 -> <\n'
        '      _var3[0](<\n'
        '        <\n'
        '          _var4[0][0],\n'
        '          _var4[1][0]\n'
        '        >\n'
        '      >[0])\n'
        '    >)),\n'
        '    (let\n'
        '      _var5=<\n'
        '        (c -> data)\n'
        '      >\n'
        '     in (_var6 -> <\n'
        '      _var5[0](_var6[0])\n'
        '    >))\n'
        '  >)\n'
        ' in <\n'
        '  federated_apply(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >)\n'
        '>)))')

  def test_identical_to_merge_tuple_intrinsics_with_different_intrinsics(self):
    called_intrinsic1 = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c',
        value_type=tf.int32)
    # These compare as not equal.
    called_intrinsic2 = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='x',
        merge_parameter_name='y',
        report_parameter_name='z',
        value_type=tf.float32)
    calls = building_blocks.Struct((called_intrinsic1, called_intrinsic2))
    comp = calls

    deduped_and_merged_comp, deduped_modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)
    directly_merged_comp, directly_modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    self.assertTrue(deduped_modified)
    self.assertTrue(directly_modified)
    self.assertEqual(deduped_and_merged_comp.formatted_representation(),
                     directly_merged_comp.formatted_representation())

  def test_aggregate_with_selection_from_block_by_index_results_in_single_aggregate(
      self):
    data = building_blocks.Reference(
        'a',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    tup_of_data = building_blocks.Struct([data, data])
    block_holding_tup = building_blocks.Block([], tup_of_data)
    index_0_from_block = building_blocks.Selection(
        source=block_holding_tup, index=0)
    index_1_from_block = building_blocks.Selection(
        source=block_holding_tup, index=1)

    result = building_blocks.Data('aggregation_result', tf.int32)
    zero = building_blocks.Data('zero', tf.int32)
    accumulate = building_blocks.Lambda('accumulate_param',
                                        [tf.int32, tf.int32], result)
    merge = building_blocks.Lambda('merge_param', [tf.int32, tf.int32], result)
    report = building_blocks.Lambda('report_param', tf.int32, result)

    called_intrinsic0 = building_block_factory.create_federated_aggregate(
        index_0_from_block, zero, accumulate, merge, report)
    called_intrinsic1 = building_block_factory.create_federated_aggregate(
        index_1_from_block, zero, accumulate, merge, report)
    calls = building_blocks.Struct((called_intrinsic0, called_intrinsic1))
    comp = calls

    deduped_and_merged_comp, deduped_modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    self.assertTrue(deduped_modified)

    fed_agg = []

    def _find_called_federated_aggregate(comp):
      if (comp.is_call() and comp.function.is_intrinsic() and
          comp.function.uri == intrinsic_defs.FEDERATED_AGGREGATE.uri):
        fed_agg.append(comp.function)
      return comp, False

    transformation_utils.transform_postorder(deduped_and_merged_comp,
                                             _find_called_federated_aggregate)
    self.assertLen(fed_agg, 1)
    self.assertEqual(
        fed_agg[0].type_signature.parameter[0].compact_representation(),
        '{<int32>}@CLIENTS')

  def test_aggregate_with_selection_from_block_by_name_results_in_single_aggregate(
      self):
    data = building_blocks.Reference(
        'a',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    tup_of_data = building_blocks.Struct([('a', data), ('b', data)])
    block_holding_tup = building_blocks.Block([], tup_of_data)
    index_0_from_block = building_blocks.Selection(
        source=block_holding_tup, name='a')
    index_1_from_block = building_blocks.Selection(
        source=block_holding_tup, name='b')

    result = building_blocks.Data('aggregation_result', tf.int32)
    zero = building_blocks.Data('zero', tf.int32)
    accumulate = building_blocks.Lambda('accumulate_param',
                                        [tf.int32, tf.int32], result)
    merge = building_blocks.Lambda('merge_param', [tf.int32, tf.int32], result)
    report = building_blocks.Lambda('report_param', tf.int32, result)

    called_intrinsic0 = building_block_factory.create_federated_aggregate(
        index_0_from_block, zero, accumulate, merge, report)
    called_intrinsic1 = building_block_factory.create_federated_aggregate(
        index_1_from_block, zero, accumulate, merge, report)
    calls = building_blocks.Struct((called_intrinsic0, called_intrinsic1))
    comp = calls

    deduped_and_merged_comp, deduped_modified = transformations.dedupe_and_merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    self.assertTrue(deduped_modified)

    fed_agg = []

    def _find_called_federated_aggregate(comp):
      if (comp.is_call() and comp.function.is_intrinsic() and
          comp.function.uri == intrinsic_defs.FEDERATED_AGGREGATE.uri):
        fed_agg.append(comp.function)
      return comp, False

    transformation_utils.transform_postorder(deduped_and_merged_comp,
                                             _find_called_federated_aggregate)
    self.assertLen(fed_agg, 1)
    self.assertEqual(
        fed_agg[0].type_signature.parameter[0].compact_representation(),
        '{<int32>}@CLIENTS')


class TensorFlowGeneratorTest(test.TestCase):

  def test_passes_on_tf(self):
    tf_comp = building_block_factory.create_compiled_identity(
        computation_types.TensorType(tf.int32))

    transformed, modified = transformations.compile_local_computation_to_tensorflow(
        tf_comp)

    self.assertFalse(modified)
    self.assertEqual(tf_comp, transformed)

  def test_generates_tf_with_lambda(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.StructType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)

    transformed, modified = transformations.compile_local_computation_to_tensorflow(
        identity_lambda)

    self.assertTrue(modified)
    self.assertIsInstance(transformed, building_blocks.CompiledComputation)
    self.assertEqual(transformed.type_signature, identity_lambda.type_signature)

  def test_generates_tf_with_block(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.StructType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)
    tf_zero = building_block_factory.create_tensorflow_constant(
        computation_types.StructType([tf.int32, tf.float32]), 0)
    ref_to_z = building_blocks.Reference('z', [tf.int32, tf.float32])
    called_lambda_on_z = building_blocks.Call(identity_lambda, ref_to_z)
    blk = building_blocks.Block([('z', tf_zero)], called_lambda_on_z)

    transformed, modified = transformations.compile_local_computation_to_tensorflow(
        blk)

    self.assertTrue(modified)
    self.assertIsInstance(transformed, building_blocks.Call)
    self.assertIsInstance(transformed.function,
                          building_blocks.CompiledComputation)
    self.assertIsNone(transformed.argument)
    self.assertEqual(transformed.type_signature, blk.type_signature)

  def test_generates_tf_with_sequence_type(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.SequenceType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)

    transformed, modified = transformations.compile_local_computation_to_tensorflow(
        identity_lambda)

    self.assertTrue(modified)
    self.assertIsInstance(transformed, building_blocks.CompiledComputation)
    self.assertEqual(transformed.type_signature, identity_lambda.type_signature)

  def test_leaves_federated_comp_alone(self):
    ref_to_federated_x = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    identity_lambda = building_blocks.Lambda(ref_to_federated_x.name,
                                             ref_to_federated_x.type_signature,
                                             ref_to_federated_x)

    transformed, modified = transformations.compile_local_computation_to_tensorflow(
        identity_lambda)

    self.assertFalse(modified)
    self.assertEqual(transformed, identity_lambda)

  def test_compiles_lambda_under_federated_comp_to_tf(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.StructType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)
    federated_data = building_blocks.Data(
        'a',
        computation_types.FederatedType(
            computation_types.StructType([tf.int32, tf.float32]),
            placement_literals.SERVER))
    applied = building_block_factory.create_federated_apply(
        identity_lambda, federated_data)

    transformed, modified = transformations.compile_local_computation_to_tensorflow(
        applied)

    self.assertTrue(modified)
    self.assertIsInstance(transformed, building_blocks.Call)
    self.assertIsInstance(transformed.function, building_blocks.Intrinsic)
    self.assertIsInstance(transformed.argument[0],
                          building_blocks.CompiledComputation)
    self.assertEqual(transformed.argument[1], federated_data)
    self.assertEqual(transformed.argument[0].type_signature,
                     identity_lambda.type_signature)

  def test_leaves_local_comp_with_unbound_reference_alone(self):
    ref_to_x = building_blocks.Reference('x', [tf.int32, tf.float32])
    ref_to_z = building_blocks.Reference('z', [tf.int32, tf.float32])
    lambda_with_unbound_ref = building_blocks.Lambda(ref_to_x.name,
                                                     ref_to_x.type_signature,
                                                     ref_to_z)
    transformed, modified = transformations.compile_local_computation_to_tensorflow(
        lambda_with_unbound_ref)

    self.assertFalse(modified)
    self.assertEqual(transformed, lambda_with_unbound_ref)

  def test_deduplicates_tensorflow_by_counting_ops(self):

    def _construct_inlined_tuple(k):
      constant_tuple_type = computation_types.TensorType(tf.int32)
      concrete_int = building_block_factory.create_tensorflow_constant(
          constant_tuple_type, 1)
      first_tf_fn = building_block_factory.create_tensorflow_binary_operator(
          concrete_int.type_signature, tf.add)
      call = building_blocks.Call(
          first_tf_fn, building_blocks.Struct([concrete_int, concrete_int]))
      for _ in range(k):
        # Simulating large TF computation
        call = building_blocks.Call(first_tf_fn,
                                    building_blocks.Struct([call, call]))
      return building_blocks.Struct([call, call])

    def _count_ops_parameterized_by_layers(k):
      inlined_tuple_with_k_layers = _construct_inlined_tuple(k)
      tf_representing_block_with_k_layers, _ = transformations.compile_local_computation_to_tensorflow(
          inlined_tuple_with_k_layers)
      block_ops_with_k_layers = tree_analysis.count_tensorflow_ops_under(
          tf_representing_block_with_k_layers)
      parser_callable = tree_to_cc_transformations.TFParser()
      naively_generated_tf_with_k_layers, _ = transformation_utils.transform_postorder(
          inlined_tuple_with_k_layers, parser_callable)
      naive_ops_with_k_layers = tree_analysis.count_tensorflow_ops_under(
          naively_generated_tf_with_k_layers)
      return block_ops_with_k_layers, naive_ops_with_k_layers

    block_ops_with_0_layers, tuple_ops_with_0_layers = _count_ops_parameterized_by_layers(
        0)
    block_ops_with_1_layers, tuple_ops_with_1_layers = _count_ops_parameterized_by_layers(
        1)
    block_ops_with_2_layers, tuple_ops_with_2_layers = _count_ops_parameterized_by_layers(
        2)
    block_ops_with_3_layers, tuple_ops_with_3_layers = _count_ops_parameterized_by_layers(
        3)

    # asserting that block ops are linear in k.
    self.assertEqual(block_ops_with_1_layers - block_ops_with_0_layers,
                     block_ops_with_2_layers - block_ops_with_1_layers)
    self.assertEqual(block_ops_with_3_layers - block_ops_with_2_layers,
                     block_ops_with_2_layers - block_ops_with_1_layers)

    # asserting that tuple ops are exponential in k.
    first_factor = (tuple_ops_with_2_layers - tuple_ops_with_1_layers) / (
        tuple_ops_with_1_layers - tuple_ops_with_0_layers)
    second_factor = (tuple_ops_with_3_layers - tuple_ops_with_2_layers) / (
        tuple_ops_with_2_layers - tuple_ops_with_1_layers)
    self.assertEqual(first_factor, second_factor)


class TestTransformToCallDominantForm(test.TestCase):

  def test_handles_called_lambda_returning_function(self):
    lower_level_lambda = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    higher_level_lambda = building_blocks.Lambda('y', tf.int32,
                                                 lower_level_lambda)

    call_dominant_rep, modified = transformations.transform_to_call_dominant(
        higher_level_lambda)

    self.assertTrue(modified)
    self.assertRegexMatch(call_dominant_rep.compact_representation(),
                          [r'\(_([a-z]{3})1 -> \(_(\1)2 -> _(\1)2\)\)'])

  def test_handles_block_returning_function(self):
    lower_level_lambda = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    blk = building_blocks.Block([], lower_level_lambda)

    call_dominant_rep, modified = transformations.transform_to_call_dominant(
        blk)
    self.assertTrue(modified)
    self.assertRegexMatch(call_dominant_rep.compact_representation(),
                          [r'\(_([a-z]{3})1 -> _(\1)1\)'])

  def test_merges_nested_blocks(self):
    data = building_blocks.Data('a', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    blk1 = building_blocks.Block([('x', data)], ref_to_x)
    blk2 = building_blocks.Block([('x', blk1)], ref_to_x)

    call_dominant_rep, modified = transformations.transform_to_call_dominant(
        blk2)

    self.assertTrue(modified)
    self.assertRegexMatch(call_dominant_rep.compact_representation(),
                          [r'\(let _([a-z]{3})1=a in _(\1)1\)'])

  def test_extracts_called_intrinsics_to_block(self):
    called_aggregate = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    tuple_holding_aggregate = building_blocks.Struct([called_aggregate])
    sel_from_tuple = building_blocks.Selection(
        source=tuple_holding_aggregate, index=0)
    lambda_to_sel = building_blocks.Lambda('x', tf.int32, sel_from_tuple)

    call_dominant_rep, modified = transformations.transform_to_call_dominant(
        lambda_to_sel)

    self.assertTrue(modified)
    self.assertIsInstance(call_dominant_rep, building_blocks.Block)
    self.assertLen(call_dominant_rep.locals, 1)
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.locals[0][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))

  def test_deduplicates_called_intrinsics(self):
    called_aggregate1 = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_aggregate2 = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    tuple_holding_aggregates = building_blocks.Struct(
        [called_aggregate1, called_aggregate2])
    lambda_to_tup = building_blocks.Lambda('x', tf.int32,
                                           tuple_holding_aggregates)

    call_dominant_rep, modified = transformations.transform_to_call_dominant(
        lambda_to_tup)

    self.assertTrue(modified)
    self.assertIsInstance(call_dominant_rep, building_blocks.Block)
    self.assertLen(call_dominant_rep.locals, 1)
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.locals[0][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))

  def test_hoists_aggregations_packed_in_tuple(self):
    called_aggregate1 = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c',
        value_type=tf.int32)
    called_aggregate2 = test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c',
        value_type=tf.float32)
    tuple_holding_aggregates = building_blocks.Struct(
        [called_aggregate1, called_aggregate2])
    lambda_to_tuple = building_blocks.Lambda('x', tf.int32,
                                             tuple_holding_aggregates)

    call_dominant_rep, modified = transformations.transform_to_call_dominant(
        lambda_to_tuple)

    self.assertTrue(modified)
    self.assertIsInstance(call_dominant_rep, building_blocks.Block)
    self.assertLen(call_dominant_rep.locals, 2)
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.locals[0][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.locals[1][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))

  def test_handles_lambda_with_lambda_parameter(self):
    int_identity_lambda = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    ref_to_fn_and_int = building_blocks.Reference(
        'y',
        computation_types.StructType([
            int_identity_lambda.type_signature,
            computation_types.TensorType(tf.int32)
        ]))
    fn = building_blocks.Selection(ref_to_fn_and_int, index=0)
    arg = building_blocks.Selection(ref_to_fn_and_int, index=1)
    called_fn = building_blocks.Call(fn, arg)
    lambda_accepting_fn = building_blocks.Lambda(
        ref_to_fn_and_int.name, ref_to_fn_and_int.type_signature, called_fn)
    ref_to_int = building_blocks.Reference('z', tf.int32)
    arg_tuple = building_blocks.Struct([int_identity_lambda, ref_to_int])
    called_lambda_with_fn = building_blocks.Call(lambda_accepting_fn, arg_tuple)
    lambda_accepting_int = building_blocks.Lambda(ref_to_int.name,
                                                  ref_to_int.type_signature,
                                                  called_lambda_with_fn)

    call_dominant_rep, modified = transformations.transform_to_call_dominant(
        lambda_accepting_int)

    self.assertTrue(modified)
    self.assertRegexMatch(call_dominant_rep.compact_representation(), [
        r'\(_([a-z]{3})1 -> \(let _(\1)3=\(_(\1)2 -> _(\1)2\)\(_(\1)1\) in _(\1)3\)\)'
    ])


if __name__ == '__main__':
  test.main()
