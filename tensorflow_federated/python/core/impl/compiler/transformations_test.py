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
"""Tests for exported, composite transformations."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations as compiler_transformations


class RemoveLambdasAndBlocksTest(common_test.TestCase):

  def assertNoLambdasOrBlocks(self, comp):

    def _transform(comp):
      if (isinstance(comp, building_blocks.Call) and
          isinstance(comp.function, building_blocks.Lambda)) or isinstance(
              comp, building_blocks.Block):
        raise AssertionError('Encountered disallowed computation: {}'.format(
            comp.compact_representation()))
      return comp, True

    transformation_utils.transform_postorder(comp, _transform)

  def test_with_simple_called_lambda(self):
    identity_lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    called_lambda = building_blocks.Call(identity_lam,
                                         building_blocks.Data('a', tf.int32))
    lambdas_and_blocks_removed, modified = compiler_transformations.remove_lambdas_and_blocks(
        called_lambda)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)
    self.assertEqual(lambdas_and_blocks_removed.compact_representation(), 'a')

  def test_with_simple_block(self):
    data = building_blocks.Data('a', tf.int32)
    simple_block = building_blocks.Block([('x', data)],
                                         building_blocks.Reference(
                                             'x', tf.int32))
    lambdas_and_blocks_removed, modified = compiler_transformations.remove_lambdas_and_blocks(
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
    arg_tuple = building_blocks.Tuple([concrete_fn, concrete_arg])
    generated_structure = building_blocks.Block([('arg', arg_tuple)], called_fn)
    lambdas_and_blocks_removed, modified = compiler_transformations.remove_lambdas_and_blocks(
        generated_structure)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)

  def test_with_structure_replacing_federated_zip(self):
    fed_tuple = building_blocks.Reference(
        'tup',
        computation_types.FederatedType([tf.int32] * 3, placements.CLIENTS))
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    zipped = building_block_factory.create_federated_zip(unzipped)
    placement_unwrapped, _ = transformations.unwrap_placement(zipped)
    placement_gone = placement_unwrapped.argument
    lambdas_and_blocks_removed, modified = compiler_transformations.remove_lambdas_and_blocks(
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
    lambdas_and_blocks_removed, modified = compiler_transformations.remove_lambdas_and_blocks(
        higher_level_lambda)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)

  def test_with_multiple_reference_indirection(self):
    identity_lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    tuple_wrapping_ref = building_blocks.Tuple(
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
    lambdas_and_blocks_removed, modified = compiler_transformations.remove_lambdas_and_blocks(
        blk)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)

  def test_with_higher_level_lambdas(self):
    self.skipTest('b/146904968')
    data = building_blocks.Data('a', tf.int32)
    dummy = building_blocks.Reference('z', tf.int32)
    lowest_lambda = building_blocks.Lambda(
        'z', tf.int32,
        building_blocks.Tuple([dummy,
                               building_blocks.Reference('x', tf.int32)]))
    middle_lambda = building_blocks.Lambda('x', tf.int32, lowest_lambda)
    lam_arg = building_blocks.Reference('x', middle_lambda.type_signature)
    rez = building_blocks.Call(lam_arg, data)
    left_lambda = building_blocks.Lambda('x', middle_lambda.type_signature, rez)
    higher_call = building_blocks.Call(left_lambda, middle_lambda)
    high_call = building_blocks.Call(higher_call, data)
    lambdas_and_blocks_removed, modified = compiler_transformations.remove_lambdas_and_blocks(
        high_call)
    self.assertTrue(modified)
    self.assertNoLambdasOrBlocks(lambdas_and_blocks_removed)


class TensorFlowCallingLambdaOnConcreteArgTest(common_test.TestCase):

  def test_raises_wrong_arguments(self):
    good_param = building_blocks.Reference('x', tf.int32)
    good_body = building_blocks.Tuple(
        [building_blocks.Reference('x', tf.int32)])
    good_arg = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_body, good_body, good_arg)
    with self.assertRaises(TypeError):
      compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_param, [good_param], good_arg)
    with self.assertRaises(TypeError):
      compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_param, good_body, [good_arg])

  def test_raises_arg_does_not_match_param(self):
    good_param = building_blocks.Reference('x', tf.int32)
    good_body = building_blocks.Tuple(
        [building_blocks.Reference('x', tf.int32)])
    bad_arg_type = building_blocks.Data('y', tf.float32)
    with self.assertRaises(TypeError):
      compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
          good_param, good_body, bad_arg_type)

  def test_constructs_called_tf_block_of_correct_type_signature(self):
    param = building_blocks.Reference('x', tf.int32)
    body = building_blocks.Tuple([building_blocks.Reference('x', tf.int32)])
    arg = building_blocks.Reference('y', tf.int32)
    tf_block = compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, arg)
    self.assertIsInstance(tf_block, building_blocks.Call)
    self.assertIsInstance(tf_block.function,
                          building_blocks.CompiledComputation)
    self.assertEqual(tf_block.type_signature, body.type_signature)

  def test_preserves_named_type(self):
    param = building_blocks.Reference('x', tf.int32)
    body = building_blocks.Tuple([('a',
                                   building_blocks.Reference('x', tf.int32))])
    arg = building_blocks.Reference('y', tf.int32)
    tf_block = compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, arg)
    self.assertIsInstance(tf_block, building_blocks.Call)
    self.assertIsInstance(tf_block.function,
                          building_blocks.CompiledComputation)
    self.assertEqual(tf_block.type_signature, body.type_signature)

  def test_generated_tensorflow_executes_correctly_int_parameter(self):
    param = building_blocks.Reference('x', tf.int32)
    body = building_blocks.Tuple([
        building_blocks.Reference('x', tf.int32),
        building_blocks.Reference('x', tf.int32)
    ])
    int_constant = building_block_factory.create_tensorflow_constant(
        tf.int32, 0)
    tf_block = compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, int_constant)
    result = test_utils.run_tensorflow(tf_block.function.proto)
    self.assertLen(result, 2)
    self.assertEqual(result[0], 0)
    self.assertEqual(result[1], 0)

  def test_generated_tensorflow_executes_correctly_tuple_parameter(self):
    param = building_blocks.Reference('x', [tf.int32, tf.float32])
    body = building_blocks.Tuple([
        building_blocks.Selection(param, index=1),
        building_blocks.Selection(param, index=0)
    ])
    int_constant = building_block_factory.create_tensorflow_constant(
        [tf.int32, tf.float32], 1)
    tf_block = compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, int_constant)
    result = test_utils.run_tensorflow(tf_block.function.proto)
    self.assertLen(result, 2)
    self.assertEqual(result[0], 1.)
    self.assertEqual(result[1], 1)

  def test_generated_tensorflow_executes_correctly_sequence_parameter(self):
    param = building_blocks.Reference('x',
                                      computation_types.SequenceType(tf.int32))
    body = building_blocks.Tuple([param])
    sequence_ref = building_blocks.Reference(
        'y', computation_types.SequenceType(tf.int32))
    tf_block = compiler_transformations.construct_tensorflow_calling_lambda_on_concrete_arg(
        param, body, sequence_ref)
    result = test_utils.run_tensorflow(tf_block.function.proto, list(range(5)))
    self.assertLen(result, 1)
    self.assertAllEqual(result[0], list(range(5)))


if __name__ == '__main__':
  common_test.main()
