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

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations as compiler_transformations


class RemoveLambdasAndBlocksTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
