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
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.compiler import building_blocks

# TODO(b/146086870) Move more tests into this file


class AstGenerationTest(test.TestCase):

  def formatted_computation(self, comp):
    self.assertIsInstance(comp, computation_impl.ComputationImpl)
    building_block = comp.to_building_block()
    self.assertIsInstance(building_block, building_blocks.CompiledComputation)
    return building_block.formatted_representation()

  def assert_ast_equal(self, comp, expected):
    self.assertIsInstance(comp, computation_impl.ComputationImpl)
    self.assertEqual(comp.to_building_block().formatted_representation(),
                     expected)

  def test_flattens_to_tf_computation(self):

    @computations.tf_computation
    def five():
      return 5

    @computations.federated_computation
    def federated_five():
      return five()

    self.assert_ast_equal(
        federated_five,
        # pyformat: disable
        '( -> (let\n'
        '  fc_federated_five_symbol_0={}()\n'
        ' in fc_federated_five_symbol_0))'.format(
            self.formatted_computation(five))
        # pyformat: enable
    )

  def test_only_one_random_only_generates_a_single_call_to_random(self):

    @computations.tf_computation
    def rand():
      return tf.random.normal([])

    @computations.federated_computation
    def same_rand_tuple():
      single_random_number = rand()
      return (single_random_number, single_random_number)

    self.assert_ast_equal(
        same_rand_tuple,
        # pyformat: disable
        '( -> (let\n'
        '  fc_same_rand_tuple_symbol_0={}()\n'
        ' in <\n'
        '  fc_same_rand_tuple_symbol_0,\n'
        '  fc_same_rand_tuple_symbol_0\n'
        '>))'.format(self.formatted_computation(rand))
        # pyformat: enable
    )


if __name__ == '__main__':
  test.main()
