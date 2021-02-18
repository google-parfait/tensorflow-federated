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

from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff

# TODO(b/146086870) Move more tests into this file


class AstGenerationTest(absltest.TestCase):

  def formatted_computation(self, comp):
    building_block = comp.to_building_block()
    return building_block.formatted_representation()

  def assert_ast_equal(self, comp, expected):
    self.assertEqual(comp.to_building_block().formatted_representation(),
                     expected)

  def test_flattens_to_tf_computation(self):

    @tff.tf_computation
    def five():
      return 5

    @tff.federated_computation
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

    @tff.tf_computation
    def rand():
      return tf.random.normal([])

    @tff.federated_computation
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
  absltest.main()
