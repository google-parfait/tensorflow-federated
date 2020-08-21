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
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl import value_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.types import placement_literals

_context_stack = context_stack_impl.context_stack


class ValueUtilsTest(parameterized.TestCase):

  def run(self, result=None):
    fc_context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack)
    with context_stack_impl.context_stack.install(fc_context):
      super(ValueUtilsTest, self).run(result)

  def test_get_curried(self):
    add_numbers = value_impl.ValueImpl(
        building_blocks.ComputationBuildingBlock.from_proto(
            computation_impl.ComputationImpl.get_proto(
                computations.tf_computation(
                    lambda a, b: tf.add(a, b),  # pylint: disable=unnecessary-lambda
                    [tf.int32, tf.int32]))),
        _context_stack)

    curried = value_utils.get_curried(add_numbers)
    self.assertEqual(str(curried.type_signature), '(int32 -> (int32 -> int32))')

    comp, _ = tree_transformations.uniquify_compiled_computation_names(
        value_impl.ValueImpl.get_comp(curried))
    self.assertEqual(comp.compact_representation(),
                     '(arg0 -> (arg1 -> comp#1(<arg0,arg1>)))')

  def test_ensure_federated_value(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def _(x):
      x = value_impl.to_value(x, None, _context_stack)
      value_utils.ensure_federated_value(x, placement_literals.CLIENTS)
      return x

  def test_ensure_federated_value_wrong_placement(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def _(x):
      x = value_impl.to_value(x, None, _context_stack)
      with self.assertRaises(TypeError):
        value_utils.ensure_federated_value(x, placement_literals.SERVER)
      return x

  def test_ensure_federated_value_implicitly_zippable(self):

    @computations.federated_computation(
        computation_types.StructType(
            (computation_types.FederatedType(tf.int32,
                                             placement_literals.CLIENTS),
             computation_types.FederatedType(tf.int32,
                                             placement_literals.CLIENTS))))
    def _(x):
      x = value_impl.to_value(x, None, _context_stack)
      value_utils.ensure_federated_value(x)
      return x

  def test_ensure_federated_value_fails_on_unzippable(self):

    @computations.federated_computation(
        computation_types.StructType(
            (computation_types.FederatedType(tf.int32,
                                             placement_literals.CLIENTS),
             computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))))
    def _(x):
      x = value_impl.to_value(x, None, _context_stack)
      with self.assertRaises(TypeError):
        value_utils.ensure_federated_value(x)
      return x


if __name__ == '__main__':
  absltest.main()
