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
"""Tests for transformations.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
import six
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements

from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import intrinsic_bodies
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import transformations


def _to_building_block(comp):
  """Deserializes `comp` into a computation building block.

  Args:
    comp: An instance of `ComputationImpl` to deserialize.

  Returns:
    A corresponding instance of `ComputationBuildingBlock`.
  """
  assert isinstance(comp, computation_impl.ComputationImpl)
  proto = computation_impl.ComputationImpl.get_proto(comp)
  return computation_building_blocks.ComputationBuildingBlock.from_proto(proto)


class TransformationsTest(absltest.TestCase):

  def test_transform_postorder_with_lambda_call_selection_and_reference(self):

    @computations.federated_computation(
        [computation_types.FunctionType(tf.int32, tf.int32), tf.int32])
    def foo(f, x):
      return f(x)

    comp = _to_building_block(foo)
    self.assertEqual(str(comp), '(arg -> arg[0](arg[1]))')

    def _transformation_func_generator():
      n = 0
      while True:
        n = n + 1

        def _func(x):
          return computation_building_blocks.Call(
              computation_building_blocks.Intrinsic(
                  'F{}'.format(n),
                  computation_types.FunctionType(x.type_signature,
                                                 x.type_signature)), x)

        yield _func

    transformation_func_sequence = _transformation_func_generator()
    # pylint: disable=unnecessary-lambda
    tx_func = lambda x: six.next(transformation_func_sequence)(x)
    # pylint: enable=unnecessary-lambda
    transfomed_comp = transformations.transform_postorder(comp, tx_func)
    self.assertEqual(
        str(transfomed_comp), 'F6((arg -> F5(F2(F1(arg)[0])(F4(F3(arg)[1])))))')

  # TODO(b/113123410): Add more tests for corner cases of `transform_preorder`.

  def test_name_compiled_computations(self):
    plus = computations.tf_computation(lambda x, y: x + y, [tf.int32, tf.int32])

    @computations.federated_computation(tf.int32)
    def add_one(x):
      return plus(x, 1)

    comp = _to_building_block(add_one)
    transformed_comp = transformations.name_compiled_computations(comp)
    self.assertEqual(str(transformed_comp), '(arg -> comp#1(<arg,comp#2()>))')

  def test_replace_intrinsic(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER, True))
    def foo(x):
      return intrinsics.federated_sum(intrinsics.federated_broadcast(x))

    comp = _to_building_block(foo)
    self.assertEqual(
        str(comp), '(arg -> federated_sum(federated_broadcast(arg)))')

    transformed_comp = transformations.replace_intrinsic(
        comp, intrinsic_defs.FEDERATED_SUM.uri, intrinsic_bodies.federated_sum)

    # TODO(b/120793862): Add a transform to eliminate unnecessary lambdas, then
    # simplify this test.

    self.assertEqual(
        str(transformed_comp), '(arg -> (arg -> '
        '(arg -> federated_reduce(<arg[0],generic_zero,generic_plus>))(<arg>)'
        ')(federated_broadcast(arg)))')


if __name__ == '__main__':
  absltest.main()
