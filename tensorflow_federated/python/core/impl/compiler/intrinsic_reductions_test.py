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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import intrinsic_reductions
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import placements


def _count_intrinsics(comp, uri):

  def _predicate(comp):
    return (isinstance(comp, building_blocks.Intrinsic) and uri is not None and
            comp.uri == uri)

  return tree_analysis.count(comp, _predicate)


class ReplaceIntrinsicsWithBodiesTest(test_case.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      intrinsic_reductions.replace_intrinsics_with_bodies(None)

  def test_federated_mean_reduces_to_aggregate(self):
    uri = intrinsic_defs.FEDERATED_MEAN.uri

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_mean(x)

    foo_building_block = building_blocks.ComputationBuildingBlock.from_proto(
        foo._computation_proto)
    count_means_before_reduction = _count_intrinsics(foo_building_block, uri)
    reduced, modified = intrinsic_reductions.replace_intrinsics_with_bodies(
        foo_building_block)
    count_means_after_reduction = _count_intrinsics(reduced, uri)
    count_aggregations = _count_intrinsics(
        reduced, intrinsic_defs.FEDERATED_AGGREGATE.uri)
    count_means_after_reduction = _count_intrinsics(reduced, uri)
    self.assertTrue(modified)
    self.assert_types_identical(foo_building_block.type_signature,
                                reduced.type_signature)
    self.assertGreater(count_means_before_reduction, 0)
    self.assertEqual(count_means_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_federated_weighted_mean_reduces_to_aggregate(self):
    uri = intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_mean(x, x)

    foo_building_block = building_blocks.ComputationBuildingBlock.from_proto(
        foo._computation_proto)
    count_means_before_reduction = _count_intrinsics(foo_building_block, uri)
    reduced, modified = intrinsic_reductions.replace_intrinsics_with_bodies(
        foo_building_block)
    count_aggregations = _count_intrinsics(
        reduced, intrinsic_defs.FEDERATED_AGGREGATE.uri)
    count_means_after_reduction = _count_intrinsics(reduced, uri)
    self.assertTrue(modified)
    self.assert_types_identical(foo_building_block.type_signature,
                                reduced.type_signature)
    self.assertGreater(count_means_before_reduction, 0)
    self.assertEqual(count_means_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_federated_sum_reduces_to_aggregate(self):
    uri = intrinsic_defs.FEDERATED_SUM.uri

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_sum(x)

    foo_building_block = building_blocks.ComputationBuildingBlock.from_proto(
        foo._computation_proto)

    count_sum_before_reduction = _count_intrinsics(foo_building_block, uri)
    reduced, modified = intrinsic_reductions.replace_intrinsics_with_bodies(
        foo_building_block)
    count_sum_after_reduction = _count_intrinsics(reduced, uri)
    count_aggregations = _count_intrinsics(
        reduced, intrinsic_defs.FEDERATED_AGGREGATE.uri)
    self.assertTrue(modified)
    self.assert_types_identical(foo_building_block.type_signature,
                                reduced.type_signature)
    self.assertGreater(count_sum_before_reduction, 0)
    self.assertEqual(count_sum_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_generic_divide_reduces(self):
    uri = intrinsic_defs.GENERIC_DIVIDE.uri
    comp = building_blocks.Intrinsic(
        uri, computation_types.FunctionType([tf.float32, tf.float32],
                                            tf.float32))

    count_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = intrinsic_reductions.replace_intrinsics_with_bodies(
        comp)
    count_after_reduction = _count_intrinsics(reduced, uri)

    self.assertTrue(modified)
    self.assert_types_identical(comp.type_signature, reduced.type_signature)
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    tree_analysis.check_contains_only_reducible_intrinsics(reduced)

  def test_generic_multiply_reduces(self):
    uri = intrinsic_defs.GENERIC_MULTIPLY.uri
    comp = building_blocks.Intrinsic(
        uri, computation_types.FunctionType([tf.float32, tf.float32],
                                            tf.float32))

    count_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = intrinsic_reductions.replace_intrinsics_with_bodies(
        comp)
    count_after_reduction = _count_intrinsics(reduced, uri)

    self.assertTrue(modified)
    self.assert_types_identical(comp.type_signature, reduced.type_signature)
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    tree_analysis.check_contains_only_reducible_intrinsics(reduced)

  def test_generic_plus_reduces(self):
    uri = intrinsic_defs.GENERIC_PLUS.uri
    comp = building_blocks.Intrinsic(
        uri, computation_types.FunctionType([tf.float32, tf.float32],
                                            tf.float32))

    count_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = intrinsic_reductions.replace_intrinsics_with_bodies(
        comp)
    count_after_reduction = _count_intrinsics(reduced, uri)

    self.assertTrue(modified)
    self.assert_types_identical(comp.type_signature, reduced.type_signature)
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    tree_analysis.check_contains_only_reducible_intrinsics(reduced)


if __name__ == '__main__':
  absltest.main()
