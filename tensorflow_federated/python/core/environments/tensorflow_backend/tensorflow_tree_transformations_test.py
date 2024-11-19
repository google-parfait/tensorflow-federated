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
import federated_language
import numpy as np

from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_tree_transformations


def _count_intrinsics(comp, uri):
  def _predicate(comp):
    return (
        isinstance(comp, federated_language.framework.Intrinsic)
        and uri is not None
        and comp.uri == uri
    )

  return federated_language.framework.computation_count(comp, _predicate)


class ReplaceIntrinsicsWithBodiesTest(parameterized.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tensorflow_tree_transformations.replace_intrinsics_with_bodies(None)

  def test_federated_mean_reduces_to_aggregate(self):
    uri = federated_language.framework.FEDERATED_MEAN.uri

    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            federated_language.FederatedType(
                np.float32, federated_language.CLIENTS
            ),
            federated_language.FederatedType(
                np.float32, federated_language.SERVER
            ),
        ),
    )

    count_means_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_means_after_reduction = _count_intrinsics(reduced, uri)
    count_aggregations = _count_intrinsics(
        reduced, federated_language.framework.FEDERATED_AGGREGATE.uri
    )
    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_means_before_reduction, 0)
    self.assertEqual(count_means_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_federated_weighted_mean_reduces_to_aggregate(self):
    uri = federated_language.framework.FEDERATED_WEIGHTED_MEAN.uri

    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            (
                federated_language.FederatedType(
                    np.float32, federated_language.CLIENTS
                ),
            )
            * 2,
            federated_language.FederatedType(
                np.float32, federated_language.SERVER
            ),
        ),
    )

    count_means_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_aggregations = _count_intrinsics(
        reduced, federated_language.framework.FEDERATED_AGGREGATE.uri
    )
    count_means_after_reduction = _count_intrinsics(reduced, uri)
    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_means_before_reduction, 0)
    self.assertEqual(count_means_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_federated_min_reduces_to_aggregate(self):
    uri = federated_language.framework.FEDERATED_MIN.uri

    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            federated_language.FederatedType(
                np.float32, federated_language.CLIENTS
            ),
            federated_language.FederatedType(
                np.float32, federated_language.SERVER
            ),
        ),
    )

    count_min_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_min_after_reduction = _count_intrinsics(reduced, uri)
    count_aggregations = _count_intrinsics(
        reduced, federated_language.framework.FEDERATED_AGGREGATE.uri
    )
    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_min_before_reduction, 0)
    self.assertEqual(count_min_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_federated_max_reduces_to_aggregate(self):
    uri = federated_language.framework.FEDERATED_MAX.uri

    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            federated_language.FederatedType(
                np.float32, federated_language.CLIENTS
            ),
            federated_language.FederatedType(
                np.float32, federated_language.SERVER
            ),
        ),
    )

    count_max_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_max_after_reduction = _count_intrinsics(reduced, uri)
    count_aggregations = _count_intrinsics(
        reduced, federated_language.framework.FEDERATED_AGGREGATE.uri
    )
    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_max_before_reduction, 0)
    self.assertEqual(count_max_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_federated_sum_reduces_to_aggregate(self):
    uri = federated_language.framework.FEDERATED_SUM.uri

    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            federated_language.FederatedType(
                np.float32, federated_language.CLIENTS
            ),
            federated_language.FederatedType(
                np.float32, federated_language.SERVER
            ),
        ),
    )

    count_sum_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_sum_after_reduction = _count_intrinsics(reduced, uri)
    count_aggregations = _count_intrinsics(
        reduced, federated_language.framework.FEDERATED_AGGREGATE.uri
    )
    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_sum_before_reduction, 0)
    self.assertEqual(count_sum_after_reduction, 0)
    self.assertGreater(count_aggregations, 0)

  def test_generic_divide_reduces(self):
    uri = federated_language.framework.GENERIC_DIVIDE.uri
    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType([np.float32, np.float32], np.float32),
    )

    count_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_after_reduction = _count_intrinsics(reduced, uri)

    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    federated_language.framework.check_contains_only_reducible_intrinsics(
        reduced
    )

  def test_generic_multiply_reduces(self):
    uri = federated_language.framework.GENERIC_MULTIPLY.uri
    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType([np.float32, np.float32], np.float32),
    )

    count_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_after_reduction = _count_intrinsics(reduced, uri)

    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    federated_language.framework.check_contains_only_reducible_intrinsics(
        reduced
    )

  def test_generic_plus_reduces(self):
    uri = federated_language.framework.GENERIC_PLUS.uri
    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType([np.float32, np.float32], np.float32),
    )

    count_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    count_after_reduction = _count_intrinsics(reduced, uri)

    self.assertTrue(modified)
    federated_language.framework.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    federated_language.framework.check_contains_only_reducible_intrinsics(
        reduced
    )


if __name__ == '__main__':
  absltest.main()
