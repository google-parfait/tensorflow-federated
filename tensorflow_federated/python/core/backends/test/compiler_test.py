# Copyright 2018 Google LLC
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
import numpy as np

from tensorflow_federated.python.core.backends.mapreduce import intrinsics
from tensorflow_federated.python.core.backends.test import compiler
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_tree_transformations
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils


def _count_intrinsics(comp, uri):
  def _predicate(comp):
    return (
        isinstance(comp, building_blocks.Intrinsic)
        and uri is not None
        and comp.uri == uri
    )

  return tree_analysis.count(comp, _predicate)


class ReplaceIntrinsicsWithBodiesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32', np.int32, np.int32),
      ('int32_struct', [np.int32, np.int32], np.int32),
      ('int64', np.int64, np.int32),
      ('mixed_struct', [np.int64, [np.int32]], np.int32),
      ('per_leaf_bitwidth', [np.int64, [np.int32]], [np.int32, [np.int32]]),
  )
  def test_federated_secure_sum(self, value_dtype, bitwidth_type):
    uri = intrinsic_defs.FEDERATED_SECURE_SUM.uri
    comp = building_blocks.Intrinsic(
        uri,
        computation_types.FunctionType(
            [
                computation_types.FederatedType(
                    value_dtype, placements.CLIENTS
                ),
                computation_types.to_type(bitwidth_type),
            ],
            computation_types.FederatedType(value_dtype, placements.SERVER),
        ),
    )
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    # First without secure intrinsics shouldn't modify anything.
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    self.assertFalse(modified)
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    type_test_utils.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    type_test_utils.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(
        _count_intrinsics(reduced, intrinsic_defs.FEDERATED_AGGREGATE.uri), 0
    )

  @parameterized.named_parameters(
      ('int32', np.int32, np.int32),
      ('int32_struct', [np.int32, np.int32], np.int32),
      ('int64', np.int64, np.int32),
      ('mixed_struct', [np.int64, [np.int32]], np.int32),
      ('per_leaf_bitwidth', [np.int64, [np.int32]], [np.int32, [np.int32]]),
  )
  def test_federated_secure_sum_bitwidth(self, value_dtype, bitwidth_type):
    uri = intrinsic_defs.FEDERATED_SECURE_SUM_BITWIDTH.uri
    comp = building_blocks.Intrinsic(
        uri,
        computation_types.FunctionType(
            parameter=[
                computation_types.FederatedType(
                    value_dtype, placements.CLIENTS
                ),
                computation_types.to_type(bitwidth_type),
            ],
            result=computation_types.FederatedType(
                value_dtype, placements.SERVER
            ),
        ),
    )
    # First without secure intrinsics shouldn't modify anything.
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    self.assertFalse(modified)
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    type_test_utils.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    type_test_utils.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(
        _count_intrinsics(reduced, intrinsic_defs.FEDERATED_AGGREGATE.uri), 0
    )

  @parameterized.named_parameters(
      ('int32', np.int32, np.int32),
      ('int32_struct', [np.int32, np.int32], np.int32),
      ('int64', np.int32, np.int32),
      ('mixed_struct', [np.int32, [np.int32]], np.int32),
      ('per_leaf_modulus', [np.int32, [np.int32]], [np.int32, [np.int32]]),
  )
  def test_federated_secure_modular_sum(self, value_dtype, modulus_type):
    uri = intrinsics.FEDERATED_SECURE_MODULAR_SUM.uri
    comp = building_blocks.Intrinsic(
        uri,
        computation_types.FunctionType(
            parameter=[
                computation_types.FederatedType(
                    value_dtype, placements.CLIENTS
                ),
                computation_types.to_type(modulus_type),
            ],
            result=computation_types.FederatedType(
                value_dtype, placements.SERVER
            ),
        ),
    )
    # First without secure intrinsics shouldn't modify anything.
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    self.assertFalse(modified)
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    type_test_utils.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    # Inserting tensorflow, as we do here, does not preserve python containers
    # currently.
    type_test_utils.assert_types_equivalent(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(
        _count_intrinsics(reduced, intrinsic_defs.FEDERATED_SUM.uri), 0
    )

  def test_federated_secure_select(self):
    uri = intrinsic_defs.FEDERATED_SECURE_SELECT.uri
    comp = building_blocks.Intrinsic(
        uri,
        computation_types.FunctionType(
            [
                computation_types.FederatedType(
                    np.int32, placements.CLIENTS
                ),  # client_keys
                computation_types.FederatedType(
                    np.int32, placements.SERVER
                ),  # max_key
                computation_types.FederatedType(
                    np.float32, placements.SERVER
                ),  # server_state
                computation_types.FunctionType(
                    [np.float32, np.int32], np.float32
                ),  # select_fn
            ],
            computation_types.FederatedType(
                computation_types.SequenceType(np.float32), placements.CLIENTS
            ),
        ),
    )
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    # First without secure intrinsics shouldn't modify anything.
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    self.assertFalse(modified)
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    type_test_utils.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    type_test_utils.assert_types_identical(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(
        _count_intrinsics(reduced, intrinsic_defs.FEDERATED_SELECT.uri), 0
    )


if __name__ == '__main__':
  absltest.main()
