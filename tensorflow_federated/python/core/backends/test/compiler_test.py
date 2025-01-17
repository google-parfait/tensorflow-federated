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
import federated_language
import numpy as np

from tensorflow_federated.python.core.backends.mapreduce import intrinsics
from tensorflow_federated.python.core.backends.test import compiler
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

  @parameterized.named_parameters(
      ('int32', np.int32, np.int32),
      ('int32_struct', [np.int32, np.int32], np.int32),
      ('int64', np.int64, np.int32),
      ('mixed_struct', [np.int64, [np.int32]], np.int32),
      ('per_leaf_bitwidth', [np.int64, [np.int32]], [np.int32, [np.int32]]),
  )
  def test_federated_secure_sum(self, value_dtype, bitwidth_type):
    uri = federated_language.framework.FEDERATED_SECURE_SUM.uri
    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            [
                federated_language.FederatedType(
                    value_dtype, federated_language.CLIENTS
                ),
                federated_language.to_type(bitwidth_type),
            ],
            federated_language.FederatedType(
                value_dtype, federated_language.SERVER
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
    self.assertEqual(comp.type_signature, reduced.type_signature)
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    self.assertEqual(comp.type_signature, reduced.type_signature)
    self.assertGreater(
        _count_intrinsics(
            reduced, federated_language.framework.FEDERATED_AGGREGATE.uri
        ),
        0,
    )

  @parameterized.named_parameters(
      ('int32', np.int32, np.int32),
      ('int32_struct', [np.int32, np.int32], np.int32),
      ('int64', np.int64, np.int32),
      ('mixed_struct', [np.int64, [np.int32]], np.int32),
      ('per_leaf_bitwidth', [np.int64, [np.int32]], [np.int32, [np.int32]]),
  )
  def test_federated_secure_sum_bitwidth(self, value_dtype, bitwidth_type):
    uri = federated_language.framework.FEDERATED_SECURE_SUM_BITWIDTH.uri
    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            parameter=[
                federated_language.FederatedType(
                    value_dtype, federated_language.CLIENTS
                ),
                federated_language.to_type(bitwidth_type),
            ],
            result=federated_language.FederatedType(
                value_dtype, federated_language.SERVER
            ),
        ),
    )
    # First without secure intrinsics shouldn't modify anything.
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    self.assertFalse(modified)
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    self.assertEqual(comp.type_signature, reduced.type_signature)
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    self.assertEqual(comp.type_signature, reduced.type_signature)
    self.assertGreater(
        _count_intrinsics(
            reduced, federated_language.framework.FEDERATED_AGGREGATE.uri
        ),
        0,
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
    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            parameter=[
                federated_language.FederatedType(
                    value_dtype, federated_language.CLIENTS
                ),
                federated_language.to_type(modulus_type),
            ],
            result=federated_language.FederatedType(
                value_dtype, federated_language.SERVER
            ),
        ),
    )
    # First without secure intrinsics shouldn't modify anything.
    reduced, modified = (
        tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
    )
    self.assertFalse(modified)
    self.assertGreater(_count_intrinsics(comp, uri), 0)
    self.assertEqual(comp.type_signature, reduced.type_signature)
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    # Inserting tensorflow, as we do here, does not preserve python containers
    # currently.
    federated_language.framework.assert_types_equivalent(
        comp.type_signature, reduced.type_signature
    )
    self.assertGreater(
        _count_intrinsics(
            reduced, federated_language.framework.FEDERATED_SUM.uri
        ),
        0,
    )

  def test_federated_secure_select(self):
    uri = federated_language.framework.FEDERATED_SECURE_SELECT.uri
    comp = federated_language.framework.Intrinsic(
        uri,
        federated_language.FunctionType(
            [
                federated_language.FederatedType(
                    np.int32, federated_language.CLIENTS
                ),  # client_keys
                federated_language.FederatedType(
                    np.int32, federated_language.SERVER
                ),  # max_key
                federated_language.FederatedType(
                    np.float32, federated_language.SERVER
                ),  # server_state
                federated_language.FunctionType(
                    [np.float32, np.int32], np.float32
                ),  # select_fn
            ],
            federated_language.FederatedType(
                federated_language.SequenceType(np.float32),
                federated_language.CLIENTS,
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
    self.assertEqual(comp.type_signature, reduced.type_signature)
    # Now replace bodies including secure intrinsics.
    reduced, modified = (
        compiler._replace_secure_intrinsics_with_insecure_bodies(comp)
    )
    self.assertTrue(modified)
    self.assertEqual(comp.type_signature, reduced.type_signature)
    self.assertGreater(
        _count_intrinsics(
            reduced, federated_language.framework.FEDERATED_SELECT.uri
        ),
        0,
    )


if __name__ == '__main__':
  absltest.main()
