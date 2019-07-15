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
"""Tests for tree_analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import tree_analysis


class IntrinsicsWhitelistedTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_analysis.check_intrinsics_whitelisted_for_reduction(None)

  def test_passes_with_federated_map(self):
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MAP.uri,
        computation_types.FunctionType([
            computation_types.FunctionType(tf.int32, tf.float32),
            computation_types.FederatedType(tf.int32, placements.CLIENTS)
        ], computation_types.FederatedType(tf.float32, placements.CLIENTS)))
    tree_analysis.check_intrinsics_whitelisted_for_reduction(intrinsic)

  def test_raises_with_federated_mean(self):
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MEAN.uri,
        computation_types.FunctionType(
            computation_types.FederatedType(tf.int32, placements.CLIENTS),
            computation_types.FederatedType(tf.int32, placements.SERVER)))

    with self.assertRaisesRegex(
        ValueError,
        computation_building_blocks.compact_representation(intrinsic)):
      tree_analysis.check_intrinsics_whitelisted_for_reduction(intrinsic)


if __name__ == '__main__':
  absltest.main()
