# Lint as: python3
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
"""Tests for compiler_pipeline.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import compiler_pipeline
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import transformation_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks


class CompilerPipelineTest(absltest.TestCase):

  def test_compile_computation(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.float32, placements.CLIENTS),
        computation_types.FederatedType(tf.float32, placements.SERVER, True)
    ])
    def foo(temperatures, threshold):
      return intrinsics.federated_sum(
          intrinsics.federated_map(
              computations.tf_computation(
                  lambda x, y: tf.cast(tf.greater(x, y), tf.int32),
                  [tf.float32, tf.float32]),
              [temperatures,
               intrinsics.federated_broadcast(threshold)]))

    pipeline = compiler_pipeline.CompilerPipeline(
        context_stack_impl.context_stack)

    compiled_foo = pipeline.compile(foo)

    def _not_federated_sum(x):
      if isinstance(x, building_blocks.Intrinsic):
        self.assertNotEqual(x.uri, intrinsic_defs.FEDERATED_SUM.uri)
      return x, False

    transformation_utils.transform_postorder(
        building_blocks.ComputationBuildingBlock.from_proto(
            computation_impl.ComputationImpl.get_proto(compiled_foo)),
        _not_federated_sum)

    # TODO(b/113123410): Expand the test with more structural invariants.


if __name__ == '__main__':
  absltest.main()
