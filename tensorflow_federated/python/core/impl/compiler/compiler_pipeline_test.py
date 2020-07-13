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
from tensorflow_federated.python.core.impl.compiler import compiler_pipeline
from tensorflow_federated.python.core.impl.types import placement_literals


class CompilerPipelineTest(absltest.TestCase):

  def test_compile_computation_with_identity(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS),
        computation_types.FederatedType(tf.float32, placement_literals.SERVER,
                                        True)
    ])
    def foo(temperatures, threshold):
      return intrinsics.federated_sum(
          intrinsics.federated_map(
              computations.tf_computation(
                  lambda x, y: tf.cast(tf.greater(x, y), tf.int32),
                  [tf.float32, tf.float32]),
              [temperatures,
               intrinsics.federated_broadcast(threshold)]))

    pipeline = compiler_pipeline.CompilerPipeline(lambda x: x)

    compiled_foo = pipeline.compile(foo)

    self.assertEqual(hash(foo), hash(compiled_foo))

    # TODO(b/113123410): Expand the test with more structural invariants.


if __name__ == '__main__':
  absltest.main()
