# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.executors import data_descriptor
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class DataDescriptorTest(absltest.TestCase):

  def test_non_federated(self):
    ds = data_descriptor.DataDescriptor(
        computations.tf_computation(lambda x: tf.cast(x + 10, tf.float32),
                                    tf.int32), 20,
        computation_types.TensorType(tf.int32))
    self.assertEqual(str(ds.type_signature), 'float32')

    @computations.tf_computation(tf.float32)
    def foo(x):
      return x * 20.0

    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory()):
      result = foo(ds)
    self.assertEqual(result, 600.0)

  def test_federated(self):
    ds = data_descriptor.DataDescriptor(
        computations.federated_computation(
            lambda x: intrinsics.federated_value(x, placements.CLIENTS),
            tf.int32), 1000, computation_types.TensorType(tf.int32), 3)
    self.assertEqual(str(ds.type_signature), 'int32@CLIENTS')

    @computations.federated_computation(
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS, all_equal=True))
    def foo(x):
      return intrinsics.federated_sum(x)

    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory()):
      result = foo(ds)
    self.assertEqual(result, 3000)

  def test_comp_none(self):
    ds = data_descriptor.DataDescriptor(
        None, [1, 2, 3],
        computation_types.FederatedType(tf.int32, placements.CLIENTS), 3)
    self.assertEqual(str(ds.type_signature), '{int32}@CLIENTS')

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_sum(x)

    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory()):
      result = foo(ds)
    self.assertEqual(result, 6)


if __name__ == '__main__':
  # TFF-CPP does not yet speak `Ingestable`; b/202336418
  absltest.main()
