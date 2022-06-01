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

from tensorflow_federated.python.core.impl.executors import data_backend_base
from tensorflow_federated.python.core.impl.executors import data_descriptor
from tensorflow_federated.python.core.impl.executors import data_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class DataDescriptorTest(absltest.TestCase):

  def test_non_federated(self):
    ds = data_descriptor.DataDescriptor(
        tensorflow_computation.tf_computation(
            lambda x: tf.cast(x + 10, tf.float32), tf.int32), 20,
        computation_types.TensorType(tf.int32))
    self.assertEqual(str(ds.type_signature), 'float32')

    @tensorflow_computation.tf_computation(tf.float32)
    def foo(x):
      return x * 20.0

    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory()):
      result = foo(ds)
    self.assertEqual(result, 600.0)

  def test_federated(self):
    ds = data_descriptor.DataDescriptor(
        federated_computation.federated_computation(
            lambda x: intrinsics.federated_value(x, placements.CLIENTS),
            tf.int32), 1000, computation_types.TensorType(tf.int32), 3)
    self.assertEqual(str(ds.type_signature), 'int32@CLIENTS')

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS, all_equal=True))
    def foo(x):
      return intrinsics.federated_sum(x)

    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory()):
      result = foo(ds)
    self.assertEqual(result, 3000)

  def test_raises_with_server_cardinality_specified(self):
    with self.assertRaises(TypeError):
      data_descriptor.DataDescriptor(
          federated_computation.federated_computation(
              lambda x: intrinsics.federated_value(x, placements.SERVER),
              tf.int32), 1000, computation_types.TensorType(tf.int32), 3)

  def test_comp_none(self):
    ds = data_descriptor.DataDescriptor(
        None, [1, 2, 3],
        computation_types.FederatedType(tf.int32, placements.CLIENTS), 3)
    self.assertEqual(str(ds.type_signature), '{int32}@CLIENTS')

    @federated_computation.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_sum(x)

    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory()):
      result = foo(ds)
    self.assertEqual(result, 6)

  def test_cardinality_free_data_descriptor_places_data(self):
    ds = data_descriptor.CardinalityFreeDataDescriptor(
        federated_computation.federated_computation(
            lambda x: intrinsics.federated_value(x, placements.CLIENTS),
            tf.int32), 1000, computation_types.TensorType(tf.int32))
    self.assertEqual(str(ds.type_signature), 'int32@CLIENTS')

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS, all_equal=True))
    def foo(x):
      return intrinsics.federated_sum(x)

    # Since this DataDescriptor does not specify its cardinality, the number of
    # values placed is inferred from the decault setting for the executor.
    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory(default_num_clients=1)):
      result = foo(ds)
    self.assertEqual(result, 1000)

    with executor_test_utils.install_executor(
        executor_stacks.local_executor_factory(default_num_clients=3)):
      result = foo(ds)
    self.assertEqual(result, 3000)

  def test_create_data_descriptor_for_data_backend(self):

    class TestDataBackend(data_backend_base.DataBackend):

      def __init__(self, value):
        self._value = value

      async def materialize(self, data, type_spec):
        return self._value

    data_constant = 1
    type_spec = computation_types.TensorType(tf.int32)

    def ex_fn(device):
      return data_executor.DataExecutor(
          eager_tf_executor.EagerTFExecutor(device),
          TestDataBackend(data_constant))

    factory = executor_stacks.local_executor_factory(leaf_executor_fn=ex_fn)

    @federated_computation.federated_computation(
        computation_types.FederatedType(type_spec, placements.CLIENTS))
    def foo(dd):
      return intrinsics.federated_sum(dd)

    with executor_test_utils.install_executor(factory):
      uris = [f'foo://bar{i}' for i in range(3)]
      dd = data_descriptor.CreateDataDescriptor(uris, type_spec)
      result = foo(dd)

    self.assertEqual(result, 3)


if __name__ == '__main__':
  # TFF-CPP does not yet speak `Ingestable`; b/202336418
  absltest.main()
