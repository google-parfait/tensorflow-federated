# Copyright 2023, The TensorFlow Federated Authors.
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

"""DTensor executor binding test with Mesh and layout_map.

These tests are added separately, since they require a module level setup
for virtual devices initialization to create logical  devices.
"""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types

TensorType = computation_types.TensorType


# Creating logical devices should be done only once before TF runtime startup
# Thus, perform it during setUpModule method.
def setUpModule():
  devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(
      devices[0],
      [
          tf.config.LogicalDeviceConfiguration(),
      ]
      * 8,
  )


class DtensorExecutorBindingTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('sharded_input', True),
      ('replicated_input', False),
  )
  def test_call_with_arg_dtensor_executor_with_mesh(self, sharded_input):
    mesh_dim_name = 'batch'
    mesh = tf.experimental.dtensor.create_mesh(
        devices=['CPU:%d' % i for i in range(8)], mesh_dims=[(mesh_dim_name, 8)]
    )
    # dtensor.run_on method is used to set mesh for the dtensor device.
    with tf.experimental.dtensor.run_on(mesh):
      executor = executor_bindings.create_dtensor_executor(
          tf.experimental.dtensor.device_name(), mesh.to_string(), -1
      )
      value_pb, _ = value_serialization.serialize_value(
          tf.constant([1, 2, 3, 4, 5, 6, 7, 8]),
          TensorType(shape=[8], dtype=tf.int64),
      )

      value_ref = executor.create_value(value_pb)
      arg = executor.create_struct((value_ref.ref, value_ref.ref))

      mesh_dim_name = 'batch'
      layout_map = {}
      if sharded_input:
        layout_map['arg_a'] = mesh_dim_name
      else:
        layout_map['arg_a'] = 'unsharded'

      @tensorflow_computation.tf_computation(
          tf.int64, tf.int64, layout_map=layout_map
      )
      def foo(a, b):
        return tf.add(a, b)

      comp_pb = executor_pb2.Value(computation=foo.get_proto(foo))
      comp = executor.create_value(comp_pb)
      result = executor.create_call(comp.ref, arg.ref)
      result_value_pb = executor.materialize(result.ref)
      result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
      self.assertAllEqual(result_tensor, [2, 4, 6, 8, 10, 12, 14, 16])


if __name__ == '__main__':
  tf.test.main()
