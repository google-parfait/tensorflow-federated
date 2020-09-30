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

import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import tensorflow_deserialization
from tensorflow_federated.python.core.impl.compiler import building_block_factory


class TensorFlowDeserializationTest(test_utils.TestCase):

  @test_utils.graph_mode_test
  def test_deserialize_and_call_tf_computation_with_add_one(self):
    identity_type = computation_types.TensorType(tf.int32)
    identity_fn = building_block_factory.create_compiled_identity(identity_type)
    init_op, result = tensorflow_deserialization.deserialize_and_call_tf_computation(
        identity_fn.proto, tf.constant(10), tf.compat.v1.get_default_graph())
    self.assertTrue(tf.is_tensor(result))
    with tf.compat.v1.Session() as sess:
      if init_op:
        sess.run(init_op)
      result_val = sess.run(result)
    self.assertEqual(result_val, 10)


if __name__ == '__main__':
  test_utils.main()
