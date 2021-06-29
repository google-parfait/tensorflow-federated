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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.executors import serialization_bindings


class SerializeTensorTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_int32', 1, tf.int32),
      ('scalar_float64', 2.0, tf.float64),
      ('scalar_string', b'abc', tf.string),
      ('tensor_int32', [1, 2, 3], tf.int32),
      ('tensor_float64', [2.0, 4.0, 6.0], tf.float64),
      ('tensor_string', [[b'abc', b'xyz']], tf.string),
  )
  def test_serialize(self, input_value, dtype):
    value_proto = serialization_bindings.Value()
    value_proto = serialization_bindings.serialize_tensor_value(
        tf.convert_to_tensor(input_value, dtype), value_proto)
    tensor_proto = tf.make_tensor_proto(values=0)
    self.assertTrue(value_proto.tensor.Unpack(tensor_proto))
    roundtrip_value = tf.make_ndarray(tensor_proto)
    self.assertAllEqual(roundtrip_value, input_value)

  @parameterized.named_parameters(
      ('scalar_int32', 1, tf.int32),
      ('scalar_float64', 2.0, tf.float64),
      ('scalar_string', b'abc', tf.string),
      ('tensor_int32', [1, 2, 3], tf.int32),
      ('tensor_float64', [2.0, 4.0, 6.0], tf.float64),
      ('tensor_string', [[b'abc', b'xyz']], tf.string),
  )
  def test_roundtrip(self, input_value, dtype):
    value_proto = serialization_bindings.Value()
    value_proto = serialization_bindings.serialize_tensor_value(
        tf.convert_to_tensor(input_value, dtype), value_proto)
    roundtrip_value = serialization_bindings.deserialize_tensor_value(
        value_proto)
    self.assertAllEqual(roundtrip_value, input_value)


if __name__ == '__main__':
  test_case.main()
