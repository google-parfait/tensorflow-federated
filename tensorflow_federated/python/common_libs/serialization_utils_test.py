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

from absl.testing import absltest
import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.python.common_libs import serialization_utils


class SerializationUtilsTest(absltest.TestCase):

  def test_pack_graph_def_returns_any_pb(self):
    input_value = tf.compat.v1.GraphDef()
    any_pb = serialization_utils.pack_graph_def(input_value)
    self.assertEqual(type(any_pb), any_pb2.Any)

  def test_pack_unpack_roundtrip(self):
    with tf.Graph().as_default() as g:
      tf.constant(1.0)
    input_value = g.as_graph_def()
    any_pb = serialization_utils.pack_graph_def(input_value)
    output_value = serialization_utils.unpack_graph_def(any_pb)
    self.assertEqual(input_value, output_value)

  def test_pack_graph_seed_set_raises(self):
    with tf.Graph().as_default() as g:
      tf.compat.v2.random.set_seed(1234)
      tf.random.normal([1])
    input_value = g.as_graph_def()
    with self.assertRaisesRegex(ValueError, 'graph-level random seed'):
      serialization_utils.pack_graph_def(input_value)

  def test_pack_graph_def_fails_non_graph_def_arg(self):
    with self.assertRaisesRegex(TypeError, 'found str'):
      serialization_utils.pack_graph_def('not a graphdef')

  def test_unpack_graph_def_not_any_arg(self):
    with self.assertRaisesRegex(TypeError, 'Any'):
      serialization_utils.unpack_graph_def('not_any')

  def test_unpack_graph_def_not_packed_graph_def(self):
    any_pb = any_pb2.Any()
    with self.assertRaisesRegex(ValueError, 'Unable to unpack'):
      serialization_utils.unpack_graph_def(any_pb)


if __name__ == '__main__':
  absltest.main()
