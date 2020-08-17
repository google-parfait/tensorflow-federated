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

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_serialization


@computations.tf_computation(tf.int32, tf.int32)
def add_int32(current, val):
  return current + val


class ComputationSerializationTest(test.TestCase):

  def test_serialize_deserialize_round_trip(self):
    serialized_comp = computation_serialization.serialize_computation(add_int32)
    deserialize_comp = computation_serialization.deserialize_computation(
        serialized_comp)
    self.assertIsInstance(deserialize_comp, computation_base.Computation)
    self.assertEqual(deserialize_comp, add_int32)


if __name__ == '__main__':
  test.main()
