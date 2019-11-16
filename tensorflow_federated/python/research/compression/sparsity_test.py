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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.compression import sparsity
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


# TODO(b/137613901): Subclass te.testing.BaseEncodingStageTest when the symbol
# is exposed in tensorflow_model_optimization package.
class OddEvenSparseEncodingStageTest(tf.test.TestCase):
  """Tests for OddEvenSparseEncodingStage."""

  def test_one_to_many_run_two_rounds(self):
    stage = sparsity.OddEvenSparseEncodingStage()
    x = tf.ones([5])
    state = self.evaluate(stage.initial_state())

    decoded_x_list = []
    for _ in range(2):
      encode_params, decode_params = stage.get_params(state)
      encoded_x, state_update_tensors = stage.encode(x, encode_params)
      state = stage.update_state(state, state_update_tensors)
      decoded_x = stage.decode(encoded_x, decode_params, shape=tf.shape(x))
      state, decoded_x = self.evaluate([state, decoded_x])
      decoded_x_list.append(decoded_x)

    # Odd and even indices should be aggregated in the two rounds.
    self.assertAllClose(np.array([1., 0., 1., 0., 1.]), decoded_x_list[0])
    self.assertAllClose(np.array([0., 1., 0., 1., 0.]), decoded_x_list[1])
    self.assertAllClose(np.ones(5), sum(decoded_x_list))


class SparseQuantizingEncoderTest(tf.test.TestCase):

  def test_sparse_quantizing_encoder(self):
    encoder = sparsity.sparse_quantizing_encoder(8)
    self.assertIsInstance(encoder, te.core.Encoder)


if __name__ == '__main__':
  tf.test.main()
