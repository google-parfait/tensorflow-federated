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
class OddEvenSparseEncodingStageTest(te.testing.BaseEncodingStageTest):
  """Tests for OddEvenSparseEncodingStage."""

  def default_encoding_stage(self):
    """See base class."""
    return sparsity.OddEvenSparseEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    num_elements = data.x.size
    num_equal = np.sum(np.where(data.x == data.decoded_x, 1, 0))
    # Checks two possible values if num_elements is an odd number.
    self.assertIn(num_equal, [num_elements // 2, num_elements // 2 + 1])

  def test_one_to_many_run_few_rounds(self):
    stage = sparsity.OddEvenSparseEncodingStage()
    x_fn = lambda: tf.ones([5])
    initial_state = self.evaluate(stage.initial_state())

    # Odd and even indices should be aggregated in different rounds.
    # Round 1.
    data = self.run_one_to_many_encode_decode(stage, x_fn, initial_state)
    self.assertAllClose(np.array([1., 0., 1., 0., 1.]), data.decoded_x)

    # Round 2.
    data = self.run_one_to_many_encode_decode(stage, x_fn, data.updated_state)
    self.assertAllClose(np.array([0., 1., 0., 1., 0.]), data.decoded_x)

    # Round 3.
    data = self.run_one_to_many_encode_decode(stage, x_fn, data.updated_state)
    self.assertAllClose(np.array([1., 0., 1., 0., 1.]), data.decoded_x)


class SparseQuantizingEncoderTest(tf.test.TestCase):

  def test_sparse_quantizing_encoder(self):
    encoder = sparsity.sparse_quantizing_encoder(8)
    self.assertIsInstance(encoder, te.core.Encoder)


if __name__ == '__main__':
  # TODO(b/148756730): Delete this explicit graph mode enforcement. This is a
  # temporary workaround due to a bug in tfmot package, and can be removed in a
  # tfmot version following 0.2.1.
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()
