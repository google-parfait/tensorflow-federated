# Copyright 2019, Google LLC.
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

from tensorflow_federated.python.simulation.baselines.shakespeare import shakespeare_models


class ModelsTest(tf.test.TestCase):

  def test_create_recurrent_model_raises_on_nonpositive_vocab_size(self):
    with self.assertRaisesRegex(
        ValueError, 'vocab_size must be a positive integer'):
      shakespeare_models.create_recurrent_model(
          vocab_size=-2, sequence_length=8)

  def test_create_recurrent_model_raises_on_nonpositive_sequence_length(self):
    with self.assertRaisesRegex(
        ValueError, 'sequence_length must be a positive integer'):
      shakespeare_models.create_recurrent_model(
          vocab_size=3, sequence_length=0)

  def test_model_input_output_shape(self):
    vocab_size = 5
    sequence_length = 7
    model = shakespeare_models.create_recurrent_model(
        vocab_size, sequence_length)

    self.assertEqual(model.input_shape, (None, sequence_length))
    self.assertEqual(model.output_shape, (None, sequence_length, vocab_size))

  def test_mask_zero_results_in_correct_mask(self):
    mask_model = shakespeare_models.create_recurrent_model(
        vocab_size=3, sequence_length=3, mask_zero=True)
    data = tf.constant([[0, 1, 1]])
    output_mask = mask_model.compute_mask(data, mask=None)
    self.assertAllEqual(output_mask, [[False, True, True]])

  def test_no_mask_zero_results_in_correct_mask(self):
    mask_model = shakespeare_models.create_recurrent_model(
        vocab_size=3, sequence_length=3, mask_zero=False)
    data = tf.constant([[0, 1, 1]])
    output_mask = mask_model.compute_mask(data, mask=None)
    self.assertIsNone(output_mask)


if __name__ == '__main__':
  tf.test.main()
