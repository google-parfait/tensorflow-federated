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
"""Tests for Stackoverflow data loader."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.research.baselines.stackoverflow import dataset

VOCAB = ['A', 'B', 'C']


class DatasetTest(test.TestCase):

  def test_split_input_target(self):
    tokens = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    expected_input = [[0, 1, 2, 3]]
    expected_target = [[1, 2, 3, 4]]
    split = dataset.split_input_target(tokens)
    self.assertAllEqual(self.evaluate(split[0]), expected_input)
    self.assertAllEqual(self.evaluate(split[1]), expected_target)

  def test_build_to_ids(self):
    data = {'tokens': 'A B X'}
    _, oov, bos, eos = dataset.get_special_tokens(len(VOCAB))
    expected = [bos, 1, 2, oov, eos]
    for max_seq_len in range(1, 8):
      to_ids_fn = dataset.build_to_ids_fn(VOCAB, max_seq_len)
      processed = to_ids_fn(data)
      seq_len = min(max_seq_len, len(expected)) + 1
      self.assertAllEqual(self.evaluate(processed), expected[:seq_len])

  def test_batch_and_split(self):
    raw_data = {'tokens': 'A Z C'}
    pad, oov, bos, eos = dataset.get_special_tokens(len(VOCAB))
    expected = [bos, 1, oov, 3, eos, pad, pad, pad]
    for max_seq_len in range(1, 8):
      to_ids_fn = dataset.build_to_ids_fn(VOCAB, max_seq_len)
      data = tf.data.Dataset.from_tensor_slices([to_ids_fn(raw_data)])
      batched = dataset.batch_and_split(data, max_seq_len, 1)
      sample_elem = next(iter(batched))
      result = self.evaluate(sample_elem)
      correct = ([expected[:max_seq_len]], [expected[1:max_seq_len+1]])
      self.assertAllEqual(result, correct)


if __name__ == '__main__':
  test.main()
