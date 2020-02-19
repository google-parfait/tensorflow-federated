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

from tensorflow_federated.python.research.optimization.stackoverflow import dataset

tf.compat.v1.enable_v2_behavior()


class DatasetTest(tf.test.TestCase):

  def test_raises_bad_args(self):
    with self.assertRaises(ValueError):
      dataset.construct_word_level_datasets(0, 1, 1, 1, 1, 1)
    with self.assertRaises(ValueError):
      dataset.construct_word_level_datasets(1, 0, 1, 1, 1, 1)
    with self.assertRaises(ValueError):
      dataset.construct_word_level_datasets(1, 1, 0, 1, 1, 1)
    with self.assertRaises(ValueError):
      dataset.construct_word_level_datasets(1, 1, 1, 0, 1, 1)
    with self.assertRaises(ValueError):
      dataset.construct_word_level_datasets(1, 1, 1, 1, -2, 1)
    with self.assertRaises(ValueError):
      dataset.construct_word_level_datasets(1, 1, 1, 1, -1, 0)

  def test_split_input_target(self):
    tokens = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    expected_input = [[0, 1, 2, 3]]
    expected_target = [[1, 2, 3, 4]]
    split = dataset.split_input_target(tokens)
    self.assertAllEqual(self.evaluate(split[0]), expected_input)
    self.assertAllEqual(self.evaluate(split[1]), expected_target)

  def test_build_to_ids_fn_truncates(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 1
    _, _, bos, _ = dataset.get_special_tokens(len(vocab))
    to_ids_fn = dataset.build_to_ids_fn(vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1])

  def test_build_to_ids_fn_embeds_all_vocab(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    _, _, bos, eos = dataset.get_special_tokens(len(vocab))
    to_ids_fn = dataset.build_to_ids_fn(vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1, 2, 3, eos])

  def test_pad_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    to_ids_fn = dataset.build_to_ids_fn(vocab, max_seq_len)
    pad, _, bos, eos = dataset.get_special_tokens(len(vocab))
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    batched_ds = tf.data.Dataset.from_tensor_slices([processed]).padded_batch(
        1, padded_shapes=[6])
    sample_elem = next(iter(batched_ds))
    self.assertAllEqual(self.evaluate(sample_elem), [[bos, 1, 2, 3, eos, pad]])

  def test_oov_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    to_ids_fn = dataset.build_to_ids_fn(vocab, max_seq_len)
    _, oov_token, _, _ = dataset.get_special_tokens(len(vocab))
    data = {'tokens': 'A B D'}
    processed = to_ids_fn(data)
    self.assertEqual(self.evaluate(processed)[3], oov_token)


class BatchAndSplitTest(tf.test.TestCase):

  def test_batch_and_split_fn_returns_dataset_with_correct_type_spec(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = dataset.batch_and_split(
        ds, max_seq_len=6, batch_size=1)
    self.assertIsInstance(padded_and_batched, tf.data.Dataset)
    self.assertEqual(padded_and_batched.element_spec, (tf.TensorSpec(
        [None, 6], dtype=tf.int64), tf.TensorSpec([None, 6], dtype=tf.int64)))

  def test_batch_and_split_fn_returns_dataset_yielding_expected_elements(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = dataset.batch_and_split(
        ds, max_seq_len=6, batch_size=1)
    num_elems = 0
    for elem in padded_and_batched:
      self.assertAllEqual(
          self.evaluate(elem[0]), np.array([[0, 1, 2, 3, 4, 0]], np.int64))
      self.assertAllEqual(
          self.evaluate(elem[1]), np.array([[1, 2, 3, 4, 0, 0]], np.int64))
      num_elems += 1
    self.assertEqual(num_elems, 1)


if __name__ == '__main__':
  tf.test.main()
