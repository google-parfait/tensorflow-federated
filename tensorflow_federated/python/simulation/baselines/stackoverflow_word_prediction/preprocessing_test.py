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
import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.baselines.stackoverflow_word_prediction import preprocessing


TEST_DATA = collections.OrderedDict(
    creation_date=(['unused date']),
    title=(['unused title']),
    score=([tf.constant(0, dtype=tf.int64)]),
    tags=(['unused test tag']),
    tokens=(['one must imagine']),
    type=(['unused type']),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class SplitInputTest(tf.test.TestCase):

  def test_split_input_returns_expected_result(self):
    tokens = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    expected_input = [[0, 1, 2, 3]]
    expected_target = [[1, 2, 3, 4]]
    split = preprocessing.split_input_target(tokens)
    self.assertAllEqual(self.evaluate(split[0]), expected_input)
    self.assertAllEqual(self.evaluate(split[1]), expected_target)


class ToIDsFnTest(tf.test.TestCase):

  def test_ids_fn_truncates_on_input_longer_than_sequence_length(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 1
    bos = preprocessing.get_special_tokens(len(vocab)).bos
    to_ids_fn = preprocessing.build_to_ids_fn(vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1])

  def test_build_to_ids_fn_embeds_all_vocab(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    special_tokens = preprocessing.get_special_tokens(len(vocab))
    bos = special_tokens.bos
    eos = special_tokens.eos
    to_ids_fn = preprocessing.build_to_ids_fn(vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1, 2, 3, eos])

  def test_pad_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    to_ids_fn = preprocessing.build_to_ids_fn(vocab, max_seq_len)
    special_tokens = preprocessing.get_special_tokens(len(vocab))
    pad, bos, eos = special_tokens.pad, special_tokens.bos, special_tokens.eos
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    batched_ds = tf.data.Dataset.from_tensor_slices([processed]).padded_batch(
        1, padded_shapes=[6])
    sample_elem = next(iter(batched_ds))
    self.assertAllEqual(self.evaluate(sample_elem), [[bos, 1, 2, 3, eos, pad]])

  def test_oov_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    num_oov_buckets = 2
    to_ids_fn = preprocessing.build_to_ids_fn(
        vocab, max_seq_len, num_oov_buckets=num_oov_buckets)
    oov_tokens = preprocessing.get_special_tokens(
        len(vocab), num_oov_buckets=num_oov_buckets).oov
    data = {'tokens': 'A B D'}
    processed = to_ids_fn(data)
    self.assertLen(oov_tokens, num_oov_buckets)
    self.assertIn(self.evaluate(processed)[3], oov_tokens)


class BatchAndSplitTest(tf.test.TestCase):

  def test_batch_and_split_fn_returns_dataset_with_correct_type_spec(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = preprocessing.batch_and_split(
        ds, sequence_length=6, batch_size=1)
    self.assertIsInstance(padded_and_batched, tf.data.Dataset)
    self.assertEqual(padded_and_batched.element_spec, (tf.TensorSpec(
        [None, 6], dtype=tf.int64), tf.TensorSpec([None, 6], dtype=tf.int64)))

  def test_batch_and_split_fn_returns_dataset_yielding_expected_elements(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = preprocessing.batch_and_split(
        ds, sequence_length=6, batch_size=1)
    num_elems = 0
    for elem in padded_and_batched:
      self.assertAllEqual(
          self.evaluate(elem[0]),
          tf.constant([[0, 1, 2, 3, 4, 0]], dtype=tf.int64))
      self.assertAllEqual(
          self.evaluate(elem[1]),
          tf.constant([[1, 2, 3, 4, 0, 0]], dtype=tf.int64))
      num_elems += 1
    self.assertEqual(num_elems, 1)


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  def test_preprocess_fn_with_negative_epochs_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'num_epochs must be a positive integer'):
      preprocessing.create_preprocess_fn(
          num_epochs=-2, batch_size=1, vocab=['A'], sequence_length=10)

  def test_preprocess_fn_with_negative_batch_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'batch_size must be a positive integer'):
      preprocessing.create_preprocess_fn(
          num_epochs=1, batch_size=-10, vocab=['A'], sequence_length=10)

  def test_preprocess_fn_with_empty_vocab_raises(self):
    with self.assertRaisesRegex(ValueError, 'vocab must be non-empty'):
      preprocessing.create_preprocess_fn(
          num_epochs=1, batch_size=1, vocab=[], sequence_length=10)

  def test_preprocess_fn_with_negative_sequence_length(self):
    with self.assertRaisesRegex(ValueError,
                                'sequence_length must be a positive integer'):
      preprocessing.create_preprocess_fn(
          num_epochs=1, batch_size=1, vocab=['A'], sequence_length=0)

  def test_preprocess_fn_with_zero_or_less_neg1_max_elements_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'max_elements must be a positive integer or -1'):
      preprocessing.create_preprocess_fn(
          num_epochs=1,
          batch_size=1,
          vocab=['A'],
          sequence_length=10,
          max_elements=-2)

    with self.assertRaisesRegex(
        ValueError, 'max_elements must be a positive integer or -1'):
      preprocessing.create_preprocess_fn(
          num_epochs=1,
          batch_size=1,
          vocab=['A'],
          sequence_length=10,
          max_elements=0)

  def test_preprocess_fn_with_negative_num_oov_buckets_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'num_oov_buckets must be a positive integer'):
      preprocessing.create_preprocess_fn(
          num_epochs=1,
          batch_size=1,
          vocab=['A'],
          sequence_length=10,
          num_oov_buckets=-1)

  @parameterized.named_parameters(('param1', 1, 1), ('param2', 4, 2),
                                  ('param3', 100, 3))
  def test_preprocess_fn_returns_correct_dataset_element_spec(
      self, sequence_length, num_oov_buckets):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = preprocessing.create_preprocess_fn(
        batch_size=32,
        num_epochs=1,
        sequence_length=sequence_length,
        max_elements=100,
        vocab=['one', 'must'],
        num_oov_buckets=num_oov_buckets)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        preprocessed_ds.element_spec,
        (tf.TensorSpec(shape=[None, sequence_length], dtype=tf.int64),
         tf.TensorSpec(shape=[None, sequence_length], dtype=tf.int64)))

  def test_preprocess_fn_returns_correct_sequence_with_1_oov_bucket(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = preprocessing.create_preprocess_fn(
        batch_size=32,
        num_epochs=1,
        sequence_length=6,
        max_elements=100,
        vocab=['one', 'must'],
        num_oov_buckets=1)

    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1
    self.assertAllEqual(
        self.evaluate(element[0]),
        tf.constant([[4, 1, 2, 3, 5, 0]], dtype=tf.int64))

  def test_preprocess_fn_returns_correct_sequence_with_3_oov_buckets(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = preprocessing.create_preprocess_fn(
        batch_size=32,
        num_epochs=1,
        sequence_length=6,
        max_elements=100,
        vocab=['one', 'must'],
        num_oov_buckets=3)
    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))
    # BOS is len(vocab)+3+1
    self.assertEqual(self.evaluate(element[0])[0][0], 6)
    self.assertEqual(self.evaluate(element[0])[0][1], 1)
    self.assertEqual(self.evaluate(element[0])[0][2], 2)
    # OOV is [len(vocab)+1, len(vocab)+2, len(vocab)+3]
    self.assertIn(self.evaluate(element[0])[0][3], [3, 4, 5])
    # EOS is len(vocab)+3+2
    self.assertEqual(self.evaluate(element[0])[0][4], 7)
    # pad is 0
    self.assertEqual(self.evaluate(element[0])[0][5], 0)

  @parameterized.named_parameters(
      ('num_epochs_1_batch_size_1', 1, 1),
      ('num_epochs_4_batch_size_2', 4, 2),
      ('num_epochs_9_batch_size_3', 9, 3),
      ('num_epochs_12_batch_size_1', 12, 1),
      ('num_epochs_3_batch_size_5', 3, 5),
      ('num_epochs_7_batch_size_2', 7, 2),
  )
  def test_ds_length_is_ceil_num_epochs_over_batch_size(self, num_epochs,
                                                        batch_size):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = preprocessing.create_preprocess_fn(
        num_epochs=num_epochs,
        batch_size=batch_size,
        vocab=['A'],
        sequence_length=10,
        shuffle_buffer_size=1)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        tf.cast(tf.math.ceil(num_epochs / batch_size), tf.int32))


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
