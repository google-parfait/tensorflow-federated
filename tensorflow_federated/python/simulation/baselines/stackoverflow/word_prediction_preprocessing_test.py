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
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines.stackoverflow import word_prediction_preprocessing


TEST_DATA = collections.OrderedDict(
    creation_date=(['unused date']),
    score=([tf.constant(0, dtype=tf.int64)]),
    tags=(['unused test tag']),
    title=(['unused title']),
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
    split = word_prediction_preprocessing.split_input_target(tokens)
    self.assertAllEqual(self.evaluate(split[0]), expected_input)
    self.assertAllEqual(self.evaluate(split[1]), expected_target)


class ToIDsFnTest(tf.test.TestCase):

  def test_ids_fn_truncates_on_input_longer_than_sequence_length(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 1
    bos = word_prediction_preprocessing.get_special_tokens(
        len(vocab)).beginning_of_sentence
    to_ids_fn = word_prediction_preprocessing.build_to_ids_fn(
        vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1])

  def test_build_to_ids_fn_embeds_all_vocab(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    special_tokens = word_prediction_preprocessing.get_special_tokens(
        len(vocab))
    bos = special_tokens.beginning_of_sentence
    eos = special_tokens.end_of_sentence
    to_ids_fn = word_prediction_preprocessing.build_to_ids_fn(
        vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1, 2, 3, eos])

  def test_pad_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    to_ids_fn = word_prediction_preprocessing.build_to_ids_fn(
        vocab, max_seq_len)
    special_tokens = word_prediction_preprocessing.get_special_tokens(
        len(vocab))
    pad = special_tokens.padding
    bos = special_tokens.beginning_of_sentence
    eos = special_tokens.end_of_sentence
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    batched_ds = tf.data.Dataset.from_tensor_slices([processed]).padded_batch(
        1, padded_shapes=[6])
    sample_elem = next(iter(batched_ds))
    self.assertAllEqual(self.evaluate(sample_elem), [[bos, 1, 2, 3, eos, pad]])

  def test_out_of_vocab_tokens_are_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    num_out_of_vocab_buckets = 2
    to_ids_fn = word_prediction_preprocessing.build_to_ids_fn(
        vocab, max_seq_len, num_out_of_vocab_buckets=num_out_of_vocab_buckets)
    out_of_vocab_tokens = word_prediction_preprocessing.get_special_tokens(
        len(vocab),
        num_out_of_vocab_buckets=num_out_of_vocab_buckets).out_of_vocab
    data = {'tokens': 'A B D'}
    processed = to_ids_fn(data)
    self.assertLen(out_of_vocab_tokens, num_out_of_vocab_buckets)
    self.assertIn(self.evaluate(processed)[3], out_of_vocab_tokens)


class BatchAndSplitTest(tf.test.TestCase):

  def test_batch_and_split_fn_returns_dataset_with_correct_type_spec(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = word_prediction_preprocessing.batch_and_split(
        ds, sequence_length=6, batch_size=1)
    self.assertIsInstance(padded_and_batched, tf.data.Dataset)
    self.assertEqual(padded_and_batched.element_spec, (tf.TensorSpec(
        [None, 6], dtype=tf.int64), tf.TensorSpec([None, 6], dtype=tf.int64)))

  def test_batch_and_split_fn_returns_dataset_yielding_expected_elements(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = word_prediction_preprocessing.batch_and_split(
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

  def test_preprocess_fn_with_empty_vocab_raises(self):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(ValueError, 'vocab must be non-empty'):
      word_prediction_preprocessing.create_preprocess_fn(
          preprocess_spec, vocab=[], sequence_length=10)

  @parameterized.named_parameters(('zero_value', 0), ('negative_value1', -1),
                                  ('negative_value2', -2))
  def test_nonpositive_sequence_length_raises(self, sequence_length):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(ValueError,
                                'sequence_length must be a positive integer'):
      word_prediction_preprocessing.create_preprocess_fn(
          preprocess_spec, vocab=['A'], sequence_length=0)

  @parameterized.named_parameters(('zero_value', 0), ('negative_value1', -1),
                                  ('negative_value2', -2))
  def test_nonpositive_num_out_of_vocab_buckets_length_raises(
      self, num_out_of_vocab_buckets):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(
        ValueError, 'num_out_of_vocab_buckets must be a positive integer'):
      word_prediction_preprocessing.create_preprocess_fn(
          preprocess_spec,
          vocab=['A'],
          sequence_length=10,
          num_out_of_vocab_buckets=num_out_of_vocab_buckets)

  @parameterized.named_parameters(('param1', 1, 1), ('param2', 4, 2),
                                  ('param3', 100, 3))
  def test_preprocess_fn_returns_correct_dataset_element_spec(
      self, sequence_length, num_out_of_vocab_buckets):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=32, max_elements=100)
    preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec,
        sequence_length=sequence_length,
        vocab=['one', 'must'],
        num_out_of_vocab_buckets=num_out_of_vocab_buckets)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        preprocessed_ds.element_spec,
        (tf.TensorSpec(shape=[None, sequence_length], dtype=tf.int64),
         tf.TensorSpec(shape=[None, sequence_length], dtype=tf.int64)))

  def test_preprocess_fn_returns_correct_sequence_with_1_out_of_vocab_bucket(
      self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=32, max_elements=100)
    preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec,
        sequence_length=6,
        vocab=['one', 'must'],
        num_out_of_vocab_buckets=1)

    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1
    self.assertAllEqual(
        self.evaluate(element[0]),
        tf.constant([[4, 1, 2, 3, 5, 0]], dtype=tf.int64))

  def test_preprocess_fn_returns_correct_sequence_with_3_out_of_vocab_buckets(
      self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=32, max_elements=100)
    preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec,
        sequence_length=6,
        vocab=['one', 'must'],
        num_out_of_vocab_buckets=3)
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
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=num_epochs, batch_size=batch_size)
    preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec, vocab=['A'], sequence_length=10)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        tf.cast(tf.math.ceil(num_epochs / batch_size), tf.int32))

  @parameterized.named_parameters(
      ('max_elements1', 1),
      ('max_elements3', 3),
      ('max_elements7', 7),
      ('max_elements11', 11),
      ('max_elements18', 18),
  )
  def test_ds_length_with_max_elements(self, max_elements):
    repeat_size = 10
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=repeat_size, batch_size=1, max_elements=max_elements)
    preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec, vocab=['A'])
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        min(repeat_size, max_elements))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
