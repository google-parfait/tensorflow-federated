# Copyright 2020, Google LLC.
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
from tensorflow_federated.python.simulation.baselines.stackoverflow import tag_prediction_preprocessing

TEST_DATA = collections.OrderedDict(
    creation_date=(['unused date']),
    score=([tf.constant(0, dtype=tf.int64)]),
    tags=(['B']),
    title=(['C']),
    tokens=(['A']),
    type=(['unused type']),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class ToIDsFnTest(tf.test.TestCase):

  def test_word_tokens_to_ids_without_oov(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C', 'title': '', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [1 / 3, 1 / 3, 1 / 3])

  def test_word_tokens_to_ids_with_duplicates_without_oov(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C A A C B B B', 'title': '', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [1 / 3, 4 / 9, 2 / 9])

  def test_word_tokens_to_ids_with_oov(self):
    word_vocab = ['A', 'B']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C', 'title': '', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [1 / 2, 1 / 2])

  def test_word_tokens_to_ids_with_duplicates_and_oov(self):
    word_vocab = ['A', 'B']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C A C C A B', 'title': '', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [3 / 5, 2 / 5])

  def test_word_tokens_all_oov(self):
    word_vocab = ['A', 'B']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'C D E F G', 'title': '', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [0, 0])

  def test_tag_tokens_to_ids_without_oov(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': '', 'title': '', 'tags': 'D|E|F'}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[1]), [1, 1, 1])

  def test_tag_tokens_to_ids_with_oov(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': '', 'title': '', 'tags': 'D|E|F'}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[1]), [1, 1])

  def test_join_word_tokens_with_title(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = tag_prediction_preprocessing.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C', 'title': 'A B', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [2 / 5, 2 / 5, 1 / 5])


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  def test_preprocess_fn_with_empty_word_vocab_raises(self):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(ValueError, 'word_vocab must be non-empty'):
      tag_prediction_preprocessing.create_preprocess_fn(
          preprocess_spec, word_vocab=[], tag_vocab=['B'])

  def test_preprocess_fn_with_empty_tag_vocab_raises(self):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(ValueError, 'tag_vocab must be non-empty'):
      tag_prediction_preprocessing.create_preprocess_fn(
          preprocess_spec, word_vocab=['A'], tag_vocab=[])

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
    preprocess_fn = tag_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec, word_vocab=['A'], tag_vocab=['B'])
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        tf.cast(tf.math.ceil(num_epochs / batch_size), tf.int32))

  def test_preprocess_fn_returns_correct_element(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    word_vocab = ['A', 'B', 'C']
    word_vocab_size = len(word_vocab)
    tag_vocab = ['A', 'B']
    tag_vocab_size = len(tag_vocab)

    preprocess_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=1, shuffle_buffer_size=1)
    preprocess_fn = tag_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec, word_vocab=word_vocab, tag_vocab=tag_vocab)

    preprocessed_ds = preprocess_fn(ds)
    expected_element_x_spec_shape = (None, word_vocab_size)
    expected_element_y_spec_shape = (None, tag_vocab_size)
    self.assertEqual(
        preprocessed_ds.element_spec,
        (tf.TensorSpec(expected_element_x_spec_shape, dtype=tf.float32),
         tf.TensorSpec(expected_element_y_spec_shape, dtype=tf.float32)))

    element = next(iter(preprocessed_ds))
    expected_element_x = tf.constant([[0.5, 0.0, 0.5]])
    expected_element_y = tf.constant([[0.0, 1.0]])
    self.assertAllClose(
        element, (expected_element_x, expected_element_y), rtol=1e-6)

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
    preprocess_fn = tag_prediction_preprocessing.create_preprocess_fn(
        preprocess_spec, word_vocab=['A'], tag_vocab=['B'])
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        min(repeat_size, max_elements))


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
