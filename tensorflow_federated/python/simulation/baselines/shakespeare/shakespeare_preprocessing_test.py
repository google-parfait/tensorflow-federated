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
from tensorflow_federated.python.simulation.baselines.shakespeare import shakespeare_preprocessing


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class TokenizeFnTest(tf.test.TestCase):

  def test_tokenize_fn_returns_expected_elements(self):
    pad, _, bos, eos = shakespeare_preprocessing.get_special_tokens()
    to_tokens = shakespeare_preprocessing._build_tokenize_fn(split_length=5)
    tokens = self.evaluate(to_tokens({'snippets': tf.constant('abc')}))
    self.assertAllEqual(tokens, [bos, 64, 42, 21, eos])
    to_tokens = shakespeare_preprocessing._build_tokenize_fn(split_length=12)
    tokens = self.evaluate(to_tokens({'snippets': tf.constant('star wars')}))
    self.assertAllEqual(tokens,
                        [bos, 25, 5, 64, 46, 14, 26, 64, 46, 25, eos, pad])

  def test_tokenize_appends_eos_token(self):
    _, oov, bos, eos = shakespeare_preprocessing.get_special_tokens()
    to_tokens = shakespeare_preprocessing._build_tokenize_fn(split_length=5)
    tokens = to_tokens({'snippets': tf.constant('a\r~')})
    self.assertAllEqual(tokens, [bos, 64, 86, oov, eos])


class SplitTargetTest(tf.test.TestCase):

  def test_split_target(self):
    example = self.evaluate(
        shakespeare_preprocessing._split_target(tf.constant([[1, 2, 3]])))
    self.assertAllEqual(([[1, 2]], [[2, 3]]), example)
    example = self.evaluate(
        shakespeare_preprocessing._split_target(
            tf.constant([[1, 2, 3], [4, 5, 6]])))
    self.assertAllEqual((
        [[1, 2], [4, 5]],
        [[2, 3], [5, 6]],
    ), example)


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('zero_value', 0), ('negative_value1', -1),
                                  ('negative_value2', -2))
  def test_nonpositive_sequence_length_raises(self, sequence_length):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(ValueError,
                                'sequence_length must be a positive integer'):
      shakespeare_preprocessing.create_preprocess_fn(
          preprocess_spec, sequence_length=sequence_length)

  def test_preprocess_fn_produces_expected_outputs(self):
    pad, _, bos, eos = shakespeare_preprocessing.get_special_tokens()
    initial_ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(snippets=['a snippet', 'different snippet']))
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=2, shuffle_buffer_size=1)
    preprocess_fn = shakespeare_preprocessing.create_preprocess_fn(
        preprocess_spec, sequence_length=10)

    ds = preprocess_fn(initial_ds)
    expected_outputs = [
        # First batch.
        ([[bos, 64, 14, 25, 45, 66, 4, 4, 65, 5],
          [bos, 1, 66, 43, 43, 65, 46, 65, 45,
           5]], [[64, 14, 25, 45, 66, 4, 4, 65, 5, eos],
                 [1, 66, 43, 43, 65, 46, 65, 45, 5, 14]]),
        # Second batch.
        ([
            [25, 45, 66, 4, 4, 65, 5, eos, pad, pad],
            [bos, 64, 14, 25, 45, 66, 4, 4, 65, 5],
        ], [
            [45, 66, 4, 4, 65, 5, eos, pad, pad, pad],
            [64, 14, 25, 45, 66, 4, 4, 65, 5, eos],
        ]),
        # Third batch.
        ([[bos, 1, 66, 43, 43, 65, 46, 65, 45, 5],
          [25, 45, 66, 4, 4, 65, 5, eos, pad,
           pad]], [[1, 66, 43, 43, 65, 46, 65, 45, 5, 14],
                   [45, 66, 4, 4, 65, 5, eos, pad, pad, pad]]),
    ]
    for batch_num, actual in enumerate(ds):
      expected = expected_outputs.pop(0)
      self.assertAllEqual(
          actual,
          expected,
          msg='Batch {:d} not equal. Actual: {!s}\nExpected: {!s}'.format(
              batch_num, actual, expected))
    self.assertEmpty(
        expected_outputs,
        msg='Actual output contained fewer than three batches.')

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
    test_sequence = 'test_sequence'
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(snippets=['test_sequence']))
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=num_epochs, batch_size=batch_size)
    preprocess_fn = shakespeare_preprocessing.create_preprocess_fn(
        preprocess_spec, sequence_length=len(test_sequence) + 1)
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
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(snippets=['test_sequence'])).repeat(repeat_size)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=1, max_elements=max_elements)
    preprocess_fn = shakespeare_preprocessing.create_preprocess_fn(
        preprocess_spec)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        min(repeat_size, max_elements))


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
