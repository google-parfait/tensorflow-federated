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
from tensorflow_federated.python.simulation.baselines.emnist import emnist_preprocessing


NUM_ONLY_DIGITS_CLIENTS = 3383
TOTAL_NUM_CLIENTS = 3400


TEST_DATA = collections.OrderedDict(
    pixels=([tf.zeros((28, 28), dtype=tf.float32)]),
    label=([tf.constant(0, dtype=tf.int32)]),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  def test_preprocess_fn_with_negative_epochs_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'num_epochs must be a positive integer'):
      emnist_preprocessing.create_preprocess_fn(
          num_epochs=-2, batch_size=1, shuffle_buffer_size=1)

  def test_non_supported_task_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        'emnist_task must be one of "digit_recognition" or "autoencoder".'):
      emnist_preprocessing.create_preprocess_fn(
          num_epochs=1,
          batch_size=1,
          shuffle_buffer_size=1,
          emnist_task='bad_task')

  @parameterized.named_parameters(
      ('paream1', 1, 1),
      ('param2', 4, 2),
      ('param3', 9, 3),
      ('param4', 12, 1),
      ('param5', 5, 3),
      ('param6', 7, 2),
  )
  def test_ds_length_is_ceil_num_epochs_over_batch_size(self, num_epochs,
                                                        batch_size):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_preprocessing.create_preprocess_fn(
        num_epochs=num_epochs, batch_size=batch_size, shuffle_buffer_size=1)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        tf.cast(tf.math.ceil(num_epochs / batch_size), tf.int32))

  def test_digit_recognition_preprocess_returns_correct_elements(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_preprocessing.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        emnist_task='digit_recognition')
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.int32)))

    element = next(iter(preprocessed_ds))
    expected_element = (tf.zeros(shape=(1, 28, 28, 1), dtype=tf.float32),
                        tf.zeros(shape=(1,), dtype=tf.int32))
    self.assertAllClose(self.evaluate(element), expected_element)

  def test_autoencoder_preprocess_returns_correct_elements(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_preprocessing.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        emnist_task='autoencoder')
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, 784), dtype=tf.float32)))

    element = next(iter(preprocessed_ds))
    expected_element = (tf.ones(shape=(1, 784), dtype=tf.float32),
                        tf.ones(shape=(1, 784), dtype=tf.float32))
    self.assertAllClose(self.evaluate(element), expected_element)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
