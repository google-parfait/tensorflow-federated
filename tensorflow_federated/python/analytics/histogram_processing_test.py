# Copyright 2021, The TensorFlow Federated Authors.
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

from absl.testing import parameterized

import tensorflow as tf

from tensorflow_federated.python.analytics import histogram_processing
from tensorflow_federated.python.analytics import histogram_test_utils


class HistogramProcessingTest(histogram_test_utils.HistogramTest,
                              parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'tensor_inputs',
          tf.constant([b'a', b'b', b'c', b'd', b'e'], dtype=tf.string),
          tf.constant([1.0, 10.3, 6.0, 4.5, 2.1], dtype=tf.float32),
          tf.constant(4.5, dtype=tf.float32),
          {
              b'b': 10.3,
              b'c': 6.0,
              b'd': 4.5
          },
      ),
      (
          'python_inputs',
          [b'a', b'b', b'c', b'd', b'e'],
          [1.0, 10.3, 6.0, 4.5, 2.1],
          4.5,
          {
              b'b': 10.3,
              b'c': 6.0,
              b'd': 4.5
          },
      ),
      (
          'mixed_input_types',
          [b'a', b'b', b'c', b'd', b'e'],
          tf.constant([1.0, 10.3, 6.0, 4.5, 2.1], dtype=tf.float32),
          4.5,
          {
              b'b': 10.3,
              b'c': 6.0,
              b'd': 4.5
          },
      ),
      (
          'tensor_float64_inputs',
          tf.constant([b'a', b'b', b'c', b'd', b'e'], dtype=tf.string),
          tf.constant([1.0, 10.3, 6.0, 4.5, 2.1], dtype=tf.float64),
          tf.constant(4.5, dtype=tf.float64),
          {
              b'b': 10.3,
              b'c': 6.0,
              b'd': 4.5
          },
      ),
      (
          'empty_inputs',
          [],
          tf.constant([], dtype=tf.float32),
          0.0,
          {},
      ),
      (
          'inf_histogram_value',
          [b'a', b'b', b'c', b'd', b'e'],
          [1.0, float('inf'), 6.0, 4.5, 2.1],
          4.5,
          {
              b'b': float('inf'),
              b'c': 6.0,
              b'd': 4.5
          },
      ),
      (
          'inf_threshold',
          [b'a', b'b', b'c', b'd', b'e'],
          [1.0, 10.3, 6.0, 4.5, 2.1],
          float('inf'),
          {},
      ),
  )
  def test_threshold_histogram_returns_expected_values(self, histogram_keys,
                                                       histogram_values,
                                                       threshold,
                                                       expected_histogram):
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    self.assert_histograms_all_close(histogram_keys_thresholded_tf,
                                     histogram_values_thresholded_tf,
                                     expected_histogram.keys(),
                                     expected_histogram.values())

  def test_threshold_histogram_duplicate_keys(self):
    histogram_keys = [b'a', b'a', b'b', b'c', b'd', b'e']
    histogram_values = tf.constant([1.0, 5.0, 10.3, 6.0, 4.5, 2.1],
                                   dtype=tf.float32)
    threshold = 4.5
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)

    expected_histogram_keys = [b'a', b'b', b'c', b'd']
    expected_histogram_values = [5.0, 10.3, 6.0, 4.5]

    # The (key, value) pairs are correctly thresholded when there are duplicate
    # keys, but values for the same key are not automatically summed up.
    self.assertAllEqual(histogram_keys_thresholded_tf, expected_histogram_keys)
    self.assertAllClose(histogram_values_thresholded_tf,
                        expected_histogram_values)

  @parameterized.named_parameters(
      (
          'len_mismatch',
          tf.constant([b'a', b'b', b'c', b'd', b'e'], dtype=tf.string),
          tf.constant([1.0, 10.3, 6.0, 4.5, 2.1, 3, 6], dtype=tf.float32),
      ),
      (
          'dimension_mismatch',
          tf.constant([[b'a', b'b', b'c', b'd', b'e']], dtype=tf.string),
          tf.constant([1.0, 10.3, 6.0, 4.5, 2.1, 3], dtype=tf.float32),
      ),
  )
  def test_threshold_histogram_raise_value_error(self, histogram_keys,
                                                 histogram_values):
    threshold = tf.constant(4.5, dtype=tf.float32)
    with self.assertRaises(ValueError):
      _, _ = histogram_processing.threshold_histogram(histogram_keys,
                                                      histogram_values,
                                                      threshold)


if __name__ == '__main__':
  tf.test.main()
