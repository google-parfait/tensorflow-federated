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
"""A set of utilities functions for histogram processing."""

import tensorflow as tf


def threshold_histogram(histogram_keys, histogram_values, threshold):
  """Thresholds a histogram by values.

  Args:
    histogram_keys: A rank-1 Tensor containing a list of strings representing
      the keys in the input histogram.
    histogram_values: A rank-1 Tensor containing a list of floats representing
      the values for each key in `histogram_keys`.
    threshold: A float tensor to threshold the input histogram. The (key, value)
      pairs where value is less than `threshold` are filtered out in the
      returned histogram.

  Returns:
    A tuple (histogram_keys_thresholded, histogram_values_thresholded) of the
    histogram after being thresholded.
  """
  histogram_mask = tf.math.greater_equal(histogram_values, threshold)
  histogram_keys_thresholded = tf.boolean_mask(histogram_keys, histogram_mask)
  histogram_values_thresholded = tf.boolean_mask(histogram_values,
                                                 histogram_mask)

  return histogram_keys_thresholded, histogram_values_thresholded
