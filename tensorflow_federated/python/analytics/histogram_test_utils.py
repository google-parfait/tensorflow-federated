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
"""TestCase for histogram processing."""

import tensorflow as tf


def create_np_histogram(keys, values):
  """Create a numpy dictionary by the input keys and values.

  Args:
    keys: A 1D tensor of tf.string or list of strings containing the keys of the
      histogram.
    values: A 1D tensor or list of ints containing the values of the histogram.

  Returns:
    A numpy dictionary.
  """
  if tf.is_tensor(keys):
    keys = keys.numpy()
  if tf.is_tensor(values):
    values = values.numpy()

  return dict(zip(keys, values))


class HistogramTest(tf.test.TestCase):
  """TestCase for histogram processing."""

  def assert_histograms_all_close(self, x_keys, x_values, y_keys, y_values):
    """Assert two histograms have the same keys, and close values.

    Each histogram is given in the format of two lists: a list of keys
    and a list of values. The keys and values are paired by their indices. The
    order of keys and values could be different in two equivalent input
    histograms.

    Args:
      x_keys: A 1D tensor of tf.string or list of strings containing the keys of
        the second histogram.
      x_values: A 1D tensor or list of ints containing the values of the first
        histogram.
      y_keys: A 1D tensor of tf.string or list of strings containing the keys of
        the second histogram.
      y_values: A 1D tensor or list of ints containing the values of the second
        histogram.

    Raises:
      ValueError: If x and y has different keys, or the value difference for the
      same key is more than the default tolerance in assertAllClose.
    """

    histogram_x = create_np_histogram(x_keys, x_values)
    histogram_y = create_np_histogram(y_keys, y_values)

    self.assertEqual(histogram_x.keys(), histogram_y.keys())
    for key in histogram_x:
      self.assertAllClose(histogram_x[key], histogram_y[key])
