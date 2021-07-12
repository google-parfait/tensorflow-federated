# Copyright 2020, The TensorFlow Federated Authors.
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
"""Tests for hierarchical_histogram."""

import math

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram as hihi
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts


def _create_hierarchical_histogram(arity, depth):
  """Utility function to create the hierarchical histogram."""

  def _shrink_hist(hist):
    return np.sum((np.reshape(hist, (-1, arity))), axis=1)

  hi_hist = [np.arange(arity**(depth - 1))]
  for _ in range(depth - 1, 0, -1):
    hi_hist = [_shrink_hist(hi_hist[0])] + hi_hist

  return tf.ragged.constant(hi_hist)


class HierarchicalHistogramTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('test_1', [1, 2, 3, 4], [1, 5], 4),)
  def test_discretized_histogram_counts(self, client_data, data_range,
                                        num_bins):
    histogram = hihi.discretized_histogram_counts(client_data, data_range,
                                                  num_bins)

    precision = (data_range[1] - data_range[0]) / num_bins
    reference_histogram = np.zeros(num_bins)

    for data_point in client_data:
      if data_point >= data_range[0] and data_point < data_range[1]:
        reference_histogram[math.floor(
            (data_point - data_range[0]) / precision)] += 1

    self.assertAllClose(histogram, reference_histogram)

  @parameterized.named_parameters(
      ('test_without_clipping', [[1, 2, 3, 4], [1, 1, 3, 3]], [1, 5], 4, 2, 10,
       [[8.], [4., 4.], [3., 1., 3., 1.]]),
      ('test_with_clipping',
       [np.ones(11).tolist(), (np.ones(12) * 2).tolist(),
        np.ones(5).tolist()], [1, 3], 2, 2, 10, [[25.], [15., 10.]]))
  def test_build_central_hierarchical_histogram_computation(
      self, client_data, data_range, num_bins, arity, max_records_per_user,
      reference_hi_hist):
    hihi_computation = hihi.build_central_hierarchical_histogram_computation(
        data_range=data_range,
        num_bins=num_bins,
        arity=arity,
        max_records_per_user=max_records_per_user,
        epsilon=0.,
        delta=0.,
    )
    hi_hist = hihi_computation(client_data)

    self.assertAllClose(hi_hist, reference_hi_hist)

  @parameterized.named_parameters(
      ('test_without_clipping', [[1, 2, 3, 4], [1, 1, 3, 3]], [1, 5], 4, 2, 10,
       [[8.], [4., 4.], [3., 1., 3., 1.]]),
      ('test_with_clipping',
       [np.ones(11).tolist(), (np.ones(12) * 2).tolist(),
        np.ones(5).tolist()], [1, 3], 2, 2, 10, [[25.], [15., 10.]]))
  def test_build_distributed_hierarchical_histogram_computation(
      self, client_data, data_range, num_bins, arity, max_records_per_user,
      reference_hi_hist):
    hihi_computation = hihi.build_distributed_hierarchical_histogram_computation(
        data_range=data_range,
        num_bins=num_bins,
        client_num_lower_bound=len(client_data),
        arity=arity,
        max_records_per_user=max_records_per_user,
        epsilon=0.,
        delta=0.,
    )
    hi_hist = hihi_computation(client_data)

    self.assertAllClose(hi_hist, reference_hi_hist)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
