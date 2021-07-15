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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram as hihi
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts


class HierarchicalHistogramTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('test_1', tf.data.Dataset.from_tensor_slices(
          [1., 2., 3., 4.]), [1, 5], 4, [1., 1., 1., 1.]),
      ('test_2', tf.data.Dataset.from_tensor_slices(
          [2., 7., 5., 3.]), [1, 8], 7, [0., 1., 1., 0., 1., 0., 1.]),
  )
  def test_discretized_histogram_counts(self, client_data, data_range, num_bins,
                                        reference_histogram):
    histogram = hihi._discretized_histogram_counts(client_data, data_range[0],
                                                   data_range[1], num_bins)

    self.assertAllClose(histogram, reference_histogram)

  @parameterized.named_parameters(
      ('test_data_range_error', [2, 1], 4),
      ('test_num_bins_error', [1, 2], 0),
  )
  def test_discretized_histogram_counts_raise(self, data_range, num_bins):
    client_data = tf.data.Dataset.from_tensor_slices(np.zeros(4, dtype=float))
    with self.assertRaises(ValueError):
      hihi._discretized_histogram_counts(client_data, data_range[0],
                                         data_range[1], num_bins)

  @parameterized.named_parameters(
      ('test_without_clipping', [[1., 2., 3., 4.], [1., 1., 3., 3.]], [1, 5], 4,
       2, 10, [[8.], [4., 4.], [3., 1., 3., 1.]]),
      ('test_with_clipping',
       [np.ones(11).tolist(), (np.ones(12) * 2).tolist(),
        np.ones(5).tolist()], [1, 3], 2, 2, 10, [[25.], [15., 10.]]),
      ('test_tf_dataset', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, 10, [[8.], [4., 4.], [3., 1., 3., 1.]]),
  )
  def test_build_central_hierarchical_histogram_computation(
      self, client_data, data_range, num_bins, arity, max_records_per_user,
      reference_hi_hist):
    hihi_computation = hihi.build_central_hierarchical_histogram_computation(
        lower_bound=data_range[0],
        upper_bound=data_range[1],
        num_bins=num_bins,
        arity=arity,
        max_records_per_user=max_records_per_user,
        epsilon=0.,
        delta=0.)
    hi_hist = hihi_computation(client_data)

    self.assertAllClose(hi_hist, reference_hi_hist)

  def test_build_central_hierarchical_histogram_computation_with_noise(self):

    hihi_computation = hihi.build_central_hierarchical_histogram_computation(
        lower_bound=0,
        upper_bound=4,
        num_bins=4,
        arity=2,
        max_records_per_user=10,
        epsilon=1.0,
        delta=1e-5)

    client_data = [[0.], [1.], [2.], [3.]]
    reference_flat_hi_hist = [4., 2., 2., 1., 1., 1., 1.]
    flat_hi_hist = hihi_computation(client_data).flat_values

    # The magic number 20. is an integer approximation of three-sigma.
    self.assertAllClose(flat_hi_hist, reference_flat_hi_hist, atol=20.)

  def test_build_central_hierarchical_histogram_computation_secure(self):

    hihi_computation = hihi.build_central_hierarchical_histogram_computation(
        lower_bound=0,
        upper_bound=4,
        num_bins=4,
        arity=2,
        max_records_per_user=10,
        epsilon=1.0,
        delta=1e-5,
        secure_sum=True)

    client_data = [[0.], [1.], [2.], [3.]]
    reference_flat_hi_hist = [4., 2., 2., 1., 1., 1., 1.]
    flat_hi_hist = hihi_computation(client_data).flat_values

    # The magic number 20. is an integer approximation of three-sigma.
    self.assertAllClose(flat_hi_hist, reference_flat_hi_hist, atol=20.)

  @parameterized.named_parameters(
      ('test_data_range_error', [5, 1], 4, 2, 10),
      ('test_num_bins_error', [1, 5], 0, 2, 10),
      ('test_arity_error', [1, 5], 4, 1, 10),
      ('test_max_records_per_user_error', [1, 5], 4, 2, 0),
  )
  def test_build_central_hierarchical_histogram_computation_raise(
      self, data_range, num_bins, arity, max_records_per_user):
    with self.assertRaises(ValueError):
      hihi.build_central_hierarchical_histogram_computation(
          lower_bound=data_range[0],
          upper_bound=data_range[1],
          num_bins=num_bins,
          arity=arity,
          max_records_per_user=max_records_per_user,
          epsilon=0.,
          delta=0.)


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
