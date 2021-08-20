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
"""Tests for hierarchical_histogram."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram as hihi
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts


class ClientWorkTest(test_case.TestCase, parameterized.TestCase):

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
  def test_raises_error(self, data_range, num_bins):
    client_data = tf.data.Dataset.from_tensor_slices(np.zeros(4, dtype=float))
    with self.assertRaises(ValueError):
      hihi._discretized_histogram_counts(client_data, data_range[0],
                                         data_range[1], num_bins)


class HierarchicalHistogramTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('data_range_error', [2, 1
                           ], 1, 2, 'sub-sampling', 1, 'central-gaussian', 0.1),
      ('num_bins_error', [1, 2
                         ], 0, 2, 'sub-sampling', 1, 'central-gaussian', 0.1),
      ('arity_error', [1, 2], 1, 1, 'sub-sampling', 1, 'central-gaussian', 0.1),
      ('clip_mechanism_error', [1, 2
                               ], 1, 2, 'invalid', 1, 'central-gaussian', 0.1),
      ('max_records_per_user_error', [1, 2], 1, 2, 'sub-sampling', 0,
       'central-gaussian', 0.1),
      ('dp_mechanism_error', [1, 2], 1, 2, 'sub-sampling', 1, 'invalid', 0.1),
      ('noise_multiplier_error', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', -0.1),
  )
  def test_raises_error(self, data_range, num_bins, arity, clip_mechanism,
                        max_records_per_user, dp_mechanism, noise_multiplier):
    with self.assertRaises(ValueError):
      hihi.build_hierarchical_histogram_computation(
          lower_bound=data_range[0],
          upper_bound=data_range[1],
          num_bins=num_bins,
          arity=arity,
          clip_mechanism=clip_mechanism,
          max_records_per_user=max_records_per_user,
          dp_mechanism=dp_mechanism,
          noise_multiplier=noise_multiplier)

  @parameterized.named_parameters(
      ('test_binary_1', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, [[8], [4, 4], [3, 1, 3, 1]]),
      ('test_binary_2', [
          tf.data.Dataset.from_tensor_slices([2., 2., 2., 2.]),
          tf.data.Dataset.from_tensor_slices([3., 3., 3., 3.])
      ], [1, 5], 4, 2, [[8], [4, 4], [0, 4, 4, 0]]),
      ('test_ternary_1', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, [[8], [3, 3, 2]]),
      ('test_ternary_2', [
          tf.data.Dataset.from_tensor_slices([2., 2., 2., 2.]),
          tf.data.Dataset.from_tensor_slices([3., 3., 3., 3.])
      ], [1, 4], 3, 3, [[8], [0, 4, 4]]),
  )
  def test_central_no_noise_hierarchical_histogram_wo_clip(
      self, client_data, data_range, num_bins, arity, reference_hi_hist):
    hihi_computation = hihi.build_hierarchical_histogram_computation(
        lower_bound=data_range[0],
        upper_bound=data_range[1],
        num_bins=num_bins,
        arity=arity,
        max_records_per_user=4,
        dp_mechanism='no-noise')
    hi_hist = hihi_computation(client_data)

    self.assertAllClose(hi_hist, reference_hi_hist)

  @parameterized.named_parameters(
      ('test_binary_sub_sampling', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, 'sub-sampling', 3, 6.),
      ('test_binary_distinct', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, 'distinct', 3, 5.),
      ('test_ternary_sub_sampling', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, 'sub-sampling', 3, 6.),
      ('test_ternary_distinct', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, 'distinct', 3, 6.),
  )
  def test_central_no_noise_hierarchical_histogram_w_clip(
      self, client_data, data_range, num_bins, arity, clip_mechanism,
      max_records_per_user, reference_layer_l1_norm):
    hihi_computation = hihi.build_hierarchical_histogram_computation(
        lower_bound=data_range[0],
        upper_bound=data_range[1],
        num_bins=num_bins,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=max_records_per_user,
        dp_mechanism='no-noise')
    hi_hist = hihi_computation(client_data)

    for layer in range(hi_hist.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hi_hist[layer]), reference_layer_l1_norm)

  @parameterized.named_parameters(
      ('test_binary_1', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, [[8.], [4., 4.], [3., 1., 3., 1.]], 5.0),
      ('test_binary_2', [
          tf.data.Dataset.from_tensor_slices([2., 2., 2., 2.]),
          tf.data.Dataset.from_tensor_slices([3., 3., 3., 3.])
      ], [1, 5], 4, 2, [[8.], [4., 4.], [0., 4., 4., 0.]], 1.0),
      ('test_ternary_1', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, [[8.], [3., 3., 2.]], 5.0),
      ('test_ternary_2', [
          tf.data.Dataset.from_tensor_slices([2., 2., 2., 2.]),
          tf.data.Dataset.from_tensor_slices([3., 3., 3., 3.])
      ], [1, 4], 3, 3, [[8.], [0., 4., 4.]], 1.0),
  )
  def test_central_gaussian_hierarchical_histogram_wo_clip(
      self, client_data, data_range, num_bins, arity, reference_hi_hist,
      noise_multiplier):
    hihi_computation = hihi.build_hierarchical_histogram_computation(
        lower_bound=data_range[0],
        upper_bound=data_range[1],
        num_bins=num_bins,
        arity=arity,
        max_records_per_user=4,
        dp_mechanism='central-gaussian',
        noise_multiplier=noise_multiplier)
    hi_hist = hihi_computation(client_data)

    # 300 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound and the privacy composition.
    self.assertAllClose(
        hi_hist, reference_hi_hist, atol=300. * noise_multiplier)

  @parameterized.named_parameters(
      ('test_binary_sub_sampling', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, 'sub-sampling', 3, 6., 1.),
      ('test_binary_distinct', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, 'distinct', 3, 5., 5.),
      ('test_ternary_sub_sampling', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, 'sub-sampling', 3, 6., 1.),
      ('test_ternary_distinct', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, 'distinct', 3, 6., 5.),
  )
  def test_central_gaussian_hierarchical_histogram_w_clip(
      self, client_data, data_range, num_bins, arity, clip_mechanism,
      max_records_per_user, reference_layer_l1_norm, noise_multiplier):
    hihi_computation = hihi.build_hierarchical_histogram_computation(
        lower_bound=data_range[0],
        upper_bound=data_range[1],
        num_bins=num_bins,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=max_records_per_user,
        dp_mechanism='central-gaussian')
    hi_hist = hihi_computation(client_data)

    # 600 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound, the privacy composition and noise accumulation via sum.
    for layer in range(hi_hist.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hi_hist[layer]),
          reference_layer_l1_norm,
          atol=600. * noise_multiplier)


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
