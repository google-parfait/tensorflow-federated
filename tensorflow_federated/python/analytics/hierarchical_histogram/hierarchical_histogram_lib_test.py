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

from typing import Tuple
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_lib as hihi
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

  def _get_hierarchical_histogram_results(
      self,
      client_data: tf.data.Dataset,
      lower_bound: float,
      upper_bound: float,
      num_bins: int,
      arity: int = 2,
      clip_mechanism: str = 'sub-sampling',
      max_records_per_user: int = 10,
      dp_mechanism: str = 'no-noise',
      noise_multiplier: float = 0.0,
      expected_clients_per_round: int = 10,
      bits: int = 22,
  ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
    """Runs the Hierarchical Histogram computation and returns the results.

    Runs the computation with both `build_hierarchical_histogram_computation`
    and `build_hierarchical_histogram_process`.

    Args:
      client_data: A tf.data.Dataset of the input client data.
      lower_bound: A `float` specifying the lower bound of the data range.
      upper_bound: A `float` specifying the upper bound of the data range.
      num_bins: The integer number of bins to compute.
      arity: The branching factor of the tree. Defaults to 2.
      clip_mechanism: A `str` representing the clipping mechanism. Currently
        supported mechanisms are
        - 'sub-sampling': (Default) Uniformly sample up to
          `max_records_per_user` records without replacement from the client
          dataset.
        - 'distinct': Uniquify client dataset and uniformly sample up to
          `max_records_per_user` records without replacement from it.
      max_records_per_user: An `int` representing the maximum of records each
        user can include in their local histogram. Defaults to 10.
      dp_mechanism: A `str` representing the differentially private mechanism to
        use. Currently supported mechanisms are
        - 'no-noise': (Default) Tree aggregation mechanism without noise.
        - 'central-gaussian': Tree aggregation with central Gaussian mechanism.
        - 'distributed-discrete-gaussian': Tree aggregation mechanism with the
          distributed discrete Gaussian mechanism in "The Distributed Discrete
          Gaussian Mechanism for Federated Learning with Secure Aggregation.
          Peter Kairouz, Ziyu Liu, Thomas Steinke".
       noise_multiplier: A `float` specifying the noise multiplier (central
         noise stddev / L2 clip norm) for model updates. Defaults to 0.0.
      expected_clients_per_round: An `int` specifying the lower bound on the
        expected number of clients. Only needed when `dp_mechanism` is
        'distributed-discrete-gaussian'. Defaults to 10.
      bits: A positive integer specifying the communication bit-width B (where
        2**B will be the field size for SecAgg operations). Only needed when
        `dp_mechanism` is 'distributed-discrete-gaussian'. Please read the below
        precautions carefully and set `bits` accordingly. Otherwise, unexpected
        overflow or accuracy degradation might happen. (1) Should be in the
        inclusive range [1, 22] to avoid overflow inside secure aggregation; (2)
        Should be at least as large as `log2(4*sqrt(expected_clients_per_round)
        * noise_multiplier * l2_norm_bound + expected_clients_per_round *
        max_records_per_user) + 1` to avoid accuracy degradation caused by
        frequent modular clipping; (3) If the number of clients exceed
        `expected_clients_per_round`, overflow might happen.

    Returns:
      A `Tuple` of two `tf.RaggedTenor`, which contain the hierarchical
      histograms computed by `build_hierarchical_histogram_computation` and
      `build_hierarchical_histogram_process`.
    """
    hihi_computation = hihi.build_hierarchical_histogram_computation(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_bins=num_bins,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=max_records_per_user,
        dp_mechanism=dp_mechanism,
        noise_multiplier=noise_multiplier,
        expected_clients_per_round=expected_clients_per_round,
        bits=bits)
    hihi_computation_result = hihi_computation(client_data)

    hihi_process = hihi.build_hierarchical_histogram_process(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_bins=num_bins,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=max_records_per_user,
        dp_mechanism=dp_mechanism,
        noise_multiplier=noise_multiplier,
        expected_clients_per_round=expected_clients_per_round,
        bits=bits)

    init_state = hihi_process.initialize()
    hihi_process_result, _ = hihi_process.next(init_state, client_data)

    return hihi_computation_result, hihi_process_result

  @parameterized.named_parameters(
      ('data_range_error', [2, 1], 1, 2, 'sub-sampling', 1, 'central-gaussian',
       0.1, 1, 1),
      ('num_bins_error', [1, 2], 0, 2, 'sub-sampling', 1, 'central-gaussian',
       0.1, 1, 1),
      ('arity_error', [1, 2], 1, 1, 'sub-sampling', 1, 'central-gaussian', 0.1,
       1, 1),
      ('clip_mechanism_error', [1, 2], 1, 2, 'invalid', 1, 'central-gaussian',
       0.1, 1, 1),
      ('max_records_per_user_error', [1, 2], 1, 2, 'sub-sampling', 0,
       'central-gaussian', 0.1, 1, 1),
      ('dp_mechanism_error', [1, 2
                             ], 1, 2, 'sub-sampling', 1, 'invalid', 0.1, 1, 1),
      ('noise_multiplier_error', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', -0.1, 1, 1),
      ('expected_clients_per_round_error', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', 0.1, 0, 1),
      ('bits_less_than_1', [1, 2], 1, 2, 'sub-sampling', 1, 'central-gaussian',
       0.1, 1, 0),
      ('bits_large_than_23', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', 0.1, 1, 23),
      ('bits_less_than_log_client_num', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', 0.1, 8, 2),
  )
  def test_raises_error_hh_computation(self, data_range, num_bins, arity,
                                       clip_mechanism, max_records_per_user,
                                       dp_mechanism, noise_multiplier,
                                       expected_clients_per_round, bits):
    with self.assertRaises(ValueError):
      hihi.build_hierarchical_histogram_computation(
          lower_bound=data_range[0],
          upper_bound=data_range[1],
          num_bins=num_bins,
          arity=arity,
          clip_mechanism=clip_mechanism,
          max_records_per_user=max_records_per_user,
          dp_mechanism=dp_mechanism,
          noise_multiplier=noise_multiplier,
          expected_clients_per_round=expected_clients_per_round,
          bits=bits)

  @parameterized.named_parameters(
      ('data_range_error', [2, 1], 1, 2, 'sub-sampling', 1, 'central-gaussian',
       0.1, 1, 1),
      ('num_bins_error', [1, 2], 0, 2, 'sub-sampling', 1, 'central-gaussian',
       0.1, 1, 1),
      ('arity_error', [1, 2], 1, 1, 'sub-sampling', 1, 'central-gaussian', 0.1,
       1, 1),
      ('clip_mechanism_error', [1, 2], 1, 2, 'invalid', 1, 'central-gaussian',
       0.1, 1, 1),
      ('max_records_per_user_error', [1, 2], 1, 2, 'sub-sampling', 0,
       'central-gaussian', 0.1, 1, 1),
      ('dp_mechanism_error', [1, 2
                             ], 1, 2, 'sub-sampling', 1, 'invalid', 0.1, 1, 1),
      ('noise_multiplier_error', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', -0.1, 1, 1),
      ('expected_clients_per_round_error', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', 0.1, 0, 1),
      ('bits_less_than_1', [1, 2], 1, 2, 'sub-sampling', 1, 'central-gaussian',
       0.1, 1, 0),
      ('bits_large_than_23', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', 0.1, 1, 23),
      ('bits_less_than_log_client_num', [1, 2], 1, 2, 'sub-sampling', 1,
       'central-gaussian', 0.1, 8, 2),
  )
  def test_raises_error_hh_process(self, data_range, num_bins, arity,
                                   clip_mechanism, max_records_per_user,
                                   dp_mechanism, noise_multiplier,
                                   expected_clients_per_round, bits):
    with self.assertRaises(ValueError):
      hihi.build_hierarchical_histogram_process(
          lower_bound=data_range[0],
          upper_bound=data_range[1],
          num_bins=num_bins,
          arity=arity,
          clip_mechanism=clip_mechanism,
          max_records_per_user=max_records_per_user,
          dp_mechanism=dp_mechanism,
          noise_multiplier=noise_multiplier,
          expected_clients_per_round=expected_clients_per_round,
          bits=bits)

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
    (hihi_computation_result,
     hihi_process_result) = self._get_hierarchical_histogram_results(
         client_data=client_data,
         lower_bound=data_range[0],
         upper_bound=data_range[1],
         num_bins=num_bins,
         arity=arity,
         max_records_per_user=4,
         dp_mechanism='no-noise')

    self.assertAllClose(hihi_computation_result, reference_hi_hist)
    self.assertAllClose(hihi_process_result, reference_hi_hist)

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
    (hihi_computation_result,
     hihi_process_result) = self._get_hierarchical_histogram_results(
         client_data=client_data,
         lower_bound=data_range[0],
         upper_bound=data_range[1],
         num_bins=num_bins,
         arity=arity,
         clip_mechanism=clip_mechanism,
         max_records_per_user=max_records_per_user,
         dp_mechanism='no-noise')

    for layer in range(hihi_computation_result.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hihi_computation_result[layer]),
          reference_layer_l1_norm)
    for layer in range(hihi_process_result.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hihi_process_result[layer]),
          reference_layer_l1_norm)

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
    (hihi_computation_result,
     hihi_process_result) = self._get_hierarchical_histogram_results(
         client_data=client_data,
         lower_bound=data_range[0],
         upper_bound=data_range[1],
         num_bins=num_bins,
         arity=arity,
         max_records_per_user=4,
         dp_mechanism='central-gaussian',
         noise_multiplier=noise_multiplier)

    # 300 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound and the privacy composition.
    self.assertAllClose(
        hihi_computation_result,
        reference_hi_hist,
        atol=300. * noise_multiplier)
    self.assertAllClose(
        hihi_process_result, reference_hi_hist, atol=300. * noise_multiplier)

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
    (hihi_computation_result,
     hihi_process_result) = self._get_hierarchical_histogram_results(
         client_data=client_data,
         lower_bound=data_range[0],
         upper_bound=data_range[1],
         num_bins=num_bins,
         arity=arity,
         clip_mechanism=clip_mechanism,
         max_records_per_user=max_records_per_user,
         dp_mechanism='central-gaussian')

    # 600 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound, the privacy composition and noise accumulation via sum.
    for layer in range(hihi_computation_result.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hihi_computation_result[layer]),
          reference_layer_l1_norm,
          atol=600. * noise_multiplier)
    for layer in range(hihi_process_result.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hihi_process_result[layer]),
          reference_layer_l1_norm,
          atol=600. * noise_multiplier)

  @parameterized.named_parameters(
      ('test_binary_1', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, [[8], [4, 4], [3, 1, 3, 1]], 5.0),
      ('test_binary_2', [
          tf.data.Dataset.from_tensor_slices([2., 2., 2., 2.]),
          tf.data.Dataset.from_tensor_slices([3., 3., 3., 3.])
      ], [1, 5], 4, 2, [[8], [4, 4], [0, 4, 4, 0]], 1.0),
      ('test_ternary_1', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, [[8], [3, 3, 2]], 5.0),
      ('test_ternary_2', [
          tf.data.Dataset.from_tensor_slices([2., 2., 2., 2.]),
          tf.data.Dataset.from_tensor_slices([3., 3., 3., 3.])
      ], [1, 4], 3, 3, [[8], [0, 4, 4]], 1.0),
  )
  def test_distributed_discrete_gaussian_hierarchical_histogram_wo_clip(
      self, client_data, data_range, num_bins, arity, reference_hi_hist,
      noise_multiplier):
    (hihi_computation_result,
     hihi_process_result) = self._get_hierarchical_histogram_results(
         client_data=client_data,
         lower_bound=data_range[0],
         upper_bound=data_range[1],
         num_bins=num_bins,
         arity=arity,
         max_records_per_user=4,
         dp_mechanism='distributed-discrete-gaussian',
         noise_multiplier=noise_multiplier)

    # 300 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound and the privacy composition.
    self.assertAllClose(
        hihi_computation_result,
        reference_hi_hist,
        atol=300. * noise_multiplier)
    self.assertAllClose(
        hihi_process_result, reference_hi_hist, atol=300. * noise_multiplier)

  @parameterized.named_parameters(
      ('test_binary_sub_sampling', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, 'sub-sampling', 3, 6, 1.),
      ('test_binary_distinct', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 4.]),
          tf.data.Dataset.from_tensor_slices([1., 1., 3., 3.])
      ], [1, 5], 4, 2, 'distinct', 3, 5, 5.),
      ('test_ternary_sub_sampling', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, 'sub-sampling', 3, 6, 1.),
      ('test_ternary_distinct', [
          tf.data.Dataset.from_tensor_slices([1., 2., 3., 1.]),
          tf.data.Dataset.from_tensor_slices([2., 3., 1., 2.])
      ], [1, 4], 3, 3, 'distinct', 3, 6, 5.),
  )
  def test_distributed_discrete_gaussian_hierarchical_histogram_w_clip(
      self, client_data, data_range, num_bins, arity, clip_mechanism,
      max_records_per_user, reference_layer_l1_norm, noise_multiplier):
    (hihi_computation_result,
     hihi_process_result) = self._get_hierarchical_histogram_results(
         client_data=client_data,
         lower_bound=data_range[0],
         upper_bound=data_range[1],
         num_bins=num_bins,
         arity=arity,
         clip_mechanism=clip_mechanism,
         max_records_per_user=max_records_per_user,
         dp_mechanism='distributed-discrete-gaussian')

    # 600 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound, the privacy composition and noise accumulation via sum.
    for layer in range(hihi_computation_result.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hihi_computation_result[layer]),
          reference_layer_l1_norm,
          atol=600. * noise_multiplier)
    for layer in range(hihi_process_result.shape[0]):
      self.assertAllClose(
          tf.math.reduce_sum(hihi_process_result[layer]),
          reference_layer_l1_norm,
          atol=600. * noise_multiplier)


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
