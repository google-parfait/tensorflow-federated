# Copyright 2019, The TensorFlow Federated Authors.
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
"""Utilities for interop with tensorflow_privacy."""

import math
import numbers
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_privacy

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process

# TODO(b/140236959): Make the nomenclature consistent (b/w 'record' and 'value')
# in this library.

# Note if the functions here change, the documentation at
# https://github.com/tensorflow/federated/blob/master/docs/tff_for_research.md
# should be updated.


def _distribute_clip(clip, vectors):

  def dim(v):
    return math.exp(sum([math.log(d.value) for d in v.shape.dims]))

  dims = tf.nest.map_structure(dim, vectors)
  total_dim = sum(tf.nest.flatten(dims))
  return tf.nest.map_structure(lambda d: clip * np.sqrt(d / total_dim), dims)


def build_dp_query(clip,
                   noise_multiplier,
                   expected_total_weight,
                   adaptive_clip_learning_rate=0,
                   target_unclipped_quantile=None,
                   clipped_count_budget_allocation=None,
                   expected_clients_per_round=None,
                   geometric_clip_update=True):
  """Makes a `DPQuery` to estimate vector averages with differential privacy.

  Supports many of the types of query available in tensorflow_privacy, including
  nested ("per-vector") queries as described in
  https://arxiv.org/pdf/1812.06210.pdf, and quantile-based adaptive clipping as
  described in https://arxiv.org/abs/1905.03871.

  Args:
    clip: The query's L2 norm bound, or the initial clip if adaptive clipping
      is used.
    noise_multiplier: The ratio of the (effective) noise stddev to the clip.
    expected_total_weight: The expected total weight of all clients, used as the
      denominator for the average computation.
    adaptive_clip_learning_rate: Learning rate for quantile-based adaptive
      clipping. If 0, fixed clipping is used.
    target_unclipped_quantile: Target unclipped quantile for adaptive clipping.
    clipped_count_budget_allocation: The fraction of privacy budget to use for
      estimating clipped counts.
    expected_clients_per_round: The expected number of clients for estimating
      clipped fractions.
    geometric_clip_update: If True, use geometric updating of the clip.

  Returns:
    A `DPQuery` suitable for use in a call to `build_dp_aggregate` and
    `build_dp_aggregate_process` to perform Federated Averaging with
    differential privacy.
  """
  py_typecheck.check_type(clip, numbers.Number, 'clip')
  py_typecheck.check_type(noise_multiplier, numbers.Number, 'noise_multiplier')
  py_typecheck.check_type(expected_total_weight, numbers.Number,
                          'expected_total_weight')

  if adaptive_clip_learning_rate:
    py_typecheck.check_type(adaptive_clip_learning_rate, numbers.Number,
                            'adaptive_clip_learning_rate')
    py_typecheck.check_type(target_unclipped_quantile, numbers.Number,
                            'target_unclipped_quantile')
    py_typecheck.check_type(clipped_count_budget_allocation, numbers.Number,
                            'clipped_count_budget_allocation')
    py_typecheck.check_type(expected_clients_per_round, numbers.Number,
                            'expected_clients_per_round')
    p = clipped_count_budget_allocation
    nm = noise_multiplier
    vectors_noise_multiplier = nm * (1 - p)**(-0.5)
    clipped_count_noise_multiplier = nm * p**(-0.5)

    # Clipped count sensitivity is 0.5.
    clipped_count_stddev = 0.5 * clipped_count_noise_multiplier

    return tensorflow_privacy.QuantileAdaptiveClipAverageQuery(
        initial_l2_norm_clip=clip,
        noise_multiplier=vectors_noise_multiplier,
        target_unclipped_quantile=target_unclipped_quantile,
        learning_rate=adaptive_clip_learning_rate,
        clipped_count_stddev=clipped_count_stddev,
        expected_num_records=expected_clients_per_round,
        geometric_update=geometric_clip_update,
        denominator=expected_total_weight)
  else:
    if target_unclipped_quantile is not None:
      warnings.warn(
          'target_unclipped_quantile is specified but '
          'adaptive_clip_learning_rate is zero. No adaptive clipping will be '
          'performed. Use adaptive_clip_learning_rate > 0 if you want '
          'adaptive clipping.')
    if clipped_count_budget_allocation is not None:
      warnings.warn(
          'clipped_count_budget_allocation is specified but '
          'adaptive_clip_learning_rate is zero. No adaptive clipping will be '
          'performed. Use adaptive_clip_learning_rate > 0 if you want '
          'adaptive clipping.')
    return tensorflow_privacy.GaussianAverageQuery(
        l2_norm_clip=clip,
        sum_stddev=clip * noise_multiplier,
        denominator=expected_total_weight)


def build_dp_aggregate_process(value_type, query):
  """Builds a `MeasuredProcess` for tensorflow_privacy DPQueries.

  The returned `MeasuredProcess` processes values of type value_type which can
  be any nested structure of tensors. Note that client weighting is not
  supported for differential privacy so the `weight` argument to the resulting
  `MeasuredProcess` will be ignored.

  Args:
    value_type: The type of values to be aggregated by the `MeasuredProcess`.
      Can be a `tff.TensorType` or a nested structure of `tff.StructType`
      that bottoms out in `tff.TensorType`.
    query: A DPQuery to aggregate. For compatibility with tensorflow_federated,
      the global_state and sample_state of the query must be structures
      supported by tf.nest.

  Returns:
    A `MeasuredProcess` implementing differentially private aggregation using
    the supplied DPQuery. Note that client weighting is not
  supported for differential privacy so the `weight` argument to the resulting
  `MeasuredProcess` will be ignored.
  """
  py_typecheck.check_type(
      value_type, (computation_types.TensorType, computation_types.StructType))

  @computations.tf_computation
  def initial_state_fn():
    return query.initial_global_state()

  @computations.federated_computation()
  def initial_state_comp():
    return intrinsics.federated_eval(initial_state_fn, placements.SERVER)

  #######################################
  # Define local tf_computations

  global_state_type = initial_state_fn.type_signature.result

  @computations.tf_computation(global_state_type)
  def derive_sample_params(global_state):
    return query.derive_sample_params(global_state)

  @computations.tf_computation(derive_sample_params.type_signature.result,
                               value_type)
  def preprocess_record(params, record):
    return query.preprocess_record(params, record)

  tensor_specs = type_conversions.type_to_tf_tensor_specs(value_type)

  @computations.tf_computation
  def zero():
    return query.initial_sample_state(tensor_specs)

  sample_state_type = zero.type_signature.result

  @computations.tf_computation(sample_state_type,
                               preprocess_record.type_signature.result)
  def accumulate(sample_state, preprocessed_record):
    return query.accumulate_preprocessed_record(sample_state,
                                                preprocessed_record)

  @computations.tf_computation(sample_state_type, sample_state_type)
  def merge(sample_state_1, sample_state_2):
    return query.merge_sample_states(sample_state_1, sample_state_2)

  @computations.tf_computation(merge.type_signature.result)
  def report(sample_state):
    return sample_state

  @computations.tf_computation(sample_state_type, global_state_type)
  def post_process(sample_state, global_state):
    result, new_global_state = query.get_noised_result(sample_state,
                                                       global_state)
    return new_global_state, result

  @computations.tf_computation(global_state_type)
  def derive_metrics(global_state):
    return query.derive_metrics(global_state)

  @computations.federated_computation(
      initial_state_comp.type_signature.result,
      computation_types.FederatedType(value_type, placements.CLIENTS),
      computation_types.FederatedType(tf.float32, placements.CLIENTS))
  def next_fn(global_state, value, weight):
    """Defines next_fn for MeasuredProcess."""
    # Weighted aggregation is not supported.
    # TODO(b/140236959): Add an assertion that weight is None here, so the
    # contract of this method is better established. Will likely cause some
    # downstream breaks.
    del weight

    sample_params = intrinsics.federated_map(derive_sample_params, global_state)
    client_sample_params = intrinsics.federated_broadcast(sample_params)
    preprocessed_record = intrinsics.federated_map(
        preprocess_record, (client_sample_params, value))
    agg_result = intrinsics.federated_aggregate(preprocessed_record, zero(),
                                                accumulate, merge, report)

    updated_state, result = intrinsics.federated_map(post_process,
                                                     (agg_result, global_state))

    metrics = intrinsics.federated_map(derive_metrics, updated_state)

    return measured_process.MeasuredProcessOutput(
        state=updated_state, result=result, measurements=metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=initial_state_comp, next_fn=next_fn)
