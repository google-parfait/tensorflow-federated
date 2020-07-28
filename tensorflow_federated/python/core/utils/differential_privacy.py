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

import collections
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
from tensorflow_federated.python.core.utils import computation_utils

# TODO(b/140236959): Make the nomenclature consistent (b/w 'record' and 'value')
# in this library.

# Note if the functions here change, the documentation at
# https://github.com/tensorflow/federated/blob/master/docs/tff_for_research.md
# should be updated.


def build_dp_query(clip,
                   noise_multiplier,
                   expected_total_weight,
                   adaptive_clip_learning_rate=0,
                   target_unclipped_quantile=None,
                   clipped_count_budget_allocation=None,
                   expected_num_clients=None,
                   per_vector_clipping=False,
                   geometric_clip_update=True,
                   model=None):
  """Makes a `DPQuery` to estimate vector averages with differential privacy.

  Supports many of the types of query available in tensorflow_privacy, including
  nested ("per-vector") queries as described in
  https://arxiv.org/pdf/1812.06210.pdf, and quantile-based adaptive clipping as
  described in https://arxiv.org/abs/1905.03871.

  Args:
    clip: The query's L2 norm bound.
    noise_multiplier: The ratio of the (effective) noise stddev to the clip.
    expected_total_weight: The expected total weight of all clients, used as the
      denominator for the average computation.
    adaptive_clip_learning_rate: Learning rate for quantile-based adaptive
      clipping. If 0, fixed clipping is used. If `per_vector_clipping=True` and
      `geometric_clip_update=False`, the learning rate of each vector is
      proportional to that vector's initial clip.
    target_unclipped_quantile: Target unclipped quantile for adaptive clipping.
    clipped_count_budget_allocation: The fraction of privacy budget to use for
      estimating clipped counts.
    expected_num_clients: The expected number of clients for estimating clipped
      fractions.
    per_vector_clipping: If True, clip each weight tensor independently.
      Otherwise, global clipping is used. The clipping norm for each vector (or
      the initial clipping norm, in the case of adaptive clipping) is
      proportional to the sqrt of the vector dimensionality such that the root
      sum squared of the individual clips equals `clip`.
    geometric_clip_update: If True, use geometric updating of the clip.
    model: A `tff.learning.Model` to determine the structure of model weights.
      Required only if per_vector_clipping is True.

  Returns:
    A `DPQuery` suitable for use in a call to `build_dp_aggregate` to perform
      Federated Averaging with differential privacy.
  """
  py_typecheck.check_type(clip, numbers.Number, 'clip')
  py_typecheck.check_type(noise_multiplier, numbers.Number, 'noise_multiplier')
  py_typecheck.check_type(expected_total_weight, numbers.Number,
                          'expected_total_weight')

  if per_vector_clipping:
    # Note we need to keep the structure of vectors (not just the num_vectors)
    # to create the subqueries below, when per_vector_clipping is True.
    vectors = model.weights.trainable
    num_vectors = len(tf.nest.flatten(vectors))
  else:
    num_vectors = 1

  if adaptive_clip_learning_rate:
    py_typecheck.check_type(adaptive_clip_learning_rate, numbers.Number,
                            'adaptive_clip_learning_rate')
    py_typecheck.check_type(target_unclipped_quantile, numbers.Number,
                            'target_unclipped_quantile')
    py_typecheck.check_type(clipped_count_budget_allocation, numbers.Number,
                            'clipped_count_budget_allocation')
    py_typecheck.check_type(expected_num_clients, numbers.Number,
                            'expected_num_clients')
    p = clipped_count_budget_allocation
    clipped_count_stddev = 0.5 * noise_multiplier * (p / num_vectors)**(-0.5)
    noise_multiplier = noise_multiplier * ((1 - p) / num_vectors)**(-0.5)

  def make_single_vector_query(vector_clip):
    """Makes a `DPQuery` for a single vector."""
    if not adaptive_clip_learning_rate:
      return tensorflow_privacy.GaussianAverageQuery(
          l2_norm_clip=vector_clip,
          sum_stddev=vector_clip * noise_multiplier * num_vectors**0.5,
          denominator=expected_total_weight)
    else:
      # Without geometric updating, the update is c = c - lr * loss, so for
      # multiple vectors we set the learning rate to be on the same scale as the
      # initial clip. That way big vectors get big updates, small vectors
      # small updates. With geometric updating, the update is
      # c = c * exp(-lr * loss) so the learning rate should be independent of
      # the initial clip.
      if geometric_clip_update:
        learning_rate = adaptive_clip_learning_rate
      else:
        learning_rate = adaptive_clip_learning_rate * vector_clip / clip
      return tensorflow_privacy.QuantileAdaptiveClipAverageQuery(
          initial_l2_norm_clip=vector_clip,
          noise_multiplier=noise_multiplier,
          target_unclipped_quantile=target_unclipped_quantile,
          learning_rate=learning_rate,
          clipped_count_stddev=clipped_count_stddev,
          expected_num_records=expected_num_clients,
          geometric_update=geometric_clip_update,
          denominator=expected_total_weight)

  if per_vector_clipping:

    def dim(v):
      return math.exp(sum([math.log(d.value) for d in v.shape.dims]))

    dims = tf.nest.map_structure(dim, vectors)
    total_dim = sum(tf.nest.flatten(dims))
    clips = tf.nest.map_structure(lambda dim: clip * np.sqrt(dim / total_dim),
                                  dims)
    subqueries = tf.nest.map_structure(make_single_vector_query, clips)
    return tensorflow_privacy.NestedQuery(subqueries)
  else:
    return make_single_vector_query(clip)


# TODO(b/123092620): When fixed, should no longer need this method.
def _default_get_value_type_fn(value):
  return value.type_signature.member


# TODO(b/140236959): The value_type_fn is needed as part of determining the
# tensor type. Is there a way to infer this inline without requiring an explicit
# method be passed as argument here?  Also, if it is necessary, is there a
# better name than value_type_fn?
def build_dp_aggregate(query, value_type_fn=_default_get_value_type_fn):
  """Builds a stateful aggregator for tensorflow_privacy DPQueries.

  The returned `StatefulAggregateFn` can be called with any nested structure for
  the values being statefully aggregated. However, it's necessary to provide two
  functions as arguments which indicate the properties (the `tff.Type` and the
  `structure.Struct` conversion) of the nested structure that will
  be used. If using a `collections.OrderedDict` as the value's nested structure,
  the defaults for the arguments suffice.

  Args:
    query: A DPQuery to aggregate. For compatibility with tensorflow_federated,
      the global_state and sample_state of the query must be structures
      supported by tf.nest.
    value_type_fn: Python function that takes the value argument of next_fn and
      returns the value type. This will be used in determining the TensorSpecs
      that establish the initial sample state. If the value being aggregated is
      an `collections.OrderedDict`, the default for this argument can be used.
      This argument probably gets removed once b/123092620 is addressed (and the
      associated processing step gets replaced with a simple call to
      `value.type_signature.member`).

  Returns:
    A tuple of:
      - a `computation_utils.StatefulAggregateFn` that aggregates according to
          the query
      - the TFF type of the DP aggregator's global state
  """
  warnings.warn(
      'Deprecation warning: tff.utils.build_dp_aggregate() is deprecated, use '
      'tff.utils.build_dp_aggregate_process() instead.', DeprecationWarning)

  @computations.tf_computation
  def initialize_fn():
    return query.initial_global_state()

  def next_fn(global_state, value, weight=None):
    """Defines next_fn for StatefulAggregateFn."""
    # Weighted aggregation is not supported.
    # TODO(b/140236959): Add an assertion that weight is None here, so the
    # contract of this method is better established. Will likely cause some
    # downstream breaks.
    del weight

    #######################################
    # Define local tf_computations

    # TODO(b/129567727): Make most of these tf_computations polymorphic
    # so type manipulation isn't needed.

    global_state_type = initialize_fn.type_signature.result

    @computations.tf_computation(global_state_type)
    def derive_sample_params(global_state):
      return query.derive_sample_params(global_state)

    @computations.tf_computation(derive_sample_params.type_signature.result,
                                 value.type_signature.member)
    def preprocess_record(params, record):
      return query.preprocess_record(params, record)

    # TODO(b/123092620): We should have the expected container type here.
    value_type = value_type_fn(value)
    value_type = computation_types.to_type(value_type)

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
      result, new_global_state = query.get_noised_result(
          sample_state, global_state)
      return new_global_state, result

    #######################################
    # Orchestration logic

    sample_params = intrinsics.federated_map(derive_sample_params, global_state)
    client_sample_params = intrinsics.federated_broadcast(sample_params)
    preprocessed_record = intrinsics.federated_map(
        preprocess_record, (client_sample_params, value))
    agg_result = intrinsics.federated_aggregate(preprocessed_record, zero(),
                                                accumulate, merge, report)

    return intrinsics.federated_map(post_process, (agg_result, global_state))

  # TODO(b/140236959): Find a way to have this method return only one thing. The
  # best approach is probably to add (to StatefulAggregateFn) a property that
  # stores the type of the global state.
  aggregate_fn = computation_utils.StatefulAggregateFn(
      initialize_fn=initialize_fn, next_fn=next_fn)
  return (aggregate_fn, initialize_fn.type_signature.result)


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

    empty_metrics = intrinsics.federated_value((), placements.SERVER)

    return collections.OrderedDict(
        state=updated_state, result=result, measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=initial_state_comp, next_fn=next_fn)
