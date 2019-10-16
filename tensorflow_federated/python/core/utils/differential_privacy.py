# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers

import numpy as np
import tensorflow as tf
import tensorflow_privacy

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core import api as tff
from tensorflow_federated.python.core import framework as tff_framework
from tensorflow_federated.python.core.utils import computation_utils

# TODO(b/140236959): Make the nomenclature consistent (b/w 'record' and 'value')
# in this library.


def build_dp_query(clip,
                   noise_multiplier,
                   expected_total_weight,
                   adaptive_clip_learning_rate=0,
                   target_unclipped_quantile=None,
                   clipped_count_budget_allocation=None,
                   expected_num_clients=None,
                   use_per_vector=False,
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
      clipping. If 0, fixed clipping is used. If per-vector clipping is enabled,
      the learning rate of each vector is proportional to that vector's initial
      clip, such that the sum of all per-vector learning rates equals this.
    target_unclipped_quantile: Target unclipped quantile for adaptive clipping.
    clipped_count_budget_allocation: The fraction of privacy budget to use for
      estimating clipped counts.
    expected_num_clients: The expected number of clients for estimating clipped
      fractions.
    use_per_vector: If True, clip each weight tensor independently. Otherwise,
      global clipping is used. The clipping norm for each vector (or the initial
      clipping norm, in the case of adaptive clipping) is proportional to the
      sqrt of the vector dimensionality while the total bound still equals
      `clip`.
    model: A `tff.learning.Model` to determine the structure of model weights.
      Required only if use_per_vector is True.

  Returns:
    A `DPQuery` suitable for use in a call to `build_dp_aggregate` to perform
      Federated Averaging with differential privacy.
  """
  py_typecheck.check_type(clip, numbers.Number, 'clip')
  py_typecheck.check_type(noise_multiplier, numbers.Number, 'noise_multiplier')
  py_typecheck.check_type(expected_total_weight, numbers.Number,
                          'expected_total_weight')

  if use_per_vector:
    # Note we need to keep the structure of vectors (not just the num_vectors)
    # to create the subqueries below, when use_per_vector is True.
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
      return tensorflow_privacy.QuantileAdaptiveClipAverageQuery(
          initial_l2_norm_clip=vector_clip,
          noise_multiplier=noise_multiplier,
          target_unclipped_quantile=target_unclipped_quantile,
          learning_rate=adaptive_clip_learning_rate * vector_clip / clip,
          clipped_count_stddev=clipped_count_stddev,
          expected_num_records=expected_num_clients,
          denominator=expected_total_weight)

  if use_per_vector:

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
  value_type = value.type_signature.member
  if hasattr(value_type, '_asdict'):
    return value_type._asdict()
  return value_type


# TODO(b/123092620): When fixed, should no longer need this method.
def _default_from_anon_tuple_fn(record):
  if hasattr(record, '_asdict'):
    return record._asdict()
  return record


# TODO(b/140236959): The value_type_fn is needed as part of determining the
# tensor type. Is there a way to infer this inline without requiring an explicit
# method be passed as argument here?  Also, if it is necessary, is there a
# better name than value_type_fn?
def build_dp_aggregate(query,
                       value_type_fn=_default_get_value_type_fn,
                       from_anon_tuple_fn=_default_from_anon_tuple_fn):
  """Builds a stateful aggregator for tensorflow_privacy DPQueries.

  The returned StatefulAggregateFn can be called with any nested structure for
  the values being statefully aggregated. However, it's necessary to provide two
  functions as arguments which indicate the properties (the tff.Type and the
  AnonymousTuple conversion) of the nested structure that will be used. If using
  an OrderedDict as the value's nested structure, the defaults for the arguments
  suffice.

  Args:
    query: A DPQuery to aggregate. For compatibility with tensorflow_federated,
      the global_state and sample_state of the query must be structures
      supported by tf.nest.
    value_type_fn: Python function that takes the value argument of next_fn and
      returns the value type. This will be used in determining the TensorSpecs
      that establish the initial sample state. If the value being aggregated is
      an OrderedDict, the default for this argument can be used. This argument
      probably gets removed once b/123092620 is addressed (and the associated
      processing step gets replaced with a simple call to
      value.type_signature.member).
    from_anon_tuple_fn: Python function that takes a client record and converts
      it to the container type that it was in before passing through TFF. (Right
      now, TFF computation causes the client record to be changed into an
      AnonymousTuple, and this method corrects for that). If the value being
      aggregated is an OrderedDict, the default for this argument can be used.
      This argument likely goes away once b/123092620 is addressed. The default
      behavior assumes that the client record (before being converted to
      AnonymousTuple) was an OrderedDict containing a flat structure of Tensors
      (as it is if using the tff.learning APIs like
      tff.learning.build_federated_averaging_process).

  Returns:
    A tuple of:
      - a `computation_utils.StatefulAggregateFn` that aggregates according to
          the query
      - the TFF type of the DP aggregator's global state
  """

  @tff.tf_computation
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

    @tff.tf_computation(global_state_type)
    def derive_sample_params(global_state):
      return query.derive_sample_params(global_state)

    @tff.tf_computation(derive_sample_params.type_signature.result,
                        value.type_signature.member)
    def preprocess_record(params, record):
      # TODO(b/123092620): Once TFF passes the expected container type (instead
      # of AnonymousTuple), we shouldn't need this.
      record = from_anon_tuple_fn(record)

      return query.preprocess_record(params, record)

    # TODO(b/123092620): We should have the expected container type here.
    value_type = value_type_fn(value)

    tensor_specs = tff_framework.type_to_tf_tensor_specs(value_type)

    @tff.tf_computation
    def zero():
      return query.initial_sample_state(tensor_specs)

    sample_state_type = zero.type_signature.result

    @tff.tf_computation(sample_state_type,
                        preprocess_record.type_signature.result)
    def accumulate(sample_state, preprocessed_record):
      return query.accumulate_preprocessed_record(sample_state,
                                                  preprocessed_record)

    @tff.tf_computation(sample_state_type, sample_state_type)
    def merge(sample_state_1, sample_state_2):
      return query.merge_sample_states(sample_state_1, sample_state_2)

    @tff.tf_computation(merge.type_signature.result)
    def report(sample_state):
      return sample_state

    @tff.tf_computation(sample_state_type, global_state_type)
    def post_process(sample_state, global_state):
      result, new_global_state = query.get_noised_result(
          sample_state, global_state)
      return new_global_state, result

    #######################################
    # Orchestration logic

    sample_params = tff.federated_apply(derive_sample_params, global_state)
    client_sample_params = tff.federated_broadcast(sample_params)
    preprocessed_record = tff.federated_map(preprocess_record,
                                            (client_sample_params, value))
    agg_result = tff.federated_aggregate(preprocessed_record, zero(),
                                         accumulate, merge, report)

    return tff.federated_apply(post_process, (agg_result, global_state))

  # TODO(b/140236959): Find a way to have this method return only one thing. The
  # best approach is probably to add (to StatefulAggregateFn) a property that
  # stores the type of the global state.
  aggregate_fn = computation_utils.StatefulAggregateFn(
      initialize_fn=initialize_fn, next_fn=next_fn)
  return (aggregate_fn, initialize_fn.type_signature.result)
