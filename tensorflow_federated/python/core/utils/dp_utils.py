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
from __future__ import google_type_annotations
from __future__ import print_function

from tensorflow_federated.python.core import api as tff
from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core.utils import computation_utils


def build_dp_aggregate(query):
  """Builds a statefeul aggregator for tensorflow_privacy DPQueries.

  Args:
    query: A DPQuery to aggregate. For compatibility with tensorflow_federated,
      the global_state and sample_state of the query must be structures
      supported by tf.nest.

  Returns:
    A tff.utils.StatefulAggregateFn that aggregates according to the query.
  """

  @tff.tf_computation
  def initialize_fn():
    return query.initial_global_state()

  def next_fn(global_state, value, weight=None):
    """Defines next_fn for StatefulAggregateFn."""
    # Weighted aggregation is not supported.
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
      if hasattr(record, '_asdict'):
        record = record._asdict()
      return query.preprocess_record(params, record)

    value_type = value.type_signature.member
    # TODO(b/123092620): We should have the expected container type here.
    if hasattr(value_type, '_asdict'):
      value_type = value_type._asdict()
    tensor_specs = framework.type_to_tf_tensor_specs(value_type)

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

  return computation_utils.StatefulAggregateFn(
      initialize_fn=initialize_fn, next_fn=next_fn)
