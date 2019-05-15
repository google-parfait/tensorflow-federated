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

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple


def build_dp_aggregate(query):
  """Builds a statefeul aggregator for tensorflow_privacy DPQueries.

  Args:
    query: A DPQuery to aggregate.
  Returns:
    A tff.utils.StatefulAggregateFn that aggregates according to the query.
  """
  def next_fn(global_state, value, weight=None):
    """next_fn for stateful aggregator."""
    del weight  # unused.

    sample_params = query.derive_sample_params(global_state)
    client_sample_params = tff.federated_broadcast(sample_params)

    @tff.tf_computation(client_sample_params.type_signature.member,
                        value.type_signature.member)
    def preprocess_record(params, record):
      return query.preprocess_record(params, record)

    preprocessed_records = tff.federated_map(
        preprocess_record, (client_sample_params, value))

    preprocessed_record_type = preprocessed_records.type_signature.member

    if isinstance(preprocessed_records.type_signature,
                  anonymous_tuple.AnonymousTuple):
      tensor_specs = preprocessed_record_type._asdict()
    else:
      tensor_specs = preprocessed_record_type

    @tff.tf_computation
    def zero():
      return query.initial_sample_state(global_state, tensor_specs)

    sample_state_type = zero.type_signature.result

    @tff.tf_computation(sample_state_type, preprocessed_record_type)
    def accumulate(sample_state, preprocessed_record):
      return query.accumulate_preprocessed_record(sample_state,
                                                  preprocessed_record)

    @tff.tf_computation(sample_state_type, sample_state_type)
    def merge(sample_state_1, sample_state_2):
      return query.merge_sample_states(sample_state_1, sample_state_2)

    @tff.tf_computation(merge.type_signature.result)
    def report(sample_state):
      return sample_state

    agg_result = tff.federated_aggregate(
        preprocessed_records,
        zero(),
        accumulate,
        merge,
        report)

    @tff.tf_computation(merge.type_signature.result,
                        global_state.type_signature.member)
    def post_process(sample_state, global_state):
      result, new_global_state = query.get_noised_result(sample_state,
                                                         global_state)
      return new_global_state, result

    return tff.federated_apply(post_process, (agg_result, global_state))

  # pylint: disable=unnecessary-lambda
  return tff.utils.StatefulAggregateFn(
      initialize_fn=tff.tf_computation(lambda: query.initial_global_state()),
      next_fn=next_fn)
  # pylint: enable=unnecessary-lambda
