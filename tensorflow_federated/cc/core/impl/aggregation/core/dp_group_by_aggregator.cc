/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_group_by_aggregator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {
DPGroupByAggregator::DPGroupByAggregator(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    int num_inputs, double epsilon, double delta,
    int64_t max_groups_contributed,
    std::optional<int> min_contributors_to_group,
    std::vector<int> contributors_to_groups, int max_string_length)
    : GroupByAggregator(input_key_specs, output_key_specs, intrinsics,
                        std::move(key_combiner), std::move(aggregators),
                        num_inputs, min_contributors_to_group,
                        contributors_to_groups),
      epsilon_(epsilon),
      delta_(delta),
      max_groups_contributed_(max_groups_contributed),
      max_string_length_(max_string_length),
      epsilon_per_agg_((epsilon < kEpsilonThreshold
                            ? epsilon / intrinsics->size()
                            : kEpsilonThreshold)),
      delta_per_agg_(delta / intrinsics->size()) {}

StatusOr<OutputTensorList> DPGroupByAggregator::Report() && {
  if (!CanReport()) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPGroupByAggregator::Report: the report goal isn't met";
  }
  TFF_RETURN_IF_ERROR(CheckValid());
  return NoisyReport();
}
}  // namespace aggregation
}  // namespace tensorflow_federated
