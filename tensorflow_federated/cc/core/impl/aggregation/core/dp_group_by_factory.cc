/*
 * Copyright 2024 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_group_by_factory.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_open_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"

namespace tensorflow_federated {
namespace aggregation {

StatusOr<std::unique_ptr<TensorAggregator>> DPGroupByFactory::Create(
    const Intrinsic& intrinsic) const {
  return CreateInternal(intrinsic, nullptr);
}

StatusOr<std::unique_ptr<TensorAggregator>> DPGroupByFactory::Deserialize(
    const Intrinsic& intrinsic, std::string serialized_state) const {
  GroupByAggregatorState aggregator_state;
  if (!aggregator_state.ParseFromString(serialized_state)) {
    return TFF_STATUS(INVALID_ARGUMENT) << "DPGroupByFactory::Deserialize: "
                                           "Failed to parse serialized state.";
  }
  return CreateInternal(intrinsic, &aggregator_state);
}

StatusOr<std::unique_ptr<TensorAggregator>> DPGroupByFactory::CreateInternal(
    const Intrinsic& intrinsic,
    const GroupByAggregatorState* aggregator_state) const {
  // Check if the intrinsic is well-formed.
  TFF_RETURN_IF_ERROR(GroupByFactory::CheckIntrinsic(intrinsic, kDPGroupByUri));

  // DPGroupByAggregator expects parameters
  constexpr int64_t kEpsilonIndex = 0;
  constexpr int64_t kDeltaIndex = 1;
  constexpr int64_t kL0Index = 2;
  constexpr int kNumInnerParameters = 3;

  // Ensure that the parameters list is valid and retrieve the values if so.
  if (intrinsic.parameters.size() != kNumInnerParameters) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Expected " << kNumInnerParameters
           << " parameters"
              " but got "
           << intrinsic.parameters.size() << " of them.";
  }

  // Epsilon must be a positive number
  if (internal::GetTypeKind(intrinsic.parameters[kEpsilonIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Epsilon must be numerical.";
  }
  double epsilon = intrinsic.parameters[kEpsilonIndex].CastToScalar<double>();
  if (epsilon <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Epsilon must be positive.";
  }
  double epsilon_per_agg = epsilon / intrinsic.nested_intrinsics.size();

  // Delta must be a number between 0 and 1
  if (internal::GetTypeKind(intrinsic.parameters[kDeltaIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Delta must be numerical.";
  }
  double delta = intrinsic.parameters[kDeltaIndex].CastToScalar<double>();
  if (delta <= 0 || delta >= 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Delta must lie between 0 and 1.";
  }
  double delta_per_agg = delta / intrinsic.nested_intrinsics.size();

  // L0 bound must be a positive number
  if (internal::GetTypeKind(intrinsic.parameters[kL0Index].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: L0 bound must be numerical.";
  }
  int64_t l0_bound = intrinsic.parameters[kL0Index].CastToScalar<int64_t>();
  if (l0_bound <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: L0 bound must be positive.";
  }
  // If no keys are given, scalar aggregation will occur. There is exactly one
  // "group" in that case.
  if (intrinsic.inputs.empty()) {
    l0_bound = 1;
  }

  // Currently, we only support nested sums.
  // The following check will be updated when this changes.
  for (const auto& intrinsic : intrinsic.nested_intrinsics) {
    if (intrinsic.uri != kDPSumUri) {
      return TFF_STATUS(UNIMPLEMENTED) << "DPGroupByFactory: Currently, only "
                                          "nested DP sums are supported.";
    }

    // Ensure that each nested intrinsic provides a positive Linfinity bound
    bool has_linfinity_bound = false;
    const Tensor& linfinity_tensor = intrinsic.parameters[kLinfinityIndex];
    DataType linfinity_dtype = linfinity_tensor.dtype();
    NUMERICAL_ONLY_DTYPE_CASES(
        linfinity_dtype, InputType,
        has_linfinity_bound = linfinity_tensor.CastToScalar<InputType>() > 0);
    if (!has_linfinity_bound) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: Each nested intrinsic must provide a "
                "positive Linfinity bound.";
    }
  }

  // Create nested aggregators.
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> nested_aggregators;
  TFF_ASSIGN_OR_RETURN(nested_aggregators, GroupByFactory::CreateAggregators(
                                               intrinsic, aggregator_state));

  // Create the DP key combiner, and only populate the key combiner with state
  // if there are keys.
  auto key_combiner = DPOpenDomainHistogram::CreateDPKeyCombiner(
      intrinsic.inputs, &intrinsic.outputs, l0_bound);
  if (aggregator_state != nullptr && key_combiner != nullptr) {
    TFF_RETURN_IF_ERROR(GroupByFactory::PopulateKeyCombinerFromState(
        *key_combiner, *aggregator_state));
  }

  int num_inputs = aggregator_state ? aggregator_state->num_inputs() : 0;

  // Use new rather than make_unique here because the factory function that uses
  // a non-public constructor can't use std::make_unique, and we don't want to
  // add a dependency on absl::WrapUnique.
  return std::unique_ptr<DPOpenDomainHistogram>(new DPOpenDomainHistogram(
      intrinsic.inputs, &intrinsic.outputs, &(intrinsic.nested_intrinsics),
      std::move(key_combiner), std::move(nested_aggregators), epsilon_per_agg,
      delta_per_agg, l0_bound, num_inputs));
}

REGISTER_AGGREGATOR_FACTORY(std::string(kDPGroupByUri), DPGroupByFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
