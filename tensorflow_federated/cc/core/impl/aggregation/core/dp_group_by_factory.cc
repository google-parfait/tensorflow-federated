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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
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
  constexpr int kMinNumParameters = 3;

  // 0. Ensure that the parameters list has a chance of being well-formed.
  if (intrinsic.parameters.size() < kMinNumParameters) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Expected at least " << kMinNumParameters
           << " parameters"
              " but got "
           << intrinsic.parameters.size() << " of them.";
  }
  int64_t num_keys = intrinsic.inputs.size();

  // 1. For any DP histogram, ensure that we have required DP parameters.
  double epsilon = 0;
  bool epsilon_found = false;
  double delta = 0;
  bool delta_found = false;
  int64_t l0_bound = 0;
  bool l0_bound_found = false;
  // We need to know the index of the key_names tensor (if present).
  int key_names_index = 0;
  // We later set this to false if and only if the key_names tensor is found.
  bool open_domain = true;
  for (const auto& parameter_tensor : intrinsic.parameters) {
    if (parameter_tensor.name() == "epsilon") {
      // Epsilon must be a positive number
      if (internal::GetTypeKind(parameter_tensor.dtype()) !=
          internal::TypeKind::kNumeric) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Epsilon must be numerical.";
      }
      epsilon = parameter_tensor.CastToScalar<double>();
      if (epsilon <= 0) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Epsilon must be positive.";
      }
      epsilon_found = true;
    }
    if (parameter_tensor.name() == "delta") {
      // Delta must be a number
      if (internal::GetTypeKind(parameter_tensor.dtype()) !=
          internal::TypeKind::kNumeric) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Delta must be numerical.";
      }
      delta = parameter_tensor.CastToScalar<double>();
      delta_found = true;
    }
    if (parameter_tensor.name() == "l0_bound") {
      // L0 bound must be a number
      if (internal::GetTypeKind(parameter_tensor.dtype()) !=
          internal::TypeKind::kNumeric) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: L0 bound must be numerical.";
      }
      // If open-domain, L0 bound must be positive
      l0_bound = parameter_tensor.CastToScalar<int64_t>();
      l0_bound_found = true;
    }
    // The key_names tensor and domain tensors should be last (if present). They
    // are processed below so we can stop here.
    if (parameter_tensor.name() == "key_names") {
      open_domain = false;
      break;
    }
    key_names_index++;
  }

  if (!epsilon_found || !delta_found || !l0_bound_found) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: For all DP histograms, epsilon, delta, and L0 "
              "bound must be provided (before key_names).";
  }

  // These checks are outside the above loop as we need to know whether we
  // are in the closed-domain case.
  if (open_domain) {
    // If open-domain, delta must lie between 0 and 1.
    if (delta <= 0 || delta >= 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: For open-domain DP histograms, delta "
                "must "
                "lie between 0 and 1.";
    }
  } else {
    // Else, delta must be less than 1. A non-positive delta requires a
    // positive L1 bound.
    if (delta >= 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: For closed-domain DP histograms, delta "
                "must "
                "be less than 1.";
    }
  }
  if (open_domain && l0_bound <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: For open-domain DP histograms, L0 bound "
              "must "
              "be positive.";
  }
  // If no keys are given, scalar aggregation will occur. There is exactly
  // one "group" in that case.
  if (intrinsic.inputs.empty()) {
    l0_bound = 1;
  }

  auto num_nested_intrinsics = intrinsic.nested_intrinsics.size();
  double epsilon_per_agg =
      (epsilon < kEpsilonThreshold ? epsilon / num_nested_intrinsics
                                   : kEpsilonThreshold);
  double delta_per_agg = delta / num_nested_intrinsics;

  // 2. For the closed-domain case, ensure that we have the specification for
  // the domain of each of the keys and they are well-formed. Store the index of
  // the key_names tensor at the same time.
  if (!open_domain) {
    // The number of domain tensors should be one more than the number of
    // grouping keys. The extra one contains the names of the keys (as strings).
    int64_t num_domain_tensors = intrinsic.parameters.size() - key_names_index;
    if (num_keys + 1 != num_domain_tensors) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: Expected " << num_keys + 1
             << " domain tensors but got " << num_domain_tensors << " of them.";
    }

    std::vector<std::string> key_names;
    bool key_names_found = false;
    int names_checked = 0;
    for (const auto& parameter_tensor : intrinsic.parameters) {
      if (parameter_tensor.name() == "key_names") {
        DataType domain_type = parameter_tensor.dtype();
        if (domain_type != DT_STRING) {
          return TFF_STATUS(INVALID_ARGUMENT)
                 << "DPGroupByFactory: Key names should be of type string but "
                    "were type "
                 << DataType_Name(domain_type) << " instead.";
        }
        key_names = parameter_tensor.ToStringVector();
        key_names_found = true;
        if (key_names.size() != num_keys) {
          return TFF_STATUS(INVALID_ARGUMENT)
                 << "DPGroupByFactory: The number of key names provided ("
                 << key_names.size()
                 << ") does not match the number of input tensors provided ("
                 << num_keys << ").";
        }
        continue;
      }
      if (key_names_found && names_checked < num_keys) {
        if (parameter_tensor.name() != key_names[names_checked]) {
          return TFF_STATUS(INVALID_ARGUMENT)
                 << "DPGroupByFactory: The name of the " << names_checked
                 << "th key tensor provided (" << parameter_tensor.name()
                 << ") does not match the key name provided in the "
                    "key_names tensor ("
                 << key_names[names_checked] << ").";
        }
        names_checked++;
      }
      if (names_checked > num_keys) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: The number of domain tensors provided "
                  "(tensors after the key names tensor in parameters) exceeded "
                  "the number of key names provided ("
               << key_names.size() << ").";
        ;
      }
    }

    if (names_checked != num_keys) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: The number of keys provided ("
             << names_checked
             << ") is less than the number of key names provided ("
             << key_names.size() << ").";
    }

    std::vector<DataType> expected_types(num_keys);
    for (auto i = 0; i < num_keys; i++) {
      expected_types[i] = intrinsic.inputs[i].dtype();
    }

    for (int i = 0; i < num_keys; i++) {
      DataType domain_type =
          intrinsic.parameters[key_names_index + i + 1].dtype();
      if (domain_type != expected_types[i]) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Domain tensor for " << i << "th key "
               << key_names[i] << " should have type "
               << DataType_Name(expected_types[i]) << " but has type "
               << DataType_Name(domain_type) << " instead.";
      }
    }
  }

  // 3. For any DP histogram, ensure that each inner aggregation is
  // well-specified.
  // Currently, we only support nested sums.
  // The following check will be updated when this changes.
  for (const auto& intrinsic : intrinsic.nested_intrinsics) {
    if (intrinsic.uri != kDPSumUri) {
      return TFF_STATUS(UNIMPLEMENTED) << "DPGroupByFactory: Currently, only "
                                          "nested DP sums are supported.";
    }

    // Verify presence of all norm bounds
    if (intrinsic.parameters.size() != kNumDPSumParameters) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: Linfinity, L1, and L2 bounds are expected.";
    }

    // Verify that the norm bounds are in numerical Tensors
    for (const auto& parameter_tensor : intrinsic.parameters) {
      if (internal::GetTypeKind(parameter_tensor.dtype()) !=
          internal::TypeKind::kNumeric) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Norm bounds must be stored in"
                  " numerical Tensors.";
      }
    }

    const auto& linfinity_tensor = intrinsic.parameters[kLinfinityIndex];
    const double l1 = intrinsic.parameters[kL1Index].CastToScalar<double>();
    const double l2 = intrinsic.parameters[kL2Index].CastToScalar<double>();

    // Check if nested intrinsic provides a positive Linfinity bound
    bool has_linfinity_bound = false;
    DataType linfinity_dtype = linfinity_tensor.dtype();
    NUMERICAL_ONLY_DTYPE_CASES(
        linfinity_dtype, InputType,
        has_linfinity_bound = linfinity_tensor.CastToScalar<InputType>() > 0);
    if (open_domain) {
      if (!has_linfinity_bound) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: For open-domain DP histograms, each "
                  "nested "
                  "intrinsic must provide a positive Linfinity bound.";
      }
    } else {
      // Closed-domain histogram requires either an L1 bound, an L2 bound, or
      // both Linfinity and L0 bounds.
      bool has_l1_bound =
          l1 > 0 && l1 != std::numeric_limits<double>::infinity();
      bool has_l2_bound =
          l2 > 0 && l2 != std::numeric_limits<double>::infinity();
      if ((!has_linfinity_bound || l0_bound <= 0) && !has_l1_bound &&
          !has_l2_bound) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Closed-domain DP histograms require "
                  "either an L1 bound, an L2 bound, or both Linfinity and L0 "
                  "bounds.";
      }
      // If delta is 0, we will employ the Laplace mechanism (Gaussian requires
      // a positive delta). But the Laplace mechanism requires a positive and
      // finite L1 sensitivity.
      bool has_l1_sensitivity =
          has_l1_bound || (has_linfinity_bound && l0_bound > 0);
      if (delta <= 0 && !has_l1_sensitivity) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Closed-domain DP histograms require "
                  "either a positive delta or one of the following: "
               << "(a) an L1 bound (b) an Linfinity bound and an L0 bound";
      }
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

  if (open_domain) {
    // Use new rather than make_unique here because the factory function that
    // uses a non-public constructor can't use std::make_unique, and we don't
    // want to add a dependency on absl::WrapUnique.
    return std::unique_ptr<DPOpenDomainHistogram>(new DPOpenDomainHistogram(
        intrinsic.inputs, &intrinsic.outputs, &(intrinsic.nested_intrinsics),
        std::move(key_combiner), std::move(nested_aggregators), epsilon_per_agg,
        delta_per_agg, l0_bound, num_inputs));
  }

  // Closed-domain case. We create a data structure containing the domain
  // spec out of intrinsic.parameters
  const Tensor* ptr = &(intrinsic.parameters[key_names_index + 1]);
  TensorSpan domain_tensors(ptr, num_keys);

  return std::unique_ptr<DPClosedDomainHistogram>(new DPClosedDomainHistogram(
      intrinsic.inputs, &intrinsic.outputs, &(intrinsic.nested_intrinsics),
      std::move(key_combiner), std::move(nested_aggregators), epsilon_per_agg,
      delta_per_agg, l0_bound, domain_tensors, num_inputs));
}

REGISTER_AGGREGATOR_FACTORY(std::string(kDPGroupByUri), DPGroupByFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
