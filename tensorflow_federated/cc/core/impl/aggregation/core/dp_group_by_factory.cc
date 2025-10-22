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
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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

namespace {

struct DPParameters {
  double epsilon;
  double delta;
  int64_t max_groups_contributed;
};

Status ValidateDPParameters(double epsilon, double delta,
                            int64_t max_groups_contributed, bool open_domain) {
  if (epsilon <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Epsilon must be positive.";
  }

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
  if (open_domain && max_groups_contributed <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: For open-domain DP histograms, "
              "max_groups_contributed "
              "must "
              "be positive.";
  }
  return TFF_STATUS(OK);
}

StatusOr<int> FindDPParameterLocationByName(
    const Intrinsic& intrinsic,
    const absl::flat_hash_map<std::string, int>& parameter_name_to_index,
    absl::string_view name) {
  auto it = parameter_name_to_index.find(name);
  if (it == parameter_name_to_index.end()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: For all DP histograms, epsilon, delta, "
              "and "
              "max_groups_contributed must be provided, but "
           << name << " was not found amongst parameters.";
  }
  if (internal::GetTypeKind(intrinsic.parameters[it->second].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: " << name << " must be numerical.";
  }
  return it->second;
}

StatusOr<DPParameters> FindDPParameters(
    const Intrinsic& intrinsic,
    const absl::flat_hash_map<std::string, int>& parameter_name_to_index) {
  // Find the indices of the DP parameters. We can't return the parameters
  // directly because they have different types, or the tensor references
  // directly because StatusOr doesn't support references.
  TFF_ASSIGN_OR_RETURN(int epsilon_index,
                       FindDPParameterLocationByName(
                           intrinsic, parameter_name_to_index, kEpsilonName));
  TFF_ASSIGN_OR_RETURN(int delta_index,
                       FindDPParameterLocationByName(
                           intrinsic, parameter_name_to_index, kDeltaName));
  TFF_ASSIGN_OR_RETURN(
      int max_groups_contributed_index,
      FindDPParameterLocationByName(intrinsic, parameter_name_to_index,
                                    kMaxGroupsContributedName));

  double epsilon = intrinsic.parameters[epsilon_index].CastToScalar<double>();
  double delta = intrinsic.parameters[delta_index].CastToScalar<double>();
  int64_t max_groups_contributed =
      intrinsic.parameters[max_groups_contributed_index]
          .CastToScalar<int64_t>();

  // If no keys are given, scalar aggregation will occur. There is exactly
  // one "group" in that case.
  if (intrinsic.inputs.empty()) {
    max_groups_contributed = 1;
  }
  return DPParameters{epsilon, delta, max_groups_contributed};
}

// Ensure that the key_names tensor has the correct length and is composed of
// strings. Should only be called in the closed-domain case, i.e. if key_names
// is present in the parameters.
StatusOr<std::vector<std::string>> FindAndValidateKeyNames(
    const Intrinsic& intrinsic,
    const absl::flat_hash_map<std::string, int>& parameter_name_to_index) {
  int64_t num_keys = intrinsic.inputs.size();
  std::vector<std::string> key_names;
  // This function should only be called in the closed-domain case, which by
  // definition means that key_names is present in the parameters.
  int key_names_index = parameter_name_to_index.find("key_names")->second;
  key_names = intrinsic.parameters.at(key_names_index).ToStringVector();
  if (key_names.size() != num_keys) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: The number of key names provided ("
           << key_names.size()
           << ") does not match the number of input tensors provided ("
           << num_keys << ").";
  }
  DataType domain_type = intrinsic.parameters[key_names_index].dtype();
  if (domain_type != DT_STRING) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Key names should be of type string but "
              "were type "
           << DataType_Name(domain_type) << " instead.";
  }
  return key_names;
}

// Given a vector of key names this function finds those keys in the
// parameters of the intrinsic and validates that they are consecutive and of
// the correct types. If those conditions are met, it returns the index of the
// first key in the parameters. parameter_name_to_index is a map of parameter
// names to their indices in the parameters of the intrinsic.
StatusOr<TensorSpan> FindAndValidateKeys(
    const Intrinsic& intrinsic, absl::Span<const std::string> key_names,
    const absl::flat_hash_map<std::string, int>& parameter_name_to_index) {
  if (key_names.empty()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: No keys were provided to FindAndValidateKeys.";
  }
  auto it = parameter_name_to_index.find(key_names[0]);
  if (it == parameter_name_to_index.end()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: The key name " << key_names[0]
           << " was listed in the key_names tensor but was not found "
              "amongst parameters.";
  }
  int first_key_index = it->second;
  for (int i = 0; i < key_names.size(); ++i) {
    const std::string& key_name = key_names[i];
    int index = first_key_index + i;
    if (index >= intrinsic.parameters.size() ||
        intrinsic.parameters[index].name() != key_name) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: The key " << key_name
             << " was not found immediately after the previous key in the "
                "parameters. All keys must be listed in order amongst the "
                "parameters.";
    }
    DataType domain_type = intrinsic.parameters[index].dtype();
    DataType expected_type = intrinsic.inputs[i].dtype();
    if (domain_type != expected_type) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: Domain tensor for key " << key_name
             << " should have type " << DataType_Name(expected_type)
             << " but has type " << DataType_Name(domain_type) << " instead.";
    }
  }

  int num_keys = key_names.size();
  const Tensor* ptr = nullptr;
  if (num_keys > 0) {
    ptr = &(intrinsic.parameters[first_key_index]);
  }

  return TensorSpan(ptr, num_keys);
}

Status ValidateNestedIntrinsics(const Intrinsic& intrinsic, bool open_domain,
                                DPParameters dp_parameters) {
  double delta = dp_parameters.delta;
  int64_t max_groups_contributed = dp_parameters.max_groups_contributed;
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
      if ((!has_linfinity_bound || max_groups_contributed <= 0) &&
          !has_l1_bound && !has_l2_bound) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Closed-domain DP histograms require "
                  "either an L1 bound, an L2 bound, or both Linfinity and L0 "
                  "bounds.";
      }
      // If delta is 0, we will employ the Laplace mechanism (Gaussian requires
      // a positive delta). But the Laplace mechanism requires a positive and
      // finite L1 sensitivity.
      bool has_l1_sensitivity =
          has_l1_bound || (has_linfinity_bound && max_groups_contributed > 0);
      if (delta <= 0 && !has_l1_sensitivity) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Closed-domain DP histograms require "
                  "either a positive delta or one of the following: "
               << "(a) an L1 bound (b) an Linfinity bound and an L0 bound";
      }
    }
  }
  return TFF_STATUS(OK);
}

// Find the parameter in the intrinsic and cast it to a scalar type. If it is
// not present, return nullopt. If it is non-positive, return a bad Status.
template <typename T>
StatusOr<std::optional<T>> FindPositiveParameter(
    const Intrinsic& intrinsic,
    const absl::flat_hash_map<std::string, int>& parameter_name_to_index,
    absl::string_view name) {
  std::optional<T> parameter = std::nullopt;
  auto it = parameter_name_to_index.find(name);
  if (it != parameter_name_to_index.end()) {
    parameter = intrinsic.parameters[it->second].CastToScalar<T>();
  }
  if (parameter.has_value() && *parameter <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: " << name << " must be positive if given.";
  }
  return parameter;
}

}  // namespace

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

  // Build a map of parameter names to indices.
  absl::flat_hash_map<std::string, int> parameter_name_to_index;
  for (int i = 0; i < intrinsic.parameters.size(); ++i) {
    if (parameter_name_to_index.contains(intrinsic.parameters[i].name())) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: Duplicate parameter name: "
             << intrinsic.parameters[i].name();
    }
    parameter_name_to_index[intrinsic.parameters[i].name()] = i;
  }

  TFF_ASSIGN_OR_RETURN(
      std::optional<int64_t> min_contributors_to_group,
      FindPositiveParameter<int64_t>(intrinsic, parameter_name_to_index,
                                     kMinContributorsToGroupName));
  TFF_ASSIGN_OR_RETURN(
      std::optional<int64_t> max_string_length_opt,
      FindPositiveParameter<int64_t>(intrinsic, parameter_name_to_index,
                                     kMaxStringLengthName));
  int64_t max_string_length =
      max_string_length_opt.value_or(kDefaultMaxStringLength);

  bool open_domain = min_contributors_to_group.has_value() ||
                     !parameter_name_to_index.contains("key_names");
  int64_t num_keys = intrinsic.inputs.size();

  // For any DP histogram, ensure that we have required DP parameters.
  TFF_ASSIGN_OR_RETURN(DPParameters dp_parameters,
                       FindDPParameters(intrinsic, parameter_name_to_index));
  double epsilon = dp_parameters.epsilon;
  double delta = dp_parameters.delta;
  int64_t max_groups_contributed = dp_parameters.max_groups_contributed;
  TFF_RETURN_IF_ERROR(ValidateDPParameters(
      epsilon, delta, max_groups_contributed, open_domain));

  // For the closed-domain case we must find the key tensors to pass on. This is
  // guaranteed to be present by the definition of the open_domain variable.
  TensorSpan key_tensors;
  if (!open_domain) {
    TFF_ASSIGN_OR_RETURN(
        std::vector<std::string> key_names,
        FindAndValidateKeyNames(intrinsic, parameter_name_to_index));
    if (num_keys != 0) {
      TFF_ASSIGN_OR_RETURN(
          key_tensors,
          FindAndValidateKeys(intrinsic, key_names, parameter_name_to_index));
    }
  }

  // For any DP histogram, ensure that each inner aggregation is
  // well-specified.
  TFF_RETURN_IF_ERROR(
      ValidateNestedIntrinsics(intrinsic, open_domain, dp_parameters));

  // Create nested aggregators.
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> nested_aggregators;
  TFF_ASSIGN_OR_RETURN(nested_aggregators, GroupByFactory::CreateAggregators(
                                               intrinsic, aggregator_state));

  // Create the DP key combiner, and only populate the key combiner with state
  // if there are keys.
  auto key_combiner = DPOpenDomainHistogram::CreateDPKeyCombiner(
      intrinsic.inputs, &intrinsic.outputs, max_groups_contributed);
  if (aggregator_state != nullptr && key_combiner != nullptr) {
    TFF_RETURN_IF_ERROR(GroupByFactory::PopulateKeyCombinerFromState(
        *key_combiner, *aggregator_state));
  }

  int num_inputs = aggregator_state ? aggregator_state->num_inputs() : 0;

  if (open_domain) {
    std::vector<int> contributors_to_groups;
    if (aggregator_state != nullptr) {
      absl::c_copy(aggregator_state->counter_of_contributors(),
                   std::back_inserter(contributors_to_groups));
    }
    return DPOpenDomainHistogram::Create(
        intrinsic.inputs, &intrinsic.outputs, &(intrinsic.nested_intrinsics),
        std::move(key_combiner), std::move(nested_aggregators), num_inputs,
        epsilon, delta, max_groups_contributed, min_contributors_to_group,
        contributors_to_groups, max_string_length);
  }

  return std::unique_ptr<DPClosedDomainHistogram>(new DPClosedDomainHistogram(
      intrinsic.inputs, &intrinsic.outputs, &(intrinsic.nested_intrinsics),
      std::move(key_combiner), std::move(nested_aggregators), num_inputs,
      epsilon, delta, max_groups_contributed, key_tensors, max_string_length));
}

REGISTER_AGGREGATOR_FACTORY(std::string(kDPGroupByUri), DPGroupByFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
