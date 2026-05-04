/*
 * Copyright 2026 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_util.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

constexpr int kMaxGroupsContributedIndex = 2;
constexpr std::array<std::pair<absl::string_view, int>, 6> kNameToIndex = {
    {{kEpsilonName, kEpsilonIndex},
     {kDeltaName, kDeltaIndex},
     {kMaxGroupsContributedName, kMaxGroupsContributedIndex},
     {kLinfName, kLinfinityIndex},
     {kL1Name, kL1Index},
     {kL2Name, kL2Index}}};

constexpr absl::string_view kL0EstimatedPrefix = "max_groups_contributed";
constexpr absl::string_view kL1EstimatedPrefix = "L1_";
constexpr absl::string_view kLinfEstimatedPrefix = "Linf_";
constexpr absl::string_view kEstimatedSuffix = "_estimated";

// Matches a string if has the specified prefix and the kEstimatedSuffix.
// The index is populated with the integer part of the string between the
// prefix and the suffix, or std::nullopt if there is no such integer part.
bool Match(absl::string_view prefix, absl::string_view str,
           std::optional<int>& index) {
  if (!absl::StartsWith(str, prefix)) {
    return false;
  }
  if (!absl::EndsWith(str, kEstimatedSuffix)) {
    return false;
  }
  int index_start = prefix.size();
  int index_end = str.size() - kEstimatedSuffix.size();
  if (index_start >= index_end) {
    return true;
  }
  int extracted_index;
  if (absl::SimpleAtoi(str.substr(index_start, index_end - index_start),
                       &extracted_index)) {
    index = extracted_index;
    return true;
  }
  return false;
}

// Returns a string that describes the expected matched parameter name.
std::string MatchDescription(absl::string_view prefix,
                             absl::string_view param = "") {
  return absl::StrCat(prefix, param, kEstimatedSuffix);
}

absl::Status MismatchedParameterError(absl::string_view function_name,
                                      absl::string_view name) {
  return absl::InvalidArgumentError(absl::StrCat(
      function_name, ": Parameter must match one of ",
      MatchDescription(kL0EstimatedPrefix), ", ",
      MatchDescription(kL1EstimatedPrefix, "<index>"), ", or ",
      MatchDescription(kLinfEstimatedPrefix, "<index>"), " but got ", name));
}

absl::Status ValidateIntrinsicUri(const Intrinsic& intrinsic) {
  if (intrinsic.uri != kDPGroupByUri) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported intrinsic uri for DP parameter population: ",
                     intrinsic.uri));
  }
  return absl::OkStatus();
}

absl::Status ValidateParameterMap(
    const absl::flat_hash_map<std::string, double>& parameters,
    int num_inner_intrinsics) {
  for (const auto& [name, value] : parameters) {
    if (value <= 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("ValidateParameterMap: Parameter values must be "
                          "positive but %s is associated with %f.",
                          name, value));
    }
    std::optional<int> index;
    if (Match(kL0EstimatedPrefix, name, index) && !index.has_value()) {
      continue;
    } else if (Match(kL1EstimatedPrefix, name, index) ||
               Match(kLinfEstimatedPrefix, name, index)) {
      if (index.has_value()) {
        if (*index < num_inner_intrinsics) {
          continue;
        }

        return absl::InvalidArgumentError(
            absl::StrFormat("ValidateParameterMap: Integer part of parameter "
                            "name must be in [0, %d) but got %d",
                            num_inner_intrinsics, *index));
      }
    } else {
      return MismatchedParameterError("ValidateParameterMap", name);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Tensor*> GetNamedParameter(Intrinsic& intrinsic,
                                          absl::string_view name) {
  // First, try the parameter by name.
  for (auto& parameter : intrinsic.parameters) {
    if (parameter.name() == name) {
      return &parameter;
    }
  }

  // If not found, try the parameter by index by looking at the kNameToIndex.
  for (const auto& [parameter_name, index] : kNameToIndex) {
    if (parameter_name == name) {
      // Check that the index is in bounds.
      if (index < intrinsic.parameters.size()) {
        // Return the parameter by index only if the name is unset; otherwise
        // the parameter should have been found already by name.
        auto parameter = &intrinsic.parameters[index];
        if (parameter->name().empty()) {
          return parameter;
        }
      }
      break;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "GetNamedParameter: Intrinsic parameter not found for name: ", name));
}

template <typename T>
absl::Status SetParameterIfUnset(Tensor& parameter, absl::string_view name,
                                 T unset_value, T new_value) {
  T current_value = parameter.CastToScalar<T>();
  if (current_value != unset_value) {
    return absl::InvalidArgumentError(
        absl::StrCat("PopulateDPParameters: Expected untuned ", name, " to be ",
                     unset_value, " but got ", current_value, " instead."));
  }
  parameter = Tensor(new_value, std::string(name));
  return absl::OkStatus();
}

absl::Status PopulateDPParameters(
    Intrinsic& intrinsic,
    const absl::flat_hash_map<std::string, double>& parameters) {
  // Validate the intrinsic and the parameters.
  TFF_RETURN_IF_ERROR(ValidateIntrinsicUri(intrinsic));
  TFF_RETURN_IF_ERROR(
      ValidateParameterMap(parameters, intrinsic.nested_intrinsics.size()));

  for (const auto& [name, value] : parameters) {
    std::optional<int> index;
    if (Match(kL0EstimatedPrefix, name, index) && !index.has_value()) {
      TFF_ASSIGN_OR_RETURN(
          Tensor * parameter,
          GetNamedParameter(intrinsic, kMaxGroupsContributedName));
      TFF_RETURN_IF_ERROR(SetParameterIfUnset(*parameter,
                                              kMaxGroupsContributedName, -1L,
                                              static_cast<int64_t>(value)));

    } else if (Match(kL1EstimatedPrefix, name, index) && index.has_value()) {
      TFF_ASSIGN_OR_RETURN(
          Tensor * parameter,
          GetNamedParameter(intrinsic.nested_intrinsics[*index], kL1Name));
      TFF_RETURN_IF_ERROR(
          SetParameterIfUnset(*parameter, kL1Name, -1.0, value));
    } else if (Match(kLinfEstimatedPrefix, name, index) && index.has_value()) {
      TFF_ASSIGN_OR_RETURN(
          Tensor * parameter,
          GetNamedParameter(intrinsic.nested_intrinsics[*index], kLinfName));
      TFF_RETURN_IF_ERROR(
          SetParameterIfUnset(*parameter, kLinfName, -1.0, value));
    } else {
      return MismatchedParameterError("PopulateDPParameters", name);
    }
  }

  return absl::OkStatus();
}

}  // namespace aggregation
}  // namespace tensorflow_federated
