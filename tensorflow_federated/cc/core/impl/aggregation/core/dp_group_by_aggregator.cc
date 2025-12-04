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

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
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

Status DPGroupByAggregator::ValidateInputs(
    const InputTensorList& tensors) const {
  TFF_RETURN_IF_ERROR(GroupByAggregator::ValidateInputs(tensors));
  // Check that each string tensor has no string that exceeds the max length.
  for (int i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->dtype() != DT_STRING) {
      continue;
    }
    for (int j = 0; j < tensors[i]->num_elements(); ++j) {
      if (tensors[i]->AsSpan<string_view>()[j].size() > max_string_length_) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "The maximum length of a string key is " << max_string_length_
               << " but got a string exceeding that length in tensor " << i;
      }
    }
  }
  return absl::OkStatus();
}

// Number of bytes to represent a number in `varint` format.
StatusOr<int64_t> CalculateVarintByteSize(int64_t value) {
  if (value <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "Value must be positive but is " << value;
  }
  // Determine how many bits are needed to represent `value`.
  double num_bits = std::floor(std::log2(value)) + 1;
  // To encode as a varint, those bits will be divided into chunks of 7. Each
  // chunk resides in a separate byte. The remaining bit in the byte indicates
  // if the next byte is part of the encoding.
  return static_cast<int64_t>(std::ceil(num_bits / 7.0));
}

// --Calculating the sensitivity of Tensors--
// The data stored in the tensor lives in a bytes field (see TensorProto and
// Tensor::ToProto). When converted to wire format, it has the LEN type. A LEN
// type contains (a) the byte-length of content, then (b) the content itself.
// ToProto creates content via EncodeContent, which has different behavior
// depending on the Tensor's data type.
StatusOr<int64_t> StringTensorSensitivity(int64_t max_groups_contributed,
                                          int64_t max_string_length) {
  // For a string tensor, first a stream of varints is written, where the ith
  // varint is the length of the ith string. Then the strings are fed in one
  // after the other.
  TFF_ASSIGN_OR_RETURN(int64_t bytes_to_encode_string_length,
                       CalculateVarintByteSize(max_string_length));
  int64_t bytes_of_content =
      // The number of bytes for the lengths:
      max_groups_contributed * bytes_to_encode_string_length
      // The number of bytes for the strings:
      + max_groups_contributed * max_string_length;

  // We have to account for (a) the byte-length of the content.
  TFF_ASSIGN_OR_RETURN(int64_t bytes_to_encode_content_length,
                       CalculateVarintByteSize(bytes_of_content));
  return bytes_of_content + bytes_to_encode_content_length;
}

StatusOr<int64_t> NumericalTensorSensitivity(int64_t max_groups_contributed,
                                             int64_t bytes_per_value) {
  // For a numerical tensor, the content consists entirely of the raw bytes
  // representing the numerical values.
  int64_t bytes_of_content = max_groups_contributed * bytes_per_value;

  // We have to account for (a) the byte-length of the content.
  TFF_ASSIGN_OR_RETURN(int64_t bytes_to_encode_content_length,
                       CalculateVarintByteSize(bytes_of_content));
  return bytes_of_content + bytes_to_encode_content_length;
}

StatusOr<int64_t> DPGroupByAggregator::CalculateSerializeSensitivity() {
  int64_t sensitivity = 0;
  // First calculate the sensitivity of the keys to one Accumulate call.
  for (DataType key_type : GroupByAggregator::key_combiner()->dtypes()) {
    int64_t tensor_sensitivity = 0;
    if (key_type == DT_STRING) {
      TFF_ASSIGN_OR_RETURN(
          tensor_sensitivity,
          StringTensorSensitivity(max_groups_contributed_, max_string_length_));
    } else if (key_type == DT_FLOAT || key_type == DT_INT32) {
      TFF_ASSIGN_OR_RETURN(tensor_sensitivity, NumericalTensorSensitivity(
                                                   max_groups_contributed_, 4));
    } else if (key_type == DT_INT64 || key_type == DT_DOUBLE ||
               key_type == DT_UINT64) {
      TFF_ASSIGN_OR_RETURN(tensor_sensitivity, NumericalTensorSensitivity(
                                                   max_groups_contributed_, 8));
    } else {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "Unsupported key type: " << key_type;
    }
    sensitivity += tensor_sensitivity;
  }
  // Then calculate the sensitivity of the state of the aggregations to one
  // Accumulate call.
  // The type of each state is `OneDimGroupingAggregatorState` but much like
  // `TensorProto` the actual data is in a `bytes` field. Moreover, the content
  // of those `bytes` consists of the raw bytes that represent the numbers (see
  // MutableVectorData::EncodeContent), which is precisely the same as the
  // `bytes` field in `TensorProto`.
  for (const Intrinsic& intrinsic : intrinsics()) {
    DataType aggregation_type = intrinsic.outputs[0].dtype();
    int64_t aggregation_sensitivity = 0;
    if (aggregation_type == DT_FLOAT || aggregation_type == DT_INT32) {
      TFF_ASSIGN_OR_RETURN(
          aggregation_sensitivity,
          NumericalTensorSensitivity(max_groups_contributed_, 4));
    } else if (aggregation_type == DT_INT64 || aggregation_type == DT_DOUBLE ||
               aggregation_type == DT_UINT64) {
      TFF_ASSIGN_OR_RETURN(
          aggregation_sensitivity,
          NumericalTensorSensitivity(max_groups_contributed_, 8));
    } else {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "Unsupported aggregation type: " << aggregation_type;
    }
    sensitivity += aggregation_sensitivity;
  }
  // Finally, we bound sensitivity of the state of contributors_to_groups.
  // This should be updated when we update the way we track contributors.
  // In the current implementation, contributors_to_groups is a list of integers
  // that represent the size of each group so when one person changes their
  // contributions, they insert at most max_groups_contributed_ groups.
  TFF_ASSIGN_OR_RETURN(int64_t bytes_to_encode_max_groups_contributed,
                       CalculateVarintByteSize(max_groups_contributed_));
  sensitivity +=
      max_groups_contributed_ + bytes_to_encode_max_groups_contributed;
  return sensitivity;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
