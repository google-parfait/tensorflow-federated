/*
 * Copyright 2023 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"

namespace tensorflow_federated {
namespace aggregation {

namespace {

// In the configuration, FedSQL intrinsics will have input and output tensor
// shapes of {} to indicate that the output is scalar. But in practice in the
// implementation, the input and output tensors are both of unknown size- the
// number of values in the input tensors can vary in each upload, and the number
// of outputs depends on the number of unique grouping keys in the
// fedsql_group_by that wraps the FedSQL intrinsic. Thus, transform the
// TensorShapes in the input and output TensorSpecs from {} to {-1} so that code
// that validates the input and output tensors match the specs will be able to
// properly account for the fact that the dimension is unknown.
// TODO: b/285201184 - Revisit the design decision to perform this
// transformation in this class; as it requires this class to have special
// knowledge about FedSQL intrinsics.
void TransformFedSqlSpecs(Intrinsic& intrinsic) {
  for (auto& input_spec : intrinsic.inputs) {
    if (input_spec.shape() == TensorShape{}) {
      input_spec =
          TensorSpec(input_spec.name(), input_spec.dtype(), TensorShape{-1});
    }
  }
  for (auto& output_spec : intrinsic.outputs) {
    if (output_spec.shape() == TensorShape{}) {
      output_spec =
          TensorSpec(output_spec.name(), output_spec.dtype(), TensorShape{-1});
    }
  }
}

Status IntrinsicConfigArgumentError(
    const Configuration::IntrinsicConfig& intrinsic_config,
    string_view error_message) {
  return TFF_STATUS(INVALID_ARGUMENT)
         << absl::StrCat("IntrinsicConfig: ", error_message, ":\n",
                         intrinsic_config.DebugString());
}

// Parses an IntrinsicConfig proto into an Intrinsic struct to
// represent the aggregation intrinsic independently from the proto.
StatusOr<Intrinsic> ParseFromConfig(
    const Configuration::IntrinsicConfig& intrinsic_config);

StatusOr<std::vector<Intrinsic>> ParseFromConfig(
    string_view parent_uri,
    const google::protobuf::RepeatedPtrField<Configuration::IntrinsicConfig>&
        intrinsic_configs) {
  std::vector<Intrinsic> intrinsics;
  std::vector<Intrinsic> wrapped_fedsql_intrinsics;
  bool need_fedsql_wrapper = false;
  // The implementation of all FedSQL intrinsics relies on the presence of a
  // wrapping fedsql_group_by intrinsic. In the case that no grouping by keys
  // should be performed and a single scalar output should be produced by each
  // FedSQL intrinsic, a fedsql_group_by with empty input and output tensors is
  // added to wrap the FedSQL intrinsic. For efficiency, use the same
  // fedsql_group_by to wrap multiple fedsql intrinsics rather than wrapping
  // each with a separate fedsql_group_by.
  //
  // For cases that involve DP, a wrapping fedsql_dp_group_by intrinsic is
  // guaranteed to always be already present because it holds required DP
  // parameters.
  //
  // TODO: b/285201184 - Revisit the design decision to perform this
  // transformation in this location; as it requires this class to have special
  // knowledge about FedSQL intrinsics.
  if (parent_uri != kGroupByUri && parent_uri != kDPGroupByUri) {
    need_fedsql_wrapper = true;
  }
  for (const Configuration::IntrinsicConfig& intrinsic_config :
       intrinsic_configs) {
    TFF_ASSIGN_OR_RETURN(Intrinsic intrinsic,
                         ParseFromConfig(intrinsic_config));
    // Disable lint checks recommending use of absl::StrContains() here, because
    // we don't want to take a dependency on absl libraries in the aggregation
    // core implementations.
    bool is_fedsql = intrinsic.uri.find(kFedSqlPrefix)  // NOLINT
                     != std::string::npos;              // NOLINT
    if (is_fedsql) {
      TransformFedSqlSpecs(intrinsic);
    }
    if (is_fedsql && need_fedsql_wrapper) {
      if (intrinsic.uri == kDPSumUri) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "Inner DP SQL intrinsics must already be wrapped with an "
                  "outer DP SQL intrinsic.";
      }
      wrapped_fedsql_intrinsics.push_back(std::move(intrinsic));
    } else {
      intrinsics.push_back(std::move(intrinsic));
    }
  }
  if (!wrapped_fedsql_intrinsics.empty()) {
    intrinsics.push_back(Intrinsic{std::string(kGroupByUri),
                                   {},
                                   {},
                                   {},
                                   std::move(wrapped_fedsql_intrinsics)});
  }
  return intrinsics;
}

StatusOr<Intrinsic> ParseFromConfig(
    const Configuration::IntrinsicConfig& intrinsic_config) {
  std::vector<TensorSpec> input_tensor_specs;
  std::vector<Tensor> params;
  for (const auto& intrinsic_arg : intrinsic_config.intrinsic_args()) {
    switch (intrinsic_arg.arg_case()) {
      case Configuration_IntrinsicConfig_IntrinsicArg::kInputTensor: {
        TFF_ASSIGN_OR_RETURN(
            TensorSpec input_spec,
            TensorSpec::FromProto(intrinsic_arg.input_tensor()));
        input_tensor_specs.push_back(std::move(input_spec));
        break;
      }
      case Configuration_IntrinsicConfig_IntrinsicArg::kParameter: {
        TFF_ASSIGN_OR_RETURN(Tensor param,
                             Tensor::FromProto(intrinsic_arg.parameter()));
        params.push_back(std::move(param));
        break;
      }
      case Configuration_IntrinsicConfig_IntrinsicArg::ARG_NOT_SET: {
        return IntrinsicConfigArgumentError(
            intrinsic_config, "All intrinsic args must have an arg value.");
      }
    }
  }
  std::vector<TensorSpec> output_tensor_specs;
  for (const auto& output_tensor_spec_proto :
       intrinsic_config.output_tensors()) {
    TFF_ASSIGN_OR_RETURN(TensorSpec output_spec,
                         TensorSpec::FromProto(output_tensor_spec_proto));
    output_tensor_specs.push_back(std::move(output_spec));
  }
  TFF_ASSIGN_OR_RETURN(std::vector<Intrinsic> intrinsics,
                       ParseFromConfig(intrinsic_config.intrinsic_uri(),
                                       intrinsic_config.inner_intrinsics()));
  return Intrinsic{
      intrinsic_config.intrinsic_uri(), std::move(input_tensor_specs),
      std::move(output_tensor_specs), std::move(params), std::move(intrinsics)};
}

}  // namespace

StatusOr<std::vector<Intrinsic>> ParseFromConfig(const Configuration& config) {
  TFF_RETURN_IF_ERROR(ValidateConfiguration(config));
  return ParseFromConfig("", config.intrinsic_configs());
}

absl::Status ValidateConfiguration(const Configuration& configuration) {
  for (const Configuration::IntrinsicConfig& intrinsic_config :
       configuration.intrinsic_configs()) {
    if (!GetAggregatorFactory(intrinsic_config.intrinsic_uri()).ok()) {
      return IntrinsicConfigArgumentError(
          intrinsic_config,
          absl::StrFormat("%s is not a supported intrinsic_uri.",
                          intrinsic_config.intrinsic_uri()));
    }
  }
  return absl::OkStatus();
}

}  // namespace aggregation
}  // namespace tensorflow_federated
