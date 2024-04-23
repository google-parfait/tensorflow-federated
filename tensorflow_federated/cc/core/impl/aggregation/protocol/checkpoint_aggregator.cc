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

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"

namespace tensorflow_federated {
namespace aggregation {

// Forward declaration of implementation utilities.
namespace {
using TensorMap = absl::flat_hash_map<std::string, Tensor>;
absl::Status AddInputsToMap(const Intrinsic& intrinsic,
                            CheckpointParser& parser, TensorMap& tensor_map);
size_t CountInputs(const Intrinsic& intrinsic);
absl::StatusOr<size_t> PopulateInputs(const Intrinsic& intrinsic,
                                      const TensorMap& tensor_map, size_t index,
                                      InputTensorList& inputs);
absl::StatusOr<int> AddOutputsToCheckpoint(
    const Intrinsic& intrinsic, const OutputTensorList& outputs,
    int output_index, CheckpointBuilder& checkpoint_builder);
absl::Status CheckCompatible(const std::vector<Intrinsic>& intrinsics,
                             const std::vector<Intrinsic>& other);
}  // namespace

absl::Status CheckpointAggregator::ValidateConfig(
    const Configuration& configuration) {
  return ValidateConfiguration(configuration);
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::Create(const Configuration& configuration) {
  TFF_ASSIGN_OR_RETURN(std::vector<Intrinsic> intrinsics,
                       ParseFromConfig(configuration));

  std::vector<std::unique_ptr<TensorAggregator>> aggregators;
  for (const Intrinsic& intrinsic : intrinsics) {
    TFF_ASSIGN_OR_RETURN(std::unique_ptr<TensorAggregator> aggregator,
                         CreateAggregator(intrinsic));
    aggregators.push_back(std::move(aggregator));
  }

  return absl::WrapUnique(
      new CheckpointAggregator(std::move(intrinsics), std::move(aggregators)));
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::Create(const std::vector<Intrinsic>* intrinsics) {
  std::vector<std::unique_ptr<TensorAggregator>> aggregators;
  for (const Intrinsic& intrinsic : *intrinsics) {
    TFF_ASSIGN_OR_RETURN(std::unique_ptr<TensorAggregator> aggregator,
                         CreateAggregator(intrinsic));
    aggregators.push_back(std::move(aggregator));
  }

  return absl::WrapUnique(
      new CheckpointAggregator(intrinsics, std::move(aggregators)));
}

CheckpointAggregator::CheckpointAggregator(
    const std::vector<Intrinsic>* intrinsics,
    std::vector<std::unique_ptr<TensorAggregator>> aggregators)
    : intrinsics_(*intrinsics), aggregators_(std::move(aggregators)) {}

CheckpointAggregator::CheckpointAggregator(
    std::vector<Intrinsic> intrinsics,
    std::vector<std::unique_ptr<TensorAggregator>> aggregators)
    : owned_intrinsics_(std::move(intrinsics)),
      intrinsics_(*owned_intrinsics_),
      aggregators_(std::move(aggregators)) {}

CheckpointAggregator::~CheckpointAggregator() {
  aggregation_finished_ = true;
  // Enter the lock to ensure that the destructor waits for any ongoing
  // operations that require *this* instance.
  absl::MutexLock lock(&aggregation_mu_);
}

absl::Status CheckpointAggregator::Accumulate(
    CheckpointParser& checkpoint_parser) {
  TensorMap tensor_map;
  for (const auto& intrinsic : intrinsics_) {
    TFF_RETURN_IF_ERROR(
        AddInputsToMap(intrinsic, checkpoint_parser, tensor_map));
  }

  absl::MutexLock lock(&aggregation_mu_);
  if (aggregation_finished_) {
    return absl::AbortedError("Aggregation has already been finished.");
  }
  for (int i = 0; i < intrinsics_.size(); ++i) {
    const Intrinsic& intrinsic = intrinsics_[i];
    InputTensorList inputs(CountInputs(intrinsic));
    TFF_RETURN_IF_ERROR(PopulateInputs(intrinsic, tensor_map, 0, inputs));
    TFF_CHECK(aggregators_[i] != nullptr)
        << "Report() has already been called.";
    TFF_RETURN_IF_ERROR(aggregators_[i]->Accumulate(std::move(inputs)));
  }
  return absl::OkStatus();
}

absl::Status CheckpointAggregator::MergeWith(CheckpointAggregator&& other) {
  TFF_RETURN_IF_ERROR(CheckCompatible(intrinsics_, other.intrinsics_));
  absl::MutexLock lock(&aggregation_mu_);
  if (!aggregation_finished_) {
    auto other_aggregators = std::move(other).TakeAggregators();
    for (int i = 0; i < intrinsics_.size(); ++i) {
      TFF_RETURN_IF_ERROR(
          aggregators_[i]->MergeWith(std::move(*other_aggregators[i])));
    }
  }
  return absl::OkStatus();
}

bool CheckpointAggregator::CanReport() const {
  absl::MutexLock lock(&aggregation_mu_);
  if (aggregation_finished_) {
    return false;
  }
  for (const auto& aggregator : aggregators_) {
    TFF_CHECK(aggregator != nullptr)
        << "CreateReport() has already been called.";
    if (!aggregator->CanReport()) {
      return false;
    }
  }
  return true;
}

absl::Status CheckpointAggregator::Report(
    CheckpointBuilder& checkpoint_builder) {
  absl::MutexLock lock(&aggregation_mu_);
  if (aggregation_finished_) {
    return absl::AbortedError("Aggregation has already been finished.");
  }

  aggregation_finished_ = true;

  for (const auto& aggregator : aggregators_) {
    TFF_CHECK(aggregator != nullptr)
        << "CreateReport() has already been called.";
    if (!aggregator->CanReport()) {
      return absl::FailedPreconditionError(
          "The aggregation can't be completed due to failed preconditions.");
    }
  }

  for (int i = 0; i < intrinsics_.size(); ++i) {
    auto tensor_aggregator = std::move(aggregators_[i]);
    TFF_ASSIGN_OR_RETURN(OutputTensorList output_tensors,
                         std::move(*tensor_aggregator).Report());
    const Intrinsic& intrinsic = intrinsics_[i];
    TFF_ASSIGN_OR_RETURN(int num_outputs,
                         AddOutputsToCheckpoint(intrinsic, output_tensors, 0,
                                                checkpoint_builder));
    TFF_CHECK(num_outputs == output_tensors.size())
        << "Number of tensors produced by TensorAggregator "
        << output_tensors.size()
        << " does not match number of output tensors with nonempty names "
        << num_outputs << ".";
  }
  return absl::OkStatus();
}

void CheckpointAggregator::Abort() { aggregation_finished_ = true; }

absl::StatusOr<std::unique_ptr<TensorAggregator>>
CheckpointAggregator::CreateAggregator(const Intrinsic& intrinsic) {
  // Resolve the intrinsic_uri to the registered TensorAggregatorFactory.
  TFF_ASSIGN_OR_RETURN(const TensorAggregatorFactory* factory,
                       GetAggregatorFactory(intrinsic.uri));

  // Use the factory to create the TensorAggregator instance.
  return factory->Create(intrinsic);
}

std::vector<std::unique_ptr<TensorAggregator>>
CheckpointAggregator::TakeAggregators() && {
  absl::MutexLock lock(&aggregation_mu_);
  return std::move(aggregators_);
}

namespace {

absl::Status AddInputsToMap(const Intrinsic& intrinsic,
                            CheckpointParser& parser, TensorMap& tensor_map) {
  for (const TensorSpec& input_spec : intrinsic.inputs) {
    auto existing_tensor_it = tensor_map.find(input_spec.name());
    if (existing_tensor_it != tensor_map.end()) {
      // Tensor with a matching name is already in the map.
      const Tensor& existing_tensor = existing_tensor_it->second;
      if (input_spec.dtype() == existing_tensor.dtype() &&
          input_spec.shape().MatchesKnownDimensions(existing_tensor.shape())) {
        continue;
      } else {
        return absl::InvalidArgumentError(
            "Tensor with same name but unmatching spec already exists.");
      }
    }

    TFF_ASSIGN_OR_RETURN(Tensor tensor, parser.GetTensor(input_spec.name()));
    if (tensor.dtype() != input_spec.dtype() ||
        !input_spec.shape().MatchesKnownDimensions(tensor.shape())) {
      // TODO: b/253099587 - Detailed diagnostics including the expected vs
      // actual data types and shapes.
      return absl::InvalidArgumentError("Input tensor spec mismatch.");
    }
    tensor_map.emplace(input_spec.name(), std::move(tensor));
  }
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    TFF_RETURN_IF_ERROR(AddInputsToMap(nested_intrinsic, parser, tensor_map));
  }
  return absl::OkStatus();
}

size_t CountInputs(const Intrinsic& intrinsic) {
  size_t count = intrinsic.inputs.size();
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    count += CountInputs(nested_intrinsic);
  }
  return count;
}

absl::StatusOr<size_t> PopulateInputs(const Intrinsic& intrinsic,
                                      const TensorMap& tensor_map, size_t index,
                                      InputTensorList& inputs) {
  size_t num_inputs = intrinsic.inputs.size();
  for (const TensorSpec& input_spec : intrinsic.inputs) {
    const auto& it = tensor_map.find(input_spec.name());
    TFF_CHECK(it != tensor_map.end());
    inputs[index++] = &it->second;
  }
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    TFF_ASSIGN_OR_RETURN(
        size_t nested_num_inputs,
        PopulateInputs(nested_intrinsic, tensor_map, index, inputs));
    index += nested_num_inputs;
    num_inputs += nested_num_inputs;
  }
  return num_inputs;
}

absl::StatusOr<int> AddOutputsToCheckpoint(
    const Intrinsic& intrinsic, const OutputTensorList& outputs,
    int output_index, CheckpointBuilder& checkpoint_builder) {
  int num_outputs = 0;
  for (const TensorSpec& output_spec : intrinsic.outputs) {
    if (output_spec.name().empty()) {
      // TensorSpecs with empty names are not included in the output.
      continue;
    }
    num_outputs++;
    const Tensor& tensor = outputs[output_index++];
    if (tensor.dtype() != output_spec.dtype()) {
      return absl::InternalError(absl::StrCat(
          "Output tensor spec mismatch for output tensor ", output_spec.name(),
          ". Tensor has dtype ", DataType_Name(tensor.dtype()),
          " and output spec has dtype ", DataType_Name(output_spec.dtype())));
    }
    if (!output_spec.shape().MatchesKnownDimensions(tensor.shape())) {
      return absl::InternalError(absl::StrCat(
          "Output tensor spec known dimensions mismatch for output tensor ",
          output_spec.name(), ". Tensor has shape ",
          tensor.shape().ToProto().DebugString(), " and output spec has shape ",
          output_spec.shape().ToProto().DebugString()));
    }
    TFF_RETURN_IF_ERROR(checkpoint_builder.Add(output_spec.name(), tensor));
  }
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    TFF_ASSIGN_OR_RETURN(
        int nested_num_outputs,
        AddOutputsToCheckpoint(nested_intrinsic, outputs, output_index,
                               checkpoint_builder));
    output_index += nested_num_outputs;
    num_outputs += nested_num_outputs;
  }
  return num_outputs;
}

absl::Status CheckCompatible(const std::vector<Intrinsic>& intrinsics,
                             const std::vector<Intrinsic>& other) {
  // TODO: b/316662605 - Implement this and consider moving this to
  // intrinsics.h / intrinsics.cpp
  return absl::OkStatus();
}

}  // namespace

}  // namespace aggregation
}  // namespace tensorflow_federated
