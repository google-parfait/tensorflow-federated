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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"

namespace tensorflow_federated {
namespace aggregation {

// Forward declaration of implementation utilities.
namespace {
using TensorMap = absl::flat_hash_map<std::string, Tensor>;
struct OutputTensorInfo {
  std::string name;
  Tensor tensor;
};

absl::Status AddInputsToMap(const Intrinsic& intrinsic,
                            CheckpointParser& parser, TensorMap& tensor_map);
std::vector<size_t> CountInputs(const std::vector<Intrinsic>& intrinsics);
absl::StatusOr<size_t> PopulateInputs(const Intrinsic& intrinsic,
                                      const TensorMap& tensor_map, size_t index,
                                      InputTensorList& inputs);
absl::StatusOr<int> CollectOutputTensorInfos(
    const Intrinsic& intrinsic, OutputTensorList& outputs, int output_index,
    std::vector<OutputTensorInfo>& collected_tensors);
absl::Status CheckCompatible(const std::vector<Intrinsic>& intrinsics,
                             const std::vector<Intrinsic>& other);
}  // namespace

absl::Status CheckpointAggregator::ValidateConfig(
    const Configuration& configuration) {
  return ValidateConfiguration(configuration);
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::Create(const Configuration& configuration) {
  return CreateInternal(configuration, nullptr);
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::Create(const std::vector<Intrinsic>* intrinsics) {
  return CreateInternal(intrinsics, nullptr);
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::Deserialize(const Configuration& configuration,
                                  std::string serialized_state) {
  CheckpointAggregatorState aggregator_state;
  if (!aggregator_state.ParseFromString(serialized_state)) {
    return absl::InvalidArgumentError("Failed to parse serialized state.");
  }
  return CreateInternal(configuration, &aggregator_state);
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::Deserialize(const std::vector<Intrinsic>* intrinsics,
                                  std::string serialized_state) {
  CheckpointAggregatorState aggregator_state;
  if (!aggregator_state.ParseFromString(serialized_state)) {
    return absl::InvalidArgumentError("Failed to parse serialized state.");
  }
  return CreateInternal(intrinsics, &aggregator_state);
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::CreateInternal(
    const Configuration& configuration,
    const CheckpointAggregatorState* aggregator_state) {
  TFF_ASSIGN_OR_RETURN(std::vector<Intrinsic> intrinsics,
                       ParseFromConfig(configuration));

  std::vector<std::unique_ptr<TensorAggregator>> aggregators;
  for (int i = 0; i < intrinsics.size(); ++i) {
    const Intrinsic& intrinsic = intrinsics[i];
    const std::string* serialized_aggregator = nullptr;
    if (aggregator_state != nullptr) {
      serialized_aggregator = &aggregator_state->aggregators(i);
    }
    TFF_ASSIGN_OR_RETURN(std::unique_ptr<TensorAggregator> aggregator,
                         CreateAggregator(intrinsic, serialized_aggregator));
    aggregators.push_back(std::move(aggregator));
  }

  return absl::WrapUnique(
      new CheckpointAggregator(std::move(intrinsics), std::move(aggregators)));
}

absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
CheckpointAggregator::CreateInternal(
    const std::vector<Intrinsic>* intrinsics,
    const CheckpointAggregatorState* aggregator_state) {
  std::vector<std::unique_ptr<TensorAggregator>> aggregators;
  for (int i = 0; i < intrinsics->size(); ++i) {
    const Intrinsic& intrinsic = (*intrinsics)[i];
    const std::string* serialized_aggregator = nullptr;
    if (aggregator_state != nullptr) {
      serialized_aggregator = &aggregator_state->aggregators(i);
    }
    TFF_ASSIGN_OR_RETURN(std::unique_ptr<TensorAggregator> aggregator,
                         CreateAggregator(intrinsic, serialized_aggregator));
    aggregators.push_back(std::move(aggregator));
  }

  return absl::WrapUnique(
      new CheckpointAggregator(intrinsics, std::move(aggregators)));
}

CheckpointAggregator::CheckpointAggregator(
    const std::vector<Intrinsic>* intrinsics,
    std::vector<std::unique_ptr<TensorAggregator>> aggregators)
    : intrinsics_(*intrinsics),
      input_counts_(CountInputs(intrinsics_)),
      aggregators_(std::move(aggregators)) {}

CheckpointAggregator::CheckpointAggregator(
    std::vector<Intrinsic> intrinsics,
    std::vector<std::unique_ptr<TensorAggregator>> aggregators)
    : owned_intrinsics_(std::move(intrinsics)),
      intrinsics_(*owned_intrinsics_),
      input_counts_(CountInputs(intrinsics_)),
      aggregators_(std::move(aggregators)) {}

absl::Status CheckpointAggregator::Accumulate(
    CheckpointParser& checkpoint_parser) {
  TensorMap tensor_map;
  for (const auto& intrinsic : intrinsics_) {
    TFF_RETURN_IF_ERROR(
        AddInputsToMap(intrinsic, checkpoint_parser, tensor_map));
  }
  std::vector<InputTensorList> inputs;
  for (int i = 0; i < intrinsics_.size(); ++i) {
    InputTensorList input_list(input_counts_[i]);
    TFF_RETURN_IF_ERROR(
        PopulateInputs(intrinsics_[i], tensor_map, 0, input_list));
    inputs.push_back(std::move(input_list));
  }
  {
    absl::MutexLock lock(&aggregation_mu_);
    if (aggregation_finished_) {
      return absl::AbortedError("Aggregation has already been finished.");
    }
    // Perform a validation pass before accumulating.
    for (int i = 0; i < intrinsics_.size(); ++i) {
      TFF_CHECK(aggregators_[i] != nullptr)
          << "Report() has already been called.";
      TFF_RETURN_IF_ERROR(aggregators_[i]->ValidateInputs(inputs[i]));
    }
    for (int i = 0; i < intrinsics_.size(); ++i) {
      TFF_RETURN_IF_ERROR(
          aggregators_[i]->AccumulateWithoutValidation(std::move(inputs[i])));
    }
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

absl::StatusOr<int> CheckpointAggregator::GetNumCheckpointsAggregated() const {
  absl::MutexLock lock(&aggregation_mu_);
  if (aggregation_finished_) {
    return absl::AbortedError("Aggregation has already been finished.");
  }
  std::optional<int> num_inputs;
  for (const auto& aggregator : aggregators_) {
    TFF_CHECK(aggregator != nullptr)
        << "CreateReport() has already been called.";
    if (!num_inputs.has_value()) {
      num_inputs = aggregator->GetNumInputs();
    } else {
      if (aggregator->GetNumInputs() != *num_inputs) {
        return absl::FailedPreconditionError(
            "The number of inputs aggregated by each inner aggregator does not "
            "match.");
      }
    }
  }
  return *num_inputs;
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
  std::vector<OutputTensorInfo> collected_tensors;
  {
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
                           CollectOutputTensorInfos(intrinsic, output_tensors,
                                                    0, collected_tensors));
      TFF_CHECK(num_outputs == output_tensors.size())
          << "Number of tensors produced by TensorAggregator "
          << output_tensors.size()
          << " does not match number of output tensors with nonempty names "
          << num_outputs << ".";
    }
  }
  // Add all collected tensors to the checkpoint builder outside the mutex.
  for (auto& [name, tensor] : collected_tensors) {
    TFF_RETURN_IF_ERROR(checkpoint_builder.Add(name, std::move(tensor)));
  }

  return absl::OkStatus();
}

void CheckpointAggregator::Abort() { aggregation_finished_ = true; }

absl::StatusOr<std::string> CheckpointAggregator::Serialize() && {
  absl::MutexLock lock(&aggregation_mu_);
  if (aggregation_finished_) {
    return absl::AbortedError("Aggregation has already been finished.");
  }
  CheckpointAggregatorState state;
  google::protobuf::RepeatedPtrField<std::string>* aggregators_proto =
      state.mutable_aggregators();
  aggregators_proto->Reserve(aggregators_.size());
  for (const auto& aggregator : aggregators_) {
    aggregators_proto->Add(std::move(*aggregator).Serialize().value());
  }
  return state.SerializeAsString();
}

absl::StatusOr<std::vector<std::string>> CheckpointAggregator::Partition(
    int num_partitions) && {
  absl::MutexLock lock(&aggregation_mu_);
  if (aggregation_finished_) {
    return absl::AbortedError("Aggregation has already been finished.");
  }

  std::vector<CheckpointAggregatorState> states(num_partitions);
  for (const auto& aggregator : aggregators_) {
    TFF_ASSIGN_OR_RETURN(std::vector<std::string> partitioned_aggregator,
                         std::move(*aggregator).Partition(num_partitions));
    TFF_CHECK(partitioned_aggregator.size() == num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      states[i].add_aggregators(std::move(partitioned_aggregator[i]));
    }
  }
  std::vector<std::string> serialized_states;
  serialized_states.reserve(states.size());
  for (const auto& state : states) {
    serialized_states.push_back(state.SerializeAsString());
  }
  return serialized_states;
}

absl::StatusOr<std::unique_ptr<TensorAggregator>>
CheckpointAggregator::CreateAggregator(
    const Intrinsic& intrinsic, const std::string* serialized_aggregator) {
  // Resolve the intrinsic_uri to the registered TensorAggregatorFactory.
  TFF_ASSIGN_OR_RETURN(const TensorAggregatorFactory* factory,
                       GetAggregatorFactory(intrinsic.uri));

  // Use the factory to create the TensorAggregator instance.
  if (serialized_aggregator == nullptr) {
    return factory->Create(intrinsic);
  }
  return factory->Deserialize(intrinsic, *serialized_aggregator);
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
    if (tensor.dtype() != input_spec.dtype()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input tensor spec mismatch. Expected dtype ",
                       DataType_Name(input_spec.dtype()), " but got ",
                       DataType_Name(tensor.dtype()), " for input tensor ",
                       input_spec.name()));
    }
    if (!input_spec.shape().MatchesKnownDimensions(tensor.shape())) {
      std::string expected_shape =
          absl::StrJoin(input_spec.shape().dim_sizes(), ",");
      std::string actual_shape = absl::StrJoin(tensor.shape().dim_sizes(), ",");
      return absl::InvalidArgumentError(
          absl::StrCat("Input tensor spec mismatch. Expected shape (",
                       expected_shape, ") but got (", actual_shape,
                       ") for input tensor ", input_spec.name()));
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

std::vector<size_t> CountInputs(const std::vector<Intrinsic>& intrinsics) {
  std::vector<size_t> counts(intrinsics.size());
  for (int i = 0; i < intrinsics.size(); ++i) {
    counts[i] = CountInputs(intrinsics[i]);
  }
  return counts;
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

absl::StatusOr<int> CollectOutputTensorInfos(
    const Intrinsic& intrinsic, OutputTensorList& outputs, int output_index,
    std::vector<OutputTensorInfo>& collected_tensors) {
  int num_outputs = 0;
  for (const TensorSpec& output_spec : intrinsic.outputs) {
    if (output_spec.name().empty()) {
      // TensorSpecs with empty names are not included in the output.
      continue;
    }
    num_outputs++;
    Tensor tensor = std::move(outputs[output_index++]);
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
    collected_tensors.push_back({output_spec.name(), std::move(tensor)});
  }
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    TFF_ASSIGN_OR_RETURN(
        int nested_num_outputs,
        CollectOutputTensorInfos(nested_intrinsic, outputs, output_index,
                                 collected_tensors));
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
