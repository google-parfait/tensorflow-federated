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
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_tensor_aggregator_bundle.h"

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"

namespace tensorflow_federated {
namespace aggregation {

DPTensorAggregatorBundle::DPTensorAggregatorBundle(
    std::vector<std::unique_ptr<DPTensorAggregator>> aggregators,
    std::vector<int> num_tensors_per_agg, double epsilon_per_agg,
    double delta_per_agg, int num_inputs)
    : aggregators_(std::move(aggregators)),
      num_tensors_per_agg_(std::move(num_tensors_per_agg)),
      epsilon_per_agg_(epsilon_per_agg),
      delta_per_agg_(delta_per_agg),
      num_inputs_(num_inputs) {
  num_tensors_per_input_ = std::accumulate(num_tensors_per_agg_.begin(),
                                           num_tensors_per_agg_.end(), 0);
}

Status DPTensorAggregatorBundle::ValidateInputs(
    const InputTensorList& tensors) const {
  if (tensors.size() != num_tensors_per_input_) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundle::ValidateInputs: Expected "
           << num_tensors_per_input_ << " tensors, got " << tensors.size();
  }

  // Verify that each batch is valid, by calling ValidateInputs() on the
  // nested aggregators.
  int current_tensor_index = 0;
  for (int i = 0; i < num_tensors_per_agg_.size(); ++i) {
    std::vector<const Tensor*> current_batch;
    current_batch.reserve(num_tensors_per_agg_[i]);
    for (int j = 0; j < num_tensors_per_agg_[i]; ++j) {
      current_batch.push_back(tensors[current_tensor_index++]);
    }
    auto current_input_tensor_list = InputTensorList(current_batch);
    TFF_RETURN_IF_ERROR(
        aggregators_[i]->ValidateInputs(current_input_tensor_list));
  }
  return absl::OkStatus();
}

Status DPTensorAggregatorBundle::AggregateTensors(InputTensorList tensors) {
  // Split input according to num_tensors_per_agg_ and delegate aggregation to
  // the nested aggregators.
  int current_tensor_index = 0;
  for (int i = 0; i < num_tensors_per_agg_.size(); ++i) {
    std::vector<const Tensor*> current_batch;
    current_batch.reserve(num_tensors_per_agg_[i]);
    for (int j = 0; j < num_tensors_per_agg_[i]; ++j) {
      current_batch.push_back(tensors[current_tensor_index++]);
    }
    TFF_RETURN_IF_ERROR(
        aggregators_[i]->Accumulate(InputTensorList(current_batch)));
  }
  num_inputs_++;

  return TFF_STATUS(OK);
}

Status DPTensorAggregatorBundle::IsCompatible(
    const TensorAggregator& other) const {
  const auto* other_ptr = dynamic_cast<const DPTensorAggregatorBundle*>(&other);
  if (other_ptr == nullptr) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundle::IsCompatible: Can only merge with "
              "another DPTensorAggregatorBundle";
  }
  // Check that the number of nested aggregators is the same.
  if (aggregators_.size() != other_ptr->aggregators_.size()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundle::IsCompatible: One bundle has "
           << aggregators_.size() << " nested aggregators, but the other has "
           << other_ptr->aggregators_.size() << ".";
  }
  // Loop over inner aggregators and check compatibility.
  for (int i = 0; i < aggregators_.size(); ++i) {
    const auto& other_aggregators = other_ptr->aggregators_;
    TFF_RETURN_IF_ERROR(aggregators_[i]->IsCompatible(*(other_aggregators[i])));
  }

  return TFF_STATUS(OK);
}

Status DPTensorAggregatorBundle::MergeWith(TensorAggregator&& other) {
  TFF_RETURN_IF_ERROR(CheckValid());
  TFF_RETURN_IF_ERROR(IsCompatible(other));
  auto* other_ptr = dynamic_cast<DPTensorAggregatorBundle*>(&other);
  TFF_CHECK(other_ptr != nullptr);
  TFF_RETURN_IF_ERROR(other_ptr->CheckValid());

  // Merge the nested aggregators.
  for (int i = 0; i < aggregators_.size(); ++i) {
    TFF_RETURN_IF_ERROR(
        aggregators_[i]->MergeWith(static_cast<TensorAggregator&&>(
            *std::move(other_ptr->aggregators_[i]))));
  }
  num_inputs_ += other_ptr->num_inputs_;

  return TFF_STATUS(OK);
}

OutputTensorList DPTensorAggregatorBundle::TakeOutputs() && {
  output_consumed_ = true;
  OutputTensorList outputs;
  for (int i = 0; i < aggregators_.size(); ++i) {
    auto value_output =
        std::move(*aggregators_[i])
            .ReportWithEpsilonAndDelta(epsilon_per_agg_, delta_per_agg_);
    TFF_CHECK(value_output.ok()) << value_output.status().message();
    for (Tensor& output_tensor : value_output.value()) {
      outputs.push_back(std::move(output_tensor));
    }
  }

  return outputs;
}

StatusOr<std::string> DPTensorAggregatorBundle::Serialize() && {
  DPTensorAggregatorBundleState state;
  state.set_num_inputs(num_inputs_);
  auto* nested_serialized_states = state.mutable_nested_serialized_states();
  for (auto& aggregator : aggregators_) {
    TFF_ASSIGN_OR_RETURN(auto nested_serialized_state,
                         std::move(*aggregator).Serialize());
    nested_serialized_states->Add(std::move(nested_serialized_state));
  }

  return state.SerializeAsString();
}

StatusOr<std::unique_ptr<TensorAggregator>>
DPTensorAggregatorBundleFactory::Deserialize(
    const Intrinsic& intrinsic, std::string serialized_state) const {
  DPTensorAggregatorBundleState aggregator_state;
  if (!aggregator_state.ParseFromString(serialized_state)) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundleFactory::Deserialize: Failed to parse "
           << "serialized aggregator.";
  }
  return CreateInternal(intrinsic, &aggregator_state);
}

StatusOr<std::unique_ptr<TensorAggregator>>
DPTensorAggregatorBundleFactory::CreateInternal(
    const Intrinsic& intrinsic,
    const DPTensorAggregatorBundleState* aggregator_state) const {
  // Check that there is at least one nested intrinsic.
  if (intrinsic.nested_intrinsics.empty()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundleFactory::CreateInternal: Expected "
           << "at least one nested intrinsic, got none.";
  }

  int num_inputs =
      (aggregator_state == nullptr ? 0 : aggregator_state->num_inputs());

  // Create the nested aggregators. Along the way, track how many input tensors
  // will be fed into each nested aggregator during Accumulate.
  std::vector<std::unique_ptr<DPTensorAggregator>> nested_aggregators;
  std::vector<int> num_tensors_per_agg;
  for (int i = 0; i < intrinsic.nested_intrinsics.size(); ++i) {
    const Intrinsic& nested = intrinsic.nested_intrinsics[i];
    // Resolve the intrinsic_uri to the registered TensorAggregatorFactory.
    TFF_ASSIGN_OR_RETURN(const TensorAggregatorFactory* factory,
                         GetAggregatorFactory(nested.uri));
    std::unique_ptr<TensorAggregator> aggregator_ptr;
    if (aggregator_state != nullptr) {
      TFF_ASSIGN_OR_RETURN(
          aggregator_ptr,
          factory->Deserialize(nested,
                               aggregator_state->nested_serialized_states(i)));
    } else {
      TFF_ASSIGN_OR_RETURN(aggregator_ptr, factory->Create(nested));
    }
    auto* dp_aggregator_ptr =
        dynamic_cast<DPTensorAggregator*>(aggregator_ptr.get());
    if (dp_aggregator_ptr == nullptr) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPTensorAggregatorBundleFactory::CreateInternal: Expected "
             << "all nested intrinsics to be DPTensorAggregators, got "
             << nested.uri;
    }
    aggregator_ptr.release();  // NOMUTANTS -- Memory ownership transfer.
    nested_aggregators.push_back(
        std::unique_ptr<DPTensorAggregator>(dp_aggregator_ptr));
    num_tensors_per_agg.push_back(nested.inputs.size());
  }

  int num_nested_intrinsics = intrinsic.nested_intrinsics.size();

  // Ensure that there are epsilon and delta parameters.
  if (intrinsic.parameters.size() != 2) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundleFactory::CreateInternal: Expected "
           << "2 parameters, got " << intrinsic.parameters.size();
  }

  // Validate epsilon and delta before splitting them.
  if (internal::GetTypeKind(intrinsic.parameters[kEpsilonIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundleFactory::CreateInternal: Epsilon must "
           << "be numerical.";
  }
  double epsilon = intrinsic.parameters[kEpsilonIndex].AsScalar<double>();
  if (internal::GetTypeKind(intrinsic.parameters[kDeltaIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundleFactory::CreateInternal: Delta must "
           << "be numerical.";
  }
  double delta = intrinsic.parameters[kDeltaIndex].AsScalar<double>();
  if (epsilon <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT) << "DPTensorAggregatorBundleFactory::"
                                           "CreateInternal: Epsilon must be "
                                           "positive, but got "
                                        << epsilon;
  }
  if (delta < 0 || delta >= 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregatorBundleFactory::CreateInternal: Delta must be "
              "non-negative and less than 1, but got "
           << delta;
  }

  double epsilon_per_agg =
      (epsilon < kEpsilonThreshold ? epsilon / num_nested_intrinsics
                                   : kEpsilonThreshold);
  double delta_per_agg = delta / num_nested_intrinsics;

  return std::make_unique<DPTensorAggregatorBundle>(
      std::move(nested_aggregators), std::move(num_tensors_per_agg),
      epsilon_per_agg, delta_per_agg, num_inputs);
}
REGISTER_AGGREGATOR_FACTORY(kDPTensorAggregatorBundleUri,
                            DPTensorAggregatorBundleFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
