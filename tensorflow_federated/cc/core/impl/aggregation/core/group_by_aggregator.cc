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

#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/partitioner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_slice_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

namespace internal {

// Given a column TensorSliceData and a permutation of the indices, apply the
// reordering to the column.
template <typename T>
Status ReorderColumn(TensorSliceData& column,
                     const std::vector<uint32_t>& new_order) {
  TFF_ASSIGN_OR_RETURN(auto column_span, column.AsSpan<T>());
  for (size_t i = 0; i < new_order.size(); ++i) {
    size_t j = new_order[i];
    std::swap(column_span[i], column_span[j]);
  }
  return absl::OkStatus();
}
}  // namespace internal

GroupByAggregator::GroupByAggregator(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    int num_inputs, std::optional<int> min_contributors_to_group,
    std::vector<int> contributors_to_groups)
    : num_inputs_(num_inputs),
      num_keys_per_input_(input_key_specs.size()),
      key_combiner_(std::move(key_combiner)),
      intrinsics_(*intrinsics),
      output_key_specs_(*output_key_specs),
      aggregators_(std::move(aggregators)),
      min_contributors_to_group_(min_contributors_to_group),
      contributors_to_groups_(std::move(contributors_to_groups)) {
  // Most invariants on construction of the GroupByAggregator such as which
  // nested intrinsics are supported should be enforced in the factory class.
  // This constructor just performs a few backup checks.
  int num_value_inputs = 0;
  TFF_CHECK(intrinsics_.size() == aggregators_.size())
      << "Intrinsics and aggregators vectors must be the same size.";
  for (int i = 0; i < intrinsics_.size(); ++i) {
    num_value_inputs += intrinsics_[i].inputs.size();
  }
  num_tensors_per_input_ = num_keys_per_input_ + num_value_inputs;
  TFF_CHECK(num_tensors_per_input_ > 0)
      << "GroupByAggregator: Must operate on a nonzero number of tensors.";
  TFF_CHECK(num_keys_per_input_ == output_key_specs->size())
      << "GroupByAggregator: Size of input_key_specs must match size of "
         "output_key_specs.";

  if (min_contributors_to_group.has_value()) {
    TFF_CHECK(*min_contributors_to_group > 0)
        << "GroupByAggregator: min_contributors_to_group must be positive.";
    max_contributors_to_group_ = min_contributors_to_group;
  }

  if (!contributors_to_groups_.empty() &&
      !min_contributors_to_group_.has_value()) {
    TFF_CHECK(false) << "GroupByAggregator: contributors_to_groups can only be "
                        "set if min_contributors_to_group is set.";
  }
}

std::unique_ptr<CompositeKeyCombiner> GroupByAggregator::CreateKeyCombiner(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs) {
  // If there are no input keys, support a columnar aggregation that aggregates
  // all the values in each column and produces a single output value per
  // column. This would be equivalent to having identical key values for all
  // rows.
  if (input_key_specs.empty()) {
    return nullptr;
  }

  return std::make_unique<CompositeKeyCombiner>(CreateKeyTypes(
      input_key_specs.size(), input_key_specs, *output_key_specs));
}

std::vector<DataType> GroupByAggregator::CreateKeyTypes(
    size_t num_keys_per_input, const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>& output_key_specs) {
  std::vector<DataType> key_types;
  key_types.reserve(num_keys_per_input);
  for (int i = 0; i < num_keys_per_input; ++i) {
    const TensorSpec& input_spec = input_key_specs[i];
    const TensorSpec& output_spec = output_key_specs[i];
    TFF_CHECK(input_spec.dtype() == output_spec.dtype())
        << "GroupByAggregator: Input and output tensor specifications must "
           "have matching data types";
    // TODO: b/279972547 - Support accumulating value tensors of multiple
    // dimensions. In that case, the size of all output dimensions but one
    // (the dimension corresponding to the number of unique composite keys)
    // should be known in advance and thus this constructor should take in a
    // shape with a single unknown dimension.
    TFF_CHECK(input_spec.shape() == TensorShape{-1} &&
              output_spec.shape() == TensorShape{-1})
        << "All input and output tensors must have one dimension of unknown "
           "size. TensorShape should be {-1}";
    key_types.push_back(input_spec.dtype());
  }
  return key_types;
}

Status GroupByAggregator::MergeWith(TensorAggregator&& other) {
  TFF_RETURN_IF_ERROR(CheckValid());
  // TODO: b/281146781 - For the bare metal environment, we will need a version
  // of this class that does not rely on dynamic_cast.
  GroupByAggregator* other_ptr = dynamic_cast<GroupByAggregator*>(&other);
  if (other_ptr == nullptr) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeOutputTensors: Can only merge with "
              "another GroupByAggregator";
  }
  TFF_RETURN_IF_ERROR((*other_ptr).CheckValid());
  TFF_RETURN_IF_ERROR(other_ptr->IsCompatible(*this));
  int other_num_inputs = other_ptr->GetNumInputs();
  OutputTensorList other_output_tensors =
      std::move(*other_ptr).TakeOutputsInternal();
  InputTensorList tensors(other_output_tensors.size());
  std::vector<int> other_contributors = other_ptr->GetContributors();
  for (int i = 0; i < other_output_tensors.size(); ++i)
    tensors[i] = &other_output_tensors[i];
  TFF_RETURN_IF_ERROR(MergeTensorsInternal(std::move(tensors), other_num_inputs,
                                           other_contributors));
  num_inputs_ += other_num_inputs;
  return absl::OkStatus();
}

bool GroupByAggregator::CanReport() const { return CheckValid().ok(); }

StatusOr<OutputTensorList> GroupByAggregator::Report() && {
  TFF_RETURN_IF_ERROR(CheckValid());
  OutputTensorList unthresholded_histogram = std::move(*this).TakeOutputs();

  if (!min_contributors_to_group_.has_value()) {
    return unthresholded_histogram;
  }

  absl::flat_hash_set<size_t> survivor_indices;
  for (size_t i = 0; i < contributors_to_groups_.size(); i++) {
    if (contributors_to_groups_[i] >= min_contributors_to_group_) {
      survivor_indices.insert(i);
    }
  }

  TFF_ASSIGN_OR_RETURN(HistogramAsSliceData histogram_as_slice_data,
                       ConvertHistogramToSliceData(unthresholded_histogram));

  return ShrinkHistogramToSurvivors(std::move(histogram_as_slice_data),
                                    survivor_indices);
}

Status GroupByAggregator::AggregateTensors(InputTensorList tensors) {
  TFF_RETURN_IF_ERROR(AggregateTensorsInternal(std::move(tensors)));
  num_inputs_++;
  return absl::OkStatus();
}

Status GroupByAggregator::CheckValid() const {
  if (output_consumed_) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "GroupByAggregator::CheckValid: Output has already been "
              "consumed.";
  }
  return absl::OkStatus();
}

OutputTensorList GroupByAggregator::TakeOutputs() && {
  size_t num_keys = num_keys_per_input_;
  OutputTensorList internal_outputs = std::move(*this).TakeOutputsInternal();
  // Keys should only be included in the final outputs if their name is nonempty
  // in the output_key_specs.
  OutputTensorList outputs;
  for (int i = 0; i < num_keys; ++i) {
    if (output_key_specs_[i].name().empty()) continue;
    outputs.push_back(std::move(internal_outputs[i]));
  }
  // Include all outputs from sub-intrinsics.
  for (size_t j = num_keys; j < internal_outputs.size(); ++j) {
    outputs.push_back(std::move(internal_outputs[j]));
  }
  return outputs;
}

Status GroupByAggregator::AddOneContributor(const Tensor& ordinals) {
  if (ordinals.dtype() != DT_INT64) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::AddOneContributor: Expected int64 ordinals "
              "but got "
           << ordinals.dtype();
  }
  if (!max_contributors_to_group_.has_value()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::AddOneContributor: Expected "
              "max_contributors_to_group_ to be set but it is not.";
  }
  auto ordinals_span = ordinals.AsSpan<int64_t>();
  if (ordinals_span.empty()) {
    return absl::OkStatus();
  }
  int64_t max_ordinal = *absl::c_max_element(ordinals_span);
  if (max_ordinal >= contributors_to_groups_.size()) {
    contributors_to_groups_.resize(max_ordinal + 1);
  }
  for (auto& ordinal : ordinals_span) {
    // A derived class may call this function on negative ordinals, which should
    // be ignored.
    if (ordinal < 0) {
      continue;
    }
    if (contributors_to_groups_[ordinal] < max_contributors_to_group_) {
      contributors_to_groups_[ordinal]++;
    }
  }
  return absl::OkStatus();
}

Status GroupByAggregator::AddMultipleContributors(
    const Tensor& ordinals, const std::vector<int>& other_contributors) {
  if (ordinals.dtype() != DT_INT64) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::AddMultipleContributor: Expected int64 "
              "ordinals but got "
           << ordinals.dtype();
  }
  if (!max_contributors_to_group_.has_value()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::AddMultipleContributor: Expected "
              "max_contributors_to_group_ to be set but it is not.";
  }
  TensorShape shape = ordinals.shape();
  if (shape.dim_sizes().size() != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::AddMultipleContributor: Expected 1D tensor "
              "of ordinals.";
  }
  int64_t num_ordinals = shape.dim_sizes()[0];
  if (num_ordinals != other_contributors.size()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::AddMultipleContributor: Expected the same "
              "number of ordinals and contributor counts but got "
           << num_ordinals << " ordinals and " << other_contributors.size()
           << " contributor counts.";
  }
  if (num_ordinals == 0) {
    return absl::OkStatus();
  }
  auto ordinals_span = ordinals.AsSpan<int64_t>();
  int64_t max_ordinal = *absl::c_max_element(ordinals_span);
  if (max_ordinal >= contributors_to_groups_.size()) {
    contributors_to_groups_.resize(max_ordinal + 1);
  }
  for (int i = 0; i < num_ordinals; ++i) {
    int64_t ordinal = ordinals_span[i];
    contributors_to_groups_[ordinal] += other_contributors[i];
    if (contributors_to_groups_[ordinal] > max_contributors_to_group_) {
      contributors_to_groups_[ordinal] = *max_contributors_to_group_;
    }
  }
  return absl::OkStatus();
}

StatusOr<std::string> GroupByAggregator::Serialize() && {
  GroupByAggregatorState state;
  state.set_num_inputs(num_inputs_);
  // If keys are being used, store the current list of output keys into state.
  if (key_combiner_ != nullptr) {
    OutputTensorList keys = key_combiner_->GetOutputKeys();
    google::protobuf::RepeatedPtrField<TensorProto>* keys_proto = state.mutable_keys();
    keys_proto->Reserve(keys.size());
    for (int i = 0; i < keys.size(); ++i) {
      keys_proto->Add(keys[i].ToProto());
    }
  }
  // Store the state of the nested aggregators.
  google::protobuf::RepeatedPtrField<OneDimGroupingAggregatorState>*
      nested_aggregators_proto = state.mutable_nested_aggregators();
  nested_aggregators_proto->Reserve(aggregators_.size());
  for (auto const& nested_aggregator : aggregators_) {
    nested_aggregators_proto->Add(nested_aggregator->ToProto());
  }
  // Store contributors to groups.
  state.mutable_counter_of_contributors()->Reserve(
      contributors_to_groups_.size());
  state.mutable_counter_of_contributors()->Add(contributors_to_groups_.begin(),
                                               contributors_to_groups_.end());
  return state.SerializeAsString();
}

StatusOr<std::vector<std::string>> GroupByAggregator::Partition(
    int num_partitions) && {
  if (num_partitions < 1) {
    return TFF_STATUS(INVALID_ARGUMENT) << "GroupByAggregator::Partition: "
                                           "num_partitions must be at least 1.";
  }

  std::vector<GroupByAggregatorState> group_by_aggregator_states(
      num_partitions);
  OutputTensorList keys = key_combiner_->GetOutputKeys();
  TFF_ASSIGN_OR_RETURN(Partitioner partitioner,
                       Partitioner::Create(keys, num_partitions));
  for (auto& group_by_aggregator_state : group_by_aggregator_states) {
    group_by_aggregator_state.mutable_keys()->Reserve(keys.size());
    group_by_aggregator_state.mutable_nested_aggregators()->Reserve(
        aggregators_.size());
    group_by_aggregator_state.set_num_inputs(num_inputs_);
  }
  for (const auto& key : keys) {
    TFF_ASSIGN_OR_RETURN(auto partitioned_keys, partitioner.PartitionKeys(key));
    for (int i = 0; i < num_partitions; ++i) {
      group_by_aggregator_states[i].mutable_keys()->Add(
          partitioned_keys[i].ToProto());
    }
  }
  for (const auto& nested_aggregator : aggregators_) {
    TFF_ASSIGN_OR_RETURN(auto partitioned_nested_aggregators,
                         nested_aggregator->PartitionToProtos(partitioner));
    for (int i = 0; i < num_partitions; ++i) {
      *group_by_aggregator_states[i].mutable_nested_aggregators()->Add() =
          partitioned_nested_aggregators[i];
    }
  }
  if (!contributors_to_groups_.empty()) {
    TFF_ASSIGN_OR_RETURN(
        std::vector<std::vector<int>> partitioned_contributors_to_groups,
        partitioner.PartitionData<int>(contributors_to_groups_));
    TFF_CHECK(partitioned_contributors_to_groups.size() == num_partitions)
        << "GroupByAggregator::Partition: partitioned_contributors_to_groups "
           "size must be equal to num_partitions.";
    for (int i = 0; i < num_partitions; ++i) {
      auto& contributors_for_partition = partitioned_contributors_to_groups[i];
      auto* counters =
          group_by_aggregator_states[i].mutable_counter_of_contributors();
      counters->Reserve(contributors_for_partition.size());
      counters->Add(contributors_for_partition.begin(),
                    contributors_for_partition.end());
    }
  }
  std::vector<std::string> serialized_states(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    serialized_states[i] =
        std::move(group_by_aggregator_states[i]).SerializeAsString();
  }
  return serialized_states;
}

StatusOr<GroupByAggregator::HistogramAsSliceData>
GroupByAggregator::ConvertHistogramToSliceData(OutputTensorList& histogram) {
  int num_columns = histogram.size();
  std::vector<std::unique_ptr<TensorSliceData>> column_data(num_columns);
  std::vector<DataType> column_dtypes(num_columns);
  size_t num_rows = 0;
  for (size_t i = 0; i < num_columns; ++i) {
    column_dtypes[i] = histogram[i].dtype();
    TFF_ASSIGN_OR_RETURN(size_t current_rows,
                         histogram[i].shape().NumElements());
    column_data[i] = std::make_unique<TensorSliceData>(std::move(histogram[i]));
    if (i == 0) {
      num_rows = current_rows;
    }
    if (num_rows != current_rows) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupByAggregator::ConvertHistogramToSliceData: Expected "
                "histogram to be a list of tensors of the same size but "
                "first "
                "tensor has size "
             << num_rows << " and tensor at index " << i << " has size "
             << current_rows;
    }
  }
  return HistogramAsSliceData{std::move(column_data), std::move(column_dtypes),
                              num_rows};
}

StatusOr<OutputTensorList> GroupByAggregator::ShrinkHistogramToSurvivors(
    HistogramAsSliceData histogram_as_slice_data,
    const absl::flat_hash_set<size_t>& survivor_indices) {
  std::vector<std::unique_ptr<TensorSliceData>>& column_data =
      histogram_as_slice_data.column_data;
  std::vector<DataType>& column_dtypes = histogram_as_slice_data.column_dtypes;
  OutputTensorList shrunk_histogram;

  // If there are no survivors, create empty tensors for each column.
  if (survivor_indices.empty()) {
    for (int i = 0; i < column_data.size(); ++i) {
      TFF_ASSIGN_OR_RETURN(
          Tensor tensor,
          Tensor::Create(column_dtypes[i], {0},
                         std::make_unique<MutableVectorData<char>>()));
      shrunk_histogram.push_back(std::move(tensor));
    }
    return shrunk_histogram;
  }

  // Select a reordering for the surviving rows, using the Fisher-Yates
  // algorithm.
  size_t num_rows = survivor_indices.size();
  absl::BitGen bitgen;
  std::vector<uint32_t> new_order(num_rows - 1, num_rows);
  for (size_t i = 0; i < num_rows - 1; i++) {
    new_order[i] = absl::Uniform(bitgen, i, num_rows);
  }

  // Shrink each column to only include the survivors, and reorder the rows.
  shrunk_histogram.reserve(column_data.size());
  for (int i = 0; i < column_data.size(); ++i) {
    DTYPE_CASES(column_dtypes[i], OutputType,
                TFF_RETURN_IF_ERROR(
                    GroupByAggregator::ShrinkTensorSliceToSurvivors<OutputType>(
                        *(column_data[i]), survivor_indices)));
    DTYPE_CASES(column_dtypes[i], OutputType,
                TFF_RETURN_IF_ERROR(internal::ReorderColumn<OutputType>(
                    *column_data[i], new_order)));
    TFF_ASSIGN_OR_RETURN(
        Tensor tensor,
        Tensor::Create(column_dtypes[i],
                       {static_cast<int64_t>(survivor_indices.size())},
                       std::move(column_data[i])));
    shrunk_histogram.push_back(std::move(tensor));
  }

  return shrunk_histogram;
}

Status GroupByAggregator::ValidateIndexedTensor(
    const Tensor& tensor, size_t input_index, DataType expected_dtype,
    const TensorShape& key_shape) const {
  // Ensure the tensor at input_index has the expected dtype and shape.
  if (tensor.dtype() != expected_dtype) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::ValidateIndexedTensor: Tensor at position "
           << input_index << " did not have expected dtype " << expected_dtype
           << " and instead had dtype " << tensor.dtype();
  }
  if (tensor.shape() != key_shape) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::ValidateIndexedTensor: Shape of value tensor "
           << "at index " << input_index
           << " does not match the shape of the first key tensor.";
  }
  if (!tensor.is_dense()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::ValidateIndexedTensor: Only dense tensors are"
           << " supported.";
  }
  return absl::OkStatus();
}

Status GroupByAggregator::ValidateInputs(const InputTensorList& tensors) const {
  if (tensors.size() != num_tensors_per_input_) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::ValidateInputs: should operate on "
           << num_tensors_per_input_ << " input tensors";
  }
  // Get the shape of the first key tensor in order to ensure that all the
  // value tensors have the same shape. CompositeKeyCombiner::Accumulate will
  // ensure that all keys have the same shape before making any changes to its
  // own internal state.
  TensorShape key_shape = tensors[0]->shape();
  if (key_shape.dim_sizes().size() > 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::ValidateInputs: Only scalar or "
           << "one-dimensional tensors are "
              "supported.";
  }
  // Check all required invariants on the input tensors via the nested
  // aggregators. The input tensors should correspond to the
  // Intrinsic input TensorSpecs since this is an Accumulate operation.
  size_t input_index = num_keys_per_input_;
  for (const Intrinsic& intrinsic : intrinsics_) {
    for (const TensorSpec& tensor_spec : intrinsic.inputs) {
      TFF_RETURN_IF_ERROR(ValidateIndexedTensor(
          *tensors[input_index], input_index, tensor_spec.dtype(), key_shape));
      ++input_index;
    }
  }

  return absl::OkStatus();
}

Status GroupByAggregator::AggregateTensorsInternal(InputTensorList tensors) {
  TFF_ASSIGN_OR_RETURN(Tensor ordinals, CreateOrdinalsByGroupingKeys(tensors));

  if (min_contributors_to_group_.has_value()) {
    TFF_RETURN_IF_ERROR(AddOneContributor(ordinals));
  }

  int input_index = num_keys_per_input_;
  for (int i = 0; i < intrinsics_.size(); ++i) {
    InputTensorList intrinsic_inputs(intrinsics_[i].inputs.size() + 1);
    intrinsic_inputs[0] = &ordinals;
    for (int j = 0; j < intrinsics_[i].inputs.size(); ++j) {
      intrinsic_inputs[j + 1] = tensors[input_index++];
    }
    // Accumulate the input tensors into the aggregator.
    Status aggregation_status =
        aggregators_[i]->Accumulate(std::move(intrinsic_inputs));
    // If the aggregation operation fails on a sub-intrinsic, the
    // key_combiner_ and any previous sub-intrinsics have already been
    // modified. Thus, exit the program with a CHECK failure rather than a
    // failed status which might leave the GroupByAggregator in an
    // inconsistent state.
    TFF_CHECK(aggregation_status.ok())
        << "GroupByAggregator::AggregateTensorsInternal "
        << aggregation_status.message();
  }
  return absl::OkStatus();
}

Status GroupByAggregator::MergeTensorsInternal(
    InputTensorList tensors, int num_merged_inputs,
    const std::vector<int>& other_contributors) {
  if (tensors.size() != num_tensors_per_input_) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeTensorsInternal should operate on "
           << num_tensors_per_input_ << " input tensors";
  }
  // Get the shape of the first key tensor in order to ensure that all the
  // value tensors have the same shape. CompositeKeyCombiner::Accumulate will
  // ensure that all keys have the same shape before making any changes to its
  // own internal state.
  TensorShape key_shape = tensors[0]->shape();
  if (key_shape.dim_sizes().size() > 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator: Only scalar or one-dimensional tensors are "
              "supported.";
  }
  // Check all required invariants on the input tensors, so this function can
  // fail before changing the state of this GroupByAggregator if there is an
  // invalid input tensor. The input tensors should correspond to the
  // Intrinsic output TensorSpecs since this is an Merge operation.
  size_t input_index = num_keys_per_input_;
  for (const Intrinsic& intrinsic : intrinsics_) {
    for (const TensorSpec& tensor_spec : intrinsic.outputs) {
      TFF_RETURN_IF_ERROR(ValidateIndexedTensor(
          *tensors[input_index], input_index, tensor_spec.dtype(), key_shape));
      ++input_index;
    }
  }

  TFF_ASSIGN_OR_RETURN(Tensor ordinals,
                       CreateOrdinalsByGroupingKeysForMerge(tensors));

  if (min_contributors_to_group_.has_value()) {
    TFF_RETURN_IF_ERROR(AddMultipleContributors(ordinals, other_contributors));
  }

  input_index = num_keys_per_input_;
  for (int i = 0; i < intrinsics_.size(); ++i) {
    InputTensorList intrinsic_inputs(intrinsics_[i].inputs.size() + 1);
    intrinsic_inputs[0] = &ordinals;
    for (int j = 0; j < intrinsics_[i].inputs.size(); ++j) {
      intrinsic_inputs[j + 1] = tensors[input_index++];
    }
    // Merge the input tensors into the aggregator.
    Status aggregation_status = aggregators_[i]->MergeTensors(
        std::move(intrinsic_inputs), num_merged_inputs);
    // If the aggregation operation fails on a sub-intrinsic, the
    // key_combiner_ and any previous sub-intrinsics have already been
    // modified. Thus, exit the program with a CHECK failure rather than a
    // failed status which might leave the GroupByAggregator in an
    // inconsistent state.
    TFF_CHECK(aggregation_status.ok())
        << "GroupByAggregator::MergeTensorsInternal "
        << aggregation_status.message();
  }
  return absl::OkStatus();
}

OutputTensorList GroupByAggregator::TakeOutputsInternal() {
  output_consumed_ = true;
  OutputTensorList outputs;
  if (key_combiner_ != nullptr) {
    outputs = key_combiner_->GetOutputKeys();
  }
  outputs.reserve(outputs.size() + intrinsics_.size());
  for (int i = 0; i < intrinsics_.size(); ++i) {
    auto tensor_aggregator = std::move(aggregators_[i]);
    StatusOr<OutputTensorList> value_output =
        std::move(*tensor_aggregator).Report();
    TFF_CHECK(value_output.ok()) << value_output.status().message();
    for (Tensor& output_tensor : value_output.value()) {
      outputs.push_back(std::move(output_tensor));
    }
  }
  return outputs;
}

StatusOr<Tensor> GroupByAggregator::CreateOrdinalsByGroupingKeys(
    const InputTensorList& inputs) {
  if (key_combiner_ != nullptr) {
    InputTensorList keys(num_keys_per_input_);
    for (int i = 0; i < num_keys_per_input_; ++i) {
      keys[i] = inputs[i];
    }
    return key_combiner_->Accumulate(std::move(keys));
  }
  // If there are no keys, we should aggregate all elements in a column into
  // one element, as if there were an imaginary key column with identical
  // values for all rows.
  auto ordinals =
      std::make_unique<MutableVectorData<int64_t>>(inputs[0]->num_elements());
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType,
                        inputs[0]->shape(), std::move(ordinals));
}
StatusOr<Tensor> GroupByAggregator::CreateOrdinalsByGroupingKeysForMerge(
    const InputTensorList& inputs) {
  // In this base class, ordinals are made the same way for
  // MergeTensorsInternal as for AggregateTensorsInternal.
  return CreateOrdinalsByGroupingKeys(inputs);
}

Status GroupByAggregator::IsCompatible(const GroupByAggregator& other) const {
  bool other_has_no_combiner = (other.key_combiner_ == nullptr);
  bool this_has_no_combiner = (key_combiner_ == nullptr);
  if (other_has_no_combiner != this_has_no_combiner) {
    return TFF_STATUS(INVALID_ARGUMENT) << "GroupByAggregator::MergeWith: "
                                           "Expected other GroupByAggregator "
                                           "to have the same key input and "
                                           "output specs";
  }
  if (this_has_no_combiner) {
    return absl::OkStatus();
  }
  if (min_contributors_to_group_ != other.min_contributors_to_group_) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeWith: "
              "Expected other GroupByAggregator to have the same "
              "min_contributors_to_group";
  }
  // The constructor validates that input key types match output key types, so
  // checking that the output key types of both aggregators match is
  // sufficient to verify key compatibility.
  if (other.output_key_specs_ != output_key_specs_) {
    return TFF_STATUS(INVALID_ARGUMENT) << "GroupByAggregator::MergeWith: "
                                           "Expected other GroupByAggregator "
                                           "to have the same key input and "
                                           "output specs";
  }
  if (other.intrinsics_.size() != intrinsics_.size()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeWith: Expected other "
              "GroupByAggregator to use the same number of inner intrinsics";
  }
  for (int i = 0; i < other.intrinsics_.size(); ++i) {
    const std::vector<Intrinsic>& other_intrinsics = other.intrinsics_;
    if (other_intrinsics[i].inputs != intrinsics_[i].inputs) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupByAggregator::MergeWith: Expected other "
                "GroupByAggregator to use inner intrinsics with the same "
                "inputs.";
    }
    if (other_intrinsics[i].outputs != intrinsics_[i].outputs) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupByAggregator::MergeWith: Expected other "
                "GroupByAggregator to use inner intrinsics with the same "
                "outputs.";
    }
  }
  return absl::OkStatus();
}

// Check that the configuration is valid for SQL grouping aggregators.
Status GroupByFactory::CheckIntrinsic(const Intrinsic& intrinsic,
                                      const char* uri) {
  if (intrinsic.uri != uri) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByFactory: Expected intrinsic URI " << uri
           << " but got uri " << intrinsic.uri;
  }
  if (intrinsic.inputs.size() != intrinsic.outputs.size()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByFactory: Exactly the same number of input args and "
              "output tensors are "
              "expected but got "
           << intrinsic.inputs.size() << " inputs vs "
           << intrinsic.outputs.size() << " outputs.";
  }
  for (int i = 0; i < intrinsic.inputs.size(); ++i) {
    const TensorSpec& input_spec = intrinsic.inputs[i];
    const TensorSpec& output_spec = intrinsic.outputs[i];
    if (input_spec.dtype() != output_spec.dtype() ||
        input_spec.shape() != output_spec.shape()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "Input and output tensors have mismatched specs.";
    }

    if (input_spec.shape() != TensorShape{-1} ||
        output_spec.shape() != TensorShape{-1}) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "All input and output tensors must have one dimension of "
                "unknown size. TensorShape should be {-1}";
    }
  }
  return absl::OkStatus();
}

// Create a vector of OneDimBaseGroupingAggregators based upon nested
// intrinsics
StatusOr<std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>>>
GroupByFactory::CreateAggregators(
    const Intrinsic& intrinsic,
    const GroupByAggregatorState* aggregator_state) {
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> nested_aggregators;
  int num_value_inputs = 0;
  for (int i = 0; i < intrinsic.nested_intrinsics.size(); ++i) {
    const Intrinsic& nested = intrinsic.nested_intrinsics[i];
    const OneDimGroupingAggregatorState* nested_state =
        aggregator_state == nullptr ? nullptr
                                    : &aggregator_state->nested_aggregators(i);
    // Resolve the intrinsic_uri to the registered TensorAggregatorFactory.
    TFF_ASSIGN_OR_RETURN(const TensorAggregatorFactory* factory,
                         GetAggregatorFactory(nested.uri));

    // Use the factory to create or deserialize the TensorAggregator instance.
    std::unique_ptr<TensorAggregator> nested_aggregator;
    if (nested_state == nullptr) {
      TFF_ASSIGN_OR_RETURN(nested_aggregator, factory->Create(nested));
    } else {
      auto one_dim_base_factory =
          dynamic_cast<const OneDimBaseGroupingAggregatorFactory*>(factory);
      TFF_CHECK(one_dim_base_factory != nullptr);
      TFF_ASSIGN_OR_RETURN(nested_aggregator, one_dim_base_factory->FromProto(
                                                  nested, *nested_state));
    }
    nested_aggregators.push_back(std::unique_ptr<OneDimBaseGroupingAggregator>(
        dynamic_cast<OneDimBaseGroupingAggregator*>(
            nested_aggregator.release())));
    num_value_inputs += nested.inputs.size();
  }
  if (num_value_inputs + intrinsic.inputs.size() == 0) {
    return TFF_STATUS(INVALID_ARGUMENT) << "GroupByFactory: Must operate on a "
                                           "nonzero number of input tensors.";
  }
  return nested_aggregators;
}

Status GroupByFactory::PopulateKeyCombinerFromState(
    CompositeKeyCombiner& key_combiner,
    const GroupByAggregatorState& aggregator_state) {
  if (aggregator_state.num_inputs() == 0) {
    return absl::OkStatus();
  }
  std::vector<Tensor> key_tensors(aggregator_state.keys().size());
  InputTensorList keys(aggregator_state.keys().size());
  for (int i = 0; i < aggregator_state.keys().size(); ++i) {
    key_tensors[i] = Tensor::FromProto(aggregator_state.keys(i)).value();
    keys[i] = &(key_tensors[i]);
  }
  return key_combiner.CompositeKeyCombiner::Accumulate(keys).status();
}

StatusOr<std::unique_ptr<TensorAggregator>> GroupByFactory::Create(
    const Intrinsic& intrinsic) const {
  return CreateInternal(intrinsic, nullptr);
}

StatusOr<std::unique_ptr<TensorAggregator>> GroupByFactory::Deserialize(
    const Intrinsic& intrinsic, std::string serialized_state) const {
  GroupByAggregatorState aggregator_state;
  if (!aggregator_state.ParseFromString(serialized_state)) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByFactory: Failed to parse serialized aggregator.";
  }
  return CreateInternal(intrinsic, &aggregator_state);
}

StatusOr<std::unique_ptr<TensorAggregator>> GroupByFactory::CreateInternal(
    const Intrinsic& intrinsic,
    const GroupByAggregatorState* aggregator_state) const {
  // Check that the configuration is valid for fedsql_group_by.
  TFF_RETURN_IF_ERROR(CheckIntrinsic(intrinsic, kGroupByUri));

  // The GroupByAggregator expects at most one parameters
  if (intrinsic.parameters.size() > 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupByFactory: At most one input parameter expected.";
  }
  std::optional<int> min_contributors_to_group = std::nullopt;
  if (intrinsic.parameters.size() == 1) {
    if (intrinsic.parameters[0].name() != "min_contributors_to_group") {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupByFactory: The name of the provided parameter does not "
                "match an expected parameter.";
    }
    min_contributors_to_group = intrinsic.parameters[0].CastToScalar<int>();
    if (*min_contributors_to_group <= 0) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupByFactory: The min_contributors_to_group parameter "
                "must "
                "be positive if provided.";
    }
  }

  // The nested intrinsics' URIs should begin with kFedSqlPrefix
  for (const Intrinsic& nested : intrinsic.nested_intrinsics) {
    if (!absl::StartsWith(nested.uri, kFedSqlPrefix)) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupByFactory: Nested intrinsic URIs must start with '"
             << kFedSqlPrefix << "'.";
    }
  }

  // Create nested aggregators.
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> nested_aggregators;
  TFF_ASSIGN_OR_RETURN(nested_aggregators,
                       CreateAggregators(intrinsic, aggregator_state));

  // Create the key combiner, and only populate the key combiner with state if
  // there are keys. Also populate the contributor counts if there is an
  // aggregator state and min_contributors_to_group is set.
  auto key_combiner = GroupByAggregator::CreateKeyCombiner(intrinsic.inputs,
                                                           &intrinsic.outputs);
  std::vector<int> contributors_to_groups;
  if (aggregator_state != nullptr) {
    if (key_combiner != nullptr) {
      TFF_RETURN_IF_ERROR(
          PopulateKeyCombinerFromState(*key_combiner, *aggregator_state));
    }
    if (min_contributors_to_group.has_value()) {
      const auto& counters = aggregator_state->counter_of_contributors();
      contributors_to_groups.assign(counters.begin(), counters.end());
    }
  }

  int num_inputs = aggregator_state ? aggregator_state->num_inputs() : 0;

  // Use new rather than make_unique here because the factory function that
  // uses a non-public constructor can't use std::make_unique, and we don't
  // want to add a dependency on absl::WrapUnique.
  return std::unique_ptr<GroupByAggregator>(new GroupByAggregator(
      intrinsic.inputs, &intrinsic.outputs, &intrinsic.nested_intrinsics,
      std::move(key_combiner), std::move(nested_aggregators), num_inputs,
      min_contributors_to_group, std::move(contributors_to_groups)));
}

// TODO: b/266497896 - Revise the registration mechanism below.
REGISTER_AGGREGATOR_FACTORY(std::string(kGroupByUri), GroupByFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
