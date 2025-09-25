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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_ONE_DIM_GROUPING_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_ONE_DIM_GROUPING_AGGREGATOR_H_

#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/partitioner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

// OneDimBaseGroupingAggregator class is a specialization of TensorAggregator
// which is used as a base class for typed OneDimGroupingAggregator instances.
//
// It operates over lists of two tensors, where the first tensor contains
// ordinals and the second tensor contains values. Currently only 1D input
// tensors are supported.
//
// The specific means of accumulating, merging, and producing default values
// are left to the subclasses.
//
// This class is not thread safe.
class OneDimBaseGroupingAggregator : public TensorAggregator {
 public:
  Status MergeWith(TensorAggregator&& other) override;

  StatusOr<std::string> Serialize() && override {
    // OneDimBaseGroupingAggregators are always nested within an outer
    // aggregator. Use ToProto to get intermediate state and then serialize the
    // outer aggregator state instead.
    return TFF_STATUS(UNIMPLEMENTED)
           << "OneDimBaseGroupingAggregator::Serialize is not supported. Use "
              "ToProto to store intermediate state.";
  }

  // Merges intermediate aggregates contained in the tensors param into the
  // current Aggregator instance. Expects a tensors param of size 2, where the
  // first tensor contains ordinals and the second tensor contains values. The
  // num_inputs param specifies how many inputs were pre-accumulated into the
  // tensors input.
  //
  // The provided ordinals should correspond to the key->ordinal mapping used by
  // the GroupByAggregator into which the values are being merged, as opposed to
  // the key->ordinal mapping used by the GroupByAggregator that accumulated the
  // values that are now being merged. The GroupBy aggregator is responsible for
  // providing the correct ordinals.
  //
  // The actual implementation of the merge operation is to be provided by a
  // derived class.
  virtual Status MergeTensors(InputTensorList tensors, int num_inputs) = 0;

  // Stores the intermediate state of the OneDimBaseGroupingAggregator as a
  // proto.
  virtual OneDimGroupingAggregatorState ToProto() = 0;

  // Partitions the OneDimBaseGroupingAggregator into multiple slices and stores
  // the intermediate states into a vector of protos.
  virtual StatusOr<std::vector<OneDimGroupingAggregatorState>>
  PartitionToProtos(const Partitioner& partitioner) = 0;

 protected:
  // Checks that the input tensors param is valid.
  Status ValidateTensorInputs(const InputTensorList& tensors);
};

class OneDimBaseGroupingAggregatorFactory : public TensorAggregatorFactory {
 public:
  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    return CreateInternal(intrinsic, nullptr);
  }

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    OneDimGroupingAggregatorState aggregator_state;
    // OneDimGroupingAggregators are always nested within an outer aggregator.
    // Use FromProto to create the aggregator from intermediate state stored by
    // the outer aggregator.
    return TFF_STATUS(UNIMPLEMENTED)
           << "OneDimBaseGroupingAggregatorFactory::Deserialize is not "
              "supported. Use FromProto to create an aggregator from "
              "intermediate state.";
  }

  // Creates a OneDimBaseGroupingAggregator from intermediate state.
  StatusOr<std::unique_ptr<TensorAggregator>> FromProto(
      const Intrinsic& intrinsic,
      const OneDimGroupingAggregatorState& aggregator_state) const {
    return CreateInternal(intrinsic, &aggregator_state);
  }

 private:
  virtual StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const OneDimGroupingAggregatorState* aggregator_state) const = 0;
};

// OneDimGroupingAggregator class is a specialization of
// OneDimBaseGroupingAggregator.
//
// The specific means of accumulating values and producing default values are
// left to the subclasses.
//
// The implementation operates on AggVector<T> instances rather than tensors.
//
// This class is not thread safe.
template <typename InputT, typename OutputT = InputT>
class OneDimGroupingAggregator : public OneDimBaseGroupingAggregator {
 public:
  // TODO: b/279972547 - Support accumulating tensors of multiple dimensions. In
  // that case, the size of all dimensions but one (the dimension corresponding
  // to the ordinal tensor) should be known in advance and thus this constructor
  // should take in a shape with a single unknown dimension.
  OneDimGroupingAggregator()
      : OneDimGroupingAggregator(std::make_unique<MutableVectorData<OutputT>>(),
                                 0) {}

  OneDimGroupingAggregator(std::unique_ptr<MutableVectorData<OutputT>> data,
                           int num_inputs)
      : data_vector_(std::move(data)), num_inputs_(num_inputs) {}

  // Implementation of the tensor merge operation.
  Status MergeTensors(InputTensorList tensors, int num_inputs) override {
    TFF_RETURN_IF_ERROR(ValidateTensorInputs(tensors));

    const Tensor* tensor = tensors[1];
    if (tensor->dtype() != internal::TypeTraits<OutputT>::kDataType) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::MergeTensors: dtype mismatch "
                "for tensor 1";
    }
    num_inputs_ += num_inputs;
    AggVector<OutputT> value_vector = tensor->AsAggVector<OutputT>();
    AggVector<int64_t> ordinals_vector = tensors[0]->AsAggVector<int64_t>();

    ResizeDataVector(ordinals_vector);
    MergeVectorByOrdinals(ordinals_vector, value_vector);
    return absl::OkStatus();
  }

  int GetNumInputs() const override { return num_inputs_; }

  Status CheckValid() const override {
    if (data_vector_ == nullptr) {
      return TFF_STATUS(FAILED_PRECONDITION)
             << "OneDimGroupingAggregator::CheckValid: Output has already been "
                "consumed.";
    }
    return absl::OkStatus();
  }

  OneDimGroupingAggregatorState ToProto() override {
    // We don't expect the data_vector_ to be used any more after ToProto is
    // called.
    return CreateStateFromMutableVectorData(std::move(*data_vector_));
  }

  StatusOr<std::vector<OneDimGroupingAggregatorState>> PartitionToProtos(
      const Partitioner& partitioner) override {
    std::vector<OneDimGroupingAggregatorState> partitioned_states;
    partitioned_states.reserve(partitioner.GetNumPartitions());
    TFF_ASSIGN_OR_RETURN(auto partitioned_nested_aggregators,
                         partitioner.PartitionData<OutputT>(*data_vector_));

    for (auto& slice : partitioned_nested_aggregators) {
      partitioned_states.push_back(CreateStateFromMutableVectorData(
          MutableVectorData<OutputT>(std::move(slice))));
    }
    return partitioned_states;
  }

 protected:
  // Provides mutable access to the aggregator data as a vector<T>
  inline std::vector<OutputT>& data() { return *data_vector_; }

  // Implementation of the tensor aggregation.
  // Expects 2 tensors as input: a tensor containing ordinals and a tensor
  // containing values.
  //
  // Accumulates the values into the positions in the output tensor which are
  // indicated by the corresponding ordinals.
  Status AggregateTensors(InputTensorList tensors) override {
    TFF_RETURN_IF_ERROR(ValidateTensorInputs(tensors));

    const Tensor* tensor = tensors[1];
    if (tensor->dtype() != internal::TypeTraits<InputT>::kDataType) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: dtype mismatch "
                "for tensor 1";
    }
    num_inputs_++;
    AggVector<InputT> value_vector = tensor->AsAggVector<InputT>();
    AggVector<int64_t> ordinals_vector = tensors[0]->AsAggVector<int64_t>();

    ResizeDataVector(ordinals_vector);
    AggregateVectorByOrdinals(ordinals_vector, value_vector);
    return absl::OkStatus();
  }

  OutputTensorList TakeOutputs() && override {
    OutputTensorList outputs = std::vector<Tensor>();
    size_t outputs_size = data_vector_->size();
    TFF_CHECK(outputs_size <= LONG_MAX)
        << "TensorShape: Dimension size too large to be represented as a "
           "signed long.";
    outputs.push_back(
        Tensor::Create(internal::TypeTraits<OutputT>::kDataType,
                       TensorShape{static_cast<int64_t>(outputs_size)},
                       std::move(data_vector_))
            .value());
    data_vector_ = nullptr;
    return outputs;
  }

  // Delegates AggVector accumulation by ordinal to a derived class.
  //
  // The size of the vector returned by data() must be greater than the largest
  // ordinal in this vector.
  //
  // To avoid making a virtual function call per value in the tensor, the whole
  // vector is passed to the subclass for aggregation, which provides better
  // performance but comes at the cost of duplicated code between subclasses for
  // iterating over the vectors.
  virtual void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<InputT>& value_vector) = 0;

  // Delegates AggVector merging by ordinal to a derived class.
  //
  // The size of the vector returned by data() must be greater than the largest
  // ordinal in this vector.
  //
  // To avoid making a virtual function call per value in the tensor, the whole
  // vector is passed to the subclass for aggregation, which provides better
  // performance but comes at the cost of duplicated code between subclasses for
  // iterating over the vectors.
  virtual void MergeVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<OutputT>& value_vector) = 0;

  // Delegates initialization of previously unseen ordinals to a derived class.
  virtual OutputT GetDefaultValue() = 0;

 private:
  // TODO: b/447139788 - Optimize memory usage when serializing the state.
  OneDimGroupingAggregatorState CreateStateFromMutableVectorData(
      MutableVectorData<OutputT>&& data) {
    OneDimGroupingAggregatorState aggregator_state;
    aggregator_state.set_num_inputs(num_inputs_);
    *(aggregator_state.mutable_vector_data()) = data.EncodeContent();
    return aggregator_state;
  }

  Status ValidateTensorInputs(const InputTensorList& tensors) {
    TFF_CHECK(tensors.size() == 2)
        << "OneDimGroupingAggregator should operate on 2 input tensors";

    const Tensor* ordinals = tensors[0];
    if (ordinals->dtype() != DT_INT64) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: dtype mismatch "
                "for tensor 0. Expected DT_INT64.";
    }
    const Tensor* tensor = tensors[1];
    if (ordinals->shape() != tensor->shape()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: tensor shape "
                "mismatch. Shape of both tensors must be the same.";
    }
    size_t num_dimensions = tensor->shape().dim_sizes().size();
    if (num_dimensions > (size_t)1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: Only 1 "
                "dimensional tensors supported. Input tensor has "
             << num_dimensions << " dimensions.";
    }
    if (!ordinals->is_dense() || !tensor->is_dense()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: Only dense "
                "tensors are supported.";
    }
    return absl::OkStatus();
  }

  void ResizeDataVector(const AggVector<int64_t>& ordinals_vector) {
    size_t final_size = data_vector_->size();
    for (auto o : ordinals_vector) {
      if (o.value >= static_cast<int64_t>(final_size)) {
        final_size = o.value + 1;
      }
    }
    // Resize once outside the loop to avoid quadratic behavior.
    data_vector_->resize(final_size, GetDefaultValue());
  }

  std::unique_ptr<MutableVectorData<OutputT>> data_vector_;
  int num_inputs_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_ONE_DIM_GROUPING_AGGREGATOR_H_
