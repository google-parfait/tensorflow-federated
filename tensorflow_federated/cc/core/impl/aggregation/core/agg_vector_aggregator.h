/*
 * Copyright 2022 Google LLC
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

// AggVectorAggregator class is a specialization of TensorAggregator which
// operates on AggVector<T> instances rather than tensors.
template <typename T>
class AggVectorAggregator : public TensorAggregator {
 public:
  AggVectorAggregator(DataType dtype, TensorShape shape)
      : AggVectorAggregator(dtype, shape, CreateData(shape), 0) {}

  AggVectorAggregator(DataType dtype, TensorShape shape,
                      std::unique_ptr<MutableVectorData<T>> data,
                      int num_inputs)
      : dtype_(dtype),
        shape_(std::move(shape)),
        data_vector_(std::move(data)),
        num_inputs_(num_inputs) {
    TFF_CHECK(internal::TypeTraits<T>::kDataType == dtype)
        << "Incompatible dtype";
  }

  // Provides mutable access to the aggregator data as a vector<T>
  inline std::vector<T>& data() { return *data_vector_; }

  int GetNumInputs() const override { return num_inputs_; }

  Status MergeWith(TensorAggregator&& other) override {
    TFF_RETURN_IF_ERROR(CheckValid());
    TFF_ASSIGN_OR_RETURN(AggVectorAggregator<T> * other_ptr, CastOther(other));
    TFF_RETURN_IF_ERROR((*other_ptr).CheckValid());
    int64_t other_num_inputs = other.GetNumInputs();
    OutputTensorList output_tensors = std::move(*other_ptr).TakeOutputs();
    TFF_CHECK(output_tensors.size() == 1)
        << "AggVectorAggregator::MergeOutputTensors: AggVectorAggregator "
           "should produce a single output tensor";
    const Tensor& output = output_tensors[0];
    if (output.shape() != shape_) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::MergeOutputTensors: tensor shape "
                "mismatch";
    }
    // Delegate the actual aggregation to the specific aggregation
    // intrinsic implementation.
    AggregateVector(output.AsAggVector<T>());
    num_inputs_ += other_num_inputs;
    return TFF_STATUS(OK);
  }

  StatusOr<std::string> Serialize() && override {
    AggVectorAggregatorState aggregator_state;
    aggregator_state.set_num_inputs(num_inputs_);
    *(aggregator_state.mutable_vector_data()) = data_vector_->EncodeContent();
    return aggregator_state.SerializeAsString();
  }

  Status ValidateInputs(const InputTensorList& tensors) const override {
    if (tensors.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::ValidateInputs: Expected 1 tensor, got "
             << tensors.size();
    }
    const Tensor* tensor = tensors[0];
    if (tensor->dtype() != dtype_) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::ValidateInputs: dtype mismatch";
    }
    if (tensor->shape() != shape_) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::ValidateInputs: tensor shape mismatch";
    }
    return TFF_STATUS(OK);
  }

 protected:
  // Implementation of the tensor aggregation.
  Status AggregateTensors(InputTensorList tensors) override {
    // Delegate the actual aggregation to the specific aggregation
    // intrinsic implementation.
    AggregateVector(tensors[0]->AsAggVector<T>());
    num_inputs_++;
    return TFF_STATUS(OK);
  }

  Status CheckValid() const override {
    if (data_vector_ == nullptr) {
      return TFF_STATUS(FAILED_PRECONDITION)
             << "AggVectorAggregator::CheckValid: Output has already been "
             << "consumed.";
    }
    return TFF_STATUS(OK);
  }

  OutputTensorList TakeOutputs() && override {
    OutputTensorList outputs = std::vector<Tensor>();
    outputs.push_back(
        Tensor::Create(dtype_, shape_, std::move(data_vector_)).value());
    return outputs;
  }

  // Delegates AggVector aggregation to a derived class.
  virtual void AggregateVector(const AggVector<T>& agg_vector) = 0;

 private:
  static std::unique_ptr<MutableVectorData<T>> CreateData(
      const TensorShape& shape) {
    StatusOr<size_t> num_elements = shape.NumElements();
    TFF_CHECK(num_elements.ok()) << "AggVectorAggregator: All dimensions of "
                                    "tensor shape must be known in advance.";
    return std::make_unique<MutableVectorData<T>>(num_elements.value());
  }

  StatusOr<AggVectorAggregator<T>*> CastOther(TensorAggregator& other) {
    AggVectorAggregator<T>* other_ptr =
        dynamic_cast<AggVectorAggregator<T>*>(&other);
    if (other_ptr == nullptr) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::MergeOutputTensors: Can only merge with"
             << "another AggVectorAggregator operating on the same dtype "
             << internal::TypeTraits<T>::kDataType;
    }
    return other_ptr;
  }

  const DataType dtype_;
  const TensorShape shape_;
  std::unique_ptr<MutableVectorData<T>> data_vector_;
  int num_inputs_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_
