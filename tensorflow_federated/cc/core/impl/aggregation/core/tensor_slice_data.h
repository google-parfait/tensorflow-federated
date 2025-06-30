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
#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_SLICE_DATA_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_SLICE_DATA_H_

#include <cstddef>
#include <utility>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {
// TensorSliceData wraps around a Tensor. It exposes an interface to reorganize
// the data in the tensor and reduce the size, thereby becoming a slice of the
// original tensor.
class TensorSliceData : public TensorData {
 public:
  explicit TensorSliceData(Tensor&& tensor)
      : tensor_(std::move(tensor)), size_(tensor_.data().byte_size()) {}

  // Reduce the size. Returns an error if the new size is larger than the
  // original size.
  Status ReduceSize(size_t new_size);

  // Swap the values at the given indices after casting to a target type.
  // Returns an error if the indices are out of range.
  template <typename T>
  Status SwapValuesAtIndices(size_t index1, size_t index2) {
    TFF_RETURN_IF_ERROR(CheckValid<T>());
    size_t num_elements = size_ / sizeof(T);
    if (index1 >= num_elements || index2 >= num_elements) {
      return TFF_STATUS(FAILED_PRECONDITION)
             << "TensorSliceData::SwapValuesAtIndices: indices must be in range"
             << "[0, " << num_elements << ") but got " << index1 << " and "
             << index2 << "\n";
    }

    T* data = const_cast<T*>(reinterpret_cast<const T*>(tensor_.data().data()));
    std::swap(data[index1], data[index2]);
    return TFF_STATUS(OK);
  }

  size_t byte_size() const override { return size_; }
  const void* data() const override { return tensor_.data().data(); }

 private:
  // the original tensor owned by this data
  Tensor tensor_;
  size_t size_;
};
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_SLICE_DATA_H_
