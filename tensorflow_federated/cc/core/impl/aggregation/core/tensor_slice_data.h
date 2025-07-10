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

#include "absl/types/span.h"
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
  // Move constructor.
  explicit TensorSliceData(Tensor&& tensor)
      : tensor_(std::move(tensor)),
        byte_size_(tensor_.data().byte_size()),
        byte_offset_(0) {}

  // Validates parameters and creates a TensorSliceData instance.
  static StatusOr<TensorSliceData> Create(Tensor&& tensor, size_t byte_size,
                                          size_t byte_offset = 0);

  // Reduce the size. Returns an error if the new size is larger than the
  // original size.
  Status ReduceByteSize(size_t new_size);

  template <typename T>
  StatusOr<absl::Span<T>> AsSpan() {
    TFF_RETURN_IF_ERROR(CheckValid<T>());
    return absl::Span<T>(const_cast<T*>(reinterpret_cast<const T*>(data())),
                         byte_size() / sizeof(T));
  }

  size_t byte_size() const override { return byte_size_; }
  const void* data() const override {
    return reinterpret_cast<const char*>(tensor_.data().data()) + byte_offset_;
  }

 private:
  TensorSliceData(Tensor&& tensor, size_t byte_size, size_t byte_offset)
      : tensor_(std::move(tensor)),
        byte_size_(byte_size),
        byte_offset_(byte_offset) {}

  Tensor tensor_;

  // Size in bytes.
  size_t byte_size_;

  // Offset in bytes.
  size_t byte_offset_;
};
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_SLICE_DATA_H_
