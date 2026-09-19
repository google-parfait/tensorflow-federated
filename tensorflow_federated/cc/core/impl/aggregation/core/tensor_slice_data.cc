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
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_slice_data.h"

#include <cstddef>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {
Status TensorSliceData::ReduceByteSize(size_t new_size) {
  if (new_size > byte_size_) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorSliceData::ReduceSize: target size ", new_size,
                     " is greater than the original size ", byte_size_));
  }
  byte_size_ = new_size;
  return absl::OkStatus();
}

absl::StatusOr<TensorSliceData> TensorSliceData::Create(Tensor&& tensor,
                                                        size_t byte_size,
                                                        size_t byte_offset) {
  if (byte_offset + byte_size > tensor.data().byte_size()) {
    return absl::InvalidArgumentError(
        "TensorSliceData::Create: byte_offset + byte_size cannot exceed "
        "the tensor's size");
  }
  return TensorSliceData(std::move(tensor), byte_size, byte_offset);
}

}  // namespace aggregation
}  // namespace tensorflow_federated
