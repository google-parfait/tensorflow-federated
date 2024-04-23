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

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

#include <cstddef>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"

namespace tensorflow_federated {
namespace aggregation {

bool TensorData::IsAligned(const void* data, size_t alignment_size) {
  return (reinterpret_cast<size_t>(data) % alignment_size) == 0;
}

Status TensorData::CheckValid(size_t value_size, size_t alignment_size) const {
  TFF_CHECK(value_size > 0);

  if (byte_size() > 0) {
    if ((byte_size() % value_size) != 0) {
      return TFF_STATUS(FAILED_PRECONDITION)
             << "TensorData: byte_size() must be a multiple of value_size "
             << value_size;
    }

    if (!IsAligned(data(), alignment_size)) {
      return TFF_STATUS(FAILED_PRECONDITION)
             << "TensorData: data() address is not aligned by alignment_size "
             << alignment_size;
    }
  }

  return TFF_STATUS(OK);
}

}  // namespace aggregation
}  // namespace tensorflow_federated
