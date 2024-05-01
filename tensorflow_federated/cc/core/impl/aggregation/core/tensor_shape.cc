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

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

namespace tensorflow_federated {
namespace aggregation {

StatusOr<size_t> TensorShape::NumElements() const {
  size_t num_elements = 1;
  for (auto dim_size : dim_sizes_) {
    // If there are any dimensions of unknown size, the total number of elements
    // is also unknown.
    if (dim_size == -1)
      return TFF_STATUS(INVALID_ARGUMENT)
             << "TensorShape with unknown size has unknown number of elements.";
    num_elements *= dim_size;
  }
  return num_elements;
}

StatusOr<TensorShape> TensorShape::FromProto(
    const TensorShapeProto& shape_proto) {
  TensorShape::DimSizesVector dim_sizes;
  for (int64_t dim_size : shape_proto.dim_sizes()) {
    dim_sizes.push_back(dim_size);
  }
  TFF_RETURN_IF_ERROR(CheckValidDimSizes(dim_sizes));
  return TensorShape(std::move(dim_sizes));
}

TensorShapeProto TensorShape::ToProto() const {
  TensorShapeProto shape_proto;
  for (auto dim_size : dim_sizes()) {
    shape_proto.add_dim_sizes(static_cast<int64_t>(dim_size));
  }
  return shape_proto;
}

bool TensorShape::MatchesKnownDimensions(const TensorShape& other) const {
  if (this->dim_sizes().size() != other.dim_sizes().size()) {
    return false;
  }
  for (size_t i = 0; i < this->dim_sizes().size(); ++i) {
    if (this->dim_sizes()[i] == -1) {
      continue;
    }
    if (this->dim_sizes()[i] != other.dim_sizes()[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
