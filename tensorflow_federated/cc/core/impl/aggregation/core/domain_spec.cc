/*
 * Copyright 2026 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_spec.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

// Templated implementation of ColumnDomainSpec, IsMember will run in constant
// time via a hash table.
template <typename T>
class TypedColumnDomainSpec : public ColumnDomainSpec {
 public:
  explicit TypedColumnDomainSpec(const Tensor& domain_tensor)
      : tensor_dtype_(internal::TypeTraits<T>::kDataType) {
    auto agg_vector = domain_tensor.AsAggVector<T>();
    domain_set_.reserve(domain_tensor.num_elements());
    for (auto [index, value] : agg_vector) {
      domain_set_.insert(value);
    }
  }

  absl::StatusOr<bool> IsMember(const void* datum_ptr,
                                DataType dtype) const override {
    if (dtype != tensor_dtype_) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "ColumnDomainSpec::IsMember: DataType (" << dtype
             << ") does not match internal tensor_dtype (" << tensor_dtype_
             << ").";
    }
    return domain_set_.contains(*static_cast<const T*>(datum_ptr));
  }

 private:
  DataType tensor_dtype_;
  absl::flat_hash_set<T> domain_set_;
};

DomainSpec::DomainSpec(TensorSpan domain_tensors) {
  for (const auto& tensor : domain_tensors) {
    DTYPE_CASES(
        tensor.dtype(), T,
        columns_.push_back(std::make_unique<TypedColumnDomainSpec<T>>(tensor)));
  }
}

}  // namespace aggregation
}  // namespace tensorflow_federated
