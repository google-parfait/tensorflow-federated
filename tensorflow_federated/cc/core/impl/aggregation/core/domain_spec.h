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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DOMAIN_SPEC_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DOMAIN_SPEC_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

// `TensorSpan` is an alias for a span of `Tensor`s, used to pass around
// lists of tensors that represent domain specifications, e.g. in
// `DPGroupByFactory` and `DomainSpec`. This alias is intended for client use,
// and can be relied on to be an `absl::Span<const Tensor>`.
using TensorSpan = absl::Span<const Tensor>;

// Abstract base class for column domain specification.
class ColumnDomainSpec {
 public:
  virtual ~ColumnDomainSpec() = default;

  // Returns whether the datum is a member of the domain.
  virtual absl::StatusOr<bool> IsMember(const void* datum_ptr,
                                        DataType dtype) const = 0;
};

// DomainSpec class that holds multiple ColumnDomainSpecs.
class DomainSpec {
 public:
  explicit DomainSpec(TensorSpan domain_tensors);
  ~DomainSpec() = default;

  // Returns whether the datum is a member of the domain for the column at the
  // given index.
  template <typename V>
  absl::StatusOr<bool> IsMember(V datum, int index) const {
    if (index < 0 || index >= columns_.size()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DomainSpec::IsMember: index " << index << " out of bounds [0, "
             << columns_.size() << ").";
    }
    return columns_[index]->IsMember(&datum,
                                     internal::TypeTraits<V>::kDataType);
  }

 private:
  std::vector<std::unique_ptr<ColumnDomainSpec>> columns_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DOMAIN_SPEC_H_
