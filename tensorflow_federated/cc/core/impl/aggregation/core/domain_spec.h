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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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

// Base class for a single column's domain specification.
class ColumnDomainSpec {
 public:
  explicit ColumnDomainSpec(DataType dtype) : dtype_(dtype) {}
  virtual ~ColumnDomainSpec() = default;

  DataType dtype() const { return dtype_; }

 private:
  DataType dtype_;
};

// Templated implementation of ColumnDomainSpec; IsMember runs in constant
// time via a hash table.
template <typename T>
class TypedColumnDomainSpec : public ColumnDomainSpec {
 public:
  explicit TypedColumnDomainSpec(const Tensor& domain_tensor)
      : ColumnDomainSpec(internal::TypeTraits<T>::kDataType) {
    auto agg_vector = domain_tensor.AsAggVector<T>();
    domain_set_.reserve(domain_tensor.num_elements());
    for (auto [index, value] : agg_vector) {
      domain_set_.insert(value);
    }
  }

  // Returns whether `datum` is a member of this column's domain.
  bool IsMember(const T& datum) const { return domain_set_.contains(datum); }

 private:
  absl::flat_hash_set<T> domain_set_;
};

// DomainSpec class that holds multiple ColumnDomainSpecs.
class DomainSpec {
 public:
  explicit DomainSpec(TensorSpan domain_tensors);
  ~DomainSpec() = default;
  DomainSpec(DomainSpec&&) = default;
  DomainSpec& operator=(DomainSpec&&) = default;

  // Returns the typed column domain for the column at the given index after
  // validating the index bounds and DataType.
  template <typename T>
  absl::StatusOr<const TypedColumnDomainSpec<T>*> GetColumnDomain(
      int index) const {
    if (index < 0 || index >= columns_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("DomainSpec::GetColumnDomain: index ", index,
                       " out of bounds [0, ", columns_.size(), ")."));
    }
    DataType expected_dtype = internal::TypeTraits<T>::kDataType;
    if (columns_[index]->dtype() != expected_dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("ColumnDomainSpec: DataType (", expected_dtype,
                       ") does not match internal tensor_dtype (",
                       columns_[index]->dtype(), ")."));
    }
    return static_cast<const TypedColumnDomainSpec<T>*>(columns_[index].get());
  }

  // Returns whether the datum is a member of the domain for the column at the
  // given index.
  template <typename V>
  absl::StatusOr<bool> IsMember(V datum, int index) const {
    TFF_ASSIGN_OR_RETURN(const TypedColumnDomainSpec<V>* column_domain,
                         GetColumnDomain<V>(index));
    return column_domain->IsMember(datum);
  }

 private:
  std::vector<std::unique_ptr<ColumnDomainSpec>> columns_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DOMAIN_SPEC_H_
