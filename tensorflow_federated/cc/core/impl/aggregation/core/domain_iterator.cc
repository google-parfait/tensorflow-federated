/*
 * Copyright 2024 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_iterator.h"

#include <cstdint>

#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

namespace internal {

// Advance through the composite key domain.
void DomainIterator::Increment(int64_t which_key) {
  ++domain_indices_[which_key];
  // If we've reached the end of a key's domain...
  if (domain_indices_[which_key] == domain_tensors_[which_key].num_elements()) {
    // ...reset to 0
    domain_indices_[which_key] = 0;
    // If there is another key after this one, increment its corresponding index
    if (which_key + 1 < domain_tensors_.size()) {
      Increment(which_key + 1);
      return;
    }
    // Otherwise, we've wrapped around the end of the composite key domain.
    wrapped_around_ = true;
  }
}

DomainIteratorForKeys& DomainIteratorForKeys::operator++() {
  Increment();
  return *this;
}

// The data of interest when we are processing keys is the value of the key
// indexed by which_key_ in the current composite key.
const void* DomainIteratorForKeys::operator*() {
  // Get the tensor of possible key values indexed by which_key.
  const Tensor& tensor = domain_tensors()[which_key_];

  // Get an index within that tensor.
  int64_t tensor_index = domain_indices()[which_key_];

  DTYPE_CASES(tensor.dtype(), T,
              return reinterpret_cast<const void*>(
                  &(tensor.AsSpan<T>()[tensor_index])));
  return nullptr;
}

DomainIteratorForAggregations& DomainIteratorForAggregations::operator++() {
  Increment();
  return *this;
}

// The data of interest when we are processing aggregates is the aggregate
// corresponding to the current composite key.
const void* DomainIteratorForAggregations::operator*() {
  // Given a combination of keys (indexed by domain_indices_), get the ordinal
  // corresponding to that combination.
  int64_t ordinal =
      dp_key_combiner_.GetOrdinal(domain_tensors(), domain_indices());
  if (ordinal >= 0) {
    // Get the aggregated value associated with the ordinal.
    DTYPE_CASES(aggregation_.dtype(), T,
                return reinterpret_cast<const void*>(
                    &(aggregation_.AsSpan<T>()[ordinal])));
  }
  return nullptr;
}

}  // namespace internal
}  // namespace aggregation
}  // namespace tensorflow_federated
