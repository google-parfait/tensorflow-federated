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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_group_by_aggregator.h"

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {
namespace internal {

namespace {

// Filters `survivor_indices` in place by checking whether each surviving row's
// key in `key_tensor` belongs to the domain specified by `domain_spec` at
// `domain_index`. Used by FilterSurvivorsByDomain to iteratively narrow down
// the surviving row indices across all key columns when a domain is specified.
template <typename T>
Status FilterRowsByKeyColumn(const Tensor& key_tensor, size_t domain_index,
                             const DomainSpec& domain_spec,
                             std::vector<size_t>& survivor_indices) {
  TFF_ASSIGN_OR_RETURN(const TypedColumnDomainSpec<T>* column_domain,
                       domain_spec.GetColumnDomain<T>(domain_index));
  absl::Span<const T> span = key_tensor.AsSpan<T>();
  size_t write_index = 0;
  for (size_t read_index = 0; read_index < survivor_indices.size();
       ++read_index) {
    size_t row = survivor_indices[read_index];
    if (column_domain->IsMember(span[row])) {
      if (write_index != read_index) {
        survivor_indices[write_index] = row;
      }
      ++write_index;
    }
  }
  survivor_indices.resize(write_index);
  return absl::OkStatus();
}

}  // namespace

Status FilterSurvivorsByDomain(const InputTensorList& tensors, size_t num_keys,
                               const DomainSpec& domain_spec,
                               std::vector<size_t>& survivor_indices) {
  for (size_t j = 0; j < num_keys; ++j) {
    DTYPE_CASES(tensors[j]->dtype(), T,
                TFF_RETURN_IF_ERROR(FilterRowsByKeyColumn<T>(
                    *tensors[j], j, domain_spec, survivor_indices)));
    if (survivor_indices.empty()) {
      break;
    }
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace aggregation
}  // namespace tensorflow_federated
