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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_CLOSED_DOMAIN_GROUP_BY_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_CLOSED_DOMAIN_GROUP_BY_AGGREGATOR_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"

namespace tensorflow_federated {
namespace aggregation {

namespace internal {

// Filters `survivor_indices` in place across the first `num_keys` tensors of
// `tensors`, retaining only rows whose keys belong to `domain_spec`.
Status FilterSurvivorsByDomain(const InputTensorList& tensors, size_t num_keys,
                               const DomainSpec& domain_spec,
                               std::vector<size_t>& survivor_indices);

}  // namespace internal

// Derived class of DPGroupByAggregator (or a subclass thereof) that takes a
// DomainSpec and filters out input rows whose keys are not in the domain
// before L0 contribution bounding.
template <typename Base = DPGroupByAggregator>
class DPClosedDomainGroupByAggregator : public Base {
 public:
  template <typename... Args>
  explicit DPClosedDomainGroupByAggregator(DomainSpec domain_spec,
                                           Args&&... args)
      : Base(std::forward<Args>(args)...),
        domain_spec_(std::move(domain_spec)) {}

 protected:
  Status FilterSurvivors(const InputTensorList& tensors,
                         std::vector<size_t>& survivor_indices) const override {
    return internal::FilterSurvivorsByDomain(
        tensors, this->num_keys_per_input(), domain_spec_, survivor_indices);
  }

 private:
  DomainSpec domain_spec_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_CLOSED_DOMAIN_GROUP_BY_AGGREGATOR_H_
