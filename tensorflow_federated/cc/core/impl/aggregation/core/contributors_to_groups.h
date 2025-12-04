// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_CONTRIBUTORS_TO_GROUPS_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_CONTRIBUTORS_TO_GROUPS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorflow_federated::aggregation {

using PrivID = int64_t;

class ContributorsToGroups {
 public:
  explicit ContributorsToGroups(int max_contributors_to_group);

  // Adds a contributor to group `group_index`. For this CL, only the case where
  // priv_id is std::nullopt is handled, incrementing the count for group
  // `group_index` up to max_contributors_to_group_.
  absl::Status AddToGroup(size_t group_index,
                          std::optional<PrivID> priv_id = std::nullopt);

  // Adds `count` to the number of contributors for group `group_index`. The
  // total count is capped at `max_contributors_to_group_`. Returns
  // InvalidArgumentError if count is negative.
  absl::Status AddCountToContributors(size_t group_index, int count);

  // Returns the number of contributors for group `group_index`, capped at
  // max_contributors_to_group_.
  absl::StatusOr<int> GetCount(size_t group_index) const;

  // Returns the number of contributors for each group, capped at
  // max_contributors_to_group_.
  const std::vector<int>& GetCounts() const { return counts_; }

 private:
  int max_contributors_to_group_;
  // The number of contributors for each group, the group indices are guaranteed
  // to be contiguous and start at 0.
  std::vector<int> counts_;
};

}  // namespace tensorflow_federated::aggregation

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_CONTRIBUTORS_TO_GROUPS_H_
