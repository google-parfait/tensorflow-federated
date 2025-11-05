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
#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_CONTRIBUTORS_TO_GROUPS_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_CONTRIBUTORS_TO_GROUPS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorflow_federated {
namespace aggregation {

using PrivID = int64_t;

class ContributorsToGroups {
 public:
  ContributorsToGroups() = default;
  explicit ContributorsToGroups(int max_contributors_to_group,
                                std::vector<int> contributor_counts = {});

  // Increases the max number of contributors to group to
  // `max_contributors_to_group`. This is needed when the max number of
  // contributors to group is not known at construction time.
  absl::Status IncreaseMaxContributorsToGroup(int max_contributors_to_group);

  // Adds a contributor to group `i`. For this CL, only the case where priv_id
  // is std::nullopt is handled, incrementing the count for group `i` up to
  // max_contributors_to_group_.
  absl::Status AddToGroup(size_t i, std::optional<PrivID> priv_id);

  // Adds `count` to the number of contributors for group `i`. The total count
  // is capped at `max_contributors_to_group_`.
  // Returns InvalidArgumentError if count is negative.
  absl::Status AddCountToContributors(size_t i, int count);

  // Returns the number of contributors for group `i`, capped at
  // max_contributors_to_group_.
  absl::StatusOr<int> GetCount(size_t i) const;

  // Returns the number of contributors for each group, capped at
  // max_contributors_to_group_.
  const std::vector<int>& GetCounts() const { return counts_; }

  // Returns the size of the counts vector.
  size_t GetSize() const { return counts_.size(); }

  // Extends the counts vector to size `i`. This is needed only to improve
  // performance.
  void ExtendTo(size_t i) {
    if (i >= counts_.size()) {
      counts_.resize(i + 1, 0);
    }
  }

 private:
  int max_contributors_to_group_;
  std::vector<int> counts_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_CONTRIBUTORS_TO_GROUPS_H_
