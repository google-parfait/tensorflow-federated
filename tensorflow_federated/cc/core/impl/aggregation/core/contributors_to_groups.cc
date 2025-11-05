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
#include "tensorflow_federated/cc/core/impl/aggregation/core/contributors_to_groups.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace tensorflow_federated {
namespace aggregation {

ContributorsToGroups::ContributorsToGroups(int max_contributors_to_group,
                                           std::vector<int> contributor_counts)
    : max_contributors_to_group_(max_contributors_to_group) {
  counts_ = std::move(contributor_counts);
}

absl::Status ContributorsToGroups::AddToGroup(size_t i,
                                              std::optional<PrivID> priv_id) {
  if (priv_id.has_value()) {
    return absl::UnimplementedError("priv_id is not supported yet.");
  }
  if (i >= counts_.size()) {
    counts_.resize(i + 1, 0);
  }
  if (counts_[i] < max_contributors_to_group_) {
    counts_[i]++;
  }
  return absl::OkStatus();
}

absl::Status ContributorsToGroups::AddCountToContributors(size_t i, int count) {
  if (count < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Count must be non-negative, but got ", count));
  }
  if (i >= counts_.size()) {
    counts_.resize(i + 1, 0);
  }
  counts_[i] = std::min(counts_[i] + count, max_contributors_to_group_);
  return absl::OkStatus();
}

absl::Status ContributorsToGroups::IncreaseMaxContributorsToGroup(
    int max_contributors_to_group) {
  if (max_contributors_to_group < max_contributors_to_group_) {
    return absl::InvalidArgumentError(
        "max_contributors_to_group cannot be decreased.");
  }
  max_contributors_to_group_ = max_contributors_to_group;
  return absl::OkStatus();
}

absl::StatusOr<int> ContributorsToGroups::GetCount(size_t i) const {
  if (i >= counts_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Group index ", i,
        " is out of bounds. Maxiumum group index is: ", counts_.size() - 1));
  }
  return counts_[i];
}

}  // namespace aggregation
}  // namespace tensorflow_federated
