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

#include <cstddef>
#include <optional>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated::aggregation {

class ContributorsToGroupsPeer {
 public:
  static size_t GetContributorsSize(const ContributorsToGroups& ctg) {
    return ctg.contributors_.size();
  }
  static const absl::flat_hash_set<PrivID>& GetContributorsForGroup(
      const ContributorsToGroups& ctg, size_t group_index) {
    return ctg.contributors_[group_index];
  }
};

namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::SizeIs;

TEST(ContributorsToGroupsTest, AddToGroupNoPrivId) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);

  // Add to group 0 twice
  TFF_EXPECT_OK(ctg.AddToGroup(0));
  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Add to group 2 (out of bounds initially)
  TFF_EXPECT_OK(ctg.AddToGroup(2));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));
  EXPECT_THAT(ctg.GetCount(2), IsOkAndHolds(1));
}

TEST(ContributorsToGroupsTest, AddToGroupCappedAtMax) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/2);

  TFF_EXPECT_OK(ctg.AddToGroup(0));
  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Adding more should not increase the count beyond max
  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));
}

TEST(ContributorsToGroupsTest, AddToGroupWithPrivIdCountsUnique) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  PrivID priv_id1 = 1;
  PrivID priv_id2 = 2;

  // Add priv_id1 to group 0. Count should be 1.
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(1));

  // Add priv_id1 to group 0 again. Count should remain 1.
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(1));

  // Add priv_id2 to group 0. Count should be 2.
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id2));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Add priv_id1 to group 1. Group 1 count should be 1 and group 0 should be 2.
  TFF_EXPECT_OK(ctg.AddToGroup(1, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));
  EXPECT_THAT(ctg.GetCount(1), IsOkAndHolds(1));
}

TEST(ContributorsToGroupsTest,
     AddToGroupWithPrivIdCountsUniqueWithMissingPrivID) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  PrivID priv_id1 = 1;

  // Add priv_id1 to group 0. Count should be 1.
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(1));

  // Add priv_id1 to group 0 again. Count should remain 1.
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(1));

  // Add anonymous contributor to group 0. Count should be 2.
  TFF_EXPECT_OK(ctg.AddToGroup(0, std::nullopt));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Add priv_id1 to group 1. Group 1 count should be 1 and group 0 should be 2.
  TFF_EXPECT_OK(ctg.AddToGroup(1, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));
  EXPECT_THAT(ctg.GetCount(1), IsOkAndHolds(1));
}

TEST(ContributorsToGroupsTest, AddToGroupWithPrivIdCappedAtMax) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/2);
  PrivID priv_id1 = 1;
  PrivID priv_id2 = 2;
  PrivID priv_id3 = 3;

  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id1));
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id2));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Adding more unique priv_ids should not increase the count beyond max
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id3));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Adding already known priv_id should not increase count.
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));
}

TEST(ContributorsToGroupsTest, AddToGroupNoMaxContributorsToGroup) {
  ContributorsToGroups ctg;
  EXPECT_THAT(ctg.AddToGroup(0, std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ContributorsToGroupsTest, AddCountToContributors) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/5);

  // Add count to group 0
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 3));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(3));

  // Add count to group 1 (out of bounds initially)
  TFF_EXPECT_OK(ctg.AddCountToContributors(1, 2));
  EXPECT_THAT(ctg.GetCount(1), IsOkAndHolds(2));

  // Add count to group 0 again
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(4));
}

TEST(ContributorsToGroupsTest, AddCountToContributorsCappedAtMax) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/5);

  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 3));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(3));

  // Adding count that would exceed max
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 3));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(5));

  // Adding more when already at max
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(5));
}

TEST(ContributorsToGroupsTest, AddCountToContributorsNegativeOrZeroCount) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/5);

  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 2));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Negative count should return an error
  EXPECT_THAT(ctg.AddCountToContributors(0, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));

  // Zero count should be ignored
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 0));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));
}

TEST(ContributorsToGroupsTest, AddCountToContributorsNoMaxContributorsToGroup) {
  ContributorsToGroups ctg;
  EXPECT_THAT(ctg.AddCountToContributors(0, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ContributorsToGroupsTest, GetCountOutOfBounds) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  EXPECT_THAT(ctg.GetCount(0), StatusIs(absl::StatusCode::kInvalidArgument));
  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_THAT(ctg.GetCount(1), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ContributorsToGroupsTest, GetCounts) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/5);
  EXPECT_THAT(ctg.GetCounts(), IsEmpty());

  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_THAT(ctg.GetCounts(), ElementsAre(1));

  TFF_EXPECT_OK(ctg.AddCountToContributors(2, 3));
  EXPECT_THAT(ctg.GetCounts(), ElementsAre(1, 0, 3));

  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_THAT(ctg.GetCounts(), ElementsAre(2, 0, 3));

  TFF_EXPECT_OK(ctg.AddCountToContributors(1, 4));
  EXPECT_THAT(ctg.GetCounts(), ElementsAre(2, 4, 3));
}

TEST(ContributorsToGroupsTest, IncreaseMaxContributorsToGroupSuccess) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 3));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(3));

  TFF_EXPECT_OK(ctg.IncreaseMaxContributorsToGroup(5));
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(4));

  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 2));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(5));
}

TEST(ContributorsToGroupsTest, IncreaseMaxContributorsToGroupNoChange) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 2));
  TFF_EXPECT_OK(ctg.IncreaseMaxContributorsToGroup(3));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(2));
}

TEST(ContributorsToGroupsTest, IncreaseMaxContributorsToGroupFailDecrease) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  EXPECT_THAT(ctg.IncreaseMaxContributorsToGroup(2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ContributorsToGroupsTest, IncreaseMaxContributorsToGroupKeepsCounts) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  TFF_EXPECT_OK(ctg.AddCountToContributors(0, 2));
  TFF_EXPECT_OK(ctg.AddCountToContributors(1, 3));
  EXPECT_THAT(ctg.GetCounts(), ElementsAre(2, 3));

  TFF_EXPECT_OK(ctg.IncreaseMaxContributorsToGroup(5));
  EXPECT_THAT(ctg.GetCounts(), ElementsAre(2, 3));
}

TEST(ContributorsToGroupsTest,
     AddToGroupWithPrivIdDoesNotAddPrivIdAfterMaxReached) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/1);
  PrivID priv_id1 = 1;
  PrivID priv_id2 = 2;

  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id1));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(1));
  EXPECT_THAT(ContributorsToGroupsPeer::GetContributorsForGroup(ctg, 0),
              ElementsAre(priv_id1));

  // Adding another priv_id should not increase count or add priv_id2 to
  // contributors set for group 0 because max_contributors_to_group has been
  // reached.
  TFF_EXPECT_OK(ctg.AddToGroup(0, priv_id2));
  EXPECT_THAT(ctg.GetCount(0), IsOkAndHolds(1));
  EXPECT_THAT(ContributorsToGroupsPeer::GetContributorsForGroup(ctg, 0),
              SizeIs(Le(1)));
}

TEST(ContributorsToGroupsTest, AddToGroupNoPrivIdDoesNotExtendContributors) {
  ContributorsToGroups ctg(/*max_contributors_to_group=*/3);
  EXPECT_EQ(ContributorsToGroupsPeer::GetContributorsSize(ctg), 0);
  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_EQ(ContributorsToGroupsPeer::GetContributorsSize(ctg), 0);
  TFF_EXPECT_OK(ctg.AddToGroup(2));
  EXPECT_EQ(ContributorsToGroupsPeer::GetContributorsSize(ctg), 0);
  EXPECT_EQ(ctg.GetCounts().size(), 3);
  TFF_EXPECT_OK(ctg.AddToGroup(1, 1234));
  EXPECT_EQ(ContributorsToGroupsPeer::GetContributorsSize(ctg), 3);
  TFF_EXPECT_OK(ctg.AddToGroup(0));
  EXPECT_EQ(ContributorsToGroupsPeer::GetContributorsSize(ctg), 3);
}

}  // namespace

}  // namespace tensorflow_federated::aggregation
