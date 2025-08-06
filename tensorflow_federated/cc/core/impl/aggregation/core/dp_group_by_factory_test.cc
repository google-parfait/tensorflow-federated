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

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_open_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::HasSubstr;

using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsicWithKeyTypes;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsicWithMinContributors;

TEST(DPGroupByFactoryTest, CreateAggregatorWithMinContributorsNoKeys) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/10, /*key_types=*/{});
  TFF_ASSERT_OK_AND_ASSIGN(auto aggregator, CreateTensorAggregator(intrinsic));
  // Check that the returned aggregator is a DPOpenDomainHistogram.
  auto* dp_open_domain_histogram =
      dynamic_cast<DPOpenDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_open_domain_histogram, nullptr);
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithMinContributorsWithKeys) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/10, /*key_types=*/{DT_STRING});
  TFF_ASSERT_OK_AND_ASSIGN(auto aggregator, CreateTensorAggregator(intrinsic));
  // Check that the returned aggregator is a DPOpenDomainHistogram.
  auto* dp_open_domain_histogram =
      dynamic_cast<DPOpenDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_open_domain_histogram, nullptr);
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithZeroMinContributors_Fails) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/0);
  EXPECT_THAT(
      CreateTensorAggregator(intrinsic),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr("min_contributors_to_group must be positive")));
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithNegativeMinContributors_Fails) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/-1);
  EXPECT_THAT(
      CreateTensorAggregator(intrinsic),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr("min_contributors_to_group must be positive")));
}

// Make sure we can successfully create a DPOpenDomainHistogram object with no
// keys.
TEST(DPGroupByFactoryTest, CreateAggregatorNoKeys_Success) {
  // Create intrinsic with default parameters except no key_types.
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
      kEpsilonThreshold, 0.001, 100, 100, -1, -1, /*key_types=*/{});
  TFF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TensorAggregator> aggregator,
                           CreateTensorAggregator(intrinsic));

  // Validate that domain_tensors is a valid empty span.
  auto* dp_open_domain_histogram =
      dynamic_cast<DPOpenDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_open_domain_histogram, nullptr);
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithKeysNoMinContributors_Success) {
  // Create intrinsic with default parameters and key_types, and no min
  // contributors.
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
      kEpsilonThreshold, 0.001, 100, 100, -1, -1, /*key_types=*/{DT_STRING});
  TFF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TensorAggregator> aggregator,
                           CreateTensorAggregator(intrinsic));

  // Validate that the aggregator is a DPClosedDomainHistogram.
  auto* dp_closed_domain_histogram =
      dynamic_cast<DPClosedDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_closed_domain_histogram, nullptr);
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
