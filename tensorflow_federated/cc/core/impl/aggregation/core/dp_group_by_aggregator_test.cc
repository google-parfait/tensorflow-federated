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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_group_by_aggregator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

// Mock class for testing the abstract DPGroupByAggregator.
class MockDPGroupByAggregator : public DPGroupByAggregator {
 public:
  MockDPGroupByAggregator(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::unique_ptr<CompositeKeyCombiner> key_combiner,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      int num_inputs, double epsilon, double delta,
      int64_t max_groups_contributed,
      std::optional<int> min_contributors_to_group = std::nullopt,
      std::vector<int> contributors_to_groups = {},
      int max_string_length = kDefaultMaxStringLength)
      : DPGroupByAggregator(input_key_specs, output_key_specs, intrinsics,
                            std::move(key_combiner), std::move(aggregators),
                            num_inputs, epsilon, delta, max_groups_contributed,
                            min_contributors_to_group, contributors_to_groups,
                            max_string_length) {}
  inline double GetEpsilonPerAgg() { return epsilon_per_agg(); }
  inline double GetDeltaPerAgg() { return delta_per_agg(); }

 protected:
  StatusOr<OutputTensorList> NoisyReport() override {
    return absl::UnimplementedError("Not implemented.");
  }
};

MockDPGroupByAggregator CreateMockForTestingEpsilonAndDeltaSplit(
    double epsilon, double delta, int num_intrinsics) {
  std::vector<TensorSpec> input_key_specs;
  input_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key", DT_STRING));
  std::vector<TensorSpec> output_key_specs;
  output_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key_out", DT_STRING));
  std::vector<Intrinsic> intrinsics;
  for (int i = 0; i < num_intrinsics; ++i) {
    intrinsics.push_back(Intrinsic());
  }
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators;
  for (int i = 0; i < num_intrinsics; ++i) {
    aggregators.push_back(nullptr);
  }
  return MockDPGroupByAggregator(input_key_specs, &output_key_specs,
                                 &intrinsics, nullptr, std::move(aggregators),
                                 0, epsilon, delta, 1);
}

// First batch of tests: ensure that epsilon is split when appropriate
// Do not split when epsilon crosses its threshold.
TEST(DPGroupByAggregatorTest, DoNotSplitEpsilonWhenExceedsThreshold) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/kEpsilonThreshold + 1, /*delta=*/0.1, /*num_intrinsics=*/2);
  EXPECT_EQ(aggregator.GetEpsilonPerAgg(), kEpsilonThreshold);
}
// Do not split when `intrinsics` has size 1.
TEST(DPGroupByAggregatorTest, DoNotSplitEpsilonWhenSingleIntrinsic) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/0.1, /*delta=*/0.1, /*num_intrinsics=*/1);
  EXPECT_EQ(aggregator.GetEpsilonPerAgg(), 0.1);
}
// Do split when epsilon is below its threshold and `intrinsics` has size > 1.
TEST(DPGroupByAggregatorTest, SplitEpsilonWhenBelowThresholdAndManyIntrinsics) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/0.1, /*delta=*/0.1, /*num_intrinsics=*/2);
  EXPECT_EQ(aggregator.GetEpsilonPerAgg(), 0.05);
}

// Second batch of tests: ensure that delta is split when appropriate
// Do not split when `intrinsics` has size 1.
TEST(DPGroupByAggregatorTest, DoNotSplitDeltaWhenSingleIntrinsic) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/0.1, /*delta=*/0.1, /*num_intrinsics=*/1);
  EXPECT_EQ(aggregator.GetDeltaPerAgg(), 0.1);
}
// Do split when `intrinsics` has size > 1.
TEST(DPGroupByAggregatorTest, SplitDeltaWhenManyIntrinsics) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/0.1, /*delta=*/0.1, /*num_intrinsics=*/2);
  EXPECT_EQ(aggregator.GetDeltaPerAgg(), 0.05);
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
