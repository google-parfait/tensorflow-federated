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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateInnerIntrinsic;
using ::testing::HasSubstr;

// Mock class for testing the abstract DPGroupByAggregator.
class MockDPGroupByAggregator : public DPGroupByAggregator {
 public:
  MockDPGroupByAggregator(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      int num_inputs, double epsilon, double delta,
      int64_t max_groups_contributed,
      std::optional<int> min_contributors_to_group = std::nullopt,
      std::vector<int> contributors_to_groups = {},
      int max_string_length = kDefaultMaxStringLength)
      : DPGroupByAggregator(
            input_key_specs, output_key_specs, intrinsics,
            std::make_unique<CompositeKeyCombiner>(CreateKeyTypes(
                input_key_specs.size(), input_key_specs, *output_key_specs)),
            std::move(aggregators), num_inputs, epsilon, delta,
            max_groups_contributed, min_contributors_to_group,
            contributors_to_groups, max_string_length) {}
  inline double GetEpsilonPerAgg() { return epsilon_per_agg(); }
  inline double GetDeltaPerAgg() { return delta_per_agg(); }
  inline StatusOr<int64_t> SerializeSensitivity() {
    return DPGroupByAggregator::CalculateSerializeSensitivity();
  }

 protected:
  StatusOr<OutputTensorList> NoisyReport() override {
    return absl::UnimplementedError("Not implemented.");
  }
};

// Creates a default inner intrinsic for testing.
template <typename InputType, typename OutputType>
Intrinsic CreateDefaultInnerIntrinsic() {
  // The parameters will not be used in this suite of tests; the types are what
  // are important.
  return CreateInnerIntrinsic<InputType, OutputType>(1, -1, -1);
}

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
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  }
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators;
  for (int i = 0; i < num_intrinsics; ++i) {
    aggregators.push_back(nullptr);
  }
  return MockDPGroupByAggregator(input_key_specs, &output_key_specs,
                                 &intrinsics, std::move(aggregators),
                                 /*num_inputs=*/0, epsilon, delta,
                                 /*max_groups_contributed=*/1);
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

// Third batch of tests: ensure that the string length is checked.
TEST(DPGroupByAggregatorTest, StringLengthCheck) {
  // The first step is to prepare the inner intrinsic and aggregator.
  Intrinsic intrinsic = CreateDefaultInnerIntrinsic<int32_t, int64_t>();
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(std::move(intrinsic));
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators;
  auto one_dim_base_aggregator = std::unique_ptr<OneDimBaseGroupingAggregator>(
      dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator.release()));
  aggregators.push_back(std::move(one_dim_base_aggregator));

  // Next we make the top-level MockDPGroupByAggregator with a max string
  // length of 9.
  std::vector<TensorSpec> input_key_specs;
  input_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key", DT_STRING));
  std::vector<TensorSpec> output_key_specs;
  output_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key_out", DT_STRING));
  MockDPGroupByAggregator top_aggregator(
      input_key_specs, &output_key_specs, &intrinsics, std::move(aggregators),
      /*num_inputs=*/0, /*epsilon=*/0.1, /*delta=*/0.1,
      /*max_groups_contributed=*/1, std::nullopt, {}, /*max_string_length=*/9);

  // Finally we show that a long string is rejected but a short one is accepted.
  Tensor short_keys =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"short"}))
          .value();
  Tensor long_keys = Tensor::Create(DT_STRING, {1},
                                    CreateTestData<string_view>({"0123456789"}))
                         .value();
  Tensor v1 =
      Tensor::Create(DT_INT32, {1}, CreateTestData<int32_t>({1})).value();
  Tensor v2 =
      Tensor::Create(DT_INT32, {1}, CreateTestData<int32_t>({2})).value();
  TFF_EXPECT_OK(top_aggregator.ValidateInputs({&short_keys, &v1}));
  EXPECT_THAT(top_aggregator.ValidateInputs({&long_keys, &v2}),
              StatusIs(INVALID_ARGUMENT, HasSubstr("tensor 0")));
}

// Fourth batch of tests: ensure that the sensitivity of Serialize is calculated
// correctly.
// As a building block, we need to calculate the number of bytes for a varint.
TEST(DPGroupByAggregatorTest, CalculateVarintByteSize) {
  EXPECT_THAT(CalculateVarintByteSize(1), IsOkAndHolds(1));
  EXPECT_THAT(CalculateVarintByteSize(150), IsOkAndHolds(2));
}
TEST(DPGroupByAggregatorTest, CalculateVarintByteSize_Negative) {
  EXPECT_THAT(CalculateVarintByteSize(-1),
              StatusIs(INVALID_ARGUMENT, HasSubstr("Value must be positive")));
}

// Creates a MockDPGroupByAggregator for testing SerializeSensitivity.
MockDPGroupByAggregator CreateMockForTestingSerializeSensitivity(
    std::vector<Intrinsic>& intrinsics, int max_groups_contributed,
    int max_string_length) {
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators;
  for (const Intrinsic& intrinsic : intrinsics) {
    auto aggregator = CreateTensorAggregator(intrinsic).value();
    auto one_dim_base_aggregator =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator.release()));
    aggregators.push_back(std::move(one_dim_base_aggregator));
  }
  std::vector<TensorSpec> input_key_specs;
  input_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key1", DT_STRING));
  input_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key2", DT_STRING));
  std::vector<TensorSpec> output_key_specs;
  output_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key1_out", DT_STRING));
  output_key_specs.push_back(
      dp_histogram_testing::CreateTensorSpec("key2_out", DT_STRING));
  return MockDPGroupByAggregator(input_key_specs, &output_key_specs,
                                 &intrinsics, std::move(aggregators),
                                 /*num_inputs=*/0, /*epsilon=*/0.1,
                                 /*delta=*/1e-6, max_groups_contributed,
                                 std::nullopt, {}, max_string_length);
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysTwoSums) {
  std::vector<Intrinsic> intrinsics;
  for (int i = 0; i < 2; ++i) {
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  }
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingSerializeSensitivity(intrinsics,
                                               /*max_groups_contributed=*/1,
                                               /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(40));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSum) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingSerializeSensitivity(intrinsics,
                                               /*max_groups_contributed=*/1,
                                               /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(31));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumFloatingType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<float, double>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingSerializeSensitivity(intrinsics,
                                               /*max_groups_contributed=*/1,
                                               /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(31));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumShorterType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingSerializeSensitivity(intrinsics,
                                               /*max_groups_contributed=*/1,
                                               /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(27));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumTwoMaxGroups) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingSerializeSensitivity(intrinsics,
                                               /*max_groups_contributed=*/2,
                                               /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(58));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumLongStrings) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingSerializeSensitivity(intrinsics,
                                               /*max_groups_contributed=*/1,
                                               /*max_string_length=*/64);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(143));
}

// Fifth behavior to test: the outcome of Serialize() has a random length that
// scales with sensitivity.
TEST(DPGroupByAggregatorTest, SerializedStateVarianceDependsOnSensitivity) {
  // Low sensitivity (27)
  size_t min_length = 1000000;
  size_t max_length = 0;
  for (int i = 0; i < 10; ++i) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
    MockDPGroupByAggregator top_aggregator =
        CreateMockForTestingSerializeSensitivity(intrinsics,
                                                 /*max_groups_contributed=*/1,
                                                 /*max_string_length=*/8);
    std::string serialized_state =
        std::move(top_aggregator).Serialize().value();
    min_length = std::min(min_length, serialized_state.size());
    max_length = std::max(max_length, serialized_state.size());
  }
  int low_sensitivity_range = max_length - min_length;
  // Randomness should cause the range to have some width.
  EXPECT_GT(low_sensitivity_range, 0);

  // High sensitivity (143)
  min_length = 1000000;
  max_length = 0;
  for (int i = 0; i < 10; ++i) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
    MockDPGroupByAggregator top_aggregator =
        CreateMockForTestingSerializeSensitivity(intrinsics,
                                                 /*max_groups_contributed=*/1,
                                                 /*max_string_length=*/64);
    std::string serialized_state =
        std::move(top_aggregator).Serialize().value();
    min_length = std::min(min_length, serialized_state.size());
    max_length = std::max(max_length, serialized_state.size());
  }
  int high_sensitivity_range = max_length - min_length;

  // The noise scale for the high-sensitivity case should be bigger.
  EXPECT_GT(high_sensitivity_range, low_sensitivity_range);
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
