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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_noise_mechanisms.h"
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
using ::testing::Eq;
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
  double GetEpsilonPerAgg() { return epsilon_per_agg(); }
  double GetDeltaPerAgg() { return delta_per_agg(); }
  StatusOr<int64_t> SerializeSensitivity() {
    return DPGroupByAggregator::CalculateSerializeSensitivity();
  }
  StatusOr<int64_t> PartitionSensitivity(int num_partitions) {
    return DPGroupByAggregator::CalculatePartitionSensitivity(num_partitions);
  }
  StatusOr<std::string> Serialize() && override {
    return (std::move(*this)).DPGroupByAggregator::Serialize();
  }
  StatusOr<std::vector<std::string>> Partition(int num_partitions) && override {
    return (std::move(*this)).DPGroupByAggregator::Partition(num_partitions);
  }

  absl::StatusOr<const DPHistogramBundle&> GetBundle(int i) const {
    return DPGroupByAggregator::GetBundle(i);
  }

  // Add to the vector of DPHistogramBundles.
  void AddBundle(DPHistogramBundle bundle) {
    DPGroupByAggregator::AddBundle(std::move(bundle));
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
MockDPGroupByAggregator CreateMockForTestingPadding(
    std::vector<Intrinsic>& intrinsics, int max_groups_contributed,
    int max_string_length, double epsilon = 0.1, double delta = 1e-6) {
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
  return MockDPGroupByAggregator(
      input_key_specs, &output_key_specs, &intrinsics, std::move(aggregators),
      /*num_inputs=*/0, epsilon, delta, max_groups_contributed, std::nullopt,
      {}, max_string_length);
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysTwoSums) {
  std::vector<Intrinsic> intrinsics;
  for (int i = 0; i < 2; ++i) {
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  }
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingPadding(intrinsics,
                                  /*max_groups_contributed=*/1,
                                  /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(40));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSum) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingPadding(intrinsics,
                                  /*max_groups_contributed=*/1,
                                  /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(31));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumFloatingType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<float, double>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingPadding(intrinsics,
                                  /*max_groups_contributed=*/1,
                                  /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(31));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumShorterType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingPadding(intrinsics,
                                  /*max_groups_contributed=*/1,
                                  /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(27));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumTwoMaxGroups) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingPadding(intrinsics,
                                  /*max_groups_contributed=*/2,
                                  /*max_string_length=*/8);
  EXPECT_THAT(top_aggregator.SerializeSensitivity(), IsOkAndHolds(58));
}
TEST(DPGroupByAggregatorTest, SerializeSensitivity_TwoKeysOneSumLongStrings) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int64_t>());
  MockDPGroupByAggregator top_aggregator =
      CreateMockForTestingPadding(intrinsics,
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
        CreateMockForTestingPadding(intrinsics,
                                    /*max_groups_contributed=*/1,
                                    /*max_string_length=*/8);
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(top_aggregator).Serialize());
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
        CreateMockForTestingPadding(intrinsics,
                                    /*max_groups_contributed=*/1,
                                    /*max_string_length=*/64);
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(top_aggregator).Serialize());
    min_length = std::min(min_length, serialized_state.size());
    max_length = std::max(max_length, serialized_state.size());
  }
  int high_sensitivity_range = max_length - min_length;

  // The noise scale for the high-sensitivity case should be bigger.
  EXPECT_GT(high_sensitivity_range, low_sensitivity_range);
}

// If we increase the number of groups that can be contributed by a single user,
// the sensitivity of Partition() should increase.
TEST(DPGroupByAggregatorTest, PartitionSensitivity_IncreasesWithMaxGroups) {
  int num_partitions = 10;
  int prev_sensitivity = 0;
  for (int max_groups_contributed : {1, 2, 4, 8, 16, 32}) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
    MockDPGroupByAggregator top_aggregator = CreateMockForTestingPadding(
        intrinsics, max_groups_contributed, /*max_string_length=*/256);
    TFF_ASSERT_OK_AND_ASSIGN(
        auto partition_sensitivity,
        top_aggregator.PartitionSensitivity(num_partitions));
    EXPECT_GT(partition_sensitivity, prev_sensitivity);
    prev_sensitivity = partition_sensitivity;
  }
}
// If we increase the number of partitions, the sensitivity of Partition()
// should not increase after a certain point.
TEST(DPGroupByAggregatorTest, PartitionSensitivity_PlateausWithNumPartitions) {
  int max_groups_contributed = 8;
  int prev_sensitivity = 0;
  for (int num_partitions : {1, 2, 4, 8, 16, 32}) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
    MockDPGroupByAggregator top_aggregator = CreateMockForTestingPadding(
        intrinsics, max_groups_contributed, /*max_string_length=*/256);
    TFF_ASSERT_OK_AND_ASSIGN(
        auto partition_sensitivity,
        top_aggregator.PartitionSensitivity(num_partitions));
    if (num_partitions <= 2 * max_groups_contributed) {
      EXPECT_GT(partition_sensitivity, prev_sensitivity);
    } else {
      EXPECT_EQ(partition_sensitivity, prev_sensitivity);
    }
    prev_sensitivity = partition_sensitivity;
  }
}

// There should be no noise in partitioning if epsilon is above the threshold.
TEST(DPGroupByAggregatorTest, Partition_LargeEpsilonNoNoise) {
  std::vector<std::string> previous_partitions;
  for (int i = 0; i < 10; ++i) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
    MockDPGroupByAggregator top_aggregator = CreateMockForTestingPadding(
        intrinsics, /*max_groups_contributed=*/1,
        /*max_string_length=*/8, /*epsilon=*/kEpsilonThreshold + 1);
    TFF_ASSERT_OK_AND_ASSIGN(auto partitions,
                             std::move(top_aggregator).Partition(10));
    if (i > 0) {
      EXPECT_EQ(partitions, previous_partitions);
    }
    previous_partitions = partitions;
  }
}

// Otherwise, there should be noise in partitioning. Increasing the number of
// partitions should increase the scale until it plateaus.

// Helper function for testing the Partition() method: returns the mean absolute
// deviation of the padding length, which estimates the Laplace parameter.
StatusOr<double> MeasureScaleOfPartitionPaddingLength(
    int max_groups_contributed, int max_string_length, int num_partitions,
    double epsilon = 1.0, double delta = 1e-6) {
  std::vector<int> lengths;
  for (int i = 0; i < 5000; ++i) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
    MockDPGroupByAggregator top_aggregator = CreateMockForTestingPadding(
        intrinsics, max_groups_contributed, max_string_length, epsilon, delta);
    TFF_ASSIGN_OR_RETURN(auto partitions,
                         std::move(top_aggregator).Partition(num_partitions));
    for (const auto& partition : partitions) {
      lengths.push_back(partition.size());
    }
  }
  std::sort(lengths.begin(), lengths.end());
  int median_length = lengths[lengths.size() / 2];
  double mean_absolute_deviation = 0;
  for (int length : lengths) {
    mean_absolute_deviation += std::abs(length - median_length);
  }
  mean_absolute_deviation /= lengths.size();
  return mean_absolute_deviation;
}

TEST(DPGroupByAggregatorTest, Partition_NoiseScaleIncreasesWithNumPartitions) {
  int max_groups_contributed = 8;
  double previous_scale = 0;
  for (int num_partitions : {1, 2, 4, 8, 16, 32}) {
    StatusOr<double> scale = MeasureScaleOfPartitionPaddingLength(
        max_groups_contributed, /*max_string_length=*/8, num_partitions);
    TFF_ASSERT_OK(scale);
    if (num_partitions <= 2 * max_groups_contributed) {
      EXPECT_GT(scale.value(), previous_scale);
    } else {
      double ratio = scale.value() / previous_scale;
      EXPECT_GT(ratio, 0.9);
      EXPECT_LT(ratio, 1.1);
    }
    previous_scale = scale.value();
  }
}

// Confirm that our new Partition function adds less noise than the old one.
TEST(DPGroupByAggregatorTest, Partition_NoiseScaleDecreasesWithNewMethod) {
  int max_groups_contributed = 8000;
  int max_string_length = 256;
  double epsilon = 1.0987;
  double delta = 1e-8;
  int num_partitions = 10;

  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
  MockDPGroupByAggregator top_aggregator = CreateMockForTestingPadding(
      intrinsics, max_groups_contributed, max_string_length);
  int64_t partitions_influenced =
      std::min<int64_t>(2 * max_groups_contributed, num_partitions);
  double per_partition_delta = delta / partitions_influenced;
  // Current technique: only split delta and rely on L1 sensitivity of
  // Partition().
  TFF_ASSERT_OK_AND_ASSIGN(int partition_sensitivity,
                           top_aggregator.PartitionSensitivity(num_partitions));
  double new_scale =
      (partition_sensitivity / epsilon) * log(1.0 / per_partition_delta);

  // Old technique: split epsilon and delta and rely on sensitivity of
  // Serialize().
  TFF_ASSERT_OK_AND_ASSIGN(int serialize_sensitivity,
                           top_aggregator.SerializeSensitivity());
  double per_partition_epsilon = epsilon / partitions_influenced;
  double old_scale = (serialize_sensitivity / per_partition_epsilon) *
                     log(1.0 / per_partition_delta);
  EXPECT_LT(new_scale, old_scale);
}

// Eight behavior to test: large epsilon switches off random padding.
TEST(DPGroupByAggregatorTest, LargeEpsilonSerializeIsNotRandom) {
  size_t min_length = 1000000;
  size_t max_length = 0;
  for (int i = 0; i < 10; ++i) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
    MockDPGroupByAggregator top_aggregator =
        CreateMockForTestingPadding(intrinsics,
                                    /*max_groups_contributed=*/1,
                                    /*max_string_length=*/8,
                                    /*epsilon=*/kEpsilonThreshold);
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(top_aggregator).Serialize());
    min_length = std::min(min_length, serialized_state.size());
    max_length = std::max(max_length, serialized_state.size());
  }
  EXPECT_EQ(max_length - min_length, 0);
}
TEST(DPGroupByAggregatorTest, LargeEpsilonPartitionIsNotRandom) {
  size_t min_length_a = 1000000;
  size_t min_length_b = 1000000;
  size_t max_length_a = 0;
  size_t max_length_b = 0;
  for (int i = 0; i < 10; ++i) {
    std::vector<Intrinsic> intrinsics;
    intrinsics.push_back(CreateDefaultInnerIntrinsic<int32_t, int32_t>());
    MockDPGroupByAggregator top_aggregator =
        CreateMockForTestingPadding(intrinsics,
                                    /*max_groups_contributed=*/1,
                                    /*max_string_length=*/8,
                                    /*epsilon=*/kEpsilonThreshold);
    TFF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> serialized_states,
                             std::move(top_aggregator).Partition(2));
    min_length_a = std::min(min_length_a, serialized_states[0].size());
    max_length_a = std::max(max_length_a, serialized_states[0].size());
    min_length_b = std::min(min_length_b, serialized_states[1].size());
    max_length_b = std::max(max_length_b, serialized_states[1].size());
  }
  EXPECT_EQ(max_length_a - min_length_a, 0);
  EXPECT_EQ(max_length_b - min_length_b, 0);
}

// Ninth behavior to test: GetBundle() returns an error if the index is out of
// range (e.g. due to empty list of bundles).
TEST(DPGroupByAggregatorTest, GetBundleReturnsError) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/0.1, /*delta=*/0.1, /*num_intrinsics=*/1);
  EXPECT_THAT(aggregator.GetBundle(0),
              StatusIs(INVALID_ARGUMENT, HasSubstr("is not in the range")));
}
// Tenth behavior to test: GetBundle() returns the right bundle if the index is
// in range.
TEST(DPGroupByAggregatorTest, GetBundleReturnsBundle) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/0.1, /*delta=*/0.1, /*num_intrinsics=*/1);
  aggregator.AddBundle({/*mechanism=*/nullptr,
                        /*threshold=*/111,
                        /*use_laplace=*/true});
  aggregator.AddBundle({/*mechanism=*/nullptr,
                        /*threshold=*/222,
                        /*use_laplace=*/false});
  EXPECT_THAT(aggregator.GetBundle(0),
              IsOkAndHolds(testing::FieldsAre(Eq(nullptr), Eq(111), Eq(true))));
  EXPECT_THAT(aggregator.GetBundle(1), IsOkAndHolds(testing::FieldsAre(
                                           Eq(nullptr), Eq(222), Eq(false))));
}

// Tenth: noise descriptions.
TEST(DPGroupByAggregatorTest, NoiseDescriptionForLargeEpsilons) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/kEpsilonThreshold + 1, /*delta=*/1e-8,
      /*num_intrinsics=*/1);
  EXPECT_THAT(aggregator.GetNoiseDescription(),
              IsOkAndHolds(testing::HasSubstr("No noise added.")));
}
TEST(DPGroupByAggregatorTest, NoiseDescriptionFailsForMissingMechanism) {
  MockDPGroupByAggregator aggregator = CreateMockForTestingEpsilonAndDeltaSplit(
      /*epsilon=*/0.1, /*delta=*/1e-8, /*num_intrinsics=*/1);
  aggregator.AddBundle({/*mechanism=*/nullptr,
                        /*threshold=*/111,
                        /*use_laplace=*/true});
  EXPECT_THAT(
      aggregator.GetNoiseDescription(),
      StatusIs(FAILED_PRECONDITION, HasSubstr("a mechanism was not set.")));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
