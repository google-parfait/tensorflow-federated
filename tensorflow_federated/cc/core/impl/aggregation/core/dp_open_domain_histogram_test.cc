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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_open_domain_histogram.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Contains;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsFalse;
using ::testing::IsTrue;
using ::testing::Lt;
using ::testing::Ne;
using ::testing::Not;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;
using ::testing::ValuesIn;

using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateInnerIntrinsic;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsic;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsicWithMinContributors;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateTensorSpec;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateTopLevelParameters;

using DPOpenDomainHistogramTest = TestWithParam<bool>;

TEST(DPOpenDomainHistogramTest, Deserialize_FailToParseProto) {
  auto intrinsic = CreateIntrinsic<int64_t, int64_t>(100, 0.01, 1);
  std::string invalid_state("invalid_state");
  Status s = DeserializeTensorAggregator(intrinsic, invalid_state).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Failed to parse"));
}

// Function to execute the DPOpenDomainHistogram on one input where there is
// just one key per contribution and each contribution is to one aggregation
template <typename InputType>
StatusOr<OutputTensorList> SingleKeySingleAgg(
    const Intrinsic& intrinsic, const TensorShape shape,
    std::initializer_list<string_view> key_list,
    std::initializer_list<InputType> value_list, bool serialize_deserialize) {
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list))
          .value();

  Tensor value_tensor =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list))
          .value();
  auto acc_status = aggregator->Accumulate({&keys, &value_tensor});

  if (serialize_deserialize) {
    auto serialized_state = std::move(*aggregator).Serialize();
    aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(1));

  return std::move(*aggregator).Report();
}

// The first batch of tests is dedicated to norm bounding when there is only one
// inner aggregation (GROUP BY key, SUM(value))
TEST_P(DPOpenDomainHistogramTest, SingleKeySingleAggWithL0Bound) {
  // L0 bounding involves randomness so we should repeat things to catch errors.
  for (int i = 0; i < 9; i++) {
    auto intrinsic =
        CreateIntrinsic<int64_t, int64_t>(kEpsilonThreshold, 0.01, 1);
    auto result = SingleKeySingleAgg<int64_t>(intrinsic, {4},
                                              {"zero", "one", "two", "zero"},
                                              {1, 3, 15, 27}, GetParam());
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output is either (zero:28) OR (one:3) OR (two:15)
    // OR their reversals
    std::vector<string_view> possible_keys({"zero", "one", "two"});
    std::vector<int64_t> possible_values({28, 3, 15});
    bool found_match = false;
    for (int j = 0; j < 3; j++) {
      auto key_matcher = IsTensor<string_view>({1}, {possible_keys[j]});
      auto callable_key_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(key_matcher);
      auto value_matcher = IsTensor<int64_t>({1}, {possible_values[j]});
      auto callable_value_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(value_matcher);

      if (callable_key_matcher("result.value()[0]", result.value()[0]) &&
          callable_value_matcher("result.value()[1]", result.value()[1])) {
        found_match = true;
        break;
      }
    }
    EXPECT_TRUE(found_match);

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

TEST_P(DPOpenDomainHistogramTest, SingleKeySingleAggWithL0LinfinityBounds) {
  for (int i = 0; i < 9; i++) {
    // Use the same setup as before but now impose a maximum magnitude of 12
    auto intrinsic =
        CreateIntrinsic<int64_t, int64_t>(kEpsilonThreshold, 0.01, 1, 12);
    auto result = SingleKeySingleAgg<int64_t>(intrinsic, {4},
                                              {"zero", "one", "two", "zero"},
                                              {1, 3, 15, 27}, GetParam());
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output is either (zero:12) OR (one:3) OR (two:12)
    // OR their reversals
    std::vector<string_view> possible_keys({"zero", "one", "two"});
    std::vector<int64_t> possible_values({12, 3, 12});
    bool found_match = false;
    for (int j = 0; j < 3; j++) {
      auto key_matcher = IsTensor<string_view>({1}, {possible_keys[j]});
      auto callable_key_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(key_matcher);
      auto value_matcher = IsTensor<int64_t>({1}, {possible_values[j]});
      auto callable_value_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(value_matcher);

      if (callable_key_matcher("result.value()[0]", result.value()[0]) &&
          callable_value_matcher("result.value()[1]", result.value()[1])) {
        found_match = true;
        break;
      }
    }
    EXPECT_TRUE(found_match);

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

TEST_P(DPOpenDomainHistogramTest, SingleKeySingleAggWithL0LinfinityL1Bounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4 (four keys), Linfinity bound is 50 (|value| <= 50),
    // and L1 bound is 100 (sum over |value| is <= 100)
    auto intrinsic =
        CreateIntrinsic<int64_t, int64_t>(kEpsilonThreshold, 0.01, 4, 50, 100);
    auto result = SingleKeySingleAgg<int64_t>(
        intrinsic, {5}, {"zero", "one", "two", "three", "four"},
        {60, 60, 60, 60, 60}, GetParam());
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output should look like (25, 25, 25, 25)
    EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {25, 25, 25, 25}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

TEST_P(DPOpenDomainHistogramTest, SingleKeySingleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4 (four keys), Linfinity bound is 50 (|value| <= 50),
    // L1 bound is 100 (sum over |value| is <= 100), and L2 bound is 10
    auto intrinsic = CreateIntrinsic<int64_t, int64_t>(kEpsilonThreshold, 0.01,
                                                       4, 50, 100, 10);
    auto result = SingleKeySingleAgg<int64_t>(
        intrinsic, {5}, {"zero", "one", "two", "three", "four"},
        {60, 60, 60, 60, 60}, GetParam());
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output should look like (5, 5, 5, 5)
    EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {5, 5, 5, 5}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

// Second batch of tests: norm bounding when there are > 1 inner aggregation.
// e.g. SUM(value1), SUM(value2)  GROUP BY key
template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsic2Agg(double epsilon = kEpsilonThreshold,
                              double delta = 0.001, int64_t l0_bound = 100,
                              InputType linfinity_bound1 = 100,
                              double l1_bound1 = -1, double l2_bound1 = -1,
                              InputType linfinity_bound2 = 100,
                              double l1_bound2 = -1, double l2_bound2 = -1) {
  Intrinsic intrinsic = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters(epsilon, delta, l0_bound)},
      .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound1, l1_bound1,
                                                  l2_bound1));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound2, l1_bound2,
                                                  l2_bound2));
  return intrinsic;
}

// Function to execute the DPOpenDomainHistogram on one input where there is
// just one key per contribution and each contribution is to two aggregations
template <typename InputType>
StatusOr<OutputTensorList> SingleKeyDoubleAgg(
    const Intrinsic& intrinsic, const TensorShape shape,
    std::initializer_list<string_view> key_list,
    std::initializer_list<InputType> value_list1,
    std::initializer_list<InputType> value_list2, bool serialize_deserialize) {
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list))
          .value();

  Tensor value_tensor1 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list1))
          .value();
  Tensor value_tensor2 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list2))
          .value();
  auto acc_status =
      aggregator->Accumulate({&keys, &value_tensor1, &value_tensor2});

  if (serialize_deserialize) {
    auto serialized_state = std::move(*aggregator).Serialize();
    aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(1));

  return std::move(*aggregator).Report();
}

TEST_P(DPOpenDomainHistogramTest, SingleKeyDoubleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4.
    // For agg 1, Linfinity bound is 20 and no other bounds provided.
    // For agg 2, Linfinity bound is 50, L1 bound is 100, L2 bound is 10.
    auto intrinsic = CreateIntrinsic2Agg<int64_t, int64_t>(
        kEpsilonThreshold, 0.01, 4, 20, -1, -1, 50, 100, 10);
    auto result = SingleKeyDoubleAgg<int64_t>(
        intrinsic, {5}, {"zero", "one", "two", "three", "four"},
        {60, 60, 60, 60, 60}, {60, 60, 60, 60, 60}, GetParam());
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(3));

    // first output should look like (20, 20, 20, 20)
    EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {20, 20, 20, 20}));

    // second output should look like (5, 5, 5, 5)
    EXPECT_THAT(result.value()[2], IsTensor<int64_t>({4}, {5, 5, 5, 5}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
    EXPECT_TRUE(result.value()[2].is_dense());
  }
}

// Third batch of tests: norm bounding when there is > 1 key and > 1 inner
// aggregation. e.g. SUM(value1), SUM(value2)  GROUP BY key1, key 2
template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsic2Key2Agg(double epsilon = kEpsilonThreshold,
                                  double delta = 0.001, int64_t l0_bound = 100,
                                  InputType linfinity_bound1 = 100,
                                  double l1_bound1 = -1, double l2_bound1 = -1,
                                  InputType linfinity_bound2 = 100,
                                  double l1_bound2 = -1,
                                  double l2_bound2 = -1) {
  Intrinsic intrinsic = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key1", DT_STRING),
                 CreateTensorSpec("key2", DT_STRING)},
      .outputs = {CreateTensorSpec("key1_out", DT_STRING),
                  CreateTensorSpec("key2_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters(epsilon, delta, l0_bound)},
      .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound1, l1_bound1,
                                                  l2_bound1));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound2, l1_bound2,
                                                  l2_bound2));
  return intrinsic;
}

template <typename InputType>
StatusOr<OutputTensorList> DoubleKeyDoubleAgg(
    const Intrinsic& intrinsic, const TensorShape shape,
    std::initializer_list<string_view> key_list1,
    std::initializer_list<string_view> key_list2,
    std::initializer_list<InputType> value_list1,
    std::initializer_list<InputType> value_list2, bool serialize_deserialize) {
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys1 =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list1))
          .value();
  Tensor keys2 =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list2))
          .value();

  Tensor value_tensor1 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list1))
          .value();
  Tensor value_tensor2 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list2))
          .value();
  auto acc_status =
      aggregator->Accumulate({&keys1, &keys2, &value_tensor1, &value_tensor2});

  if (serialize_deserialize) {
    auto serialized_state = std::move(*aggregator).Serialize();
    aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(1));

  return std::move(*aggregator).Report();
}

TEST_P(DPOpenDomainHistogramTest, DoubleKeyDoubleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4.
    // For agg 1, Linfinity bound is 20 and no other bounds provided.
    // For agg 2, Linfinity bound is 50, L1 bound is 100, L2 bound is 10.
    auto intrinsic = CreateIntrinsic2Key2Agg<int64_t, int64_t>(
        kEpsilonThreshold, 0.01, 4, 20, -1, -1, 50, 100, 10);
    auto result = DoubleKeyDoubleAgg<int64_t>(
        intrinsic, {5}, {"red", "green", "green", "blue", "gray"},
        {"zero", "one", "two", "three", "four"}, {60, 60, 60, 60, 60},
        {60, 60, 60, 60, 60}, GetParam());
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(4));

    // first output should look like (20, 20, 20, 20)
    EXPECT_THAT(result.value()[2], IsTensor<int64_t>({4}, {20, 20, 20, 20}));

    // second output should look like (5, 5, 5, 5)
    EXPECT_THAT(result.value()[3], IsTensor<int64_t>({4}, {5, 5, 5, 5}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
    EXPECT_TRUE(result.value()[2].is_dense());
    EXPECT_TRUE(result.value()[3].is_dense());
  }
}

// Fourth batch of tests: norm bounding on key-less data (norm bound = magnitude
// bound)
template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsicNoKeys(double epsilon = kEpsilonThreshold,
                                double delta = 0.001, int64_t l0_bound = 100,
                                InputType linfinity_bound1 = 100,
                                double l1_bound1 = -1, double l2_bound1 = -1,
                                InputType linfinity_bound2 = 100,
                                double l1_bound2 = -1, double l2_bound2 = -1,
                                InputType linfinity_bound3 = 100,
                                double l1_bound3 = -1, double l2_bound3 = -1) {
  Intrinsic intrinsic = {
      .uri = kDPGroupByUri,
      .inputs = {},
      .outputs = {},
      .parameters = {CreateTopLevelParameters(epsilon, delta, l0_bound)},
      .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound1, l1_bound1,
                                                  l2_bound1));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound2, l1_bound2,
                                                  l2_bound2));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound3, l1_bound3,
                                                  l2_bound3));

  return intrinsic;
}

TEST_P(DPOpenDomainHistogramTest, NoKeyTripleAggWithAllBounds) {
  Intrinsic intrinsic = CreateIntrinsicNoKeys<int32_t, int64_t>(
      kEpsilonThreshold, 0.01, 100, 10, 9, 8,  // limit to 8
      100, 9, -1,                              // limit to 9
      100, -1, -1);                            // 100

  auto aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  EXPECT_THAT(aggregator->Accumulate({&t1, &t2, &t3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        aggregator, DeserializeTensorAggregator(intrinsic, serialized_state));
  }

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {8}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {9}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({1}, {11}));
}

// Fifth batch of tests: check that noise is added. The noised sum should not be
// the same as the unnoised sum. The odds of a false negative shrinks with
// epsilon.
TEST_P(DPOpenDomainHistogramTest, NoiseAddedForSmallEpsilons) {
  Intrinsic intrinsic = CreateIntrinsic<int32_t, int64_t>(0.05, 1e-8, 2, 1);
  auto dp_aggregator = CreateTensorAggregator(intrinsic).value();
  int num_inputs = 4000;
  for (int i = 0; i < num_inputs; i++) {
    Tensor keys = Tensor::Create(DT_STRING, {2},
                                 CreateTestData<string_view>({"key0", "key1"}))
                      .value();
    Tensor values =
        Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({1, 1})).value();
    auto acc_status = dp_aggregator->Accumulate({&keys, &values});
    EXPECT_THAT(acc_status, IsOk());
  }
  EXPECT_EQ(dp_aggregator->GetNumInputs(), num_inputs);
  EXPECT_TRUE(dp_aggregator->CanReport());

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*dp_aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(dp_aggregator, DeserializeTensorAggregator(
                                                intrinsic, serialized_state));
  }

  auto report = std::move(*dp_aggregator).Report();
  EXPECT_THAT(report, IsOk());
  EXPECT_EQ(report->size(), 2);
  const auto& values = report.value()[1].AsSpan<int64_t>();
  ASSERT_THAT(values.size(), Eq(2));
  EXPECT_TRUE(values[0] != num_inputs || values[1] != num_inputs);
}

// Sixth batch of tests: check that the right groups get dropped

// Test that we will drop groups with any small aggregate and keep groups with
// large aggregates. The surviving aggregates should have noise in them.
// This test has a small probability of failing due to false positives: noise
// could push the 0 past the threshold. It also has a small probability of
// failing due to false negatives: noise could push the 100 below the threshold.
TEST_P(DPOpenDomainHistogramTest, SingleKeyDropAggregatesWithValueZero) {
  // epsilon = 1, delta= 1e-8, L0 bound = 2, Linfinity bound = 1
  Intrinsic intrinsic =
      CreateIntrinsic2Agg<int32_t, int64_t>(1.0, 1e-8, 2, 1, -1, -1, 1, -1, -1);
  auto dp_aggregator = CreateTensorAggregator(intrinsic).value();
  // Simulate many clients where they contribute
  // aggregation 1: zeroes to key0 and ones to key1
  // aggregation 2: ones to both
  std::string key0 = "drop me";
  std::string key1 = "keep me";
  int num_inputs = 400;
  for (int i = 0; i < num_inputs; i++) {
    Tensor keys = Tensor::Create(DT_STRING, {2},
                                 CreateTestData<string_view>({key0, key1}))
                      .value();

    Tensor value_tensor1 =
        Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({0, 1})).value();
    Tensor value_tensor2 =
        Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({1, 1})).value();

    auto acc_status =
        dp_aggregator->Accumulate({&keys, &value_tensor1, &value_tensor2});
    EXPECT_THAT(acc_status, IsOk());
  }

  EXPECT_THAT(dp_aggregator->GetNumInputs(), Eq(num_inputs));
  EXPECT_THAT(dp_aggregator->CanReport(), IsTrue());

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*dp_aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(dp_aggregator, DeserializeTensorAggregator(
                                                intrinsic, serialized_state));
  }

  auto report = std::move(*dp_aggregator).Report();
  EXPECT_THAT(report, IsOk());
  EXPECT_THAT(report->size(), Eq(3));

  // The report should not include key0 because it has an aggregate of 0
  // and a key survives only when *all* its values are above their thresholds
  EXPECT_THAT(report.value()[0], IsTensor<string_view>({1}, {key1}));

  // The values associated with the surviving key (key1) must be large
  int64_t threshold = static_cast<int64_t>(ceil(1 + 2 * std::log(1e8)));
  EXPECT_THAT(report.value()[1].AsScalar<int64_t>(), Gt(threshold));
  EXPECT_THAT(report.value()[2].AsScalar<int64_t>(), Gt(threshold));
}

// When there are no grouping keys, aggregation will be scalar. Hence, the sole
// "group" does not need to be dropped for DP (because it exists whether or not
// a given client contributed data)
// This test should never fail because DPOpenDomainHistogram & its factory check
// for the no key case and force aggregates to survive.
TEST_P(DPOpenDomainHistogramTest, NoKeyNoDrop) {
  Intrinsic intrinsic = CreateIntrinsicNoKeys<int32_t, int64_t>(
      1.0, 1e-8, 3, 10, 9, 8,  // limit to 8
      100, 9, -1,              // limit to 9
      10, -1, -1);             // limit to 10
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  for (int i = 0; i < 100; i++) {
    Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
    Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
    Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
    EXPECT_THAT(aggregator->Accumulate({&t1, &t2, &t3}), IsOk());
  }

  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        aggregator, DeserializeTensorAggregator(intrinsic, serialized_state));
  }

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));

  // report should have all the noisy data (one number per Tensor)
  EXPECT_THAT(result.value()[0].num_elements(), 1);  // 800 + noise
  EXPECT_THAT(result.value()[1].num_elements(), 1);  // 900 + noise
  EXPECT_THAT(result.value()[2].num_elements(), 1);  // 1000 + noise
}

// Test to verify that Report() still drops key columns that were given empty
// labels.
TEST_P(DPOpenDomainHistogramTest,
       Accumulate_MultipleKeyTensors_SomeKeysNotInOutput_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic = {.uri = kDPGroupByUri,
                         .inputs = {CreateTensorSpec("key1", DT_STRING),
                                    CreateTensorSpec("key2", DT_STRING)},
                         // An empty string in the output keys means that the
                         // key should not be included in the output.
                         .outputs = {CreateTensorSpec("", DT_STRING),
                                     CreateTensorSpec("animals", DT_STRING)},
                         .parameters = {CreateTopLevelParameters(1.0, 1e-8, 3)},
                         .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int32_t, int64_t>(4, 8, -1));
  // L0 = 3, Linf = 4, L1 = 8
  // These bounds should not affect any individual input data below

  auto aggregator = CreateTensorAggregator(intrinsic).value();
  constexpr int num_inputs = 200;
  for (int i = 0; i < num_inputs; i++) {
    Tensor sizeKeys1 =
        Tensor::Create(
            DT_STRING, shape,
            CreateTestData<string_view>({"large", "large", "small", "large"}))
            .value();
    Tensor animalKeys1 =
        Tensor::Create(
            DT_STRING, shape,
            CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
            .value();
    Tensor t1 =
        Tensor::Create(DT_INT32, shape, CreateTestData({1, 2, 1, 4})).value();
    EXPECT_THAT(aggregator->Accumulate({&sizeKeys1, &animalKeys1, &t1}),
                IsOk());
    // Totals: [3, 1, 4]
    Tensor sizeKeys2 =
        Tensor::Create(
            DT_STRING, shape,
            CreateTestData<string_view>({"small", "large", "small", "small"}))
            .value();
    Tensor animalKeys2 =
        Tensor::Create(
            DT_STRING, shape,
            CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
            .value();
    Tensor t2 =
        Tensor::Create(DT_INT32, shape, CreateTestData({2, 0, 2, 4})).value();
    EXPECT_THAT(aggregator->Accumulate({&sizeKeys2, &animalKeys2, &t2}),
                IsOk());
    // Totals: [3, 5, 4, 4]
    EXPECT_THAT(aggregator->GetNumInputs(), Eq(2 * (i + 1)));
  }

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        aggregator, DeserializeTensorAggregator(intrinsic, serialized_state));
  }

  // Totals: [600, 1000, 800, 800]
  ASSERT_THAT(aggregator->CanReport(), IsTrue());
  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensors.
  // Only the second key tensor should be included in the output.
  EXPECT_EQ(result.value().size(), 2);

  // That tensor should have all animal keys because the threshold is less than
  // the noised values.
  ASSERT_EQ(result.value()[0].num_elements(), 4);

  // The order of the keys may differ from what is output by GroupByAggregator
  // ("cat", "cat", "dog", "dog"). This is because DPCompositeKeyCombiner's
  // AccumulateWithBound function samples l0_bound_ keys from its input; in our
  // case, sampling 3 keys from 3 keys results in a random permutation.
  int num_cat = 0;
  int num_dog = 0;
  for (int i = 0; i < 4; i++) {
    auto key = result.value()[0].AsSpan<string_view>()[i];
    num_cat += (key == "cat") ? 1 : 0;
    num_dog += (key == "dog") ? 1 : 0;
  }
  EXPECT_EQ(num_cat, 2);
  EXPECT_EQ(num_dog, 2);
}

// Seventh test batch: merge should not clip or noise intermediary aggregates.

TEST_P(DPOpenDomainHistogramTest, MergeDoesNotDistortData_SingleKey) {
  // For any single user's data we will give to the aggregators, the norm bounds
  // below do nothing: each Accumulate call has 1 distinct key and a value of 1,
  // which satisfies the L0 bound and Linfinity bound constraints.
  Intrinsic intrinsic =
      CreateIntrinsic<int64_t, int64_t>(kEpsilonThreshold, 0.001, 1, 1, -1, -1);
  auto agg1 = CreateTensorAggregator(intrinsic).value();
  auto agg2 = CreateTensorAggregator(intrinsic).value();

  // agg1 gets one person's data, mapping "key" to 1
  Tensor key1 =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"key"}))
          .value();
  Tensor data1 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  EXPECT_THAT(agg1->Accumulate({&key1, &data1}), IsOk());

  // agg2 gets data from a lot more people. At the end, it will map "other key"
  // to 1000 and "yet another key" to 1000.
  for (int i = 0; i < 1000; i++) {
    Tensor key_a = Tensor::Create(DT_STRING, {1},
                                  CreateTestData<string_view>({"other key"}))
                       .value();
    Tensor data_a =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
    EXPECT_THAT(agg2->Accumulate({&key_a, &data_a}), IsOk());

    Tensor key_b =
        Tensor::Create(DT_STRING, {1},
                       CreateTestData<string_view>({"yet another key"}))
            .value();
    Tensor data_b =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
    EXPECT_THAT(agg2->Accumulate({&key_b, &data_b}), IsOk());
  }

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state1,
                             std::move(*agg1).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> agg1,
        DeserializeTensorAggregator(intrinsic, serialized_state1));
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state2,
                             std::move(*agg2).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> agg2,
        DeserializeTensorAggregator(intrinsic, serialized_state2));
  }

  // Merge the aggregators. The result should contain the 3 different keys.
  // "key" should map to 1 while the other two keys should map to 1000.
  // If we wrote merge wrong, the code might do the following to agg2:
  // (1) pick one of "other key" or "yet another key" at random (l0_bound = 1),
  // or (2) force one of the sums from 1000 to 1 (linfinity_bound_ = 1)
  // or both
  auto merge_status = agg1->MergeWith(std::move(*agg2));
  EXPECT_THAT(merge_status, IsOk());
  auto result = std::move(*agg1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value()[0].num_elements(), Eq(3));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({3}, {1, 1000, 1000}));
}

TEST_P(DPOpenDomainHistogramTest, MergeDoesNotDistortData_MultiKey) {
  Intrinsic intrinsic = CreateIntrinsic2Key2Agg<int64_t, int64_t>(
      kEpsilonThreshold, 0.001, 1, 1, -1, -1);
  auto agg1 = CreateTensorAggregator(intrinsic).value();
  auto agg2 = CreateTensorAggregator(intrinsic).value();

  // agg1 gets one person's data, mapping "red apple" to 1,1
  Tensor red =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"red"}))
          .value();
  Tensor apple =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"apple"}))
          .value();
  Tensor data1 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  Tensor data2 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  EXPECT_THAT(agg1->Accumulate({&red, &apple, &data1, &data2}), IsOk());

  // agg2 gets data from a lot more people. At the end, it will map "red apple"
  // to 1000,1000 and "white grape" to 1000,1000.
  for (int i = 0; i < 1000; i++) {
    EXPECT_THAT(agg2->Accumulate({&red, &apple, &data1, &data2}), IsOk());

    Tensor white =
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"white"}))
            .value();
    Tensor grape =
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"grape"}))
            .value();
    EXPECT_THAT(agg2->Accumulate({&white, &grape, &data1, &data2}), IsOk());
  }

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state1,
                             std::move(*agg1).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> agg1,
        DeserializeTensorAggregator(intrinsic, serialized_state1));
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state2,
                             std::move(*agg2).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> agg2,
        DeserializeTensorAggregator(intrinsic, serialized_state2));
  }

  // Merge the aggregators. The result should contain two different keys.
  // "red apple" should map to 1001, 1001 while "white grape" should map to
  // 1000, 1000.
  auto merge_status = agg1->MergeWith(std::move(*agg2));
  EXPECT_THAT(merge_status, IsOk());
  auto result = std::move(*agg1).Report();
  ASSERT_THAT(result, IsOk());
  ASSERT_THAT(result.value()[0].num_elements(), Eq(2));
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({2}, {"red", "white"}));
  EXPECT_THAT(result.value()[1],
              IsTensor<string_view>({2}, {"apple", "grape"}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({2}, {1001, 1000}));
}

TEST_P(DPOpenDomainHistogramTest, MergeDoesNotDistortData_NoKeys) {
  Intrinsic intrinsic = {
      .uri = "fedsql_dp_group_by",
      .inputs = {},
      .outputs = {},
      .parameters = {CreateTopLevelParameters(kEpsilonThreshold, 0.01, 100)},
      .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int64_t, int64_t>(10, 9, 8));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int64_t, int64_t>(100, 9, -1));
  auto agg1 = CreateTensorAggregator(intrinsic).value();
  auto agg2 = CreateTensorAggregator(intrinsic).value();
  Tensor data1 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  Tensor data2 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  TFF_EXPECT_OK(agg1->Accumulate({&data1, &data2}));
  // Aggregate should be 1, 1

  for (int i = 0; i < 1000; i++) {
    Tensor data3 =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({10})).value();
    Tensor data4 =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({10})).value();
    TFF_EXPECT_OK(agg2->Accumulate({&data3, &data4}));
  }
  // Aggregate should be 8000, 9000 due to contribution bounding.

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state1,
                             std::move(*agg1).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> agg1,
        DeserializeTensorAggregator(intrinsic, serialized_state1));
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state2,
                             std::move(*agg2).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> agg2,
        DeserializeTensorAggregator(intrinsic, serialized_state2));
  }

  auto merge_status = agg1->MergeWith(std::move(*agg2));
  TFF_ASSERT_OK(merge_status);
  auto result = std::move(*agg1).Report();
  TFF_ASSERT_OK(result.status());
  ASSERT_EQ(result.value().size(), 2);
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {8001}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {9001}));
}

// Eighth test batch: order of outputs doesn't leak which keys were added.

TEST_P(DPOpenDomainHistogramTest, RowsAreShuffled) {
  // We will simulate 26000 clients each contributing to the count of 1 letter.
  std::string alphabet[26] = {"a", "b", "c", "d", "e", "f", "g", "h", "i",
                              "j", "k", "l", "m", "n", "o", "p", "q", "r",
                              "s", "t", "u", "v", "w", "x", "y", "z"};

  // We perform DP aggregation with a large epsilon, so that the standard
  // deviation of each random variable is small.
  Intrinsic intrinsic = CreateIntrinsic<int32_t, int32_t>(50.0, 0.01, 1, 1);
  auto aggregator = CreateTensorAggregator(intrinsic).value();

  for (int i = 0; i < 26000; i++) {
    Tensor keys =
        Tensor::Create(DT_STRING, {},
                       CreateTestData<string_view>({alphabet[i % 26]}))
            .value();

    Tensor value_tensor =
        Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({1})).value();
    TFF_EXPECT_OK(aggregator->Accumulate({&keys, &value_tensor}));
  }
  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        aggregator, DeserializeTensorAggregator(intrinsic, serialized_state));
  }
  auto report = std::move(*aggregator).Report();
  TFF_ASSERT_OK(report.status());
  ASSERT_EQ(report.value().size(), 2);

  // Because each letter receives 1000 contributions and the stdev is small, it
  // is likely that all keys survive NoiseAndThreshold. The output is some
  // permutation of the alphabet; the chance of being identical is 1/(26!).
  auto report_span = report.value()[0].AsSpan<string_view>();
  EXPECT_THAT(report_span, UnorderedElementsAreArray(alphabet));
  EXPECT_THAT(report_span, Not(ElementsAreArray(alphabet)));
}

// Ninth test batch: creation of aggregator with min_contributors_to_group set.

TEST(DPOpenDomainHistogramTest,
     CreateWithMinContributorsSetsSelectorAndMaxContributors) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/5, /*key_types=*/{}, /*epsilon=*/1.0);
  auto aggregator_or_status = CreateTensorAggregator(intrinsic);
  ASSERT_OK(aggregator_or_status);
  auto aggregator = *std::move(aggregator_or_status);

  DPOpenDomainHistogramPeer peer(std::move(aggregator));
  EXPECT_TRUE(peer.HasSelector());
  // max_contributors_to_group should be greater than the min_contributors
  // parameter in the intrinsic, but we don't want to test the exact value just
  // that it is set and not obviously too low.
  EXPECT_THAT(peer.GetMaxContributorsToGroup(), Gt(5));
}

TEST(DPOpenDomainHistogramTest,
     CreateWithoutMinContributorsDoesNotSetSelectorAndMaxContributors) {
  Intrinsic intrinsic = CreateIntrinsic<int64_t, int64_t>();

  auto aggregator_or_status = CreateTensorAggregator(intrinsic);
  ASSERT_OK(aggregator_or_status);
  auto aggregator = *std::move(aggregator_or_status);

  DPOpenDomainHistogramPeer peer(std::move(aggregator));
  EXPECT_THAT(peer.HasSelector(), IsFalse());
  EXPECT_THAT(peer.GetMaxContributorsToGroup(), Eq(std::nullopt));
}

// Tenth test batch: the k-thresholding feature does not break Accumulate - the
// logic that updates the contribution counts should account for groups that
// were dropped due to enforcing the L0 bound a.k.a. max_groups_contributed.
TEST(DPOpenDomainHistogramTest, AccumulateWhileKThresholding) {
  Intrinsic intrinsic = CreateIntrinsic2Agg<int64_t, int64_t>(
      /*epsilon=*/1.0, /*delta=*/1e-8, /*l0_bound=*/2, /*linfinity_bound1=*/1);
  intrinsic.parameters.push_back(Tensor::Create(DT_INT64, {},
                                                CreateTestData<int64_t>({5}),
                                                "min_contributors_to_group")
                                     .value());
  // A simulated client will offer contributions to 3 groups but our L0 value
  // is 2 so one of those groups will be dropped (an ordinal will be -1).
  auto agg = CreateTensorAggregator(intrinsic).value();
  Tensor keys = Tensor::Create(DT_STRING, {3},
                               CreateTestData<string_view>({"a", "b", "c"}))
                    .value();
  Tensor value_tensor1 =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({1, 1, 1})).value();
  Tensor value_tensor2 =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({1, 1, 1})).value();
  auto acc_status = agg->Accumulate({&keys, &value_tensor1, &value_tensor2});
  TFF_ASSERT_OK(acc_status);
  DPOpenDomainHistogramPeer peer(std::move(agg));
  EXPECT_THAT(peer.GetContributorsToGroups(), ElementsAreArray({1, 1}));
}

// Eleventh test batch: Report applies k-thresholding and DP noise correctly
// when min_contributors_to_group is set.

TEST_P(DPOpenDomainHistogramTest,
       SingleKeyAndMinContributorsAggregatesWithValueZeroCanSurvive) {
  Intrinsic intrinsic = CreateIntrinsic2Agg<int32_t, int64_t>(
      /*epsilon=*/1.0, /*delta=*/1e-8, /*l0_bound=*/2, /*linfinity_bound1=*/1,
      /*l1_bound1=*/-1, /*l2_bound1=*/-1, /*linfinity_bound2=*/1,
      /*l1_bound2=*/-1, /*l2_bound2=*/-1);
  intrinsic.parameters.push_back(
      Tensor::Create(internal::TypeTraits<int64_t>::kDataType, {},
                     CreateTestData<int64_t>({5}), "min_contributors_to_group")
          .value());
  TFF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TensorAggregator> dp_aggregator,
                           CreateTensorAggregator(intrinsic));
  // Simulate many clients where they contribute
  // aggregation 1: zeroes to key0 and ones to key1
  // aggregation 2: zeroes to both
  // The number of clients contributing is such that no key is removed by
  // k-thresholding.
  std::string key0 = "keep me";
  std::string key1 = "keep me also";
  constexpr int num_inputs = 400;
  for (int i = 0; i < num_inputs; i++) {
    Tensor keys = Tensor::Create(DT_STRING, {2},
                                 CreateTestData<string_view>({key0, key1}))
                      .value();

    Tensor value_tensor1 =
        Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({0, 1})).value();
    Tensor value_tensor2 =
        Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({0, 0})).value();

    EXPECT_THAT(
        dp_aggregator->Accumulate({&keys, &value_tensor1, &value_tensor2}),
        IsOk());
  }

  EXPECT_THAT(dp_aggregator->GetNumInputs(), Eq(num_inputs));
  EXPECT_THAT(dp_aggregator->CanReport(), IsTrue());

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*dp_aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> dp_aggregator,
        DeserializeTensorAggregator(intrinsic, serialized_state));
  }

  TFF_ASSERT_OK_AND_ASSIGN(OutputTensorList report,
                           std::move(*dp_aggregator).Report());
  ASSERT_THAT(report.size(), Eq(3));

  // The report should include both keys because we don't drop aggregates with
  // value zero when k-thresholding.
  EXPECT_THAT(report[0].AsSpan<string_view>(),
              UnorderedElementsAre(key0, key1));
}

TEST_P(DPOpenDomainHistogramTest, ReportAppliesKThresholdingEvenWithoutDP) {
  // Test that when k-thresholding is enabled and no group survives, the
  // output is empty.
  const int64_t min_contributors_to_group = 5;
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      min_contributors_to_group, /*key_types=*/{DT_STRING});
  TFF_ASSERT_OK_AND_ASSIGN(auto dp_aggregator,
                           CreateTensorAggregator(intrinsic));
  std::string key0 = "drop me";
  std::string key1 = "keep me";
  // Simulate too few clients for key0 to survive k-thresholding.
  constexpr int kNumInputsKey0 = 4;
  for (int i = 0; i < kNumInputsKey0; i++) {
    Tensor keys =
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({key0}))
            .value();

    Tensor value_tensor =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();

    TFF_ASSERT_OK(dp_aggregator->Accumulate({&keys, &value_tensor}));
  }
  // Simulate enough clients for key1 to survive k-thresholding.
  constexpr int kNumInputsKey1 = 100;
  for (int i = 0; i < kNumInputsKey1; i++) {
    Tensor keys =
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({key1}))
            .value();

    Tensor value_tensor =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();

    TFF_ASSERT_OK(dp_aggregator->Accumulate({&keys, &value_tensor}));
  }

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*dp_aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> dp_aggregator,
        DeserializeTensorAggregator(intrinsic, serialized_state));
  }

  TFF_ASSERT_OK_AND_ASSIGN(OutputTensorList report,
                           std::move(*dp_aggregator).Report());
  ASSERT_THAT(report.size(), Eq(2));

  // The report should not include key0 because it has too few
  // contributors.
  EXPECT_THAT(report[0].AsSpan<string_view>(), UnorderedElementsAre(key1));
}

TEST(DPOpenDomainHistogramTest, ReportAppliesDPDuringSelection) {
  // Check that the protocol doesn't publish (with probability much more than
  // delta) values that have only k contributors.
  const int64_t min_contributors_to_group = 5;
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int32_t, int32_t>(
      min_contributors_to_group, /*key_types=*/{DT_INT32}, /*epsilon=*/1.0);
  int key = 1;
  constexpr int kNumInputs = 5;
  constexpr int kNumRepeats = 100;
  int times_kept = 0;
  for (int i = 0; i < kNumRepeats; i++) {
    TFF_ASSERT_OK_AND_ASSIGN(auto dp_aggregator,
                             CreateTensorAggregator(intrinsic));
    for (int i = 0; i < kNumInputs; i++) {
      Tensor keys =
          Tensor::Create(DT_INT32, {1}, CreateTestData<int32_t>({key})).value();

      Tensor value_tensor =
          Tensor::Create(DT_INT32, {1}, CreateTestData<int32_t>({1})).value();

      TFF_EXPECT_OK(dp_aggregator->Accumulate({&keys, &value_tensor}));
    }
    TFF_ASSERT_OK_AND_ASSIGN(OutputTensorList report,
                             std::move(*dp_aggregator).Report());
    ASSERT_THAT(report.size(), Eq(2));
    if (report[0].num_elements() > 0) {
      times_kept++;
    }
  }
  // The key should be kept with probability at most delta = 0.001 so
  // probability of keeping at least 10 is at most 10^-30 times 100 choose 10 <
  // 10^-16.
  EXPECT_THAT(times_kept, Lt(10));
}

TEST_P(DPOpenDomainHistogramTest, NoiseAddedForSmallEpsilonsWithKThresholding) {
  const int64_t min_contributors_to_group = 5;
  // We need to add not too much noise to avoid flakiness from the
  // thresholding, but not so little to get flakiness from happening to add zero
  // noise. Testing four keys and only requiring one of them to have changed
  // makes this easier without kNumInputs being too large.
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int32_t, int64_t>(
      min_contributors_to_group, /*key_types=*/{DT_STRING}, /*epsilon=*/0.05,
      /*delta=*/0.001, /*l0_bound=*/4);
  TFF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TensorAggregator> dp_aggregator,
                           CreateTensorAggregator(intrinsic));
  constexpr int kNumInputs = 4000;
  for (int i = 0; i < kNumInputs; i++) {
    Tensor keys =
        Tensor::Create(
            DT_STRING, {4},
            CreateTestData<string_view>({"key0", "key1", "key2", "key3"}))
            .value();
    Tensor values =
        Tensor::Create(DT_INT32, {4}, CreateTestData<int32_t>({1, 1, 1, 1}))
            .value();
    TFF_EXPECT_OK(dp_aggregator->Accumulate({&keys, &values}));
  }
  EXPECT_EQ(dp_aggregator->GetNumInputs(), kNumInputs);
  EXPECT_TRUE(dp_aggregator->CanReport());

  if (GetParam()) {
    TFF_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                             std::move(*dp_aggregator).Serialize());
    TFF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TensorAggregator> dp_aggregator,
        DeserializeTensorAggregator(intrinsic, serialized_state));
  }

  TFF_ASSERT_OK_AND_ASSIGN(OutputTensorList report,
                           std::move(*dp_aggregator).Report());
  EXPECT_EQ(report.size(), 2);
  const absl::Span<const int64_t>& values = report[1].AsSpan<int64_t>();
  ASSERT_THAT(values.size(), Eq(4));
  EXPECT_THAT(values, Contains(Ne(kNumInputs)));
}

INSTANTIATE_TEST_SUITE_P(
    DPOpenDomainHistogramTestInstantiation, DPOpenDomainHistogramTest,
    ValuesIn<bool>({false, true}),
    [](const TestParamInfo<DPOpenDomainHistogramTest::ParamType>& info) {
      return info.param ? "SerializeDeserialize" : "None";
    });

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
