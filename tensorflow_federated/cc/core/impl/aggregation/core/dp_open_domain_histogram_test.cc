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
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;
using testing::Gt;
using ::testing::HasSubstr;
using testing::IsTrue;
using testing::TestWithParam;

using DPOpenDomainHistogramTest = TestWithParam<bool>;

TensorSpec CreateTensorSpec(std::string name, DataType dtype) {
  return TensorSpec(name, dtype, {-1});
}

// A simple Sum Aggregator
template <typename T>
class SumAggregator final : public AggVectorAggregator<T> {
 public:
  using AggVectorAggregator<T>::AggVectorAggregator;
  using AggVectorAggregator<T>::data;

 private:
  void AggregateVector(const AggVector<T>& agg_vector) override {
    // This aggregator is not expected to actually be used for aggregating in
    // the current tests.
    ASSERT_TRUE(false);
  }
};

template <typename EpsilonType, typename DeltaType, typename L0_BoundType>
std::vector<Tensor> CreateTopLevelParameters(EpsilonType epsilon,
                                             DeltaType delta,
                                             L0_BoundType l0_bound) {
  std::vector<Tensor> parameters;

  std::unique_ptr<MutableVectorData<EpsilonType>> epsilon_tensor =
      CreateTestData<EpsilonType>({epsilon});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<EpsilonType>::kDataType, {},
                     std::move(epsilon_tensor))
          .value());

  std::unique_ptr<MutableVectorData<DeltaType>> delta_tensor =
      CreateTestData<DeltaType>({delta});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<DeltaType>::kDataType, {},
                     std::move(delta_tensor))
          .value());

  std::unique_ptr<MutableVectorData<L0_BoundType>> l0_bound_tensor =
      CreateTestData<L0_BoundType>({l0_bound});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<L0_BoundType>::kDataType, {},
                     std::move(l0_bound_tensor))
          .value());
  return parameters;
}

std::vector<Tensor> CreateTopLevelParameters(double epsilon = 1000.0,
                                             double delta = 0.001,
                                             int64_t l0_bound = 100) {
  return CreateTopLevelParameters<double, double, int64_t>(epsilon, delta,
                                                           l0_bound);
}

std::vector<Tensor> CreateFewTopLevelParameters() {
  std::vector<Tensor> parameters;

  auto epsilon_tensor = CreateTestData({1.0});
  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, std::move(epsilon_tensor)).value());

  return parameters;
}

template <typename InputType>
std::vector<Tensor> CreateNestedParameters(InputType linfinity_bound,
                                           double l1_bound, double l2_bound) {
  std::vector<Tensor> parameters;

  parameters.push_back(
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, {},
                     CreateTestData<InputType>({linfinity_bound}))
          .value());

  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({l1_bound}))
          .value());

  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({l2_bound}))
          .value());

  return parameters;
}

template <typename InputType, typename OutputType>
Intrinsic CreateInnerIntrinsic(InputType linfinity_bound, double l1_bound,
                               double l2_bound) {
  return Intrinsic{
      kDPSumUri,
      {CreateTensorSpec("value", internal::TypeTraits<InputType>::kDataType)},
      {CreateTensorSpec("value", internal::TypeTraits<OutputType>::kDataType)},
      {CreateNestedParameters<InputType>(linfinity_bound, l1_bound, l2_bound)},
      {}};
}

template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsic(double epsilon = kEpsilonThreshold,
                          double delta = 0.001, int64_t l0_bound = 100,
                          InputType linfinity_bound = 100, double l1_bound = -1,
                          double l2_bound = -1) {
  Intrinsic intrinsic{kDPGroupByUri,
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {CreateTopLevelParameters(epsilon, delta, l0_bound)},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound, l1_bound,
                                                  l2_bound));
  return intrinsic;
}

// First batch of tests validate the intrinsic(s)
TEST(DPOpenDomainHistogramTest, CatchWrongNumberOfParameters) {
  Intrinsic too_few{kDPGroupByUri,
                    {CreateTensorSpec("key", DT_STRING)},
                    {CreateTensorSpec("key_out", DT_STRING)},
                    {CreateFewTopLevelParameters()},
                    {}};
  auto too_few_status = CreateTensorAggregator(too_few).status();
  EXPECT_THAT(too_few_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(too_few_status.message(),
              HasSubstr("Expected at least 3 parameters but got 1 of them"));
}

TEST(DPOpenDomainHistogramTest, CatchInvalidParameterTypes) {
  Intrinsic intrinsic0{
      kDPGroupByUri,
      {CreateTensorSpec("key", DT_STRING)},
      {CreateTensorSpec("key_out", DT_STRING)},
      {CreateTopLevelParameters<string_view, double, int64_t>("x", 0.1, 10)},
      {}};
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_epsilon.message(), HasSubstr("must be numerical"));

  Intrinsic intrinsic1{
      kDPGroupByUri,
      {CreateTensorSpec("key", DT_STRING)},
      {CreateTensorSpec("key_out", DT_STRING)},
      {CreateTopLevelParameters<double, string_view, int64_t>(1.0, "x", 10)},
      {}};
  auto bad_delta = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_delta, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta.message(), HasSubstr("must be numerical"));

  Intrinsic intrinsic2{
      kDPGroupByUri,
      {CreateTensorSpec("key", DT_STRING)},
      {CreateTensorSpec("key_out", DT_STRING)},
      {CreateTopLevelParameters<double, double, string_view>(1.0, 0.1, "x")},
      {}};
  auto bad_l0_bound = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_l0_bound, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_l0_bound.message(), HasSubstr("must be numerical"));
}

TEST(DPOpenDomainHistogramTest, CatchInvalidParameterValues) {
  Intrinsic intrinsic0 = CreateIntrinsic<int64_t, int64_t>(-1, 0.001, 10);
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_epsilon.message(), HasSubstr("Epsilon must be positive"));

  Intrinsic intrinsic1 = CreateIntrinsic<int64_t, int64_t>(1.0, -1, 10);
  auto bad_delta = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_delta, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta.message(), HasSubstr("delta must lie between 0 and 1"));

  Intrinsic intrinsic2 = CreateIntrinsic<int64_t, int64_t>(1.0, 0.001, -1);
  auto bad_l0_bound = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_l0_bound, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_l0_bound.message(), HasSubstr("L0 bound must be positive"));
}

TEST(DPOpenDomainHistogramTest, CatchInvalidLinfinityBound) {
  Intrinsic intrinsic =
      CreateIntrinsic<int64_t, int64_t>(1.0, 0.001, 10, -1, 2, 3);
  auto aggregator_status = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(aggregator_status,
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("must provide a positive Linfinity bound.")));
}

TEST(DPOpenDomainHistogramTest, Deserialize_FailToParseProto) {
  auto intrinsic = CreateIntrinsic<int64_t, int64_t>(100, 0.01, 1);
  std::string invalid_state("invalid_state");
  Status s = DeserializeTensorAggregator(intrinsic, invalid_state).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Failed to parse"));
}

TEST(DPOpenDomainHistogramTest, CatchUnsupportedNestedIntrinsic) {
  Intrinsic intrinsic = {kDPGroupByUri,
                         {CreateTensorSpec("key", DT_STRING)},
                         {CreateTensorSpec("key_out", DT_STRING)},
                         {CreateTopLevelParameters()},
                         {}};
  intrinsic.nested_intrinsics.push_back(Intrinsic{
      "GoogleSQL:$not_differential_privacy_sum",
      {CreateTensorSpec("value", internal::TypeTraits<int32_t>::kDataType)},
      {CreateTensorSpec("value", internal::TypeTraits<int32_t>::kDataType)},
      {CreateNestedParameters<int32_t>(1000, -1, -1)},
      {}});
  auto aggregator_status = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(aggregator_status, StatusIs(UNIMPLEMENTED));
  EXPECT_THAT(aggregator_status.message(), HasSubstr("Currently, only nested "
                                                     "DP sums are supported"));
}

// Function to execute the DPOpenDomainHistogram on one input where there is
// just one key per contribution and each contribution is to one aggregation
template <typename InputType>
StatusOr<OutputTensorList> SingleKeySingleAgg(
    const Intrinsic& intrinsic, const TensorShape shape,
    std::initializer_list<string_view> key_list,
    std::initializer_list<InputType> value_list, bool serialize_deserialize) {
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list))
          .value();

  Tensor value_tensor =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list))
          .value();
  auto acc_status = group_by_aggregator->Accumulate({&keys, &value_tensor});

  if (serialize_deserialize) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(1));

  return std::move(*group_by_aggregator).Report();
}

// Second batch of tests are dedicated to norm bounding when there is only one
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

// Third: test norm bounding when there are multiple inner aggregations
// (SUM(value1), SUM(value2)  GROUP BY key)
template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsic2Agg(double epsilon = kEpsilonThreshold,
                              double delta = 0.001, int64_t l0_bound = 100,
                              InputType linfinity_bound1 = 100,
                              double l1_bound1 = -1, double l2_bound1 = -1,
                              InputType linfinity_bound2 = 100,
                              double l1_bound2 = -1, double l2_bound2 = -1) {
  Intrinsic intrinsic{kDPGroupByUri,
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {CreateTopLevelParameters(epsilon, delta, l0_bound)},
                      {}};
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
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
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
      group_by_aggregator->Accumulate({&keys, &value_tensor1, &value_tensor2});

  if (serialize_deserialize) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(1));

  return std::move(*group_by_aggregator).Report();
}

TEST_P(DPOpenDomainHistogramTest, SingleKeyDoubleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4.
    // For agg 1, Linfinity bound is 20 and no other bounds provided.
    // For agg 2, Linfinity bound is 50, L1 bound is 100, L2 bound is 10.
    auto intrinsic = CreateIntrinsic2Agg<int64_t, int64_t>(
        2 * kEpsilonThreshold, 0.01, 4, 20, -1, -1, 50, 100, 10);
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

// Fourth: test norm bounding, when there are multiple keys and multiple inner
// aggregations. (SUM(value1), SUM(value2)  GROUP BY key1, key 2)
template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsic2Key2Agg(double epsilon = kEpsilonThreshold,
                                  double delta = 0.001, int64_t l0_bound = 100,
                                  InputType linfinity_bound1 = 100,
                                  double l1_bound1 = -1, double l2_bound1 = -1,
                                  InputType linfinity_bound2 = 100,
                                  double l1_bound2 = -1,
                                  double l2_bound2 = -1) {
  Intrinsic intrinsic{kDPGroupByUri,
                      {CreateTensorSpec("key1", DT_STRING),
                       CreateTensorSpec("key2", DT_STRING)},
                      {CreateTensorSpec("key1_out", DT_STRING),
                       CreateTensorSpec("key2_out", DT_STRING)},
                      {CreateTopLevelParameters(epsilon, delta, l0_bound)},
                      {}};
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
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
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
  auto acc_status = group_by_aggregator->Accumulate(
      {&keys1, &keys2, &value_tensor1, &value_tensor2});

  if (serialize_deserialize) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(1));

  return std::move(*group_by_aggregator).Report();
}

TEST_P(DPOpenDomainHistogramTest, DoubleKeyDoubleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4.
    // For agg 1, Linfinity bound is 20 and no other bounds provided.
    // For agg 2, Linfinity bound is 50, L1 bound is 100, L2 bound is 10.
    auto intrinsic = CreateIntrinsic2Key2Agg<int64_t, int64_t>(
        2 * kEpsilonThreshold, 0.01, 4, 20, -1, -1, 50, 100, 10);
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

// Fifth: test norm bounding on key-less data (norm bound = magnitude bound)
template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsicNoKeys(double epsilon = kEpsilonThreshold,
                                double delta = 0.001, int64_t l0_bound = 100,
                                InputType linfinity_bound1 = 100,
                                double l1_bound1 = -1, double l2_bound1 = -1,
                                InputType linfinity_bound2 = 100,
                                double l1_bound2 = -1, double l2_bound2 = -1,
                                InputType linfinity_bound3 = 100,
                                double l1_bound3 = -1, double l2_bound3 = -1) {
  Intrinsic intrinsic{kDPGroupByUri,
                      {},
                      {},
                      {CreateTopLevelParameters(epsilon, delta, l0_bound)},
                      {}};
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
      3 * kEpsilonThreshold, 0.01, 100, 10, 9, 8,  // limit to 8
      100, 9, -1,                                  // limit to 9
      100, -1, -1);                                // 100

  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&t1, &t2, &t3}), IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {8}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {9}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({1}, {11}));
}

// Sixth: check for proper noise addition.

// Check that noise is added at all: the noised sum should not be the same as
// the unnoised sum. The chance of a false negative shrinks with epsilon.
TEST_P(DPOpenDomainHistogramTest, NoiseAddedForSmallEpsilons) {
  Intrinsic intrinsic = CreateIntrinsic<int32_t, int64_t>(0.05, 1e-8, 2, 1);
  auto dpgba = CreateTensorAggregator(intrinsic).value();
  int num_inputs = 4000;
  for (int i = 0; i < num_inputs; i++) {
    Tensor keys = Tensor::Create(DT_STRING, {2},
                                 CreateTestData<string_view>({"key0", "key1"}))
                      .value();
    Tensor values =
        Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({1, 1})).value();
    auto acc_status = dpgba->Accumulate({&keys, &values});
    EXPECT_THAT(acc_status, IsOk());
  }
  EXPECT_EQ(dpgba->GetNumInputs(), num_inputs);
  EXPECT_TRUE(dpgba->CanReport());

  if (GetParam()) {
    auto serialized_state = std::move(*dpgba).Serialize();
    dpgba = DeserializeTensorAggregator(intrinsic, serialized_state.value())
                .value();
  }

  auto report = std::move(*dpgba).Report();
  EXPECT_THAT(report, IsOk());
  EXPECT_EQ(report->size(), 2);
  const auto& values = report.value()[1].AsSpan<int64_t>();
  ASSERT_THAT(values.size(), Eq(2));
  EXPECT_TRUE(values[0] != num_inputs || values[1] != num_inputs);
}

// Check that SetupNoiseAndThreshold is capable of switching between
// distributions
TEST_P(DPOpenDomainHistogramTest, SetupNoiseAndThreshold_CorrectDistribution) {
  Intrinsic intrinsic1{kDPGroupByUri,
                       {CreateTensorSpec("key", DT_STRING)},
                       {CreateTensorSpec("key_out", DT_STRING)},
                       {CreateTopLevelParameters(1.0, 1e-10, 2)},
                       {}};
  // "Baseline" aggregation where Laplace was chosen
  intrinsic1.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int32_t, int64_t>(10, -1, -1));

  // Aggregation where a given L2 norm bound is sufficiently smaller than L_0 *
  // L_inf, which means Gaussian is preferred.
  intrinsic1.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int32_t, int64_t>(10, -1, 2));

  auto agg1 = CreateTensorAggregator(intrinsic1).value();
  auto report = std::move(*agg1).Report();
  auto laplace_was_used =
      dynamic_cast<DPOpenDomainHistogram&>(*agg1).laplace_was_used();
  ASSERT_EQ(laplace_was_used.size(), 2);
  EXPECT_TRUE(laplace_was_used[0]);
  EXPECT_FALSE(laplace_was_used[1]);

  // If a user can contribute to L0 = x groups and there is only an L_inf bound,
  // Laplace noise is linear in x while Gaussian noise scales with sqrt(x).
  // Hence, we should use Gaussian when we loosen x (from 2 to 20)
  Intrinsic intrinsic2 =
      CreateIntrinsic<int32_t, int64_t>(1.0, 1e-10, 20, 10, -1, -1);
  auto agg2 = CreateTensorAggregator(intrinsic2).value();
  auto report2 = std::move(*agg2).Report();
  laplace_was_used =
      dynamic_cast<DPOpenDomainHistogram&>(*agg2).laplace_was_used();
  ASSERT_EQ(laplace_was_used.size(), 1);
  EXPECT_FALSE(laplace_was_used[0]);

  // Gaussian noise should also be used if delta was loosened enough
  Intrinsic intrinsic3 =
      CreateIntrinsic<int32_t, int64_t>(1.0, 1e-3, 2, 10, -1, -1);
  auto agg3 = CreateTensorAggregator(intrinsic3).value();
  auto report3 = std::move(*agg3).Report();
  laplace_was_used =
      dynamic_cast<DPOpenDomainHistogram&>(*agg3).laplace_was_used();
  ASSERT_EQ(laplace_was_used.size(), 1);
  EXPECT_FALSE(laplace_was_used[0]);
}

// Check that CalculateLaplaceThreshold computes the right threshold
TEST(DPOpenDomainHistogramTest, CalculateLaplaceThreshold_Succeeds) {
  // Case 1: adjusted delta less than 1/2
  double delta = 0.468559;  // = 1-(9/10)^6
  double linfinity_bound = 1;
  int64_t l0_bound = 1;

  // under replacement DP:
  int64_t l0_sensitivity = 2 * l0_bound;
  double l1_sensitivity = 2;  // = min(2 * l0_bound * linf_bound, 2 * l1_bound)

  // We'll work with eps = 1 for simplicity
  auto threshold_wrapper = internal::CalculateLaplaceThreshold<double>(
      1.0, delta, l0_sensitivity, linfinity_bound, l1_sensitivity);
  TFF_ASSERT_OK(threshold_wrapper.status());

  double laplace_tail_bound = 1.22497855;
  // = -(l1_sensitivity / 1.0) * std::log(2.0 * adjusted_delta),
  // where adjusted_delta = 1 - sqrt(1-delta) = 1 - (9/10)^3 = 1 - 0.729 = 0.271

  EXPECT_NEAR(threshold_wrapper.value(), linfinity_bound + laplace_tail_bound,
              1e-5);

  // Case 2: adjusted delta greater than 1/2
  delta = 0.77123207545039;  // 1-(9/10)^14
  threshold_wrapper = internal::CalculateLaplaceThreshold<double>(
      1.0, delta, l0_sensitivity, linfinity_bound, l1_sensitivity);
  TFF_ASSERT_OK(threshold_wrapper.status());

  laplace_tail_bound = -0.0887529;
  // = (l1_sensitivity / 1.0) * std::log(2.0 - 2.0 * adjusted_delta),
  // where adjusted_delta = 1 - sqrt(1-delta) = 1 - (9/10)^7 = 0.5217031
  EXPECT_NEAR(threshold_wrapper.value(), linfinity_bound + laplace_tail_bound,
              1e-5);
}

// Seventh: check that the right groups get dropped

// Test that we will drop groups with any small aggregate and keep groups with
// large aggregates. The surviving aggregates should have noise in them.
// This test has a small probability of failing due to false positives: noise
// could push the 0 past the threshold. It also has a small probability of
// failing due to false negatives: noise could push the 100 below the threshold.
TEST_P(DPOpenDomainHistogramTest, SingleKeyDropAggregatesWithValueZero) {
  // epsilon = 1, delta= 1e-8, L0 bound = 2, Linfinity bound = 1
  Intrinsic intrinsic =
      CreateIntrinsic2Agg<int32_t, int64_t>(1.0, 1e-8, 2, 1, -1, -1, 1, -1, -1);
  auto dpgba = CreateTensorAggregator(intrinsic).value();
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
        dpgba->Accumulate({&keys, &value_tensor1, &value_tensor2});
    EXPECT_THAT(acc_status, IsOk());
  }

  EXPECT_THAT(dpgba->GetNumInputs(), Eq(num_inputs));
  EXPECT_THAT(dpgba->CanReport(), IsTrue());

  if (GetParam()) {
    auto serialized_state = std::move(*dpgba).Serialize();
    dpgba = DeserializeTensorAggregator(intrinsic, serialized_state.value())
                .value();
  }

  auto report = std::move(*dpgba).Report();
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
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  for (int i = 0; i < 100; i++) {
    Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
    Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
    Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
    EXPECT_THAT(group_by_aggregator->Accumulate({&t1, &t2, &t3}), IsOk());
  }

  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
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
  Intrinsic intrinsic{
      kDPGroupByUri,
      {CreateTensorSpec("key1", DT_STRING),
       CreateTensorSpec("key2", DT_STRING)},
      // An empty string in the output keys means that the key should not be
      // included in the output.
      {CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)},
      {CreateTopLevelParameters(1.0, 1e-8, 3)},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int32_t, int64_t>(4, 8, -1));
  // L0 = 3, Linf = 4, L1 = 8
  // These bounds should not affect any individual input data below

  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
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
    EXPECT_THAT(
        group_by_aggregator->Accumulate({&sizeKeys1, &animalKeys1, &t1}),
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
    EXPECT_THAT(
        group_by_aggregator->Accumulate({&sizeKeys2, &animalKeys2, &t2}),
        IsOk());
    // Totals: [3, 5, 4, 4]
    EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(2 * (i + 1)));
  }

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  // Totals: [600, 1000, 800, 800]
  ASSERT_THAT(group_by_aggregator->CanReport(), IsTrue());
  auto result = std::move(*group_by_aggregator).Report();
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

// Finally, test merge: intermediary aggregates should not be clipped or noised

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
    auto serialized_state1 = std::move(*agg1).Serialize().value();
    auto serialized_state2 = std::move(*agg2).Serialize().value();
    agg1 = DeserializeTensorAggregator(intrinsic, serialized_state1).value();
    agg2 = DeserializeTensorAggregator(intrinsic, serialized_state2).value();
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
      2 * kEpsilonThreshold, 0.001, 1, 1, -1, -1);
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
    auto serialized_state1 = std::move(*agg1).Serialize().value();
    auto serialized_state2 = std::move(*agg2).Serialize().value();
    agg1 = DeserializeTensorAggregator(intrinsic, serialized_state1).value();
    agg2 = DeserializeTensorAggregator(intrinsic, serialized_state2).value();
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
  Intrinsic intrinsic{
      "fedsql_dp_group_by",
      {},
      {},
      {CreateTopLevelParameters(2 * kEpsilonThreshold, 0.01, 100)},
      {}};
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
    auto serialized_state1 = std::move(*agg1).Serialize().value();
    auto serialized_state2 = std::move(*agg2).Serialize().value();
    agg1 = DeserializeTensorAggregator(intrinsic, serialized_state1).value();
    agg2 = DeserializeTensorAggregator(intrinsic, serialized_state2).value();
  }

  auto merge_status = agg1->MergeWith(std::move(*agg2));
  TFF_ASSERT_OK(merge_status);
  auto result = std::move(*agg1).Report();
  TFF_ASSERT_OK(result.status());
  ASSERT_EQ(result.value().size(), 2);
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {8001}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {9001}));
}

INSTANTIATE_TEST_SUITE_P(
    DPOpenDomainHistogramTestInstantiation, DPOpenDomainHistogramTest,
    testing::ValuesIn<bool>({false, true}),
    [](const testing::TestParamInfo<DPOpenDomainHistogramTest::ParamType>&
           info) { return info.param ? "SerializeDeserialize" : "None"; });

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
