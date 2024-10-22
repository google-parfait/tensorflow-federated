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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_quantile_aggregator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {

namespace {
using ::testing::HasSubstr;

TensorSpec CreateTensorSpec(std::string name, DataType dtype) {
  return TensorSpec(name, dtype, {-1});
}

// Function to create a variable number of parameters. If is_string is false,
// the parameters will be copies of the target_quantile. Otherwise, they will
// be strings "invalid value".
std::vector<Tensor> CreateDPQuantileParameters(double target_quantile,
                                               int copies = 1,
                                               bool is_string = false) {
  std::vector<Tensor> parameters;
  for (int i = 0; i < copies; ++i) {
    if (is_string) {
      parameters.push_back(
          Tensor::Create(DT_STRING, {},
                         CreateTestData<string_view>({"invalid value"}))
              .value());
    } else {
      parameters.push_back(
          Tensor::Create(DT_DOUBLE, {},
                         CreateTestData<double>({target_quantile}))
              .value());
    }
  }
  return parameters;
}

// The first batch of tests is dedicated to the factory.

// Show that it is possible to create a DPQuantileAggregator. It should have
// zero inputs to begin with.
TEST(DPQuantileAggregatorTest, CreateWorks) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_INT32)},
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateDPQuantileParameters(0.5)},
                                  {}};
  auto aggregator_status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status);
  // Just a check to ensure we can access the object as a DPQuantileAggregator.
  auto& dp_aggregator =
      dynamic_cast<DPQuantileAggregator<int32_t>&>(*aggregator_status.value());
  EXPECT_EQ(dp_aggregator.GetNumInputs(), 0);
}

// Multiple parameters in the intrinsic are not allowed.
TEST(DPQuantileAggregatorTest, MultipleParametersInIntrinsic) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_INT32)},
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateDPQuantileParameters(0.5, 2)},
                                  {}};
  auto aggregator_status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(aggregator_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status.status().message(),
              HasSubstr("Expected exactly one parameter, but got 2"));
}

// A string parameter in the intrinsic is not allowed.
TEST(DPQuantileAggregatorTest, OneStringParameterInIntrinsic) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_INT32)},
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateDPQuantileParameters(0.5, 1, true)},
                                  {}};
  auto aggregator_status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(aggregator_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status.status().message(),
              HasSubstr("Expected a double for the `target_quantile` parameter"
                        " of DPQuantileAggregator, but got DT_STRING"));
}

// A target quantile <= 0 is not allowed.
TEST(DPQuantileAggregatorTest, NegativeParameterInIntrinsic) {
  // Negative
  Intrinsic intrinsic1 = Intrinsic{kDPQuantileUri,
                                   {CreateTensorSpec("value", DT_INT32)},
                                   {CreateTensorSpec("value", DT_DOUBLE)},
                                   {CreateDPQuantileParameters(-0.5)},
                                   {}};
  auto aggregator_status1 = CreateTensorAggregator(intrinsic1);
  EXPECT_THAT(aggregator_status1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status1.status().message(),
              HasSubstr("Target quantile must be in (0, 1)"));

  // Zero
  Intrinsic intrinsic2 = Intrinsic{kDPQuantileUri,
                                   {CreateTensorSpec("value", DT_INT32)},
                                   {CreateTensorSpec("value", DT_DOUBLE)},
                                   {CreateDPQuantileParameters(0)},
                                   {}};
  auto aggregator_status2 = CreateTensorAggregator(intrinsic2);
  EXPECT_THAT(aggregator_status2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status2.status().message(),
              HasSubstr("Target quantile must be in (0, 1)"));
}

// A target quantile >= 1 is not allowed.
TEST(DPQuantileAggregatorTest, LargeParameterInIntrinsic) {
  // Larger than 1
  Intrinsic intrinsic1 = Intrinsic{kDPQuantileUri,
                                   {CreateTensorSpec("value", DT_INT32)},
                                   {CreateTensorSpec("value", DT_DOUBLE)},
                                   {CreateDPQuantileParameters(1.5)},
                                   {}};
  auto aggregator_status1 = CreateTensorAggregator(intrinsic1);
  EXPECT_THAT(aggregator_status1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status1.status().message(),
              HasSubstr("Target quantile must be in (0, 1)"));

  // One
  Intrinsic intrinsic2 = Intrinsic{kDPQuantileUri,
                                   {CreateTensorSpec("value", DT_INT32)},
                                   {CreateTensorSpec("value", DT_DOUBLE)},
                                   {CreateDPQuantileParameters(1)},
                                   {}};
  auto aggregator_status2 = CreateTensorAggregator(intrinsic2);
  EXPECT_THAT(aggregator_status2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status2.status().message(),
              HasSubstr("Target quantile must be in (0, 1)"));
}

// Input and output types must be numeric.
TEST(DPQuantileAggregatorTest, NonNumericInput) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_STRING)},
                                  {CreateTensorSpec("value", DT_STRING)},
                                  {CreateDPQuantileParameters(0.5)},
                                  {}};
  auto aggregator_status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(aggregator_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      aggregator_status.status().message(),
      HasSubstr("DPQuantileAggregator only supports numeric datatypes"));
}

// Outputs are doubles.
TEST(DPQuantileAggregatorTest, WrongOutputType) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateTensorSpec("value", DT_INT32)},
                                  {CreateDPQuantileParameters(0.5)},
                                  {}};
  auto aggregator_status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(aggregator_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status.status().message(),
              HasSubstr("Output type must be double"));
}

// The second batch of tests is on aggregating data.

StatusOr<std::unique_ptr<TensorAggregator>> CreateDPQuantileAggregator(
    DataType dtype, double target_quantile = 0.5) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", dtype)},
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateDPQuantileParameters(target_quantile)},
                                  {}};
  return CreateTensorAggregator(intrinsic);
}

// Cannot aggregate multiple tensors.
TEST(DPQuantileAggregatorTest, AggregateMultipleTensors) {
  auto aggregator_status = CreateDPQuantileAggregator(DT_INT32);
  TFF_EXPECT_OK(aggregator_status);
  auto aggregator = std::move(aggregator_status.value());
  Tensor t1 =
      Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({1})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({2})).value();
  auto accumulate_stauts = aggregator->Accumulate(InputTensorList({&t1, &t2}));
  EXPECT_THAT(accumulate_stauts, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(accumulate_stauts.message(),
              HasSubstr("Expected exactly one tensor, but got 2"));
}

// Cannot aggregate a tensor with the wrong shape.
TEST(DPQuantileAggregatorTest, AggregateTensorWithWrongShape) {
  auto aggregator_status = CreateDPQuantileAggregator(DT_INT64);
  TFF_EXPECT_OK(aggregator_status);
  auto aggregator = std::move(aggregator_status.value());
  Tensor t =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 1})).value();
  auto accumulate_stauts = aggregator->Accumulate(InputTensorList({&t}));
  EXPECT_THAT(accumulate_stauts, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      accumulate_stauts.message(),
      HasSubstr("Expected a scalar tensor, but got a tensor with 2 elements"));
}

// Cannot aggregate a tensor with the wrong dtype.
TEST(DPQuantileAggregatorTest, AggregateTensorWithWrongDtype) {
  auto aggregator_status = CreateDPQuantileAggregator(DT_FLOAT);
  TFF_EXPECT_OK(aggregator_status);
  auto aggregator = std::move(aggregator_status.value());
  Tensor t =
      Tensor::Create(DT_DOUBLE, {1}, CreateTestData<double>({1.01})).value();
  auto accumulate_stauts = aggregator->Accumulate(InputTensorList({&t}));
  EXPECT_THAT(accumulate_stauts, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(accumulate_stauts.message(),
              HasSubstr("Expected a DT_FLOAT tensor, but got a DT_DOUBLE"
                        " tensor"));
}

// Can aggregate scalars. Expect buffer size to be <= kDPQuantileMaxInputs.
TEST(DPQuantileAggregatorTest, AggregateTensorsSuccessful) {
  auto aggregator_status = CreateDPQuantileAggregator(DT_DOUBLE);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status.value());

  for (int i = 1; i <= kDPQuantileMaxInputs + 10; ++i) {
    double val = 0.5 + i;
    Tensor t =
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({val})).value();
    auto accumulate_status = aggregator.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
    EXPECT_EQ(aggregator.GetBufferSize(),
              i < kDPQuantileMaxInputs ? i : kDPQuantileMaxInputs);
    EXPECT_EQ(aggregator.GetNumInputs(), i);
  }
  EXPECT_EQ(aggregator.GetReservoirSamplingCount(), 10);
}

// The third batch of tests is on merging with another DPQuantileAggregator.

// Cannot merge with the wrong type.
TEST(DPQuantileAggregatorTest, MergeWithWrongType) {
  // Cannot merge DPQualtileAggregator<double> with DPQuantileAggregator<float>.
  auto aggregator_status = CreateDPQuantileAggregator(DT_DOUBLE);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator = *aggregator_status.value();

  auto mismatched_aggregator_status1 = CreateDPQuantileAggregator(DT_FLOAT);
  TFF_EXPECT_OK(mismatched_aggregator_status1);
  auto& mismatched_aggregator1 = *mismatched_aggregator_status1.value();
  auto merge_status = aggregator.MergeWith(std::move(mismatched_aggregator1));
  EXPECT_THAT(merge_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(merge_status.message(),
              HasSubstr("Can only merge with another DPQuantileAggregator of"
                        " the same input type."));

  // Cannot merge DPQualtileAggregator<double> with DPGroupingFederatedSum.
  std::vector<Tensor> parameters;
  for (int i = 0; i < 3; ++i) {
    parameters.push_back(
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({10})).value());
  }
  Intrinsic intrinsic = Intrinsic{kDPSumUri,
                                  {CreateTensorSpec("value", DT_INT64)},
                                  {CreateTensorSpec("value", DT_INT64)},
                                  std::move(parameters),
                                  {}};
  auto mismatched_aggregator_status2 = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(mismatched_aggregator_status2);
  auto& mismatched_aggregator2 = *mismatched_aggregator_status2.value();
  merge_status = aggregator.MergeWith(std::move(mismatched_aggregator2));
  EXPECT_THAT(merge_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(merge_status.message(),
              HasSubstr("Can only merge with another DPQuantileAggregator of"
                        " the same input type."));
}

// Cannot merge with a different target quantile.
TEST(DPQuantileAggregatorTest, MergeWithDifferentTargetQuantile) {
  auto aggregator_status1 = CreateDPQuantileAggregator(DT_DOUBLE, 0.5);
  TFF_EXPECT_OK(aggregator_status1);
  auto& aggregator1 = *aggregator_status1.value();

  auto aggregator_status2 = CreateDPQuantileAggregator(DT_DOUBLE, 0.75);
  TFF_EXPECT_OK(aggregator_status2);
  auto& aggregator2 = *aggregator_status2.value();
  auto merge_status = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(merge_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(merge_status.message(),
              HasSubstr("Target quantiles must match."));
}

// Can merge with the same target quantile. The size of the buffer should grow
// but stay <= kDPQuantileMaxInputs. The total number of inputs should be the
// sum of the two aggregators' input counts.
TEST(DPQuantileAggregatorTest, MergeWithSameTargetQuantile) {
  auto aggregator_status1 = CreateDPQuantileAggregator(DT_DOUBLE, 0.5);
  TFF_EXPECT_OK(aggregator_status1);
  auto& aggregator1 =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status1.value());
  int kNumInputs1 = kDPQuantileMaxInputs - 10;
  for (int i = 0; i < kNumInputs1; ++i) {
    double val = 0.5 + i;
    Tensor t =
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({val})).value();
    auto accumulate_status = aggregator1.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
  }
  EXPECT_EQ(aggregator1.GetBufferSize(),
            std::min(kNumInputs1, kDPQuantileMaxInputs));

  auto aggregator_status2 = CreateDPQuantileAggregator(DT_DOUBLE, 0.5);
  TFF_EXPECT_OK(aggregator_status2);
  auto& aggregator2 =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status2.value());
  int kNumInputs2 = kDPQuantileMaxInputs + 10;
  for (int i = 0; i < kNumInputs2; ++i) {
    double val = 0.5 + i;
    Tensor t =
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({val})).value();
    auto accumulate_status = aggregator2.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
  }
  EXPECT_EQ(aggregator2.GetBufferSize(),
            std::min(kNumInputs2, kDPQuantileMaxInputs));

  auto aggregator2_buffer_size = aggregator2.GetBufferSize();
  auto merge_status = aggregator1.MergeWith(std::move(aggregator2));
  TFF_EXPECT_OK(merge_status);
  EXPECT_EQ(aggregator1.GetBufferSize(), kDPQuantileMaxInputs);
  EXPECT_EQ(aggregator1.GetNumInputs(), kNumInputs1 + kNumInputs2);
  EXPECT_EQ(aggregator1.GetReservoirSamplingCount(),
            std::max(0, kNumInputs1 + aggregator2_buffer_size -
                            kDPQuantileMaxInputs));
}

// The fourth batch of tests is on ReportWithEpsilonAndDelta. The DP quantile
// algorithm should produce an output that reasonably approximates the target
// quantile.

// The fifth batch of tests is on serialization & deserialization.

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
