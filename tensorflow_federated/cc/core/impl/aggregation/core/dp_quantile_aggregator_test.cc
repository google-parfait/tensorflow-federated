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
#include <random>
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
  return TensorSpec(name, dtype, {});
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

// Input and Output specs must each contain one tensor.
TEST(DPQuantileAggregatorTest, MultipleInputOrOutputSpecs) {
  Intrinsic two_output_intrinsic =
      Intrinsic{kDPQuantileUri,
                {CreateTensorSpec("value", DT_INT32)},
                {CreateTensorSpec("value", DT_DOUBLE),
                 CreateTensorSpec("value", DT_DOUBLE)},
                {CreateDPQuantileParameters(0.5)},
                {}};
  auto aggregator_status1 = CreateTensorAggregator(two_output_intrinsic);
  EXPECT_THAT(aggregator_status1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status1.status().message(),
              HasSubstr("Expected one output tensor, but got 2"));

  Intrinsic two_input_intrinsic =
      Intrinsic{kDPQuantileUri,
                {CreateTensorSpec("value", DT_INT32),
                 CreateTensorSpec("value", DT_INT32)},
                {CreateTensorSpec("value", DT_DOUBLE)},
                {CreateDPQuantileParameters(0.5)},
                {}};
  auto aggregator_status2 = CreateTensorAggregator(two_input_intrinsic);
  EXPECT_THAT(aggregator_status2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status2.status().message(),
              HasSubstr("Expected one input tensor, but got 2"));
}

// The tensors in the specs must be scalars.
TEST(DPQuantileAggregatorTest, NonScalarInputOrOutputSpecs) {
  Intrinsic vector_output_intrinsic =
      Intrinsic{kDPQuantileUri,
                {CreateTensorSpec("value", DT_INT32)},
                {TensorSpec("value", DT_DOUBLE, {2})},
                {CreateDPQuantileParameters(0.5)},
                {}};
  auto aggregator_status1 = CreateTensorAggregator(vector_output_intrinsic);
  EXPECT_THAT(aggregator_status1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status1.status().message(),
              HasSubstr("Expected a scalar output tensor, but got a tensor with"
                        " 2 elements."));

  Intrinsic vector_input_intrinsic =
      Intrinsic{kDPQuantileUri,
                {TensorSpec("value", DT_INT32, {2})},
                {CreateTensorSpec("value", DT_DOUBLE)},
                {CreateDPQuantileParameters(0.5)},
                {}};
  auto aggregator_status2 = CreateTensorAggregator(vector_input_intrinsic);
  EXPECT_THAT(aggregator_status2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status2.status().message(),
              HasSubstr("Expected a scalar input tensor, but got a tensor with"
                        " 2 elements."));
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

// Can aggregate scalars. Expect buffer size to be <= kDPQuantileMaxInputs.
TEST(DPQuantileAggregatorTest, AggregateTensorsSuccessful_Fractional) {
  auto aggregator_status = CreateDPQuantileAggregator(DT_FLOAT);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator<float>&>(*aggregator_status.value());

  for (int i = 1; i <= kDPQuantileMaxInputs + 10; ++i) {
    float val = 0.5 + i;
    Tensor t =
        Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({val})).value();
    auto accumulate_status = aggregator.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
    EXPECT_EQ(aggregator.GetBufferSize(),
              i < kDPQuantileMaxInputs ? i : kDPQuantileMaxInputs);
    EXPECT_EQ(aggregator.GetNumInputs(), i);
  }
  EXPECT_EQ(aggregator.GetReservoirSamplingCount(), 10);
}
TEST(DPQuantileAggregatorTest, AggregateTensorsSuccessful_Integer) {
  auto aggregator_status = CreateDPQuantileAggregator(DT_INT32);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator<int32_t>&>(*aggregator_status.value());

  for (int i = 1; i <= kDPQuantileMaxInputs + 10; ++i) {
    int32_t val = i;
    Tensor t =
        Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({val})).value();
    auto accumulate_status = aggregator.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
    EXPECT_EQ(aggregator.GetBufferSize(),
              i < kDPQuantileMaxInputs ? i : kDPQuantileMaxInputs);
    EXPECT_EQ(aggregator.GetNumInputs(), i);
  }
  EXPECT_EQ(aggregator.GetReservoirSamplingCount(), 10);
}

// The third batch of tests is on merging with another DPQuantileAggregator.
// The IsCompatible function is tested as part of this batch.

// Cannot merge with a different type.
TEST(DPQuantileAggregatorTest, DifferentTypeIncompatible) {
  // Cannot merge DPQualtileAggregator<double> with DPQuantileAggregator<float>.
  auto aggregator_status = CreateDPQuantileAggregator(DT_DOUBLE);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status.value());

  auto mismatched_aggregator_status1 = CreateDPQuantileAggregator(DT_FLOAT);
  TFF_EXPECT_OK(mismatched_aggregator_status1);
  auto& mismatched_aggregator1 = *mismatched_aggregator_status1.value();
  auto compatibility = aggregator.IsCompatible(mismatched_aggregator1);
  EXPECT_THAT(compatibility, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(compatibility.message(),
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
  compatibility = aggregator.IsCompatible(mismatched_aggregator2);
  EXPECT_THAT(compatibility, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(compatibility.message(),
              HasSubstr("Can only merge with another DPQuantileAggregator of"
                        " the same input type."));
}

// Cannot merge with a different target quantile.
TEST(DPQuantileAggregatorTest, DifferentTargetQuantileIncompatible) {
  auto aggregator_status1 = CreateDPQuantileAggregator(DT_DOUBLE, 0.5);
  TFF_EXPECT_OK(aggregator_status1);
  auto& aggregator1 =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status1.value());

  auto aggregator_status2 = CreateDPQuantileAggregator(DT_DOUBLE, 0.75);
  TFF_EXPECT_OK(aggregator_status2);
  auto& aggregator2 = *aggregator_status2.value();
  auto compatibility = aggregator1.IsCompatible(aggregator2);
  EXPECT_THAT(compatibility, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(compatibility.message(),
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

  auto merge_status = aggregator1.MergeWith(std::move(aggregator2));
  TFF_EXPECT_OK(merge_status);
  EXPECT_EQ(aggregator1.GetBufferSize(), kDPQuantileMaxInputs);
  EXPECT_EQ(aggregator1.GetNumInputs(), kNumInputs1 + kNumInputs2);
  EXPECT_EQ(aggregator1.GetReservoirSamplingCount(),
            std::max(0, kNumInputs1 + kNumInputs2 - kDPQuantileMaxInputs));
}

// The fourth batch of tests is on ReportWithEpsilonAndDelta. The DP quantile
// algorithm should produce an output that reasonably approximates the target
// quantile.

// Given a DPQuantileAggregator and samples that were fed into it, check that
// its report is not too far from the target quantile.
void AssessDPQuantile(DPQuantileAggregator<double>& aggregator,
                      std::vector<double>& samples, double target_quantile) {
  // Obtain the report (estimated quantile)
  auto report_status = std::move(aggregator).ReportWithEpsilonAndDelta(1, 1e-7);
  TFF_EXPECT_OK(report_status);
  auto& output = report_status.value();
  EXPECT_EQ(output.size(), 1);
  EXPECT_EQ(output[0].dtype(), DT_DOUBLE);
  double estimate = output[0].AsScalar<double>();

  // Sort the samples and find the values whose ranks are
  // (target_quantile - 0.05) * # samples
  // & (target_quantile + 0.05) * # samples.
  std::sort(samples.begin(), samples.end());
  double left = samples[int_ceil((target_quantile - 0.05) * samples.size())];
  double right = samples[int_ceil((target_quantile + 0.05) * samples.size())];
  // the estimate should be in one of the following intervals:
  // (left-kDPQuantileLinearRate, right+kDPQuantileLinearRate) or
  // (left / kDPQuantileExponentialRate, right * kDPQuantileExponentialRate)
  bool in_linear_interval = estimate > left - kDPQuantileLinearRate &&
                            estimate < right + kDPQuantileLinearRate;
  bool in_exponential_interval = estimate > left / kDPQuantileExponentialRate &&
                                 estimate < right * kDPQuantileExponentialRate;
  EXPECT_TRUE(in_linear_interval || in_exponential_interval);
}

// Create a DPQuantileAggregator and feed it samples from a distribution given
// to it. Then assess the DPQuantileAggregator's report against real quantile.
template <typename T>
void AccumulateAndAssessDPQuantile(double target_quantile, int num_samples,
                                   std::default_random_engine& rng,
                                   T& distribution) {
  // Generate & send samples into a DPQuantileAggregator. Store the samples in a
  // vector so that we have ground truth to compare against.
  std::vector<double> samples;
  auto aggregator_status =
      CreateDPQuantileAggregator(DT_DOUBLE, target_quantile);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status.value());
  for (int i = 0; i < num_samples; ++i) {
    double sample = distribution(rng);
    samples.push_back(sample);
    Tensor t =
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({sample})).value();
    auto accumulate_status = aggregator.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
  }
  // Assess
  AssessDPQuantile(aggregator, samples, target_quantile);
}

TEST(DPQuantileAggregatorTest, GaussianData) {
  std::default_random_engine rng;

  // Vary quantile, # samples, and mean.
  for (double target_quantile : {0.1, 0.2, 0.5, 0.8, 0.9}) {
    for (int num_samples :
         {kDPQuantileMaxInputs / 2, 2 * kDPQuantileMaxInputs}) {
      for (double mean :
           {kDPQuantileEndOfLinearGrowth / 2, kDPQuantileEndOfLinearGrowth * 2,
            kDPQuantileEndOfLinearGrowth * 10}) {
        for (double stddev : {mean / 5.0, mean / 4.0, mean / 3.0}) {
          std::normal_distribution<double> distribution(mean, stddev);
          AccumulateAndAssessDPQuantile<std::normal_distribution<double>>(
              target_quantile, num_samples, rng, distribution);
        }
      }
    }
  }
}

TEST(DPQuantileAggregatorTest, ExponentialData) {
  std::default_random_engine rng;
  // Vary quantile, # samples, and scale
  for (double target_quantile : {0.1, 0.2, 0.5, 0.8, 0.9}) {
    for (int num_samples :
         {kDPQuantileMaxInputs / 2, 2 * kDPQuantileMaxInputs}) {
      for (double scale :
           {kDPQuantileEndOfLinearGrowth / 2, kDPQuantileEndOfLinearGrowth * 2,
            kDPQuantileEndOfLinearGrowth * 10}) {
        std::exponential_distribution<double> distribution(scale);
        AccumulateAndAssessDPQuantile<std::exponential_distribution<double>>(
            target_quantile, num_samples, rng, distribution);
      }
    }
  }
}

// There should be no error due to DP when epsilon >= kEpsilonThreshold.
TEST(DPQuantileAggregatorTest, ExactQuantileForLargeEpsilon) {
  for (double target_quantile : {0.001, 0.2, 0.8, 0.999}) {
    auto aggregator_status =
        CreateDPQuantileAggregator(DT_DOUBLE, target_quantile);
    TFF_EXPECT_OK(aggregator_status);
    auto& aggregator =
        dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status.value());
    for (int i = 0; i < 100; ++i) {
      double val = 0.1 * (99 - i);
      Tensor t =
          Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({val})).value();
      auto accumulate_status = aggregator.Accumulate(InputTensorList({&t}));
      TFF_EXPECT_OK(accumulate_status);
    }
    auto report_status =
        std::move(aggregator)
            .ReportWithEpsilonAndDelta(kEpsilonThreshold, 1e-7);
    TFF_EXPECT_OK(report_status);
    auto& output = report_status.value();
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].dtype(), DT_DOUBLE);
    double estimate = output[0].AsScalar<double>();
    int target_rank = static_cast<int>(target_quantile * 100);
    EXPECT_THAT(estimate, testing::DoubleEq(0.1 * target_rank));
  }
}

// ReportWithEpsilonAndDelta invokes some helper functions. This test
// checks that they work as intended.
TEST(DPQuantileAggregatorTest, CorrectHelperFunctions) {
  auto aggregator_status = CreateDPQuantileAggregator(DT_INT32, 0.75);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator<int32_t>&>(*aggregator_status.value());

  // Check that GetBucket works as intended.
  EXPECT_EQ(aggregator.GetBucket(0.0), 0.0);
  EXPECT_EQ(aggregator.GetBucket(kDPQuantileLinearRate), 1);
  EXPECT_EQ(aggregator.GetBucket(1.5 * kDPQuantileLinearRate), 2);
  // The bucket at which the linear growth ends.
  int liminal_bucket =
      int_ceil(kDPQuantileEndOfLinearGrowth / kDPQuantileLinearRate);
  EXPECT_EQ(aggregator.GetBucket(kDPQuantileEndOfLinearGrowth), liminal_bucket);
  EXPECT_EQ(aggregator.GetBucket(kDPQuantileEndOfLinearGrowth *
                                 kDPQuantileExponentialRate *
                                 kDPQuantileExponentialRate),
            liminal_bucket + 2);

  // Check that BucketUpperBound works as intended.
  EXPECT_EQ(aggregator.BucketUpperBound(1), kDPQuantileLinearRate);
  EXPECT_EQ(aggregator.BucketUpperBound(10), 10 * kDPQuantileLinearRate);
  EXPECT_EQ(aggregator.BucketUpperBound(liminal_bucket),
            kDPQuantileEndOfLinearGrowth);
  EXPECT_EQ(aggregator.BucketUpperBound(liminal_bucket + 2),
            kDPQuantileEndOfLinearGrowth * kDPQuantileExponentialRate *
                kDPQuantileExponentialRate);

  // Check that GetTargetRank works as intended.
  EXPECT_EQ(aggregator.GetTargetRank(), 0.0);
  for (int i = 0; i < 1000; ++i) {
    Tensor t =
        Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({i})).value();
    auto accumulate_status = aggregator.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
  }
  EXPECT_EQ(aggregator.GetTargetRank(), 750);
}

// The fifth batch tests serialization and deserialization.
TEST(DPQuantileAggregatorTest, SerializeAndDeserialize) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateDPQuantileParameters(0.1)},
                                  {}};

  auto aggregator_status1 = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status1);
  auto& aggregator1 =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status1.value());
  for (int i = 0; i < kDPQuantileMaxInputs + 1; ++i) {
    Tensor t =
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({50})).value();
    auto accumulate_status = aggregator1.Accumulate(InputTensorList({&t}));
    TFF_EXPECT_OK(accumulate_status);
  }
  auto serialized_state_statusor = std::move(aggregator1).Serialize();
  TFF_EXPECT_OK(serialized_state_statusor);
  auto serialized_state = serialized_state_statusor.value();

  auto factory = dynamic_cast<const DPQuantileAggregatorFactory*>(
      GetAggregatorFactory(kDPQuantileUri).value());
  auto aggregator_status2 = factory->Deserialize(intrinsic, serialized_state);
  TFF_EXPECT_OK(aggregator_status2);

  auto& aggregator2 =
      dynamic_cast<DPQuantileAggregator<double>&>(*aggregator_status2.value());
  EXPECT_EQ(aggregator2.GetBufferSize(), kDPQuantileMaxInputs);
  EXPECT_EQ(aggregator2.GetNumInputs(), kDPQuantileMaxInputs + 1);
  EXPECT_EQ(aggregator2.GetReservoirSamplingCount(), 1);

  auto report_status =
      std::move(aggregator2).ReportWithEpsilonAndDelta(1, 1e-7);
  TFF_EXPECT_OK(report_status);
  auto& output = report_status.value();
  EXPECT_EQ(output.size(), 1);
  EXPECT_EQ(output[0].dtype(), DT_DOUBLE);
  double estimate = output[0].AsScalar<double>();
  EXPECT_EQ(estimate, 50);
}
}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
