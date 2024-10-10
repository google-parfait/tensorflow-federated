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

#include <memory>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
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

// Show that it is possible to create a DPQuantileAggregator and observe that it
// has missing DP parameters.
TEST(DPQuantileAggregatorTest, CreateWorks) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_INT32)},
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateDPQuantileParameters(0.5)},
                                  {}};
  auto aggregator_status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator&>(*aggregator_status.value());
  EXPECT_EQ(aggregator.GetEpsilon(), -1);
  EXPECT_EQ(aggregator.GetDelta(), -1);
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

// The second batch of tests is on setting and verifying DP params.
TEST(DPQuantileAggregatorTest, EpsilonAndDelta) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateTensorSpec("value", DT_DOUBLE)},
                                  {CreateDPQuantileParameters(0.5)},
                                  {}};
  auto aggregator_status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status);
  auto& aggregator =
      dynamic_cast<DPQuantileAggregator&>(*(aggregator_status.value()));

  // We do not initially have valid epsilon and delta.
  EXPECT_FALSE(aggregator.HaveValidEpsilon());
  EXPECT_FALSE(aggregator.HaveValidDelta());
  EXPECT_FALSE(aggregator.HaveValidEpsilonAndDelta());

  // We cannot set epsilon to a negative value.
  EXPECT_THAT(aggregator.SetEpsilon(-0.5), StatusIs(INVALID_ARGUMENT));
  EXPECT_FALSE(aggregator.HaveValidEpsilon());

  // We can set epsilon to a positive value.
  TFF_EXPECT_OK(aggregator.SetEpsilon(0.5));
  EXPECT_TRUE(aggregator.HaveValidEpsilon());

  // We cannot set delta to zero, a negative value, or a value larger than 1.
  EXPECT_THAT(aggregator.SetDelta(0), StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator.SetDelta(-0.5), StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator.SetDelta(1.5), StatusIs(INVALID_ARGUMENT));
  EXPECT_FALSE(aggregator.HaveValidDelta());

  // We can set delta to a positive fraction.
  TFF_EXPECT_OK(aggregator.SetDelta(0.5));
  EXPECT_TRUE(aggregator.HaveValidEpsilon());
  EXPECT_TRUE(aggregator.HaveValidDelta());
  EXPECT_TRUE(aggregator.HaveValidEpsilonAndDelta());
}

// The third batch of tests is on aggregating and merging data.

// The fourth batch of tests is on TakeOutputs: the DP quantile algorithm should
// produce outputs that reasonably approximate the target quantile.

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
