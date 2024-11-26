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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_tensor_aggregator_bundle.h"

#include <string>
#include <utility>
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
  return TensorSpec(name, dtype, {});
}

double kDefaultEpsilon = 1.0;
double kDefaultDelta = 1e-7;

// The first batch of tests concerns creation (via the factory)

std::vector<Tensor> CreateBundleParameters(double epsilon = kDefaultEpsilon,
                                           double delta = kDefaultDelta) {
  std::vector<Tensor> parameters;
  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({epsilon})).value());
  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({delta})).value());
  return parameters;
}

template <typename T>
Intrinsic CreateDPQuantileIntrinsic() {
  Tensor t =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({0.5})).value();
  Intrinsic intrinsic =
      Intrinsic{kDPQuantileUri,
                {CreateTensorSpec("in", internal::TypeTraits<T>::kDataType)},
                {CreateTensorSpec("out", DT_DOUBLE)},
                {},
                {}};
  intrinsic.parameters.push_back(std::move(t));
  return intrinsic;
}

// A bundle consists of at least one aggregator.
TEST(DPTensorAggregatorBundleTest, CreateZeroAggregators) {
  Intrinsic intrinsic = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(), {}};
  auto status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Expected at least one nested intrinsic, got none"));
}

// A bundle can only contain DPTensorAggregators.
TEST(DPTensorAggregatorBundleTest, CreateBadAggregators) {
  Intrinsic intrinsic = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(), {}};
  intrinsic.nested_intrinsics.push_back(
      Intrinsic{"GoogleSQL:sum",
                {CreateTensorSpec("x", DT_FLOAT)},
                {CreateTensorSpec("y", DT_FLOAT)},
                {},
                {}});
  auto status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Expected all nested intrinsics to be "
                        "DPTensorAggregators, got GoogleSQL:sum"));
}

// There are two parameters of the bundle.
TEST(DPTensorAggregatorBundleTest, CreateWrongNumberOfParameters) {
  // Too few parameters.
  Tensor t =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1.0})).value();
  Intrinsic intrinsic = Intrinsic{kDPTensorAggregatorBundleUri, {}, {}, {}, {}};
  intrinsic.parameters.push_back(std::move(t));
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Expected 2 parameters, got 1"));

  // Too many parameters.
  Tensor t1 =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1.0})).value();
  Tensor t2 =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1.0})).value();
  Tensor t3 =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1.0})).value();
  Intrinsic intrinsic2 =
      Intrinsic{kDPTensorAggregatorBundleUri, {}, {}, {}, {}};
  intrinsic2.parameters.push_back(std::move(t1));
  intrinsic2.parameters.push_back(std::move(t2));
  intrinsic2.parameters.push_back(std::move(t3));
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  status = CreateTensorAggregator(intrinsic2);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Expected 2 parameters, got 3"));
}

// Epsilon must be numerical.
TEST(DPTensorAggregatorBundleTest, CreateWrongTypeOfEpsilon) {
  Tensor epsilon =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"blah"}))
          .value();
  Tensor delta =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1e-7})).value();
  Intrinsic intrinsic = Intrinsic{kDPTensorAggregatorBundleUri, {}, {}, {}, {}};
  intrinsic.parameters.push_back(std::move(epsilon));
  intrinsic.parameters.push_back(std::move(delta));
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Epsilon must be numerical"));
}

// Epsilon must be positive.
TEST(DPTensorAggregatorBundleTest, CreateBadEpsilon) {
  // Exclude 0
  Intrinsic intrinsic = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(0), {}};
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Epsilon must be positive, but got 0"));

  // Exclude negatives.
  Intrinsic intrinsic2 = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(-1.0), {}};
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  status = CreateTensorAggregator(intrinsic2);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Epsilon must be positive, but got -1"));
}

// Delta must be numerical.
TEST(DPTensorAggregatorBundleTest, CreateWrongTypeOfDelta) {
  Tensor epsilon =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1.0})).value();
  Tensor delta =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"blah"}))
          .value();
  Intrinsic intrinsic = Intrinsic{kDPTensorAggregatorBundleUri, {}, {}, {}, {}};
  intrinsic.parameters.push_back(std::move(epsilon));
  intrinsic.parameters.push_back(std::move(delta));
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(), HasSubstr("Delta must be numerical"));
}

// Delta must be non-negative and less than 1.
TEST(DPTensorAggregatorBundleTest, CreateBadDelta) {
  // Exclude larger-than-1.
  Intrinsic intrinsic = Intrinsic{kDPTensorAggregatorBundleUri,
                                  {},
                                  {},
                                  CreateBundleParameters(1.0, 1.5),
                                  {}};
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status = CreateTensorAggregator(intrinsic);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Delta must be non-negative and less than 1, "
                        "but got 1.5"));

  // Exclude negatives.
  Intrinsic intrinsic2 = Intrinsic{kDPTensorAggregatorBundleUri,
                                   {},
                                   {},
                                   CreateBundleParameters(1.0, -0.5),
                                   {}};
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  status = CreateTensorAggregator(intrinsic2);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Delta must be non-negative and less than 1, "
                        "but got -0.5"));
}

// If there is no problem in an intrinsic with 1 quantile aggregator, the
// bundle should be created successfully. Epsilon and delta should not be split.
TEST(DPTensorAggregatorBundleTest, CreateSucceeds) {
  Intrinsic intrinsic = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(), {}};
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<int>());
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto* bundle = dynamic_cast<DPTensorAggregatorBundle*>(status.value().get());
  EXPECT_EQ(bundle->GetEpsilonPerAgg(), kDefaultEpsilon);
  EXPECT_EQ(bundle->GetDeltaPerAgg(), kDefaultDelta);
}
// If there is no problem in an intrinsic with 2 quantile aggregators, the
// bundle should be created successfully. Epsilon and delta should be split,
// unless the value of epsilon is at least kEpsilonThreshold.
TEST(DPTensorAggregatorBundleTest, CreateSucceedsWithSplitEpsilonAndDelta) {
  Intrinsic intrinsic1 = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(), {}};
  intrinsic1.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<int>());
  intrinsic1.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status1 = CreateTensorAggregator(intrinsic1);
  TFF_EXPECT_OK(status1);
  auto* bundle = dynamic_cast<DPTensorAggregatorBundle*>(status1.value().get());
  EXPECT_EQ(bundle->GetEpsilonPerAgg(), kDefaultEpsilon / 2);
  EXPECT_EQ(bundle->GetDeltaPerAgg(), kDefaultDelta / 2);

  Intrinsic intrinsic2 = Intrinsic{kDPTensorAggregatorBundleUri,
                                   {},
                                   {},
                                   CreateBundleParameters(kEpsilonThreshold),
                                   {}};
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<int>());
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status2 = CreateTensorAggregator(intrinsic2);
  TFF_EXPECT_OK(status2);
  bundle = dynamic_cast<DPTensorAggregatorBundle*>(status2.value().get());
  EXPECT_EQ(bundle->GetEpsilonPerAgg(), kEpsilonThreshold);
  EXPECT_EQ(bundle->GetDeltaPerAgg(), kDefaultDelta / 2);
}

}  // namespace

}  // namespace aggregation
}  // namespace tensorflow_federated
