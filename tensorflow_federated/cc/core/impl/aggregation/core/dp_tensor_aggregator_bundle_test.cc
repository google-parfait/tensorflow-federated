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
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
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

// The second batch of tests concerns aggregation.
Intrinsic CreateBundleOfTwo() {
  Intrinsic intrinsic = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(), {}};
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<int>());
  intrinsic.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<float>());
  return intrinsic;
}

// If a bundle contains k aggregators and each expects c Tensors for input,
// the bundle expects c*k Tensors for input.
TEST(DPTensorAggregatorBundleTest, AggregateWrongNumberOfInputs) {
  Intrinsic intrinsic = CreateBundleOfTwo();
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto aggregator = std::move(status.value());

  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData<int>({1})).value();
  Tensor t2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData<int>({1})).value();

  // Too few inputs.
  auto accumulate_status = aggregator->Accumulate(InputTensorList{&t1});
  EXPECT_THAT(accumulate_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(accumulate_status.message(),
              HasSubstr("Expected 2 tensors, got 1"));

  // Too many inputs.
  auto accumulate_status2 =
      aggregator->Accumulate(InputTensorList{&t1, &t2, &t3});
  EXPECT_THAT(accumulate_status2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(accumulate_status2.message(),
              HasSubstr("Expected 2 tensors, got 3"));
}

// If a bundle has two inner aggregators and receives inputs such that those
// meant for aggregator 0 are correct but those meant for aggregator 1 are
// incorrect, the bundle should return an error status *and not update state*.
TEST(DPTensorAggregatorBundleTest, PartiallyCorrectInputDoesNotUpdateState) {
  Intrinsic intrinsic = CreateBundleOfTwo();
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto aggregator = std::move(status.value());

  // Give the first aggregator the right type of input, but not the second.
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData<int>({1})).value();
  Tensor t2 =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1})).value();
  auto accumulate_status = aggregator->Accumulate(InputTensorList{&t1, &t2});
  EXPECT_THAT(accumulate_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(accumulate_status.message(),
              HasSubstr("Expected an input of type 1,"  // DT_FLOAT
                        " but got 2"));                 // DT_DOUBLE
  EXPECT_EQ(aggregator->GetNumInputs(), 0);
}

// If all n inputs are valid, GetNumInputs() should return n after accumulation.
TEST(DPTensorAggregatorBundleTest, AllCorrectInputs) {
  Intrinsic intrinsic = CreateBundleOfTwo();
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto aggregator = std::move(status.value());

  for (int i = 0; i < 10; i++) {
    Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData<int>({1})).value();
    Tensor t2 =
        Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
    auto accumulate_status = aggregator->Accumulate(InputTensorList{&t1, &t2});
    TFF_EXPECT_OK(accumulate_status);
  }
  EXPECT_EQ(aggregator->GetNumInputs(), 10);
}

// The third batch of tests concerns merging. Tests of compatibility are also
// covered here.

// MergeWith is only compatible with other bundles.
TEST(DPTensorAggregatorBundleTest, DifferentTypesIncompatible) {
  Intrinsic intrinsic = CreateBundleOfTwo();
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto aggregator_ptr = std::move(status.value());
  auto& aggregator =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr.get());

  Intrinsic intrinsic2 = Intrinsic{kDPQuantileUri,
                                   {CreateTensorSpec("unused", DT_DOUBLE)},
                                   {CreateTensorSpec("unused", DT_DOUBLE)},
                                   {},
                                   {}};
  intrinsic2.parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({0.8})).value());
  auto status2 = CreateTensorAggregator(intrinsic2);
  TFF_EXPECT_OK(status2);
  auto aggregator_ptr2 = std::move(status2.value());
  auto compatible = aggregator.IsCompatible(*std::move(aggregator_ptr2));
  EXPECT_THAT(compatible, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      compatible.message(),
      HasSubstr("Can only merge with another DPTensorAggregatorBundle"));
}

// Two merging bundles need to have the same number of inner aggregators.
TEST(DPTensorAggregatorBundleTest, DifferentSizesIncompatible) {
  Intrinsic intrinsic1 = CreateBundleOfTwo();
  auto status1 = CreateTensorAggregator(intrinsic1);
  TFF_EXPECT_OK(status1);
  auto aggregator_ptr1 = std::move(status1.value());
  auto& aggregator1 =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr1.get());

  Intrinsic intrinsic2 = CreateBundleOfTwo();
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<int>());
  auto status2 = CreateTensorAggregator(intrinsic2);
  TFF_EXPECT_OK(status2);
  auto aggregator_ptr2 = std::move(status2.value());
  auto& aggregator2 =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr2.get());

  auto compatible = aggregator1.IsCompatible(aggregator2);
  EXPECT_THAT(compatible, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      compatible.message(),
      HasSubstr("One bundle has 2 nested aggregators, but the other has 3"));
}

// If any of the inner aggregators are incompatible, the bundles are as well.
TEST(DPTensorAggregatorBundleTest, DifferentInnerAggregatorsIncompatible) {
  Intrinsic intrinsic1 = CreateBundleOfTwo();
  auto status1 = CreateTensorAggregator(intrinsic1);
  TFF_EXPECT_OK(status1);
  auto aggregator_ptr1 = std::move(status1.value());
  auto& aggregator1 =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr1.get());

  Intrinsic intrinsic2 = Intrinsic{
      kDPTensorAggregatorBundleUri, {}, {}, CreateBundleParameters(), {}};
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<int>());
  intrinsic2.nested_intrinsics.push_back(CreateDPQuantileIntrinsic<double>());
  auto status2 = CreateTensorAggregator(intrinsic2);
  TFF_EXPECT_OK(status2);
  auto aggregator_ptr2 = std::move(status2.value());
  auto& aggregator2 =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr2.get());

  auto compatible = aggregator1.IsCompatible(aggregator2);
  EXPECT_THAT(compatible, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      compatible.message(),
      HasSubstr("Can only merge with another DPQuantileAggregator of the same"
                " input type."));
}

// After a successful merge, num_inputs_ should be the sum of the number of
// inputs of the pre-merge bundles.
TEST(DPTensorAggregatorBundleTest, SuccessfulMerge) {
  Intrinsic intrinsic1 = CreateBundleOfTwo();
  auto status1 = CreateTensorAggregator(intrinsic1);
  TFF_EXPECT_OK(status1);
  auto aggregator_ptr1 = std::move(status1.value());
  auto& aggregator1 =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr1.get());
  for (int i = 0; i < 4; i++) {
    Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData<int>({1})).value();
    Tensor t2 =
        Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
    auto accumulate_status = aggregator1.Accumulate(InputTensorList{&t1, &t2});
    TFF_EXPECT_OK(accumulate_status);
  }

  Intrinsic intrinsic2 = CreateBundleOfTwo();
  auto status2 = CreateTensorAggregator(intrinsic2);
  TFF_EXPECT_OK(status2);
  auto aggregator_ptr2 = std::move(status2.value());
  auto& aggregator2 =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr2.get());
  for (int i = 0; i < 6; i++) {
    Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData<int>({1})).value();
    Tensor t2 =
        Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
    auto accumulate_status = aggregator2.Accumulate(InputTensorList{&t1, &t2});
    TFF_EXPECT_OK(accumulate_status);
  }

  auto merge_status = aggregator2.MergeWith(std::move(aggregator1));
  EXPECT_EQ(aggregator2.GetNumInputs(), 10);
}

// The fourth batch of tests concerns serialization and deserialization.

// If a string is not a valid serialization of a bundle, the bundle should not
// be created.
TEST(DPTensorAggregatorBundleTest, InvalidSerialization) {
  std::string invalid_serialization = "invalid";
  const auto* factory =
      GetAggregatorFactory(kDPTensorAggregatorBundleUri).value();
  auto status =
      factory->Deserialize(CreateBundleOfTwo(), invalid_serialization);
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.status().message(),
              HasSubstr("Failed to parse serialized aggregator"));
}

// If a string is a valid serialization of a bundle, the bundle should be
// created with the right number of inputs.
TEST(DPTensorAggregatorBundleTest, ValidSerialization) {
  auto status1 = CreateTensorAggregator(CreateBundleOfTwo());
  TFF_EXPECT_OK(status1);
  auto aggregator_ptr1 = std::move(status1.value());
  auto& aggregator1 =
      dynamic_cast<DPTensorAggregatorBundle&>(*aggregator_ptr1.get());
  for (int i = 0; i < 4; i++) {
    Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData<int>({1})).value();
    Tensor t2 =
        Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
    auto accumulate_status = aggregator1.Accumulate(InputTensorList{&t1, &t2});
    TFF_EXPECT_OK(accumulate_status);
  }
  auto serialize_status = std::move(aggregator1).Serialize();
  TFF_EXPECT_OK(serialize_status);
  std::string serialized_string = serialize_status.value();
  const auto* factory =
      GetAggregatorFactory(kDPTensorAggregatorBundleUri).value();
  auto status2 = factory->Deserialize(CreateBundleOfTwo(), serialized_string);
  TFF_EXPECT_OK(status2);
  EXPECT_EQ(status2.value()->GetNumInputs(), 4);
}

}  // namespace

}  // namespace aggregation
}  // namespace tensorflow_federated
