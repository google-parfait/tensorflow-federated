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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_tensor_aggregator.h"

#include <memory>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
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

// To test ValidateInputs, we instantiate DPQuantileAggregator and throw a
// variety of inputs at it.

// Multiple tensors are not valid input.
TEST(DPTensorAggregatorTest, WrongNumberOfInputs) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {},
                                  {}};
  intrinsic.parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({0.5})).value());

  auto aggregator_status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status);
  auto aggregator = std::move(aggregator_status.value());
  Tensor t1 =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1})).value();
  Tensor t2 =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({2})).value();
  auto* ptr = dynamic_cast<DPTensorAggregator*>(aggregator.get());
  auto status = ptr->ValidateInputs(InputTensorList({&t1, &t2}));
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.message(),
              HasSubstr("Expected exactly 1 tensors, but got 2"));
}

// Tensors that are not scalars are not valid input.
TEST(DPTensorAggregatorTest, WrongShape) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {},
                                  {}};
  intrinsic.parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({0.5})).value());

  auto aggregator_status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status);
  auto aggregator = std::move(aggregator_status.value());
  Tensor t =
      Tensor::Create(DT_DOUBLE, {2}, CreateTestData<double>({1, 2})).value();
  auto* ptr = dynamic_cast<DPTensorAggregator*>(aggregator.get());
  auto status = ptr->ValidateInputs(InputTensorList({&t}));
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.message(),
              HasSubstr("Expected input with shape {}, but got {2}"));
}

// Tensors with the wrong dtype are not valid input.
TEST(DPTensorAggregatorTest, WrongDType) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {},
                                  {}};
  intrinsic.parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({0.5})).value());

  auto aggregator_status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status);
  auto aggregator = std::move(aggregator_status.value());
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1.0})).value();
  auto* ptr = dynamic_cast<DPTensorAggregator*>(aggregator.get());
  auto status = ptr->ValidateInputs(InputTensorList({&t}));
  EXPECT_THAT(status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(status.message(),
              HasSubstr("Expected an input of type 2,"  // DT_DOUBLE
                        " but got 1"));                 // DT_FLOAT
}

// ValidateInputs returns OK for valid input.
TEST(DPTensorAggregatorTest, ValidInput) {
  Intrinsic intrinsic = Intrinsic{kDPQuantileUri,
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {TensorSpec("value", DT_DOUBLE, {})},
                                  {},
                                  {}};
  intrinsic.parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({0.5})).value());

  auto aggregator_status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator_status);
  auto aggregator = std::move(aggregator_status.value());
  Tensor t =
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({1.0})).value();
  auto* ptr = dynamic_cast<DPTensorAggregator*>(aggregator.get());
  auto status = ptr->ValidateInputs(InputTensorList({&t}));
  TFF_EXPECT_OK(status);
}

}  // namespace

}  // namespace aggregation
}  // namespace tensorflow_federated
