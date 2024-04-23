/*
 * Copyright 2023 Google LLC
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

#include <memory>
#include <string>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using testing::IsFalse;
using testing::IsTrue;
using testing::TestWithParam;

using FederatedMeanTest = TestWithParam<bool>;

TEST_P(FederatedMeanTest, ScalarAggregation_Succeeds) {
  Intrinsic federated_mean_intrinsic{"federated_mean",
                                     {TensorSpec{"foo", DT_FLOAT, {}}},
                                     {TensorSpec{"foo_out", DT_FLOAT, {}}},
                                     {},
                                     {}};
  auto aggregator = CreateTensorAggregator(federated_mean_intrinsic).value();
  Tensor v1 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
  Tensor v2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({2})).value();
  Tensor v3 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({3})).value();
  EXPECT_THAT(aggregator->Accumulate(v1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(v2), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize();
    aggregator = DeserializeTensorAggregator(federated_mean_intrinsic,
                                             serialized_state.value())
                     .value();
  }

  EXPECT_THAT(aggregator->Accumulate(v3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsFalse());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<float>({}, {2}));
}

TEST_P(FederatedMeanTest, WeightedScalarAggregation_Succeeds) {
  Intrinsic federated_mean_intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_FLOAT, {}}, TensorSpec{"bar", DT_FLOAT, {}}},
      {TensorSpec{"foo_out", DT_FLOAT, {}}},
      {},
      {}};
  auto aggregator = CreateTensorAggregator(federated_mean_intrinsic).value();
  Tensor v1 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
  Tensor w1 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
  Tensor v2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({2})).value();
  Tensor w2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({4})).value();
  Tensor v3 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({3})).value();
  Tensor w3 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({5})).value();
  EXPECT_THAT(aggregator->Accumulate({&v1, &w1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&v2, &w2}), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize();
    aggregator = DeserializeTensorAggregator(federated_mean_intrinsic,
                                             serialized_state.value())
                     .value();
  }

  EXPECT_THAT(aggregator->Accumulate({&v3, &w3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsFalse());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  float expected_value =
      static_cast<float>(1 * 1 + 2 * 4 + 3 * 5) / (1 + 4 + 5);
  EXPECT_THAT(result.value()[0], IsTensor<float>({}, {expected_value}));
}

TEST_P(FederatedMeanTest, DenseAggregation_Succeeds) {
  Intrinsic federated_mean_intrinsic{"federated_mean",
                                     {TensorSpec{"foo", DT_FLOAT, {4}}},
                                     {TensorSpec{"foo_out", DT_FLOAT, {4}}},
                                     {},
                                     {}};
  auto aggregator = CreateTensorAggregator(federated_mean_intrinsic).value();
  Tensor v1 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({1, 3, 15, 27}))
          .value();
  Tensor v2 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({10, 5, 1, 1}))
          .value();
  Tensor v3 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({4, 13, 8, 20}))
          .value();
  EXPECT_THAT(aggregator->Accumulate(v1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(v2), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize();
    aggregator = DeserializeTensorAggregator(federated_mean_intrinsic,
                                             serialized_state.value())
                     .value();
  }

  EXPECT_THAT(aggregator->Accumulate(v3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<float>({4}, {5, 7, 8, 16}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(FederatedMeanTest, WeightedDenseAggregation_Succeeds) {
  Intrinsic federated_mean_intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_FLOAT, {4}}, TensorSpec{"bar", DT_FLOAT, {}}},
      {TensorSpec{"foo_out", DT_FLOAT, {4}}},
      {},
      {}};
  auto aggregator = CreateTensorAggregator(federated_mean_intrinsic).value();
  Tensor v1 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({1, 3, 15, 27}))
          .value();
  Tensor w1 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
  Tensor v2 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({10, 5, 1, 1}))
          .value();
  Tensor w2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({4})).value();
  Tensor v3 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({4, 13, 8, 20}))
          .value();
  Tensor w3 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({5})).value();
  EXPECT_THAT(aggregator->Accumulate({&v1, &w1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&v2, &w2}), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize();
    aggregator = DeserializeTensorAggregator(federated_mean_intrinsic,
                                             serialized_state.value())
                     .value();
  }

  EXPECT_THAT(aggregator->Accumulate({&v3, &w3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  float e1 = static_cast<float>(1 * 1 + 10 * 4 + 4 * 5) / (1 + 4 + 5);
  float e2 = static_cast<float>(3 * 1 + 5 * 4 + 13 * 5) / (1 + 4 + 5);
  float e3 = static_cast<float>(15 * 1 + 1 * 4 + 8 * 5) / (1 + 4 + 5);
  float e4 = static_cast<float>(27 * 1 + 1 * 4 + 20 * 5) / (1 + 4 + 5);
  EXPECT_THAT(result.value()[0], IsTensor<float>({4}, {e1, e2, e3, e4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(FederatedMeanTest, Merge_Succeeds) {
  Intrinsic federated_mean_intrinsic{"federated_mean",
                                     {TensorSpec{"foo", DT_FLOAT, {}}},
                                     {TensorSpec{"foo_out", DT_FLOAT, {}}},
                                     {},
                                     {}};
  auto aggregator1 = CreateTensorAggregator(federated_mean_intrinsic).value();
  auto aggregator2 = CreateTensorAggregator(federated_mean_intrinsic).value();
  Tensor v1 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
  Tensor v2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({2})).value();
  Tensor v3 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({3})).value();
  EXPECT_THAT(aggregator1->Accumulate(v1), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(v2), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(v3), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator1 = DeserializeTensorAggregator(federated_mean_intrinsic,
                                              serialized_state1.value())
                      .value();
    aggregator2 = DeserializeTensorAggregator(federated_mean_intrinsic,
                                              serialized_state2.value())
                      .value();
  }

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator2->CanReport(), IsFalse());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<float>({}, {2}));
}

TEST_P(FederatedMeanTest, WeightedDenseMerge_Succeeds) {
  Intrinsic federated_mean_intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_FLOAT, {4}}, TensorSpec{"bar", DT_FLOAT, {}}},
      {TensorSpec{"foo_out", DT_FLOAT, {4}}},
      {},
      {}};
  auto aggregator1 = CreateTensorAggregator(federated_mean_intrinsic).value();
  auto aggregator2 = CreateTensorAggregator(federated_mean_intrinsic).value();
  Tensor v1 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({1, 3, 15, 27}))
          .value();
  Tensor w1 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1})).value();
  Tensor v2 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({10, 5, 1, 1}))
          .value();
  Tensor w2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({4})).value();
  Tensor v3 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({4, 13, 8, 20}))
          .value();
  Tensor w3 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({5})).value();
  EXPECT_THAT(aggregator1->Accumulate({&v1, &w1}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&v2, &w2}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&v3, &w3}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator1 = DeserializeTensorAggregator(federated_mean_intrinsic,
                                              serialized_state1.value())
                      .value();
    aggregator2 = DeserializeTensorAggregator(federated_mean_intrinsic,
                                              serialized_state2.value())
                      .value();
  }

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator2->CanReport(), IsFalse());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  float e1 = static_cast<float>(1 * 1 + 10 * 4 + 4 * 5) / (1 + 4 + 5);
  float e2 = static_cast<float>(3 * 1 + 5 * 4 + 13 * 5) / (1 + 4 + 5);
  float e3 = static_cast<float>(15 * 1 + 1 * 4 + 8 * 5) / (1 + 4 + 5);
  float e4 = static_cast<float>(27 * 1 + 1 * 4 + 20 * 5) / (1 + 4 + 5);
  EXPECT_THAT(result.value()[0], IsTensor<float>({4}, {e1, e2, e3, e4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(FederatedMeanTest, Create_WrongUri) {
  Intrinsic intrinsic{"wrong_uri",
                      {TensorSpec{"foo", DT_FLOAT, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}}},
                      {},
                      {}};

  Status s =
      (*GetAggregatorFactory("federated_mean"))->Create(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected intrinsic URI federated_mean"));
}

TEST(FederatedMeanTest, Create_UnsupportedNumberOfInputs) {
  Intrinsic intrinsic{
      "federated_mean",
      {TensorSpec{"foo", DT_FLOAT, {}}, TensorSpec{"bar", DT_FLOAT, {}}},
      {TensorSpec{"foo_out", DT_FLOAT, {}}},
      {},
      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      HasSubstr("Exactly one input is expected for federated_mean intrinsic"));
}

TEST(FederatedMeanTest, Create_WeightedUnsupportedNumberOfInputs) {
  Intrinsic intrinsic{"federated_weighted_mean",
                      {TensorSpec{"foo", DT_FLOAT, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}}},
                      {},
                      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly two inputs are expected for "
                                     "federated_weighted_mean intrinsic"));
}

TEST(FederatedMeanTest, Create_UnsupportedNumberOfOutputs) {
  Intrinsic intrinsic{"federated_mean",
                      {TensorSpec{"foo", DT_FLOAT, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}},
                       TensorSpec{"bar_out", DT_FLOAT, {}}},
                      {},
                      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one output tensor is expected"));
}

TEST(FederatedMeanTest,
     Create_UnsupportedUnmatchingInputValueAndOutputDataType) {
  Intrinsic intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_FLOAT, {}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_INT32, {}}},
      {},
      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      HasSubstr("Input value tensor and output tensor have mismatched specs"));
}

TEST(FederatedMeanTest, Create_UnsupportedInputValueDataType) {
  Intrinsic intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_INT32, {}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_INT32, {}}},
      {},
      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Input value tensor type must be DT_FLOAT"));
}

TEST(FederatedMeanTest, Create_UnsupportedUnmatchingInputValueAndOutputShape) {
  Intrinsic intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_FLOAT, {1}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_FLOAT, {2}}},
      {},
      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      HasSubstr("Input value tensor and output tensor have mismatched specs"));
}

TEST(FederatedMeanTest, Create_UnsupportedUndefinedInputValueShape) {
  Intrinsic intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_FLOAT, {-1}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_FLOAT, {-1}}},
      {},
      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      HasSubstr(
          "All dimensions of value tensor shape must be known in advance"));
}

TEST(FederatedMeanTest, Create_UnsupportedNonScalarWeight) {
  Intrinsic intrinsic{
      "federated_weighted_mean",
      {TensorSpec{"foo", DT_FLOAT, {2}}, TensorSpec{"bar", DT_INT32, {2}}},
      {TensorSpec{"foo_out", DT_FLOAT, {2}}},
      {},
      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("The weight must be a scalar"));
}

TEST(FederatedMeanTest, Create_UnsupportedIntrinsicWithParameter) {
  Tensor tensor = Tensor::Create(DT_FLOAT, {2, 3},
                                 CreateTestData<float>({1, 2, 3, 4, 5, 6}))
                      .value();
  Intrinsic intrinsic{"federated_mean",
                      {TensorSpec{"foo", DT_FLOAT, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}}},
                      {},
                      {}};
  intrinsic.parameters.push_back(std::move(tensor));

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected no parameters"));
}

TEST(FederatedMeanTest, Create_UnsupportedNestedIntrinsic) {
  Intrinsic inner{"federated_mean",
                  {TensorSpec{"foo", DT_FLOAT, {}}},
                  {TensorSpec{"foo_out", DT_FLOAT, {}}},
                  {},
                  {}};
  Intrinsic intrinsic{"federated_mean",
                      {TensorSpec{"bar", DT_FLOAT, {}}},
                      {TensorSpec{"bar_out", DT_FLOAT, {}}},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(std::move(inner));

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected no nested intrinsics"));
}

TEST(FederatedMeanTest, Deserialize_FailToParseProto) {
  Intrinsic federated_mean_intrinsic{"federated_mean",
                                     {TensorSpec{"foo", DT_FLOAT, {}}},
                                     {TensorSpec{"foo_out", DT_FLOAT, {}}},
                                     {},
                                     {}};
  std::string invalid_state("invalid_state");
  Status s =
      DeserializeTensorAggregator(federated_mean_intrinsic, invalid_state)
          .status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Failed to parse"));
}

INSTANTIATE_TEST_SUITE_P(
    FederatedMeanTestInstantiation, FederatedMeanTest,
    testing::ValuesIn<bool>({false, true}),
    [](const testing::TestParamInfo<FederatedMeanTest::ParamType>& info) {
      return info.param ? "SerializeDeserialize" : "None";
    });

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
