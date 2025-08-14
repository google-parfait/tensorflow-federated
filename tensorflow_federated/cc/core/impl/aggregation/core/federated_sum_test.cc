/*
 * Copyright 2022 Google LLC
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
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using testing::IsTrue;
using ::testing::SizeIs;

Intrinsic GetDefaultIntrinsic() {
  // One "federated_sum" intrinsic with a single scalar int32 tensor.
  return Intrinsic{"federated_sum",
                   {TensorSpec{"foo", DT_INT32, {}}},
                   {TensorSpec{"foo_out", DT_INT32, {}}},
                   {},
                   {}};
}

TEST(FederatedSumTest, ScalarAggregation_Succeeds) {
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(FederatedSumTest, DenseAggregation_Succeeds) {
  Intrinsic federated_sum_intrinsic{"federated_sum",
                                    {TensorSpec{"foo", DT_INT32, {4}}},
                                    {TensorSpec{"foo_out", DT_INT32, {4}}},
                                    {},
                                    {}};
  auto aggregator = CreateTensorAggregator(federated_sum_intrinsic).value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({4}, {14, 19, 23, 49}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(FederatedSumTest, Merge_Succeeds) {
  auto aggregator1 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  auto aggregator2 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator1->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(t3), IsOk());

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(FederatedSumTest, SerializeDeserialize_Succeeds) {
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  auto serialized_state =
      std::move(*aggregator).Serialize(/*num_partitions=*/1);
  EXPECT_THAT(serialized_state, IsOkAndHolds(SizeIs(1)));
  auto deserialized_aggregator =
      DeserializeTensorAggregator(GetDefaultIntrinsic(),
                                  serialized_state.value()[0])
          .value();

  EXPECT_THAT(deserialized_aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(deserialized_aggregator->GetNumInputs(), Eq(3));
  EXPECT_THAT(deserialized_aggregator->CanReport(), IsTrue());

  auto result = std::move(*deserialized_aggregator).Report();
  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(FederatedSumTest, Create_WrongUri) {
  Intrinsic intrinsic{"wrong_uri",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_INT32, {}}},
                      {},
                      {}};

  Status s =
      (*GetAggregatorFactory("federated_sum"))->Create(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected intrinsic URI federated_sum"));
}

TEST(FederatedSumTest, Create_UnsupportedNumberOfInputs) {
  Intrinsic intrinsic{
      "federated_sum",
      {TensorSpec{"foo", DT_INT32, {}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_INT32, {}}},
      {},
      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one input is expected"));
}

TEST(FederatedSumTest, Create_UnsupportedEmptyIntrinsic) {
  Status s = (*GetAggregatorFactory("federated_sum"))
                 ->Create(Intrinsic{"federated_sum", {}, {}, {}, {}})
                 .status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one input is expected"));
}

TEST(FederatedSumTest, Create_UnsupportedNumberOfOutputs) {
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_INT32, {}},
                       TensorSpec{"bar_out", DT_INT32, {}}},
                      {},
                      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one output tensor is expected"));
}

TEST(FederatedSumTest, Create_UnsupportedUnmatchingInputAndOutputDataType) {
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}}},
                      {},
                      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Input and output tensors have mismatched specs"));
}

TEST(FederatedSumTest, Create_UnsupportedUnmatchingInputAndOutputShape) {
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {1}}},
                      {TensorSpec{"foo", DT_INT32, {2}}},
                      {},
                      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Input and output tensors have mismatched specs"));
}

TEST(FederatedSumTest, Create_UnsupportedIntrinsicWithParameter) {
  Tensor tensor = Tensor::Create(DT_FLOAT, {2, 3},
                                 CreateTestData<float>({1, 2, 3, 4, 5, 6}))
                      .value();
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {1}}},
                      {TensorSpec{"foo", DT_INT32, {2}}},
                      {},
                      {}};
  intrinsic.parameters.push_back(std::move(tensor));

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected no parameters"));
}

TEST(FederatedSumTest, Create_UnsupportedNestedIntrinsic) {
  Intrinsic inner{"federated_sum",
                  {TensorSpec{"foo", DT_INT32, {8}}},
                  {TensorSpec{"foo_out", DT_INT32, {16}}},
                  {},
                  {}};
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"bar", DT_INT32, {1}}},
                      {TensorSpec{"bar_out", DT_INT32, {2}}},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(std::move(inner));

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected no nested intrinsics"));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
