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

#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {

class GroupByAggregatorPeer {
 public:
  explicit GroupByAggregatorPeer(GroupByAggregator* aggregator)
      : aggregator_(aggregator) {}

  Status AddOneContributor(Tensor ordinals) {
    return aggregator_->AddOneContributor(std::move(ordinals));
  }

  Status AddMultipleContributors(Tensor ordinals,
                                 std::vector<int> num_contributors) {
    return aggregator_->AddMultipleContributors(ordinals, num_contributors);
  }

  const std::vector<int>& GetContributors() const {
    return aggregator_->GetContributors();
  }

 private:
  GroupByAggregator* aggregator_;
};

namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using testing::IsFalse;
using testing::IsTrue;
using testing::TestWithParam;

using GroupByAggregatorTest = TestWithParam<bool>;

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

Intrinsic CreateDefaultInnerIntrinsic(DataType input_dtype,
                                      DataType output_dtype) {
  return Intrinsic{"GoogleSQL:sum",
                   {CreateTensorSpec("value", input_dtype)},
                   {CreateTensorSpec("value", output_dtype)},
                   {},
                   {}};
}

Intrinsic CreateDefaultIntrinsic() {
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  return intrinsic;
}

// Creates a default intrinsic and adds the min_contributors_to_group parameter.
Intrinsic CreateIntrinsicWithMinContributors(int min_contributors) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  intrinsic.parameters.push_back(
      Tensor::Create(DT_INT32, {}, CreateTestData({min_contributors})).value());
  return intrinsic;
}

TEST_P(GroupByAggregatorTest, EmptyReport) {
  // Intrinsic lifetime must outlast that of the TensorAggregator.
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({0}, {}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({0}, {}));
}

TEST_P(GroupByAggregatorTest, ScalarAggregation_Succeeds) {
  // Intrinsic lifetime must outlast that of the TensorAggregator.
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"key_string"}))
          .value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&key, &t1}), IsOk());
  EXPECT_THAT(group_by_aggregator->Accumulate({&key, &t2}), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(group_by_aggregator->Accumulate({&key, &t3}), IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({1}, {"key_string"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {6}));
}

TEST_P(GroupByAggregatorTest, AggregateOnlyEmptyTensorsSucceeds) {
  // Intrinsic lifetime must outlast that of the TensorAggregator.
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key =
      Tensor::Create(DT_STRING, {0}, CreateTestData<string_view>({})).value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {0}, CreateTestData<int32_t>({})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&key, &t1}), IsOk());
  EXPECT_THAT(group_by_aggregator->Accumulate({&key, &t1}), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(group_by_aggregator->Accumulate({&key, &t1}), IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({0}, {}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({0}, {}));
}

TEST_P(GroupByAggregatorTest, DenseAggregation_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "one", "two", "three"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys, &t1}), IsOk());
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys, &t2}), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  EXPECT_THAT(group_by_aggregator->Accumulate({&keys, &t3}), IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(2));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>(shape, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensors are dense.
  EXPECT_TRUE(result.value()[0].is_dense());
  EXPECT_TRUE(result.value()[1].is_dense());
}

TEST_P(GroupByAggregatorTest, AccumulateEmptyInputDoesNotAffectResult) {
  const TensorShape shape = {4};
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "one", "two", "three"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys, &t1}), IsOk());
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys, &t2}), IsOk());
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys, &t3}), IsOk());

  // Now accumulate an empty input. This will increase NumInputs to 4 but
  // otherwise has no effect on the result.
  Tensor empty_keys =
      Tensor::Create(DT_STRING, {0}, CreateTestData<string_view>({})).value();
  Tensor empty_tensor =
      Tensor::Create(DT_INT32, {0}, CreateTestData<int32_t>({})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&empty_keys, &empty_tensor}),
              IsOk());

  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(4));

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(2));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>(shape, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensors are dense.
  EXPECT_TRUE(result.value()[0].is_dense());
  EXPECT_TRUE(result.value()[1].is_dense());
}

TEST_P(GroupByAggregatorTest, DifferentKeysPerAccumulate_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys1, &t1}), IsOk());
  // Totals: [4, 15, 27]
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys2, &t2}), IsOk());
  // Totals: [9, 26, 27, 2]

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  Tensor keys3 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"two", "two", "four", "one"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys3, &t3}), IsOk());
  // Totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensors.
  EXPECT_THAT(
      result.value()[0],
      IsTensor<string_view>({5}, {"zero", "one", "two", "three", "four"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({5}, {9, 46, 41, 2, 7}));
}

TEST_P(GroupByAggregatorTest, DifferentShapesPerAccumulate_Succeeds) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys1, &t1}), IsOk());
  // Totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys2, &t2}), IsOk());
  // Totals: [2, 10, 19, 4]

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  Tensor keys3 =
      Tensor::Create(
          DT_STRING, {5},
          CreateTestData<string_view>({"two", "two", "one", "zero", "four"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 6, 3})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys3, &t3}), IsOk());
  // Totals: [8, 17, 33, 4, 3]
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensors.
  EXPECT_THAT(
      result.value()[0],
      IsTensor<string_view>({5}, {"zero", "one", "two", "three", "four"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({5}, {8, 17, 33, 4, 3}));
}

TEST_P(GroupByAggregatorTest, Accumulate_MultipleValueTensors_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  Tensor tA1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor tB1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({14, 11, 7, 14})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys1, &tA1, &tB1}), IsOk());
  // Totals: [4, 15, 27], [25, 7, 14]
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  Tensor tA2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor tB2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 2, 8})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys2, &tA2, &tB2}), IsOk());
  // Totals: [9, 26, 27, 2], [28, 10, 14, 8]
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(2));

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {9, 26, 27, 2}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({4}, {28, 10, 14, 8}));
}

TEST_P(GroupByAggregatorTest, Accumulate_NoValueTensors_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {},
                      {}};
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys1}), IsOk());
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&keys2}), IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(2));

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
}

TEST_P(GroupByAggregatorTest, Accumulate_MultipleKeyTensors_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key1", DT_STRING),
                       CreateTensorSpec("key2", DT_STRING)},
                      {CreateTensorSpec("key1_out", DT_STRING),
                       CreateTensorSpec("key2_out", DT_STRING)},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&sizeKeys1, &animalKeys1, &t1}),
              IsOk());
  // Totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&sizeKeys2, &animalKeys2, &t2}),
              IsOk());
  // Totals: [9, 26, 27, 2]

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  Tensor sizeKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "small"}))
          .value();
  Tensor animalKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"dog", "dog", "rabbit", "cat"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&sizeKeys3, &animalKeys3, &t3}),
              IsOk());
  // Totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>(
                  {5}, {"large", "small", "large", "small", "small"}));
  EXPECT_THAT(
      result.value()[1],
      IsTensor<string_view>({5}, {"cat", "cat", "dog", "dog", "rabbit"}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({5}, {9, 46, 41, 2, 7}));
}

TEST_P(GroupByAggregatorTest,
       Accumulate_MultipleKeyTensors_SomeKeysNotInOutput_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{
      "fedsql_group_by",
      {CreateTensorSpec("key1", DT_STRING),
       CreateTensorSpec("key2", DT_STRING)},
      // An empty string in the output keys means that the key should not be
      // included in the output.
      {CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)},
      {},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();

  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&sizeKeys1, &animalKeys1, &t1}),
              IsOk());
  // Totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&sizeKeys2, &animalKeys2, &t2}),
              IsOk());
  // Totals: [9, 26, 27, 2]
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(2));

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensors.
  // Only the second key tensor should be included in the output.
  EXPECT_THAT(result.value().size(), Eq(2));
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"cat", "cat", "dog", "dog"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {9, 26, 27, 2}));
}

TEST_P(GroupByAggregatorTest,
       MultipleKeyTensorsSomeKeysNotInOutputSucceedsWhenAllInputsEmpty) {
  const TensorShape shape = {0};
  Intrinsic intrinsic{
      "fedsql_group_by",
      {CreateTensorSpec("key1", DT_STRING),
       CreateTensorSpec("key2", DT_STRING)},
      // An empty string in the output keys means that the key should not be
      // included in the output.
      {CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)},
      {},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();

  Tensor size_keys =
      Tensor::Create(DT_STRING, {0}, CreateTestData<string_view>({})).value();
  Tensor animal_keys =
      Tensor::Create(DT_STRING, {0}, CreateTestData<string_view>({})).value();
  Tensor t = Tensor::Create(DT_INT32, {0}, CreateTestData<int32_t>({})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&size_keys, &animal_keys, &t}),
              IsOk());
  EXPECT_THAT(group_by_aggregator->Accumulate({&size_keys, &animal_keys, &t}),
              IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(2));

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensors, which should be empty.
  // Only the second key tensor should be included in the output.
  EXPECT_THAT(result.value().size(), Eq(2));
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({0}, {}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({0}, {}));
}

TEST_P(GroupByAggregatorTest, Accumulate_NoKeyTensors) {
  Intrinsic intrinsic{"fedsql_group_by", {}, {}, {}, {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();

  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&t1}), IsOk());
  Tensor t2 = Tensor::Create(DT_INT32, {3}, CreateTestData({10, 5, 1})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&t2}), IsOk());

  if (GetParam()) {
    auto serialized_state = std::move(*group_by_aggregator).Serialize();
    group_by_aggregator =
        DeserializeTensorAggregator(intrinsic, serialized_state.value())
            .value();
  }

  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 20, 5})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&t3}), IsOk());

  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {108}));
}

TEST_P(GroupByAggregatorTest, Merge_Succeeds) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();

  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}))
          .value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  Tensor t4 = Tensor::Create(DT_INT32, {}, CreateTestData({4})).value();
  Tensor t5 = Tensor::Create(DT_INT32, {}, CreateTestData({5})).value();
  EXPECT_THAT(aggregator1->Accumulate({&key, &t1}), IsOk());
  EXPECT_THAT(aggregator1->Accumulate({&key, &t2}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&key, &t3}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&key, &t4}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&key, &t5}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(5));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({1}, {"foo"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {15}));
}

TEST_P(GroupByAggregatorTest, Merge_MultipleValueTensors_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  Tensor tA1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor tB1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({14, 11, 7, 14})).value();
  EXPECT_THAT(aggregator1->Accumulate({&keys1, &tA1, &tB1}), IsOk());
  // Totals: [4, 15, 27], [25, 7, 14]
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  Tensor tA2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor tB2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 2, 8})).value();
  EXPECT_THAT(aggregator1->Accumulate({&keys2, &tA2, &tB2}), IsOk());
  // aggregator1 totals: [9, 26, 27, 2], [28, 10, 14, 8]
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(2));

  // Create a second aggregator and accumulate an input with overlapping keys.
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();
  Tensor keys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"three", "two", "three", "two"}))
          .value();
  Tensor tA3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({11, 3, 4, 2})).value();
  Tensor tB3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({6, 1, 4, 12})).value();
  EXPECT_THAT(aggregator2->Accumulate({&keys3, &tA3, &tB3}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {9, 26, 32, 17}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({4}, {28, 10, 27, 18}));
}

TEST_P(GroupByAggregatorTest, Merge_NoValueTensors_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {},
                      {}};
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  EXPECT_THAT(aggregator1->Accumulate({&keys1}), IsOk());
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  EXPECT_THAT(aggregator1->Accumulate({&keys2}), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(2));

  // Create a second aggregator and accumulate an input with overlapping keys.
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();
  Tensor keys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"three", "two", "three", "two"}))
          .value();
  EXPECT_THAT(aggregator2->Accumulate({&keys3}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
}

TEST_P(GroupByAggregatorTest, Merge_MultipleKeyTensors_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key1", DT_STRING),
                       CreateTensorSpec("key2", DT_STRING)},
                      {CreateTensorSpec("key1_out", DT_STRING),
                       CreateTensorSpec("key2_out", DT_STRING)},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();
  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1->Accumulate({&sizeKeys1, &animalKeys1, &t1}), IsOk());
  // aggregator1 totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1->Accumulate({&sizeKeys2, &animalKeys2, &t2}), IsOk());
  // aggregator1 totals: [9, 26, 27, 2]

  // Create a second GroupByAggregator.
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();
  Tensor sizeKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "small"}))
          .value();
  Tensor animalKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"dog", "dog", "rabbit", "cat"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator2->Accumulate({&sizeKeys3, &animalKeys3, &t3}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  // Merged totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>(
                  {5}, {"large", "small", "large", "small", "small"}));
  EXPECT_THAT(
      result.value()[1],
      IsTensor<string_view>({5}, {"cat", "cat", "dog", "dog", "rabbit"}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({5}, {9, 46, 41, 2, 7}));
}

TEST_P(GroupByAggregatorTest,
       Merge_MultipleKeyTensors_SomeKeysNotInOutput_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic intrinsic{
      "fedsql_group_by",
      {CreateTensorSpec("key1", DT_STRING),
       CreateTensorSpec("key2", DT_STRING)},
      {CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)},
      {},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1->Accumulate({&sizeKeys1, &animalKeys1, &t1}), IsOk());
  // aggregator1 totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1->Accumulate({&sizeKeys2, &animalKeys2, &t2}), IsOk());
  // aggregator1 totals: [9, 26, 27, 2]

  // Create a second GroupByAggregator.
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();
  Tensor sizeKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "small"}))
          .value();
  Tensor animalKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"dog", "dog", "rabbit", "cat"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator2->Accumulate({&sizeKeys3, &animalKeys3, &t3}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  // Merged totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensors.
  // Only the second key tensor should be included in the output.
  EXPECT_THAT(result.value().size(), Eq(2));
  EXPECT_THAT(
      result.value()[0],
      IsTensor<string_view>({5}, {"cat", "cat", "dog", "dog", "rabbit"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({5}, {9, 46, 41, 2, 7}));
}

TEST_P(GroupByAggregatorTest, Merge_NoKeyTensors) {
  Intrinsic intrinsic{"fedsql_group_by", {}, {}, {}, {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1->Accumulate({&t1}), IsOk());
  Tensor t2 = Tensor::Create(DT_INT32, {3}, CreateTestData({10, 5, 1})).value();
  EXPECT_THAT(aggregator1->Accumulate({&t2}), IsOk());

  auto aggregator2 = CreateTensorAggregator(intrinsic).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 20, 5})).value();
  EXPECT_THAT(aggregator2->Accumulate({&t3}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {108}));
}

TEST_P(GroupByAggregatorTest, MergeThisOutputReceivedNoInputsSucceeds) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(aggregator2->Accumulate({&keys1, &t1}), IsOk());
  // aggregator2 totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(aggregator2->Accumulate({&keys2, &t2}), IsOk());
  // aggregator2 totals: [2, 10, 19, 4]

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge aggregator2 into aggregator1 which has not received any inputs.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(2));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {2, 10, 19, 4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(GroupByAggregatorTest, MergeThisOutputReceivedOnlyEmptyInputsSucceeds) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(aggregator2->Accumulate({&keys1, &t1}), IsOk());
  // aggregator2 totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(aggregator2->Accumulate({&keys2, &t2}), IsOk());
  // aggregator2 totals: [2, 10, 19, 4]

  // aggregator1 receives only an empty input.
  Tensor empty_keys =
      Tensor::Create(DT_STRING, {0}, CreateTestData<string_view>({})).value();
  Tensor empty_tensor =
      Tensor::Create(DT_INT32, {0}, CreateTestData<int32_t>({})).value();
  EXPECT_THAT(aggregator1->Accumulate({&empty_keys, &empty_tensor}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge aggregator2 into aggregator1 which has not received any inputs.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {2, 10, 19, 4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(GroupByAggregatorTest, MergeOtherAggregatorReceivedNoInputsSucceeds) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(aggregator1->Accumulate({&keys1, &t1}), IsOk());
  // aggregator1 totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(aggregator1->Accumulate({&keys2, &t2}), IsOk());
  // aggregator1 totals: [2, 10, 19, 4]

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge with aggregator2 which has not received any inputs.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(2));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {2, 10, 19, 4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(GroupByAggregatorTest,
       MergeOtherAggregatorReceivedOnlyEmptyInputsSucceeds) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();
  auto aggregator2 = CreateTensorAggregator(intrinsic).value();

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(aggregator1->Accumulate({&keys1, &t1}), IsOk());
  // aggregator1 totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(aggregator1->Accumulate({&keys2, &t2}), IsOk());
  // aggregator1 totals: [2, 10, 19, 4]

  // aggregator2 receives only an empty input.
  Tensor empty_keys =
      Tensor::Create(DT_STRING, {0}, CreateTestData<string_view>({})).value();
  Tensor empty_tensor =
      Tensor::Create(DT_INT32, {0}, CreateTestData<int32_t>({})).value();
  EXPECT_THAT(aggregator2->Accumulate({&empty_keys, &empty_tensor}), IsOk());

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize();
    aggregator1 =
        DeserializeTensorAggregator(intrinsic, serialized_state1.value())
            .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize();
    aggregator2 =
        DeserializeTensorAggregator(intrinsic, serialized_state2.value())
            .value();
  }

  // Merge with aggregator2 which has only received empty inputs.
  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {2, 10, 19, 4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

// TODO: b/277982238 - Expand on the tests below to check that even when
// Accumulate or MergeWith return INVALID_ARGUMENT, the internal state of the
// GroupByAggregator remains unaffected, exactly the same as if the failed
// operation had never been called.
TEST(GroupByAggregatorTest, AccumulateKeyTensorHasIncompatibleDataType) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key =
      Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1.2})).value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({0})).value();
  Status s = group_by_aggregator->Accumulate({&key, &t});
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      ::testing::HasSubstr("Tensor at position 0 did not have expected dtype"));
}

TEST(GroupByAggregatorTest, AccumulateValueTensorHasIncompatibleDataType) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"key_string"}))
          .value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1.2})).value();
  Status s = group_by_aggregator->Accumulate({&key, &t});
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      ::testing::HasSubstr("Tensor at position 1 did not have expected dtype"));
}

TEST(GroupByAggregatorTest, Accumulate_FewerTensorsThanExpected) {
  Intrinsic intrinsic = {"fedsql_group_by",
                         {CreateTensorSpec("key1", DT_STRING),
                          CreateTensorSpec("key2", DT_STRING)},
                         {CreateTensorSpec("key1_out", DT_STRING),
                          CreateTensorSpec("key2_out", DT_STRING)},
                         {},
                         {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"key_string"}))
          .value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Status s = group_by_aggregator->Accumulate({&key, &t});
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), ::testing::HasSubstr(
                               "GroupByAggregator::AggregateTensorsInternal "
                               "should operate on 3 input tensors"));
}

TEST(GroupByAggregatorTest, Accumulate_MoreTensorsThanExpected) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key1 = Tensor::Create(DT_STRING, {},
                               CreateTestData<string_view>({"key_string_1"}))
                    .value();
  Tensor key2 = Tensor::Create(DT_STRING, {},
                               CreateTestData<string_view>({"key_string_2"}))
                    .value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Status s = group_by_aggregator->Accumulate({&key1, &key2, &t});
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), ::testing::HasSubstr(
                               "GroupByAggregator::AggregateTensorsInternal "
                               "should operate on 2 input tensors"));
}

TEST(GroupByAggregatorTest, Accumulate_KeyTensorSmallerThanValueTensor) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key = Tensor::Create(DT_STRING, {},
                              CreateTestData<string_view>({"key_string_1"}))
                   .value();
  Tensor t = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 2})).value();
  Status s = group_by_aggregator->Accumulate({&key, &t});
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Shape of value tensor at index 1 does not "
                                   "match the shape of the first key tensor."));
}

TEST(GroupByAggregatorTest, Accumulate_KeyTensorLargerThanValueTensor) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key =
      Tensor::Create(DT_STRING, {3},
                     CreateTestData<string_view>(
                         {"key_string_1", "key_string_2", "key_string_3"}))
          .value();
  Tensor t = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 2})).value();
  Status s = group_by_aggregator->Accumulate({&key, &t});
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Shape of value tensor at index 1 does not "
                                   "match the shape of the first key tensor."));
}

TEST(GroupByAggregatorTest, Accumulate_MultidimensionalTensorsNotSupported) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor key = Tensor::Create(DT_STRING, {2, 2},
                              CreateTestData<string_view>({"a", "b", "c", "d"}))
                   .value();
  Tensor t =
      Tensor::Create(DT_INT32, {2, 2}, CreateTestData({1, 2, 3, 4})).value();
  Status s = group_by_aggregator->Accumulate({&key, &t});
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr(
                  "Only scalar or one-dimensional tensors are supported."));
}

TEST(GroupByAggregatorTest, Merge_IncompatibleKeyType) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Intrinsic intrinsic2 = {"fedsql_group_by",
                          {CreateTensorSpec("key", DT_FLOAT)},
                          {CreateTensorSpec("key_out", DT_FLOAT)},
                          {},
                          {}};
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator2 = CreateTensorAggregator(intrinsic2).value();
  Status s = aggregator1->MergeWith(std::move(*aggregator2));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Expected other GroupByAggregator to have "
                                   "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_IncompatibleOutputKeySpec) {
  Intrinsic intrinsic = {"fedsql_group_by",
                         {CreateTensorSpec("key1", DT_STRING),
                          CreateTensorSpec("key2", DT_STRING)},
                         {CreateTensorSpec("", DT_STRING),
                          CreateTensorSpec("key2_out", DT_STRING)},
                         {},
                         {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  // Key1 is included in the output of aggregator2 but not included in the
  // output of aggregator1.
  Intrinsic intrinsic2 = {"fedsql_group_by",
                          {CreateTensorSpec("key1", DT_STRING),
                           CreateTensorSpec("key2", DT_STRING)},
                          {CreateTensorSpec("key1_out", DT_STRING),
                           CreateTensorSpec("key2_out", DT_STRING)},
                          {},
                          {}};
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator2 = CreateTensorAggregator(intrinsic2).value();
  Status s = aggregator1->MergeWith(std::move(*aggregator2));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Expected other GroupByAggregator to have "
                                   "the same key input and output specs"));
}

TEST(GroupByAggregatorTest,
     Merge_IncompatibleKeyType_InputTensorListTypesMatch) {
  Intrinsic intrinsic = {
      "fedsql_group_by",
      {CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_INT32)},
      {CreateTensorSpec("key1_out", DT_STRING),
       CreateTensorSpec("key2_out", DT_INT32)},
      {},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Intrinsic intrinsic2 = {"fedsql_group_by",
                          {CreateTensorSpec("key1", DT_STRING)},
                          {CreateTensorSpec("key1_out", DT_STRING)},
                          {},
                          {}};
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator2 = CreateTensorAggregator(intrinsic2).value();
  Status s = aggregator1->MergeWith(std::move(*aggregator2));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Expected other GroupByAggregator to have "
                                   "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_IncompatibleValueType) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Intrinsic intrinsic2 = {"fedsql_group_by",
                          {CreateTensorSpec("key", DT_STRING)},
                          {CreateTensorSpec("key_out", DT_STRING)},
                          {},
                          {}};
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_FLOAT, DT_DOUBLE));
  auto aggregator2 = CreateTensorAggregator(intrinsic2).value();
  Status s = aggregator1->MergeWith(std::move(*aggregator2));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), ::testing::HasSubstr(
                               "Expected other GroupByAggregator to "
                               "use inner intrinsics with the same inputs"));
}

TEST(GroupByAggregatorTest, Merge_DifferentNumKeys) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Intrinsic intrinsic2 = {
      "fedsql_group_by",
      {CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_INT32)},
      {CreateTensorSpec("key2", DT_STRING),
       CreateTensorSpec("key2_out", DT_INT32)},
      {},
      {}};
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator2 = CreateTensorAggregator(intrinsic2).value();
  Status s = aggregator1->MergeWith(std::move(*aggregator2));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Expected other GroupByAggregator to have "
                                   "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_NonzeroVsZeroNumKeys) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Intrinsic intrinsic2 = {"fedsql_group_by", {}, {}, {}, {}};
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator2 = CreateTensorAggregator(intrinsic2).value();
  Status s = aggregator1->MergeWith(std::move(*aggregator2));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Expected other GroupByAggregator to have "
                                   "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_DifferentNumValues) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Intrinsic intrinsic2 = {"fedsql_group_by",
                          {CreateTensorSpec("key", DT_STRING)},
                          {CreateTensorSpec("key_out", DT_STRING)},
                          {},
                          {}};
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  intrinsic2.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator2 = CreateTensorAggregator(intrinsic2).value();
  Status s = aggregator1->MergeWith(std::move(*aggregator2));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Expected other GroupByAggregator to "
                                   "use the same number of inner intrinsics"));
}

TEST(GroupByAggregatorTest, Merge_DifferentTensorAggregatorImpl) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  SumAggregator<int32_t> sum_aggregator(DT_INT32, TensorShape{});
  Status s = aggregator1->MergeWith(std::move(sum_aggregator));
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      ::testing::HasSubstr("Can only merge with another GroupByAggregator"));
}

TEST(GroupByAggregatorTest, FailsAfterBeingConsumed) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}))
          .value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();

  EXPECT_THAT(aggregator1->Accumulate({&key, &t}), IsOk());
  EXPECT_THAT(std::move(*aggregator1).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator1->CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(*aggregator1).Report(),
              StatusIs(FAILED_PRECONDITION));       // NOLINT
  EXPECT_THAT(aggregator1->Accumulate({&key, &t}),  // NOLINT
              StatusIs(FAILED_PRECONDITION));

  auto aggregator2 = CreateTensorAggregator(intrinsic).value();

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)),
              // NOLINT
              StatusIs(FAILED_PRECONDITION));

  // Passing this aggregator as an argument to another MergeWith must fail
  // too.
  auto aggregator3 = CreateTensorAggregator(intrinsic).value();
  EXPECT_THAT(aggregator3->MergeWith(std::move(*aggregator1)),  // NOLINT
              StatusIs(FAILED_PRECONDITION));
}

TEST(GroupByAggregatorTest, FailsAfterBeingConsumed_WhenNoKeys) {
  Intrinsic intrinsic{"fedsql_group_by", {}, {}, {}, {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  auto aggregator1 = CreateTensorAggregator(intrinsic).value();

  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  EXPECT_THAT(aggregator1->Accumulate({&t}), IsOk());
  EXPECT_THAT(std::move(*aggregator1).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator1->CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(*aggregator1).Report(),
              StatusIs(FAILED_PRECONDITION));  // NOLINT
  EXPECT_THAT(aggregator1->Accumulate({&t}),   // NOLINT
              StatusIs(FAILED_PRECONDITION));

  auto aggregator2 = CreateTensorAggregator(intrinsic).value();

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)),
              // NOLINT
              StatusIs(FAILED_PRECONDITION));

  // Passing this aggregator as an argument to another MergeWith must fail
  // too.
  auto aggregator3 = CreateTensorAggregator(intrinsic).value();
  EXPECT_THAT(aggregator3->MergeWith(std::move(*aggregator1)),  // NOLINT
              StatusIs(FAILED_PRECONDITION));
}

TEST(GroupByAggregatorTest, AddOneContributor_Success) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(10);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor ordinals =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({2, 0, 3})).value();

  EXPECT_THAT(peer.AddOneContributor(std::move(ordinals)), IsOk());

  EXPECT_THAT(peer.GetContributors(),
              testing::ContainerEq(std::vector<int>{1, 0, 1, 1}));
}

TEST(GroupByAggregatorTest, AddOneContributor_StopsAtMaxValue) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(2);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor ordinals_1 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({0, 1})).value();
  Tensor ordinals_2 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({0})).value();

  EXPECT_THAT(peer.AddOneContributor(std::move(ordinals_1)), IsOk());
  EXPECT_THAT(peer.AddOneContributor(std::move(ordinals_2)), IsOk());
  EXPECT_THAT(peer.GetContributors(),
              testing::ContainerEq(std::vector<int>{2, 1}));

  // Add one more contributor; the count for group 0 should not increase.
  Tensor ordinals_3 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({0})).value();
  EXPECT_THAT(peer.AddOneContributor(std::move(ordinals_3)), IsOk());

  // The count for group 0 should not increase further.
  EXPECT_THAT(peer.GetContributors(),
              testing::ContainerEq(std::vector<int>{2, 1}));
}

TEST(GroupByAggregatorTest, AddOneContributor_HandlesEmptyTensor) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(5);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor empty_ordinals =
      Tensor::Create(DT_INT64, {0}, CreateTestData<int64_t>({})).value();
  EXPECT_THAT(peer.AddOneContributor(std::move(empty_ordinals)), IsOk());
  EXPECT_TRUE(peer.GetContributors().empty());
}

TEST(GroupByAggregatorTest, AddOneContributor_FailsWithInvalidDtype) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(5);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor ordinals =
      Tensor::Create(DT_INT32, {2}, CreateTestData({0, 1})).value();

  EXPECT_THAT(peer.AddOneContributor(std::move(ordinals)),
              StatusIs(INVALID_ARGUMENT, HasSubstr("Expected int64 ordinals")));
}

TEST(GroupByAggregatorTest, AddOneContributor_FailsWhenMaxContributorsNotSet) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor ordinals =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({0, 1, 2})).value();

  EXPECT_THAT(
      peer.AddOneContributor(std::move(ordinals)),
      StatusIs(
          INVALID_ARGUMENT,
          HasSubstr("max_contributors_to_group_ to be set but it is not")));
}

TEST(GroupByAggregatorTest, AddMultipleContributors_Success) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(10);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({0, 2})).value();
  std::vector<int> num_contributors = std::vector<int>{2, 5};

  EXPECT_THAT(
      peer.AddMultipleContributors(std::move(ordinals), num_contributors),
      IsOk());
  EXPECT_THAT(peer.GetContributors(),
              testing::ContainerEq(std::vector<int>{2, 0, 5}));
}

TEST(GroupByAggregatorTest, AddMultipleContributors_SuccessWithMultipleCalls) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(20);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  // First, add 3 contributors to group 0 and 5 contributors to group 2.
  Tensor ordinals1 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({0, 2})).value();
  std::vector<int> num_contributors1 = {3, 5};
  EXPECT_THAT(
      peer.AddMultipleContributors(std::move(ordinals1), num_contributors1),
      IsOk());

  // Verify the intermediate state.
  EXPECT_THAT(peer.GetContributors(),
              testing::ContainerEq(std::vector<int>{3, 0, 5}));

  // Second, add 2 contributors to group 0, 4 to group 1, and 1 to group 2.
  Tensor ordinals2 =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({0, 1, 2})).value();
  std::vector<int> num_contributors2 = {2, 4, 1};
  EXPECT_THAT(
      peer.AddMultipleContributors(std::move(ordinals2), num_contributors2),
      IsOk());

  // Verify the final accumulated state.
  // Group 0 should have 3 + 2 = 5 contributors.
  // Group 1 should have 0 + 4 = 4 contributors.
  // Group 2 should have 5 + 1 = 6 contributors.
  EXPECT_THAT(peer.GetContributors(),
              testing::ContainerEq(std::vector<int>{5, 4, 6}));
}

TEST(GroupByAggregatorTest, AddMultipleContributors_ClampsToMaxContributors) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(5);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  // First, add 3 contributors to group 0.
  Tensor ordinals1 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({0})).value();
  std::vector<int> num_contributors1 = {3};
  EXPECT_THAT(
      peer.AddMultipleContributors(std::move(ordinals1), num_contributors1),
      IsOk());

  // Second, add 4 contributors to each of group 0 and 1.
  Tensor ordinals2 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({0, 1})).value();
  std::vector<int> num_contributors2 = {4, 4};
  EXPECT_THAT(
      peer.AddMultipleContributors(std::move(ordinals2), num_contributors2),
      IsOk());

  // Verify that the number of contributors for group 0 is clamped to 5 and
  // the number of contributors for group 1 is set correctly.
  EXPECT_THAT(peer.GetContributors(),
              testing::ContainerEq(std::vector<int>{5, 4}));
}

TEST(GroupByAggregatorTest,
     AddMultipleContributors_FailsWhenMaxContributorsNotSet) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor ordinals =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({0})).value();
  std::vector<int> num_contributors = {1};

  EXPECT_THAT(
      peer.AddMultipleContributors(std::move(ordinals), num_contributors),
      StatusIs(
          INVALID_ARGUMENT,
          HasSubstr("max_contributors_to_group_ to be set but it is not")));
}

TEST(GroupByAggregatorTest, AddMultipleContributors_FailsOnSizeMismatch) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(10);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  // Ordinals tensor has 2 elements, but num_contributors vector has 3.
  Tensor ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({0, 1})).value();
  std::vector<int> num_contributors = {1, 1, 1};

  EXPECT_THAT(
      peer.AddMultipleContributors(std::move(ordinals), num_contributors),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr("same number of ordinals and contributor counts")));
}

TEST(GroupByAggregatorTest, AddMultipleContributors_HandlesEmptyInputs) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors(5);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  GroupByAggregatorPeer peer(
      dynamic_cast<GroupByAggregator*>(aggregator.get()));

  Tensor empty_ordinals =
      Tensor::Create(DT_INT64, {0}, CreateTestData<int64_t>({})).value();
  std::vector<int> empty_num_contributors = {};

  EXPECT_THAT(peer.AddMultipleContributors(std::move(empty_ordinals),
                                           empty_num_contributors),
              IsOk());
  EXPECT_TRUE(peer.GetContributors().empty());
}

TEST(GroupByFactoryTest, WrongUri) {
  Intrinsic intrinsic{"wrong_uri",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  Status s =
      (*GetAggregatorFactory("fedsql_group_by"))->Create(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected intrinsic URI fedsql_group_by"));
}

TEST(GroupByFactoryTest, NoInputTensors) {
  Intrinsic intrinsic{"fedsql_group_by", {}, {}, {}, {}};
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Must operate on a nonzero number of input tensors."));
}

TEST(GroupByFactoryTest, InputAndOutputKeySizeMismatch) {
  Intrinsic intrinsic{
      "fedsql_group_by",
      {CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_FLOAT)},
      {CreateTensorSpec("animals", DT_STRING)},
      {},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Exactly the same number of input args and "
                        "output tensors are expected"));
}

TEST(GroupByFactoryTest, InputAndOutputDtypeMismatch) {
  Intrinsic intrinsic{
      "fedsql_group_by",
      {CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_FLOAT)},
      {CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)},
      {},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Input and output tensors have mismatched specs"));
}

TEST(GroupByFactoryTest, InputAndOutputShapeInvalid) {
  Intrinsic intrinsic{"fedsql_group_by",
                      {TensorSpec("key", DT_STRING, {8})},
                      {TensorSpec("key", DT_STRING, {8})},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateDefaultInnerIntrinsic(DT_INT32, DT_INT64));

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("All input and output tensors must have one "
                        "dimension of unknown size."));
}

TEST(GroupByFactoryTest, SubIntrinsicNotGroupingAggregator) {
  Intrinsic intrinsic{"fedsql_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      Intrinsic{"federated_sum",
                {CreateTensorSpec("value", DT_INT32)},
                {CreateTensorSpec("value", DT_INT64)},
                {},
                {}});

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Nested intrinsic URIs must start with 'GoogleSQL:'"));
}

TEST(GroupByFactoryDeathTest, MinContributorsToGroupTensorNotScalar) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  intrinsic.parameters.push_back(
      Tensor::Create(DT_INT32, {3}, CreateTestData({0, 0, 0})).value());
  EXPECT_DEATH(CreateTensorAggregator(intrinsic).IgnoreError(),
               "used on scalar tensors");
}

TEST(GroupByFactoryTest, TooManyParameters) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  intrinsic.parameters.push_back(
      Tensor::Create(DT_INT32, {}, CreateTestData({5})).value());
  intrinsic.parameters.push_back(
      Tensor::Create(DT_INT32, {}, CreateTestData({5})).value());
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("At most one input parameter expected"));
}

TEST(GroupByFactoryTest, MinContributorsToGroupNegative) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  intrinsic.parameters.push_back(
      Tensor::Create(DT_INT32, {}, CreateTestData({-5})).value());
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("must be positive"));
}

TEST(GroupByFactoryTest, MinContributorsToGroupZero) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  intrinsic.parameters.push_back(
      Tensor::Create(DT_INT32, {}, CreateTestData({0})).value());
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("must be positive"));
}

TEST(GroupByFactoryTest, MinContributorsToGroupSetDoesNotCauseError) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  intrinsic.parameters.push_back(
      Tensor::Create(DT_INT32, {}, CreateTestData({5})).value());
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(OK));
}

TEST(GroupByAggregatorTest, Deserialize_FailToParseProto) {
  Intrinsic intrinsic = CreateDefaultIntrinsic();
  std::string invalid_state("invalid_state");
  Status s = DeserializeTensorAggregator(intrinsic, invalid_state).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Failed to parse"));
}

INSTANTIATE_TEST_SUITE_P(
    GroupByAggregatorTestInstantiation, GroupByAggregatorTest,
    testing::ValuesIn<bool>({false, true}),
    [](const testing::TestParamInfo<GroupByAggregatorTest::ParamType>& info) {
      return info.param ? "SerializeDeserialize" : "None";
    });

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
