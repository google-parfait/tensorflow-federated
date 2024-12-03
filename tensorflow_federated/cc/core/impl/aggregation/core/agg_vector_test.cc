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

#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"

#include <cstddef>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_iterator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;

template <typename T>
using Pair = typename AggVectorIterator<T>::IndexValuePair;

TEST(AggVectorTest, Size) {
  auto t1 = Tensor::Create(DT_INT32, {}, CreateTestData({0}));
  EXPECT_EQ(t1->AsAggVector<int>().size(), 1);

  auto t2 = Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({0, 1, 2}));
  EXPECT_EQ(t2->AsAggVector<float>().size(), 3);
}

TEST(AggVectorTest, PostIncrementIterator_ScalarTensor) {
  auto t = Tensor::Create(DT_INT32, {}, CreateTestData({5}));
  EXPECT_THAT(t->AsAggVector<int>(), ElementsAre(Pair<int>{0, 5}));
}

TEST(AggVectorTest, PostIncrementIterator_DenseTensor) {
  auto t = Tensor::Create(DT_INT32, {2}, CreateTestData({3, 14}));
  EXPECT_THAT(t->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 3}, Pair<int>{1, 14}));
}

TEST(AggVectorTest, PostIncrementIterator_ForLoopIterator) {
  auto t = Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({2, 3, 4, 5}));
  float sum = 0;
  size_t expected_index = 0;
  for (auto [index, value] : t->AsAggVector<float>()) {
    EXPECT_THAT(index, Eq(expected_index++));
    sum += value;
  }
  EXPECT_THAT(sum, Eq(14));
}

TEST(AggVectorTest, PreIncrementIterator) {
  auto t = Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({2, 3, 4, 5}));
  auto agg_vector = t->AsAggVector<float>();
  float sum = 0;
  size_t expected_index = 0;
  for (auto it = agg_vector.begin(); it != agg_vector.end(); it++) {
    EXPECT_THAT(it.index(), Eq(expected_index++));
    sum += it.value();
  }
  EXPECT_THAT(sum, Eq(14));
}

TEST(AggVectorTest, Empty_DenseTensor_WithNullTensorDataBuffer) {
  auto data = CreateTestData<int>({});
  EXPECT_THAT(data->data(), Eq(nullptr));
  auto t = Tensor::Create(DT_INT32, {0}, std::move(data));
  auto agg_vector = t->AsAggVector<int>();
  // Iterating over the agg vector should return zero elements.
  EXPECT_THAT(agg_vector.begin(), Eq(agg_vector.end()));
}

TEST(AggVectorTest, Empty_DenseTensor_WithReservedTensorDataBuffer) {
  auto data = CreateTestData<int>({});
  // Ensures that the test data buffer is not simply null (empty vectors
  // generally don't allocate a buffer at all). The triggers the condition where
  // the AggVectorIterator start pointer is non-null and non-equal to end(),
  // which should still be handled correctly.
  data->reserve(100);
  EXPECT_THAT(data->data(), testing::NotNull());
  auto t = Tensor::Create(DT_INT32, {0}, std::move(data));
  auto agg_vector = t->AsAggVector<int>();
  // Iterating over the agg vector should return zero elements.
  EXPECT_THAT(agg_vector.begin(), Eq(agg_vector.end()));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
