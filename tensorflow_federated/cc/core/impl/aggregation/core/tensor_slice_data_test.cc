/*
 * Copyright 2025 Google LLC
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
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_slice_data.h"

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(TensorSliceDataTest, ConstructTensorSliceData) {
  TensorSliceData tensor_slice_data(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  EXPECT_EQ(tensor_slice_data.byte_size(), 12);
  absl::Span<const int32_t> span(
      reinterpret_cast<const int32_t*>(tensor_slice_data.data()), 3);
  EXPECT_THAT(span, ElementsAre(1, 2, 3));
}

TEST(TensorSliceDataTest, CanShrinkTensorSliceData) {
  TensorSliceData tensor_slice_data(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  TFF_EXPECT_OK(tensor_slice_data.ReduceSize(8));
  EXPECT_EQ(tensor_slice_data.byte_size(), 8);
  absl::Span<const int32_t> span(
      reinterpret_cast<const int32_t*>(tensor_slice_data.data()), 2);
  EXPECT_THAT(span, ElementsAre(1, 2));
}

TEST(TensorSliceDataTest, CannotGrowTensorSliceData) {
  TensorSliceData tensor_slice_data(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  EXPECT_THAT(
      tensor_slice_data.ReduceSize(16),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("target size 16 is greater than the original size 12")));
}

// Successfully swap two values.
TEST(TensorSliceDataTest, CanSwapValues) {
  TensorSliceData tensor_slice_data(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  TFF_EXPECT_OK(tensor_slice_data.SwapValuesAtIndices<int32_t>(0, 2));
  EXPECT_EQ(tensor_slice_data.byte_size(), 12);
  absl::Span<const int32_t> span(
      reinterpret_cast<const int32_t*>(tensor_slice_data.data()), 3);
  EXPECT_THAT(span, ElementsAre(3, 2, 1));
}

TEST(TensorSliceDataTest, CannotSwapValues_WrongType) {
  TensorSliceData tensor_slice_data(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  EXPECT_THAT(
      tensor_slice_data.SwapValuesAtIndices<int64_t>(0, 2),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("byte_size() must be a multiple of value_size")));
}

TEST(TensorSliceDataTest, CannotSwapValues_InvalidIndices) {
  TensorSliceData tensor_slice_data(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  EXPECT_THAT(tensor_slice_data.SwapValuesAtIndices<int32_t>(0, 3),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("indices must be in range")));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
