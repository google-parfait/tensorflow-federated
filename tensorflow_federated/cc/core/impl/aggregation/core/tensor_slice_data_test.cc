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
#include <memory>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::HasSubstr;

TEST(TensorSliceDataTest, ConstructTensorSliceData) {
  auto tensor_slice_data = std::make_unique<TensorSliceData>(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  EXPECT_EQ(tensor_slice_data->byte_size(), 12);
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor tensor,
      Tensor::Create(DT_INT32, {3}, std::move(tensor_slice_data)));
  EXPECT_THAT(tensor, IsTensor<int32_t>({3}, {1, 2, 3}));
}

TEST(TensorSliceDataTest, CanShrinkTensorSliceData) {
  auto tensor_slice_data = std::make_unique<TensorSliceData>(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  TFF_EXPECT_OK(tensor_slice_data->ReduceByteSize(8));
  EXPECT_EQ(tensor_slice_data->byte_size(), 8);
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor tensor,
      Tensor::Create(DT_INT32, {2}, std::move(tensor_slice_data)));
  EXPECT_THAT(tensor, IsTensor<int32_t>({2}, {1, 2}));
}

TEST(TensorSliceDataTest, CannotGrowTensorSliceData) {
  TensorSliceData tensor_slice_data(
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
          .value());
  EXPECT_THAT(
      tensor_slice_data.ReduceByteSize(16),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("target size 16 is greater than the original size 12")));
}

// Create a prefix (default offset 0)
TEST(TensorSliceDataTest, CreatePrefix) {
  TFF_ASSERT_OK_AND_ASSIGN(
      TensorSliceData tensor_slice_data,
      TensorSliceData::Create(
          Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
              .value(),
          /*byte_size=*/8));

  TFF_ASSERT_OK_AND_ASSIGN(Tensor tensor,
                           Tensor::Create(DT_INT32, {2},
                                          std::make_unique<TensorSliceData>(
                                              std::move(tensor_slice_data))));
  EXPECT_THAT(tensor, IsTensor<int32_t>({2}, {1, 2}));
}
// Create a suffix (custom offset)
TEST(TensorSliceDataTest, CreateSuffix) {
  TFF_ASSERT_OK_AND_ASSIGN(
      TensorSliceData tensor_slice_data,
      TensorSliceData::Create(
          Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
              .value(),
          /*byte_size=*/8, /*byte_offset=*/4));

  TFF_ASSERT_OK_AND_ASSIGN(Tensor tensor,
                           Tensor::Create(DT_INT32, {2},
                                          std::make_unique<TensorSliceData>(
                                              std::move(tensor_slice_data))));
  EXPECT_THAT(tensor, IsTensor<int32_t>({2}, {2, 3}));
}

// Cannot create when byte_offset + byte_size > tensor's byte size
TEST(TensorSliceDataTest, CannotCreateInvalidSlice) {
  EXPECT_THAT(
      TensorSliceData::Create(
          Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
              .value(),
          /*byte_size=*/16),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "byte_offset + byte_size cannot exceed the tensor's size")));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
