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

#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_spec.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(DomainSpecTest, IsMember_NumericalTypes) {
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor t1,
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})));
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor t2,
      Tensor::Create(DT_FLOAT, {2}, CreateTestData<float>({1.1, 2.2})));

  std::vector<Tensor> tensors;
  tensors.push_back(std::move(t1));
  tensors.push_back(std::move(t2));
  DomainSpec domain_spec(TensorSpan{tensors});

  // Column 0 (INT32)
  TFF_ASSERT_OK_AND_ASSIGN(bool result0_1, domain_spec.IsMember<int32_t>(1, 0));
  EXPECT_TRUE(result0_1);

  TFF_ASSERT_OK_AND_ASSIGN(bool result0_4, domain_spec.IsMember<int32_t>(4, 0));
  EXPECT_FALSE(result0_4);

  // Column 1 (FLOAT)
  TFF_ASSERT_OK_AND_ASSIGN(bool result1_1, domain_spec.IsMember<float>(1.1, 1));
  EXPECT_TRUE(result1_1);

  TFF_ASSERT_OK_AND_ASSIGN(bool result1_3, domain_spec.IsMember<float>(3.3, 1));
  EXPECT_FALSE(result1_3);
}

TEST(DomainSpecTest, IsMember_StringType) {
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor t1, Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"abc", "def"})));

  std::vector<Tensor> tensors_str;
  tensors_str.push_back(std::move(t1));
  DomainSpec domain_spec(TensorSpan{tensors_str});

  TFF_ASSERT_OK_AND_ASSIGN(bool result_abc,
                           domain_spec.IsMember<string_view>("abc", 0));
  EXPECT_TRUE(result_abc);

  TFF_ASSERT_OK_AND_ASSIGN(bool result_ghi,
                           domain_spec.IsMember<string_view>("ghi", 0));
  EXPECT_FALSE(result_ghi);
}

TEST(DomainSpecTest, IsMember_WrongType_ReturnsInvalidArgument) {
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor t1,
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})));
  std::vector<Tensor> tensors_err;
  tensors_err.push_back(std::move(t1));
  DomainSpec domain_spec(TensorSpan{tensors_err});

  auto result = domain_spec.IsMember<float>(1.1, 0);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(DomainSpecTest, IsMember_OutOfBoundsIndex_ReturnsInvalidArgument) {
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor t1,
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})));
  std::vector<Tensor> tensors_oob;
  tensors_oob.push_back(std::move(t1));
  DomainSpec domain_spec(TensorSpan{tensors_oob});

  EXPECT_THAT(domain_spec.IsMember<int32_t>(1, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(domain_spec.IsMember<int32_t>(1, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
