// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/in_memory_checkpoint_parser.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"

namespace tensorflow_federated::aggregation {
namespace {

using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(InMemoryCheckpointParserTest, GetTensors) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({1l, 2l, 3l}, "t1"));
  tensors.push_back(Tensor({"value1", "value2"}, "t2"));

  InMemoryCheckpointParser parser(std::move(tensors));

  auto tensor1 = parser.GetTensor("t1");
  ASSERT_OK(tensor1.status());
  auto tensor2 = parser.GetTensor("t2");
  ASSERT_OK(tensor2.status());

  EXPECT_THAT(*tensor1, IsTensor<int64_t>({3}, {1, 2, 3}));
  EXPECT_THAT(*tensor2, IsTensor<absl::string_view>({2}, {"value1", "value2"}));
}

TEST(InMemoryCheckpointParserTest, GetTensorNotFound) {
  InMemoryCheckpointParser parser(std::vector<Tensor>{});

  auto tensor = parser.GetTensor("unknown");
  EXPECT_EQ(tensor.status().code(), absl::StatusCode::kNotFound);
}

TEST(InMemoryCheckpointParserTest, LoadAllTensors) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({1l, 2l, 3l}, "t1"));
  tensors.push_back(Tensor({"value1", "value2"}, "t2"));

  InMemoryCheckpointParser parser(std::move(tensors));

  auto loaded_tensors = parser.LoadAllTensors();
  ASSERT_OK(loaded_tensors.status());
  EXPECT_THAT(
      *loaded_tensors,
      UnorderedElementsAre(
          Pair("t1", IsTensor<int64_t>({3}, {1, 2, 3})),
          Pair("t2", IsTensor<absl::string_view>({2}, {"value1", "value2"}))));
}

TEST(InMemoryCheckpointParserTest, GetTensorTwiceFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({1l, 2l, 3l}, "t1"));

  InMemoryCheckpointParser parser(std::move(tensors));

  auto tensor1 = parser.GetTensor("t1");
  ASSERT_OK(tensor1.status());

  auto tensor1_again = parser.GetTensor("t1");
  EXPECT_EQ(tensor1_again.status().code(), absl::StatusCode::kNotFound);
}

TEST(InMemoryCheckpointParserTest, GetTensorAfterLoadAllTensorsFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({1l, 2l, 3l}, "t1"));

  InMemoryCheckpointParser parser(std::move(tensors));

  auto loaded_tensors = parser.LoadAllTensors();
  ASSERT_OK(loaded_tensors.status());

  auto tensor1 = parser.GetTensor("t1");
  EXPECT_EQ(tensor1.status().code(), absl::StatusCode::kNotFound);
}

TEST(InMemoryCheckpointParserTest, SecondLoadAllTensorsReturnsEmptyMap) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({1l, 2l, 3l}, "t1"));

  InMemoryCheckpointParser parser(std::move(tensors));

  auto loaded_tensors1 = parser.LoadAllTensors();
  ASSERT_OK(loaded_tensors1.status());
  EXPECT_THAT(*loaded_tensors1, UnorderedElementsAre(Pair(
                                    "t1", IsTensor<int64_t>({3}, {1, 2, 3}))));

  auto loaded_tensors2 = parser.LoadAllTensors();
  ASSERT_OK(loaded_tensors2.status());
  EXPECT_THAT(*loaded_tensors2, IsEmpty());
}
}  // namespace
}  // namespace tensorflow_federated::aggregation
