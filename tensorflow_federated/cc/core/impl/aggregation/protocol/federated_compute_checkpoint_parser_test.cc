// Copyright 2024 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

#include <cstdint>
#include <memory>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"

namespace tensorflow_federated::aggregation {
namespace {

TEST(FederatedComputeCheckpointParserTest, GetTensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DT_INT64, TensorShape({3}), CreateTestData<uint64_t>({1, 2, 3}));
  ASSERT_OK(t1.status());
  absl::StatusOr<Tensor> t2 =
      Tensor::Create(DT_STRING, TensorShape({2}),
                     CreateTestData<absl::string_view>({"value1", "value2"}));
  ASSERT_OK(t2.status());
  absl::StatusOr<Tensor> t3 = Tensor::Create(
      DataType::DT_INT32, TensorShape({2}), CreateTestData<int32_t>({1, 2}));
  ASSERT_OK(t3.status());

  EXPECT_OK(builder->Add("t1", *t1));
  EXPECT_OK(builder->Add("t2", *t2));
  EXPECT_OK(builder->Add("t3", *t3));
  auto checkpoint = builder->Build();
  ASSERT_OK(checkpoint.status());

  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  ASSERT_OK(parser.status());
  auto tensor1 = (*parser)->GetTensor("t1");
  ASSERT_OK(tensor1.status());
  auto tensor2 = (*parser)->GetTensor("t2");
  ASSERT_OK(tensor2.status());
  auto tensor3 = (*parser)->GetTensor("t3");
  ASSERT_OK(tensor2.status());
  EXPECT_THAT(*tensor1, IsTensor<int64_t>({3}, {1, 2, 3}));
  EXPECT_THAT(*tensor2, IsTensor<absl::string_view>({2}, {"value1", "value2"}));
  EXPECT_THAT(*tensor3, IsTensor<int32_t>({2}, {1, 2}));
}

}  // namespace
}  // namespace tensorflow_federated::aggregation
