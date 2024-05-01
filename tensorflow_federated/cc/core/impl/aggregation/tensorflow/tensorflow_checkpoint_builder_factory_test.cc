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

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"

#include <memory>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

using ::testing::AllOf;
using ::testing::Each;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::StartsWith;
using ::testing::UnorderedElementsAre;

TEST(TensorflowCheckpointBuilderFactoryTest, BuildCheckpoint) {
  TensorflowCheckpointBuilderFactory factory;
  std::unique_ptr<CheckpointBuilder> builder = factory.Create();

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DT_FLOAT, TensorShape({4}), CreateTestData<float>({1.0, 2.0, 3.0, 4.0}));
  ASSERT_OK(t1.status());
  absl::StatusOr<Tensor> t2 = Tensor::Create(DT_FLOAT, TensorShape({2}),
                                             CreateTestData<float>({5.0, 6.0}));
  ASSERT_OK(t2.status());

  EXPECT_OK(builder->Add("t1", *t1));
  EXPECT_OK(builder->Add("t2", *t2));
  absl::StatusOr<absl::Cord> checkpoint = builder->Build();
  ASSERT_OK(checkpoint.status());
  auto summary = SummarizeCheckpoint(*checkpoint);
  ASSERT_OK(summary.status());
  EXPECT_THAT(*summary,
              UnorderedElementsAre(Pair("t1", "1 2 3 4"), Pair("t2", "5 6")));
}

// Check that multiple checkpoints can be built simultanously.
TEST(TensorflowCheckpointBuilderFactoryTest, SimultaneousWrites) {
  TensorflowCheckpointBuilderFactory factory;

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DT_FLOAT, TensorShape({4}), CreateTestData<float>({1.0, 2.0, 3.0, 4.0}));
  ASSERT_OK(t1.status());
  absl::StatusOr<Tensor> t2 = Tensor::Create(DT_FLOAT, TensorShape({2}),
                                             CreateTestData<float>({5.0, 6.0}));
  ASSERT_OK(t2.status());

  std::unique_ptr<CheckpointBuilder> builder1 = factory.Create();
  std::unique_ptr<CheckpointBuilder> builder2 = factory.Create();
  EXPECT_OK(builder1->Add("t1", *t1));
  EXPECT_OK(builder2->Add("t2", *t2));
  absl::StatusOr<absl::Cord> checkpoint1 = builder1->Build();
  ASSERT_OK(checkpoint1.status());
  absl::StatusOr<absl::Cord> checkpoint2 = builder2->Build();
  ASSERT_OK(checkpoint2.status());
  auto summary1 = SummarizeCheckpoint(*checkpoint1);
  ASSERT_OK(summary1.status());
  EXPECT_THAT(*summary1, UnorderedElementsAre(Pair("t1", "1 2 3 4")));
  auto summary2 = SummarizeCheckpoint(*checkpoint2);
  ASSERT_OK(summary2.status());
  EXPECT_THAT(*summary2, UnorderedElementsAre(Pair("t2", "5 6")));
}

TEST(TensorflowCheckpointBuilderFactoryTest, LargeCheckpoint) {
  TensorflowCheckpointBuilderFactory factory;
  std::unique_ptr<CheckpointBuilder> builder = factory.Create();

  // Add 10 tensors that each require at least 8kB to exercise reading and
  // writing in multiple chunks.
  static constexpr int kTensorSize = 1024;
  absl::StatusOr<Tensor> t =
      Tensor::Create(DT_INT64, TensorShape({kTensorSize}),
                     std::make_unique<MutableVectorData<int64_t>>(kTensorSize));
  ASSERT_OK(t.status());
  for (int i = 0; i < 10; ++i) {
    EXPECT_OK(builder->Add(absl::StrCat("t", i), *t));
  }
  absl::StatusOr<absl::Cord> checkpoint = builder->Build();
  ASSERT_OK(checkpoint.status());
  auto summary = SummarizeCheckpoint(*checkpoint);
  ASSERT_OK(summary.status());
  EXPECT_THAT(*summary,
              AllOf(SizeIs(10), Each(Pair(StartsWith("t"),
                                          StartsWith("0 0 0 0 0 0 0 0 0")))));
}

}  // namespace
}  // namespace tensorflow_federated::aggregation::tensorflow
