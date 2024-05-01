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

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_parser_factory.h"

#include <memory>
#include <string>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/platform.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

TEST(TensorflowCheckpointParserFactoryTest, ReadCheckpoint) {
  std::string filename = aggregation::TemporaryTestFile(".ckpt");
  ASSERT_OK(CreateTfCheckpoint(filename, {"t1", "t2"},
                               {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f}}));
  absl::StatusOr<absl::Cord> checkpoint = ReadFileToCord(filename);
  ASSERT_OK(checkpoint.status());

  TensorflowCheckpointParserFactory factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      factory.Create(*checkpoint);
  ASSERT_OK(parser.status());

  auto t1 = (*parser)->GetTensor("t1");
  ASSERT_OK(t1.status());
  EXPECT_THAT(*t1, IsTensor<float>({4}, {1.0, 2.0, 3.0, 4.0}));
  auto t2 = (*parser)->GetTensor("t2");
  ASSERT_OK(t2.status());
  EXPECT_THAT(*t2, IsTensor<float>({2}, {5.0, 6.0}));
  EXPECT_FALSE((*parser)->GetTensor("t3").ok());
}

TEST(TensorflowCheckpointParserFactoryTest, InvalidCheckpoint) {
  TensorflowCheckpointParserFactory factory;
  EXPECT_FALSE(factory.Create(absl::Cord("invalid")).ok());
}

}  // namespace
}  // namespace tensorflow_federated::aggregation::tensorflow
