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

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/checkpoint_reader.h"

#include <cstdint>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/platform.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

using ::testing::Key;
using ::testing::UnorderedElementsAre;

TEST(CheckpointReaderTest, ReadTensors) {
  // Write a test TF checkpoint with 3 tensors
  auto temp_filename = aggregation::TemporaryTestFile(".ckpt");
  auto tensor_a =
      CreateTfTensor<float>(tf::DT_FLOAT, {4}, {1.0, 2.0, 3.0, 4.0});
  auto tensor_b =
      CreateTfTensor<int32_t>(tf::DT_INT32, {2, 3}, {11, 12, 13, 14, 15, 16});
  auto tensor_c = CreateStringTfTensor({}, {"foobar"});
  EXPECT_TRUE(CreateTfCheckpoint(temp_filename, {"a", "b", "c"},
                                 {tensor_a, tensor_b, tensor_c})
                  .ok());

  // Read the checkpoint using the Aggregation Core checkpoint reader.
  auto checkpoint_reader_or_status = CheckpointReader::Create(temp_filename);
  EXPECT_OK(checkpoint_reader_or_status.status());

  auto checkpoint_reader = std::move(checkpoint_reader_or_status).value();
  EXPECT_THAT(checkpoint_reader->GetDataTypeMap(),
              UnorderedElementsAre(Key("a"), Key("b"), Key("c")));
  EXPECT_THAT(checkpoint_reader->GetTensorShapeMap(),
              UnorderedElementsAre(Key("a"), Key("b"), Key("c")));

  // Read and verify the tensors.
  EXPECT_THAT(*checkpoint_reader->GetTensor("a"),
              IsTensor<float>({4}, {1.0, 2.0, 3.0, 4.0}));
  EXPECT_THAT(*checkpoint_reader->GetTensor("b"),
              IsTensor<int32_t>({2, 3}, {11, 12, 13, 14, 15, 16}));
  EXPECT_THAT(*checkpoint_reader->GetTensor("c"),
              IsTensor<string_view>({}, {"foobar"}));
}

TEST(CheckpointReaderTest, InvalidFileName) {
  auto checkpoint_reader_or_status = CheckpointReader::Create("foo/bar");
  EXPECT_THAT(checkpoint_reader_or_status, StatusIs(INTERNAL));
}

TEST(CheckpointReaderTest, MalformedFile) {
  auto temp_filename = aggregation::TemporaryTestFile(".ckpt");
  WriteStringToFile(temp_filename, "foobar").IgnoreError();
  auto checkpoint_reader_or_status = CheckpointReader::Create(temp_filename);
  EXPECT_THAT(checkpoint_reader_or_status, StatusIs(INTERNAL));
}

}  // namespace
}  // namespace tensorflow_federated::aggregation::tensorflow
