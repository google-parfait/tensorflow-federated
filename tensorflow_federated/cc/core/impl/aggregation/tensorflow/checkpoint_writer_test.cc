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

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/checkpoint_writer.h"

#include <cstdint>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/checkpoint_reader.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

using ::testing::Key;
using ::testing::UnorderedElementsAre;

TEST(CheckpointWriterTest, WriteTensors) {
  // Write the checkpoint using Aggregation Core checkpoint writer.
  auto temp_filename = aggregation::TemporaryTestFile(".ckpt");

  auto t1 = Tensor::Create(DT_FLOAT, TensorShape({4}),
                           CreateTestData<float>({1.0, 2.0, 3.0, 4.0}))
                .value();
  auto t2 = Tensor::Create(DT_INT32, TensorShape({2, 3}),
                           CreateTestData<int32_t>({11, 12, 13, 14, 15, 16}))
                .value();
  auto t3 =
      Tensor::Create(
          DT_STRING, TensorShape({3}),
          CreateTestData<string_view>({"foo", "bar", "bazzzzzzzzzzzzzzzzzzz"}))
          .value();

  CheckpointWriter checkpoint_writer(temp_filename);
  EXPECT_OK(checkpoint_writer.Add("a", t1));
  EXPECT_OK(checkpoint_writer.Add("b", t2));
  EXPECT_OK(checkpoint_writer.Add("c", t3));
  EXPECT_OK(checkpoint_writer.Finish());

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
  EXPECT_THAT(
      *checkpoint_reader->GetTensor("c"),
      IsTensor<string_view>({3}, {"foo", "bar", "bazzzzzzzzzzzzzzzzzzz"}));
}

}  // namespace
}  // namespace tensorflow_federated::aggregation::tensorflow
