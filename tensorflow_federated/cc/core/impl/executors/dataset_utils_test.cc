/* Copyright 2021, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#include "tensorflow_federated/cc/core/impl/executors/dataset_utils.h"

#include <cstdint>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {

namespace {

using ::tensorflow_federated::testing::SequenceV;

TEST(DatasetConversionsTest, TestGraphDefTensorFromSequenceRoundTrip) {
  v0::Value value_pb = SequenceV(0, 10, 1);
  tensorflow::Tensor graph_def_tensor =
      TFF_ASSERT_OK(GraphDefTensorFromSequence(value_pb.sequence()));

  // Verify the result is a scalar string tensor containing a serialized
  // GraphDef.
  EXPECT_EQ(graph_def_tensor.dtype(), tensorflow::DT_STRING);
  EXPECT_EQ(graph_def_tensor.dims(), 0);
}

TEST(DatasetConversionsTest, TestExtractOutputTypesAndShapes) {
  v0::Value value_pb = SequenceV(0, 5, 1);
  tensorflow::Tensor graph_def_tensor =
      TFF_ASSERT_OK(GraphDefTensorFromSequence(value_pb.sequence()));

  auto types_and_shapes =
      TFF_ASSERT_OK(ExtractOutputTypesAndShapesFromGraphDef(graph_def_tensor));

  EXPECT_EQ(types_and_shapes.first.size(), 1);
  EXPECT_EQ(types_and_shapes.first[0], tensorflow::DT_INT64);
  EXPECT_EQ(types_and_shapes.second.size(), 1);
}

TEST(DatasetConversionsTest, TestIterateDatasetFromGraphDef) {
  v0::Value value_pb = SequenceV(0, 10, 1);
  tensorflow::Tensor graph_def_tensor =
      TFF_ASSERT_OK(GraphDefTensorFromSequence(value_pb.sequence()));

  auto types_and_shapes =
      TFF_ASSERT_OK(ExtractOutputTypesAndShapesFromGraphDef(graph_def_tensor));

  std::vector<std::vector<tensorflow::Tensor>> elements =
      TFF_ASSERT_OK(IterateDatasetFromGraphDef(
          graph_def_tensor, types_and_shapes.first, types_and_shapes.second));

  EXPECT_EQ(elements.size(), 10);
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(elements[i].size(), 1);
    EXPECT_EQ(elements[i][0].dtype(), tensorflow::DT_INT64);
    EXPECT_EQ(elements[i][0].scalar<int64_t>()(), i);
  }
}

TEST(DatasetConversionsTest, TestIterateMultiElementDataset) {
  v0::Value value_pb = SequenceV({{1, 2}, {3, 4}, {5, 6}});
  tensorflow::Tensor graph_def_tensor =
      TFF_ASSERT_OK(GraphDefTensorFromSequence(value_pb.sequence()));

  auto types_and_shapes =
      TFF_ASSERT_OK(ExtractOutputTypesAndShapesFromGraphDef(graph_def_tensor));

  std::vector<std::vector<tensorflow::Tensor>> elements =
      TFF_ASSERT_OK(IterateDatasetFromGraphDef(
          graph_def_tensor, types_and_shapes.first, types_and_shapes.second));

  EXPECT_EQ(elements.size(), 3);
  ASSERT_EQ(elements[0].size(), 2);
  EXPECT_EQ(elements[0][0].scalar<int64_t>()(), 1);
  EXPECT_EQ(elements[0][1].scalar<int64_t>()(), 2);
  EXPECT_EQ(elements[1][0].scalar<int64_t>()(), 3);
  EXPECT_EQ(elements[1][1].scalar<int64_t>()(), 4);
  EXPECT_EQ(elements[2][0].scalar<int64_t>()(), 5);
  EXPECT_EQ(elements[2][1].scalar<int64_t>()(), 6);
}

}  // namespace
}  // namespace tensorflow_federated
