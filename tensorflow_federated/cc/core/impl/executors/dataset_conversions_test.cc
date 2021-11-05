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
#include "tensorflow_federated/cc/core/impl/executors/dataset_conversions.h"

#include <memory>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::EqualsProto;
using ::tensorflow_federated::testing::SequenceV;

TEST(DatasetConversionsTest, TestDatasetCreationFromSequence) {
  v0::Value value_pb = SequenceV(0, 10, 1);
  TFF_ASSERT_OK(SequenceValueToDataset(value_pb.sequence()));
}

TEST(DatasetConversionsTest, TestBadStringReturnsInternalError) {
  tensorflow::tstring bad_graph_def = "bad_graph_def";
  v0::Value value_pb;
  v0::Value::Sequence* sequence_pb = value_pb.mutable_sequence();
  *sequence_pb->mutable_serialized_graph_def() =
      std::string(bad_graph_def.data(), bad_graph_def.size());
  EXPECT_THAT(SequenceValueToDataset(value_pb.sequence()),
              StatusIs(StatusCode::kInternal));
}

TEST(DatasetConversionsTest, TestIterationOverCreatedDataset) {
  v0::Value value_pb = SequenceV(0, 10, 1);
  TFF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<tensorflow::data::standalone::Dataset> ds,
      SequenceValueToDataset(value_pb.sequence()));
  std::unique_ptr<tensorflow::data::standalone::Iterator> iterator;
  // We avoid taking a dependency on TF testing internals.
  auto status = ds->MakeIterator(&iterator);
  ASSERT_TRUE(status.ok());
  bool end_of_values = false;
  std::vector<tensorflow::Tensor> values = {};
  int64_t idx = 0;
  while (true) {
    std::vector<tensorflow::Tensor> output_tensors = {};
    auto status = iterator->GetNext(&output_tensors, &end_of_values);
    ASSERT_TRUE(status.ok());
    if (end_of_values) {
      break;
    }
    ASSERT_EQ(output_tensors.size(), 1);
    tensorflow::TensorProto actual_tensor_proto;
    tensorflow::Tensor output_tensor = output_tensors.at(0);
    values.emplace_back(output_tensor);
    output_tensor.AsProtoTensorContent(&actual_tensor_proto);
    tensorflow::TensorProto expected_tensor_proto;
    tensorflow::Tensor expected_tensor = tensorflow::Tensor(idx++);
    expected_tensor.AsProtoTensorContent(&expected_tensor_proto);
    EXPECT_THAT(expected_tensor_proto, EqualsProto(actual_tensor_proto));
  }
  EXPECT_EQ(values.size(), 10);
}

}  // namespace
}  // namespace tensorflow_federated
