/* Copyright 2022, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/sequence_executor.h"

#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_test_base.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/sequence_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::ComputationV;
using ::tensorflow_federated::testing::CreateSerializedRangeDatasetGraphDef;
using ::tensorflow_federated::testing::IntrinsicV;
using ::tensorflow_federated::testing::LambdaComputation;
using ::tensorflow_federated::testing::ReferenceComputation;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::TensorV;

class SequenceExecutorTest : public ExecutorTestBase {
 public:
  explicit SequenceExecutorTest()
      : mock_executor_(
            std::make_shared<::testing::StrictMock<MockExecutor>>()) {
    test_executor_ = CreateSequenceExecutor(mock_executor_);
  }
  ~SequenceExecutorTest() override {}

  std::shared_ptr<::testing::StrictMock<MockExecutor>> mock_executor_;
};

TEST_F(SequenceExecutorTest, CreateMaterializeTFFSequence) {
  tensorflow::tstring graph_def =
      CreateSerializedRangeDatasetGraphDef(0, 10, 1);
  v0::Value value_pb;
  v0::Value::Sequence* sequence_pb = value_pb.mutable_sequence();
  *sequence_pb->mutable_serialized_graph_def() =
      std::string(graph_def.data(), graph_def.size());
  mock_executor_->ExpectCreateMaterialize(value_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(SequenceExecutorTest, CreateMaterializeLocalComputation) {
  v0::Value value_pb =
      ComputationV(LambdaComputation("x", ReferenceComputation("x")));
  mock_executor_->ExpectCreateMaterialize(value_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(SequenceExecutorTest, MaterializeSequenceIntrinsicFails) {
  v0::Value value_pb = IntrinsicV(std::string(kSequenceReduceUri));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(value_pb));
  EXPECT_THAT(test_executor_->Materialize(id),
              StatusIs(StatusCode::kUnimplemented));
}

TEST_F(SequenceExecutorTest, CreateMaterializePassesThroughUnknownIntrinsic) {
  v0::Value value_pb = IntrinsicV("unknown_intrinsic");
  mock_executor_->ExpectCreateMaterialize(value_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(SequenceExecutorTest, CreateSelectionFromComputationFails) {
  v0::Value value_pb = IntrinsicV(std::string(kSequenceReduceUri));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId computation,
                           test_executor_->CreateValue(value_pb));
  // Note that there is no concurrency introduced yet, so this failure should
  // happen eagerly.
  EXPECT_THAT(test_executor_->CreateSelection(computation, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(SequenceExecutorTest, CreateStructureOfTensors) {
  v0::Value five_tensor = TensorV(5.);
  v0::Value ten_tensor = TensorV(10.);
  v0::Value struct_val = StructV({five_tensor, ten_tensor});

  auto embedded_five_id = mock_executor_->ExpectCreateValue(five_tensor);
  auto embedded_ten_id = mock_executor_->ExpectCreateValue(ten_tensor);
  // We lazily create this struct inside the child on materialization.
  auto embedded_struct_id =
      mock_executor_->ExpectCreateStruct({embedded_five_id, embedded_ten_id});
  mock_executor_->ExpectMaterialize(embedded_struct_id, struct_val);

  TFF_ASSERT_OK_AND_ASSIGN(auto five_id,
                           test_executor_->CreateValue(five_tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto ten_id,
                           test_executor_->CreateValue(ten_tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateStruct({five_id, ten_id}));
  ExpectMaterialize(struct_id, struct_val);
}

TEST_F(SequenceExecutorTest, CreateSelectionFromStructure) {
  v0::Value five_tensor = TensorV(5.);
  v0::Value ten_tensor = TensorV(10.);
  mock_executor_->ExpectCreateMaterialize(five_tensor);
  mock_executor_->ExpectCreateMaterialize(ten_tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto five_id,
                           test_executor_->CreateValue(five_tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto ten_id,
                           test_executor_->CreateValue(ten_tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateStruct({five_id, ten_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto selected_five,
                           test_executor_->CreateSelection(struct_id, 0));
  TFF_ASSERT_OK_AND_ASSIGN(auto selected_ten,
                           test_executor_->CreateSelection(struct_id, 1));
  ExpectMaterialize(selected_five, five_tensor);
  ExpectMaterialize(selected_ten, ten_tensor);
}

TEST_F(SequenceExecutorTest, CreateSelectionFromEmbedded) {
  v0::Value five_tensor = TensorV(5.);
  v0::Value ten_tensor = TensorV(10.);
  v0::Value struct_value = StructV({five_tensor, ten_tensor});
  auto embedded_struct_id = mock_executor_->ExpectCreateValue(struct_value);
  auto embedded_five_id =
      mock_executor_->ExpectCreateSelection(embedded_struct_id, 0);
  auto embedded_ten_id =
      mock_executor_->ExpectCreateSelection(embedded_struct_id, 1);
  mock_executor_->ExpectMaterialize(embedded_five_id, five_tensor);
  mock_executor_->ExpectMaterialize(embedded_ten_id, ten_tensor);

  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateValue(struct_value));
  TFF_ASSERT_OK_AND_ASSIGN(auto selected_five,
                           test_executor_->CreateSelection(struct_id, 0));
  TFF_ASSERT_OK_AND_ASSIGN(auto selected_ten,
                           test_executor_->CreateSelection(struct_id, 1));
  ExpectMaterialize(selected_five, five_tensor);
  ExpectMaterialize(selected_ten, ten_tensor);
}

TEST_F(SequenceExecutorTest, TestCreateSelectionFromStructureOutOfBounds) {
  v0::Value five_tensor = TensorV(5.);
  v0::Value ten_tensor = TensorV(10.);
  // Notice no materialize calls should go through to the underlying mock.
  mock_executor_->ExpectCreateValue(five_tensor);
  mock_executor_->ExpectCreateValue(ten_tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto five_id,
                           test_executor_->CreateValue(five_tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto ten_id,
                           test_executor_->CreateValue(ten_tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateStruct({five_id, ten_id}));
  // Note that there is no concurrency introduced yet, so this failure should
  // happen eagerly.
  EXPECT_THAT(test_executor_->CreateSelection(struct_id, 2),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace tensorflow_federated
