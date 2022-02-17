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
#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::ComputationV;
using ::tensorflow_federated::testing::CreateSerializedRangeDatasetGraphDef;
using ::tensorflow_federated::testing::IntrinsicV;
using ::tensorflow_federated::testing::LambdaComputation;
using ::tensorflow_federated::testing::ReferenceComputation;
using ::tensorflow_federated::testing::SequenceV;
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

TEST_F(SequenceExecutorTest, CallSequenceReduceNoargFails) {
  int dataset_len = 10;
  v0::Value sequence_value_pb = SequenceV(1, dataset_len, 1);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(std::string(kSequenceReduceUri))));
  auto reduce_call_id = TFF_ASSERT_OK(
      test_executor_->CreateCall(sequence_reduce_id, absl::nullopt));
  EXPECT_THAT(test_executor_->Materialize(reduce_call_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(SequenceExecutorTest, CreateCreateCallTensorSequenceReduce) {
  int dataset_len = 10;
  v0::Value expected_sum_result = TensorV(45l);
  v0::Value sequence_value_pb = SequenceV(1, 10, 1);
  v0::Value zero = TensorV(static_cast<int64_t>(0));
  v0::Value reduce_fn = IntrinsicV("some_passthru_intrinsic");

  // Since we pull elements out of the sequence in the sequence
  // executor, we never create the sequence in the target.
  auto embedded_accumulator_id = mock_executor_->ExpectCreateValue(zero);
  auto embedded_reduce_fn_id = mock_executor_->ExpectCreateValue(reduce_fn);

  for (int i = 1; i < dataset_len; i++) {
    auto embedded_dataset_element =
        mock_executor_->ExpectCreateValue(TensorV(static_cast<int64_t>(i)));
    auto embedded_arg_struct = mock_executor_->ExpectCreateStruct(
        {embedded_accumulator_id, embedded_dataset_element});
    embedded_accumulator_id = mock_executor_->ExpectCreateCall(
        embedded_reduce_fn_id, embedded_arg_struct);
  }
  mock_executor_->ExpectMaterialize(embedded_accumulator_id,
                                    expected_sum_result);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(std::string(kSequenceReduceUri))));
  auto sequence_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(sequence_value_pb));
  auto fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(reduce_fn));
  auto zero_id = TFF_ASSERT_OK(test_executor_->CreateValue(zero));
  auto struct_id = TFF_ASSERT_OK(
      test_executor_->CreateStruct({sequence_id, zero_id, fn_id}));
  auto call_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(sequence_reduce_id, struct_id));
  ExpectMaterialize(call_id, expected_sum_result);
}

TEST_F(SequenceExecutorTest, EmbedMappedSequenceFails) {
  int dataset_len = 10;
  v0::Value sequence_value_pb = SequenceV(1, dataset_len, 1);
  v0::Value mapping_fn = IntrinsicV("some_passthru_mapping_fn");

  mock_executor_->ExpectCreateValue(mapping_fn);

  auto sequence_map_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(std::string(kSequenceMapUri))));
  auto sequence_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(sequence_value_pb));
  auto mapping_fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(mapping_fn));
  auto map_struct_id =
      TFF_ASSERT_OK(test_executor_->CreateStruct({mapping_fn_id, sequence_id}));
  auto map_call_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(sequence_map_id, map_struct_id));
  EXPECT_THAT(test_executor_->Materialize(map_call_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(SequenceExecutorTest, CreateCreateCallTensorSequenceMapThenReduce) {
  int dataset_len = 10;
  v0::Value sequence_value_pb = SequenceV(1, dataset_len, 1);
  v0::Value mapping_fn = IntrinsicV("some_passthru_mapping_fn");

  // Only the mapping function may be embedded while we create the sequence map,
  // the rest will happen lazily upon iteration.
  auto embedded_mapping_fn_id = mock_executor_->ExpectCreateValue(mapping_fn);

  auto sequence_map_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(std::string(kSequenceMapUri))));
  auto sequence_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(sequence_value_pb));
  auto mapping_fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(mapping_fn));
  auto map_struct_id =
      TFF_ASSERT_OK(test_executor_->CreateStruct({mapping_fn_id, sequence_id}));
  auto map_call_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(sequence_map_id, map_struct_id));

  // We reduce so that we can materialize the result
  v0::Value expected_sum_result = TensorV(45l);
  v0::Value zero = TensorV(static_cast<int64_t>(0));
  v0::Value reduce_fn = IntrinsicV("some_passthru_reduce_fn");

  auto embedded_accumulator_id = mock_executor_->ExpectCreateValue(zero);
  auto embedded_reduce_fn_id = mock_executor_->ExpectCreateValue(reduce_fn);

  for (int i = 1; i < dataset_len; i++) {
    auto embedded_dataset_element =
        mock_executor_->ExpectCreateValue(TensorV(static_cast<int64_t>(i)));
    auto embedded_mapped_fn = mock_executor_->ExpectCreateCall(
        embedded_mapping_fn_id, embedded_dataset_element);
    auto embedded_reduce_arg_struct = mock_executor_->ExpectCreateStruct(
        {embedded_accumulator_id, embedded_mapped_fn});
    embedded_accumulator_id = mock_executor_->ExpectCreateCall(
        embedded_reduce_fn_id, embedded_reduce_arg_struct);
  }

  mock_executor_->ExpectMaterialize(embedded_accumulator_id,
                                    expected_sum_result);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(std::string(kSequenceReduceUri))));
  auto reduce_fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(reduce_fn));
  auto zero_id = TFF_ASSERT_OK(test_executor_->CreateValue(zero));
  // Packages mapped sequence as argument to sequence reduce.
  auto reduce_struct_id = TFF_ASSERT_OK(
      test_executor_->CreateStruct({map_call_id, zero_id, reduce_fn_id}));
  auto reduce_call_id = TFF_ASSERT_OK(
      test_executor_->CreateCall(sequence_reduce_id, reduce_struct_id));
  ExpectMaterialize(reduce_call_id, expected_sum_result);
}

}  // namespace
}  // namespace tensorflow_federated
