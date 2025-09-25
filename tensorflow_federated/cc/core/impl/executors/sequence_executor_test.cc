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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "federated_language/proto/computation.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_test_base.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/sequence_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::ComputationV;
using ::tensorflow_federated::testing::IntrinsicV;
using ::tensorflow_federated::testing::LambdaComputation;
using ::tensorflow_federated::testing::MakeInt64ScalarType;
using ::tensorflow_federated::testing::ReferenceComputation;
using ::tensorflow_federated::testing::SequenceV;
using ::tensorflow_federated::testing::StructV;

class SequenceExecutorTest : public ExecutorTestBase {
 public:
  explicit SequenceExecutorTest()
      : mock_executor_(
            std::make_shared<::testing::StrictMock<MockExecutor>>()) {
    test_executor_ = CreateSequenceExecutor(mock_executor_);
  }
  ~SequenceExecutorTest() override = default;

  std::shared_ptr<::testing::StrictMock<MockExecutor>> mock_executor_;
};

TEST_F(SequenceExecutorTest, CreateMaterializeTFFSequence) {
  federated_language_executor::Value value_pb = SequenceV(0, 10, 1);
  mock_executor_->ExpectCreateMaterialize(value_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(SequenceExecutorTest, CreateMaterializeTFFSequenceYieldingStructures) {
  federated_language_executor::Value value_pb = SequenceV({
      {1, 2, 3},
      {10, 20, 30},
      {100, 200, 300},
  });
  mock_executor_->ExpectCreateMaterialize(value_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(SequenceExecutorTest, CreateMaterializeLocalComputation) {
  federated_language_executor::Value value_pb =
      ComputationV(LambdaComputation("x", ReferenceComputation("x")));
  mock_executor_->ExpectCreateMaterialize(value_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(SequenceExecutorTest, MaterializeSequenceIntrinsicFails) {
  federated_language_executor::Value value_pb = IntrinsicV(kSequenceReduceUri);
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(value_pb));
  EXPECT_THAT(test_executor_->Materialize(id),
              StatusIs(StatusCode::kUnimplemented));
}

TEST_F(SequenceExecutorTest, CreateMaterializePassesThroughUnknownIntrinsic) {
  federated_language_executor::Value value_pb = IntrinsicV("unknown_intrinsic");
  mock_executor_->ExpectCreateMaterialize(value_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(SequenceExecutorTest, CreateSelectionFromComputationFails) {
  federated_language_executor::Value value_pb = IntrinsicV(kSequenceReduceUri);
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId computation,
                           test_executor_->CreateValue(value_pb));
  // Note that there is no concurrency introduced yet, so this failure should
  // happen eagerly.
  EXPECT_THAT(test_executor_->CreateSelection(computation, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(SequenceExecutorTest, CreateStructureOfTensors) {
  federated_language::Array five_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {5.0}));
  federated_language_executor::Value five_tensor;
  five_tensor.mutable_array()->Swap(&five_array_pb);
  federated_language::Array ten_array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_FLOAT,
                           testing::CreateArrayShape({}), {10.0}));
  federated_language_executor::Value ten_tensor;
  ten_tensor.mutable_array()->Swap(&ten_array_pb);
  federated_language_executor::Value struct_val =
      StructV({five_tensor, ten_tensor});

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
  federated_language::Array five_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {5.0}));
  federated_language_executor::Value five_tensor;
  five_tensor.mutable_array()->Swap(&five_array_pb);
  federated_language::Array ten_array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_FLOAT,
                           testing::CreateArrayShape({}), {10.0}));
  federated_language_executor::Value ten_tensor;
  ten_tensor.mutable_array()->Swap(&ten_array_pb);
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
  federated_language::Array five_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {5.0}));
  federated_language_executor::Value five_tensor;
  five_tensor.mutable_array()->Swap(&five_array_pb);
  federated_language::Array ten_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {5.0}));
  federated_language_executor::Value ten_tensor;
  ten_tensor.mutable_array()->Swap(&ten_array_pb);
  federated_language_executor::Value struct_value =
      StructV({five_tensor, ten_tensor});
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
  federated_language::Array five_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {5.0}));
  federated_language_executor::Value five_tensor;
  five_tensor.mutable_array()->Swap(&five_array_pb);
  federated_language::Array ten_array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_FLOAT,
                           testing::CreateArrayShape({}), {10.0}));
  federated_language_executor::Value ten_tensor;
  ten_tensor.mutable_array()->Swap(&ten_array_pb);
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

TEST_F(SequenceExecutorTest, CallPassThrough) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value tensor_value;
  tensor_value.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value passthru_fn =
      IntrinsicV("some_passthru_intrinsic");

  auto embedded_tensor_id = mock_executor_->ExpectCreateValue(tensor_value);
  auto embedded_fn_id = mock_executor_->ExpectCreateValue(passthru_fn);
  auto embedded_call_id =
      mock_executor_->ExpectCreateCall(embedded_fn_id, embedded_tensor_id);
  mock_executor_->ExpectMaterialize(embedded_call_id, tensor_value);

  auto arg_id = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_value));
  auto fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(passthru_fn));
  auto call_id = TFF_ASSERT_OK(test_executor_->CreateCall(fn_id, arg_id));
  ExpectMaterialize(call_id, tensor_value);
}

TEST_F(SequenceExecutorTest, CallPassThroughNoArg) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value tensor_value;
  tensor_value.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value passthru_fn =
      IntrinsicV("some_passthru_intrinsic");

  {
    auto embedded_fn_id = mock_executor_->ExpectCreateValue(passthru_fn);
    auto embedded_call_id =
        mock_executor_->ExpectCreateCall(embedded_fn_id, std::nullopt);
    mock_executor_->ExpectMaterialize(embedded_call_id, tensor_value);

    auto fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(passthru_fn));
    auto call_id =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn_id, std::nullopt));
    ExpectMaterialize(call_id, tensor_value);
  }
}

TEST_F(SequenceExecutorTest, CallSequenceReduceNoargFails) {
  int dataset_len = 10;
  federated_language_executor::Value sequence_value_pb =
      SequenceV(1, dataset_len, 1);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(kSequenceReduceUri)));
  auto reduce_call_id = TFF_ASSERT_OK(
      test_executor_->CreateCall(sequence_reduce_id, std::nullopt));
  EXPECT_THAT(test_executor_->Materialize(reduce_call_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(SequenceExecutorTest, EmbedFailsWithBadType) {
  int dataset_len = 10;
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {45}));
  federated_language_executor::Value expected_sum_result;
  expected_sum_result.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value sequence_pb = SequenceV(1, dataset_len, 1);
  // We mutate the element type of this Sequence value to a non-embeddable type.

  federated_language::Type function_type;
  *function_type.mutable_function()->mutable_result() =
      sequence_pb.sequence().element_type();

  *sequence_pb.mutable_sequence()->mutable_element_type() = function_type;

  federated_language::Array zero_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {0}));
  federated_language_executor::Value zero;
  zero.mutable_array()->Swap(&zero_array_pb);
  federated_language_executor::Value reduce_fn =
      IntrinsicV("some_passthru_intrinsic");

  mock_executor_->ExpectCreateValue(zero);
  mock_executor_->ExpectCreateValue(reduce_fn);

  // None of the elements will be embedded, we should err out trying to embed
  // them

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(kSequenceReduceUri)));
  auto sequence_id = TFF_ASSERT_OK(test_executor_->CreateValue(sequence_pb));
  auto fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(reduce_fn));
  auto zero_id = TFF_ASSERT_OK(test_executor_->CreateValue(zero));
  auto struct_id = TFF_ASSERT_OK(
      test_executor_->CreateStruct({sequence_id, zero_id, fn_id}));
  auto call_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(sequence_reduce_id, struct_id));
  EXPECT_THAT(test_executor_->Materialize(call_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(SequenceExecutorTest, CreateCallStructureSequenceReduce) {
  int dataset_len = 10;
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {45}));
  federated_language_executor::Value expected_sum_result;
  expected_sum_result.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value sequence_value_pb = SequenceV({
      {1, 11},
      {2, 12},
      {3, 13},
      {4, 14},
      {5, 15},
      {6, 16},
      {7, 17},
      {8, 18},
      {9, 19},
  });

  federated_language::Array zero_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {0}));
  federated_language_executor::Value zero;
  zero.mutable_array()->Swap(&zero_array_pb);
  federated_language_executor::Value reduce_fn =
      IntrinsicV("some_passthru_intrinsic");

  auto embedded_accumulator_id = mock_executor_->ExpectCreateValue(zero);
  auto embedded_reduce_fn_id = mock_executor_->ExpectCreateValue(reduce_fn);

  for (int i = 1; i < dataset_len; i++) {
    federated_language::Array first_array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT64,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value first_element_pb;
    first_element_pb.mutable_array()->Swap(&first_array_pb);
    auto embedded_dataset_first_element =
        mock_executor_->ExpectCreateValue(first_element_pb);
    federated_language::Array second_array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT64,
                             testing::CreateArrayShape({}), {i + dataset_len}));
    federated_language_executor::Value second_element_pb;
    second_element_pb.mutable_array()->Swap(&second_array_pb);
    auto embedded_dataset_second_element =
        mock_executor_->ExpectCreateValue(second_element_pb);
    auto embedded_dataset_element = mock_executor_->ExpectCreateStruct(
        {embedded_dataset_first_element, embedded_dataset_second_element});
    auto embedded_arg_struct = mock_executor_->ExpectCreateStruct(
        {embedded_accumulator_id, embedded_dataset_element});
    embedded_accumulator_id = mock_executor_->ExpectCreateCall(
        embedded_reduce_fn_id, embedded_arg_struct);
  }
  mock_executor_->ExpectMaterialize(embedded_accumulator_id,
                                    expected_sum_result);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(kSequenceReduceUri)));
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

TEST_F(SequenceExecutorTest, CreateCallNestedStructureSequenceReduce) {
  int dataset_len = 10;
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {45}));
  federated_language_executor::Value expected_sum_result;
  expected_sum_result.mutable_array()->Swap(&array_pb);

  federated_language_executor::Value sequence_value_pb = SequenceV({
      {1, 11, 21},
      {2, 12, 22},
      {3, 13, 23},
      {4, 14, 24},
      {5, 15, 25},
      {6, 16, 26},
      {7, 17, 27},
      {8, 18, 28},
      {9, 19, 29},
  });

  // We make a nested type corresponding to <int,<y=int,x=int>>
  // Notice that the names appear in non-sorted order in the TFF type signature;
  // we explicitly test this case to ensure that our traversal corresponds to
  // tf.nest's traversal order, where the keys of ordered dicts are sorted.
  federated_language::Type sequence_element_type;
  *sequence_element_type.mutable_struct_()->add_element()->mutable_value() =
      MakeInt64ScalarType();
  federated_language::Type* nested_struct_type =
      sequence_element_type.mutable_struct_()->add_element()->mutable_value();
  for (int i = 0; i < 2; i++) {
    federated_language::StructType_Element* struct_elem =
        nested_struct_type->mutable_struct_()->add_element();
    *struct_elem->mutable_value() = MakeInt64ScalarType();
    if (i == 0) {
      *struct_elem->mutable_name() = "y";
    } else {
      *struct_elem->mutable_name() = "x";
    }
  }

  *sequence_value_pb.mutable_sequence()->mutable_element_type() =
      sequence_element_type;

  federated_language::Array zero_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {0}));
  federated_language_executor::Value zero;
  federated_language_executor::Value reduce_fn =
      IntrinsicV("some_passthru_intrinsic");

  // Since we pull elements out of the sequence in the sequence
  // executor, we never create the sequence in the target.
  auto embedded_accumulator_id = mock_executor_->ExpectCreateValue(zero);
  auto embedded_reduce_fn_id = mock_executor_->ExpectCreateValue(reduce_fn);

  for (int i = 1; i < dataset_len; i++) {
    // Each of the tensors above will be embedded
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT64,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value element_pb;
    element_pb.mutable_array()->Swap(&array_pb);
    auto embedded_dataset_top_level_element =
        mock_executor_->ExpectCreateValue(element_pb);
    // Notice that the 'x' element should be embedded first, since this is the
    // way TF.data will yield the tensors (with sorted keys).

    federated_language::Array x_array_pb = TFF_ASSERT_OK(testing::CreateArray(
        federated_language::DataType::DT_INT64, testing::CreateArrayShape({}),
        {i + 2 * dataset_len}));
    federated_language_executor::Value x_element_pb;
    x_element_pb.mutable_array()->Swap(&x_array_pb);
    auto embedded_dataset_struct_x_element =
        mock_executor_->ExpectCreateValue(x_element_pb);
    federated_language::Array y_array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT64,
                             testing::CreateArrayShape({}), {i + dataset_len}));
    federated_language_executor::Value y_element_pb;
    y_element_pb.mutable_array()->Swap(&y_array_pb);
    auto embedded_dataset_struct_y_element =
        mock_executor_->ExpectCreateValue(y_element_pb);
    // Two of them will be tupled together, back in the order corresponding to
    // the TFF type.
    auto embedded_nested_struct = mock_executor_->ExpectCreateStruct(
        {embedded_dataset_struct_y_element, embedded_dataset_struct_x_element});
    // Then tupled with the final, top-level tensor in the expected type.
    auto embedded_dataset_element = mock_executor_->ExpectCreateStruct(
        {embedded_dataset_top_level_element, embedded_nested_struct});
    auto embedded_arg_struct = mock_executor_->ExpectCreateStruct(
        {embedded_accumulator_id, embedded_dataset_element});
    embedded_accumulator_id = mock_executor_->ExpectCreateCall(
        embedded_reduce_fn_id, embedded_arg_struct);
  }
  mock_executor_->ExpectMaterialize(embedded_accumulator_id,
                                    expected_sum_result);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(kSequenceReduceUri)));
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

TEST_F(SequenceExecutorTest, CreateCreateSequenceReduceStructuredZero) {
  int dataset_len = 10;
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {45}));
  federated_language_executor::Value expected_sum_result;
  expected_sum_result.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value sequence_value_pb =
      SequenceV(1, dataset_len, 1);
  federated_language::Array zero_one_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {100}));
  federated_language_executor::Value zero_one;
  zero_one.mutable_array()->Swap(&zero_one_array_pb);
  federated_language::Array zero_two_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {101}));
  federated_language_executor::Value zero_two;
  zero_two.mutable_array()->Swap(&zero_two_array_pb);
  federated_language_executor::Value reduce_fn =
      IntrinsicV("some_passthru_intrinsic");

  // Since we pull elements out of the sequence in the sequence
  // executor, we never create the sequence in the target.
  auto embedded_accumulator_element0_id =
      mock_executor_->ExpectCreateValue(zero_one);
  auto embedded_accumulator_element1_id =
      mock_executor_->ExpectCreateValue(zero_two);
  auto embedded_accumulator_id = mock_executor_->ExpectCreateStruct(
      {embedded_accumulator_element0_id, embedded_accumulator_element1_id});
  auto embedded_reduce_fn_id = mock_executor_->ExpectCreateValue(reduce_fn);

  for (int i = 1; i < dataset_len; i++) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT64,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value element_pb;
    element_pb.mutable_array()->Swap(&array_pb);
    auto embedded_dataset_element =
        mock_executor_->ExpectCreateValue(element_pb);
    auto embedded_arg_struct = mock_executor_->ExpectCreateStruct(
        {embedded_accumulator_id, embedded_dataset_element});
    embedded_accumulator_id = mock_executor_->ExpectCreateCall(
        embedded_reduce_fn_id, embedded_arg_struct);
  }
  mock_executor_->ExpectMaterialize(embedded_accumulator_id,
                                    expected_sum_result);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(kSequenceReduceUri)));
  auto sequence_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(sequence_value_pb));
  auto fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(reduce_fn));
  auto zero_element0_id = TFF_ASSERT_OK(test_executor_->CreateValue(zero_one));
  auto zero_element1_id = TFF_ASSERT_OK(test_executor_->CreateValue(zero_two));
  auto zero_id = TFF_ASSERT_OK(
      test_executor_->CreateStruct({zero_element0_id, zero_element1_id}));
  auto struct_id = TFF_ASSERT_OK(
      test_executor_->CreateStruct({sequence_id, zero_id, fn_id}));
  auto call_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(sequence_reduce_id, struct_id));
  ExpectMaterialize(call_id, expected_sum_result);
}

TEST_F(SequenceExecutorTest, CreateCreateCallTensorSequenceReduce) {
  int dataset_len = 10;
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {45}));
  federated_language_executor::Value expected_sum_result;
  expected_sum_result.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value sequence_value_pb =
      SequenceV(1, dataset_len, 1);
  federated_language::Array zero_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {0}));
  federated_language_executor::Value zero;
  zero.mutable_array()->Swap(&zero_array_pb);
  federated_language_executor::Value reduce_fn =
      IntrinsicV("some_passthru_intrinsic");

  // Since we pull elements out of the sequence in the sequence
  // executor, we never create the sequence in the target.
  auto embedded_accumulator_id = mock_executor_->ExpectCreateValue(zero);
  auto embedded_reduce_fn_id = mock_executor_->ExpectCreateValue(reduce_fn);

  for (int i = 1; i < dataset_len; i++) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT64,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value element_pb;
    element_pb.mutable_array()->Swap(&array_pb);
    auto embedded_dataset_element =
        mock_executor_->ExpectCreateValue(element_pb);
    auto embedded_arg_struct = mock_executor_->ExpectCreateStruct(
        {embedded_accumulator_id, embedded_dataset_element});
    embedded_accumulator_id = mock_executor_->ExpectCreateCall(
        embedded_reduce_fn_id, embedded_arg_struct);
  }
  mock_executor_->ExpectMaterialize(embedded_accumulator_id,
                                    expected_sum_result);

  auto sequence_reduce_id = TFF_ASSERT_OK(
      test_executor_->CreateValue(IntrinsicV(kSequenceReduceUri)));
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
  federated_language_executor::Value sequence_value_pb =
      SequenceV(1, dataset_len, 1);
  federated_language_executor::Value mapping_fn =
      IntrinsicV("some_passthru_mapping_fn");

  mock_executor_->ExpectCreateValue(mapping_fn);

  auto sequence_map_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(IntrinsicV(kSequenceMapUri)));
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
  federated_language_executor::Value sequence_value_pb =
      SequenceV(1, dataset_len, 1);
  federated_language_executor::Value mapping_fn =
      IntrinsicV("some_passthru_mapping_fn");

  // Only the mapping function may be embedded while we create the sequence map,
  // the rest will happen lazily upon iteration.
  auto embedded_mapping_fn_id = mock_executor_->ExpectCreateValue(mapping_fn);

  auto sequence_map_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(IntrinsicV(kSequenceMapUri)));
  auto sequence_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(sequence_value_pb));
  auto mapping_fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(mapping_fn));
  auto map_struct_id =
      TFF_ASSERT_OK(test_executor_->CreateStruct({mapping_fn_id, sequence_id}));
  auto map_call_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(sequence_map_id, map_struct_id));

  // We reduce so that we can materialize the result
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {45}));
  federated_language_executor::Value expected_sum_result;
  expected_sum_result.mutable_array()->Swap(&array_pb);
  federated_language::Array zero_array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {0}));
  federated_language_executor::Value zero;
  zero.mutable_array()->Swap(&zero_array_pb);
  federated_language_executor::Value reduce_fn =
      IntrinsicV("some_passthru_reduce_fn");

  auto embedded_accumulator_id = mock_executor_->ExpectCreateValue(zero);
  auto embedded_reduce_fn_id = mock_executor_->ExpectCreateValue(reduce_fn);

  for (int i = 1; i < dataset_len; i++) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT64,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value element_pb;
    element_pb.mutable_array()->Swap(&array_pb);
    auto embedded_dataset_element =
        mock_executor_->ExpectCreateValue(element_pb);
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
      test_executor_->CreateValue(IntrinsicV(kSequenceReduceUri)));
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
