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

#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/py/federated_language_executor/executor.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_test_base.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::ClientsV;
using ::tensorflow_federated::testing::IntrinsicV;
using ::tensorflow_federated::testing::SequenceV;
using ::tensorflow_federated::testing::ServerV;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::intrinsic::ArgsIntoSequenceV;
using ::tensorflow_federated::testing::intrinsic::FederatedAggregateV;
using ::tensorflow_federated::testing::intrinsic::FederatedBroadcastV;
using ::tensorflow_federated::testing::intrinsic::FederatedEvalAtClientsV;
using ::tensorflow_federated::testing::intrinsic::FederatedEvalAtServerV;
using ::tensorflow_federated::testing::intrinsic::FederatedMapAllEqualV;
using ::tensorflow_federated::testing::intrinsic::FederatedMapV;
using ::tensorflow_federated::testing::intrinsic::FederatedSelectV;
using ::tensorflow_federated::testing::intrinsic::FederatedValueAtClientsV;
using ::tensorflow_federated::testing::intrinsic::FederatedValueAtServerV;
using ::tensorflow_federated::testing::intrinsic::FederatedZipAtClientsV;
using ::tensorflow_federated::testing::intrinsic::FederatedZipAtServerV;
using ::testing::Cardinality;
using ::testing::HasSubstr;

const uint16_t NUM_CLIENTS = 10;

const Cardinality ONCE = ::testing::Exactly(1);
const Cardinality ONCE_PER_CLIENT = ::testing::Exactly(NUM_CLIENTS);

struct IdPair {
  OwnedValueId id;
  ValueId child_id;
};

class FederatingExecutorTest : public ExecutorTestBase {
 public:
  FederatingExecutorTest() {
    // Extra method required in order to use `TFF_ASSERT_OK_AND_ASSIGN`.
    Initialize();
  }

  ~FederatingExecutorTest() override = default;

  void Initialize() {
    TFF_ASSERT_OK_AND_ASSIGN(test_executor_,
                             tensorflow_federated::CreateFederatingExecutor(
                                 mock_server_executor_, mock_client_executor_,
                                 {{"clients", NUM_CLIENTS}}));
  }

 protected:
  std::shared_ptr<::testing::StrictMock<MockExecutor>> mock_server_executor_ =
      std::make_shared<::testing::StrictMock<MockExecutor>>();
  std::shared_ptr<::testing::StrictMock<MockExecutor>> mock_client_executor_ =
      std::make_shared<::testing::StrictMock<MockExecutor>>();

  ValueId ExpectCreateInServerChild(
      const federated_language_executor::Value& expected,
      Cardinality repeatedly = ONCE) {
    return mock_server_executor_->ExpectCreateValue(expected, repeatedly);
  }
  ValueId ExpectCreateInClientChild(
      const federated_language_executor::Value& expected,
      Cardinality repeatedly = ONCE) {
    return mock_client_executor_->ExpectCreateValue(expected, repeatedly);
  }

  void ExpectMaterializeInServerChild(
      ValueId id, federated_language_executor::Value to_return,
      Cardinality repeatedly = ONCE) {
    mock_server_executor_->ExpectMaterialize(id, std::move(to_return),
                                             repeatedly);
  }

  void ExpectMaterializeInClientChild(
      ValueId id, federated_language_executor::Value to_return,
      Cardinality repeatedly = ONCE) {
    mock_client_executor_->ExpectMaterialize(id, std::move(to_return),
                                             repeatedly);
  }

  ValueId ExpectCreateCallInServerChild(ValueId fn_id,
                                        std::optional<const ValueId> arg_id,
                                        Cardinality repeatedly = ONCE) {
    return mock_server_executor_->ExpectCreateCall(fn_id, arg_id, repeatedly);
  }

  ValueId ExpectCreateCallInClientChild(ValueId fn_id,
                                        std::optional<const ValueId> arg_id,
                                        Cardinality repeatedly = ONCE) {
    return mock_client_executor_->ExpectCreateCall(fn_id, arg_id, repeatedly);
  }

  ValueId ExpectCreateStructInServerChild(
      const absl::Span<const ValueId> elements, Cardinality repeatedly = ONCE) {
    return mock_server_executor_->ExpectCreateStruct(elements, repeatedly);
  }

  ValueId ExpectCreateStructInClientChild(
      const absl::Span<const ValueId> elements, Cardinality repeatedly = ONCE) {
    return mock_client_executor_->ExpectCreateStruct(elements, repeatedly);
  }
  void ExpectCreateMaterializeInServerChild(
      federated_language_executor::Value value, Cardinality repeatedly = ONCE) {
    ValueId id = ExpectCreateInServerChild(value, repeatedly);
    ExpectMaterializeInServerChild(id, value, repeatedly);
  }
  void ExpectCreateMaterializeInClientChild(
      federated_language_executor::Value value, Cardinality repeatedly = ONCE) {
    ValueId id = ExpectCreateInClientChild(value, repeatedly);
    ExpectMaterializeInClientChild(id, value, repeatedly);
  }

  absl::StatusOr<IdPair> CreatePassthroughValue(
      const federated_language_executor::Value& value) {
    ValueId child_id = ExpectCreateInServerChild(value);
    OwnedValueId id = TFF_TRY(test_executor_->CreateValue(value));
    return IdPair{std::move(id), child_id};
  }
};

TEST_F(FederatingExecutorTest, ConstructsExecutorWithEmptyCardinalities) {
  EXPECT_THAT(tensorflow_federated::CreateFederatingExecutor(
                  mock_server_executor_, mock_client_executor_, {}),
              StatusIs(StatusCode::kNotFound));
}

TEST_F(FederatingExecutorTest, CreateMaterializeTensor) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(FederatingExecutorTest, CreateValueIntrinsic) {
  EXPECT_THAT(test_executor_->CreateValue(FederatedMapV()), IsOk());
}

TEST_F(FederatingExecutorTest,
       CreateValueNonFederatedIntrinsicForwardedToChild) {
  const federated_language_executor::Value intrinsic_pb =
      IntrinsicV("sequence_reduce");
  TFF_ASSERT_OK(test_executor_->CreateValue(intrinsic_pb));
}

TEST_F(FederatingExecutorTest, MaterializeIntrinsicFails) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateValue(FederatedMapV()));
  EXPECT_THAT(test_executor_->Materialize(id),
              StatusIs(StatusCode::kUnimplemented));
}

TEST_F(FederatingExecutorTest, CreateMaterializeEmptyStruct) {
  ExpectCreateMaterialize(StructV({}));
}

TEST_F(FederatingExecutorTest, CreateMaterializeFlatStruct) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {3.0}));
  federated_language_executor::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  auto elements = {value1_pb, value2_pb, value3_pb};
  ExpectCreateMaterialize(StructV(elements));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtServer) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ExpectCreateMaterializeInServerChild(value_pb);
  ExpectCreateMaterialize(ServerV(value_pb));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtClients) {
  std::vector<federated_language_executor::Value> values;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    values.emplace_back(value_pb);
    ExpectCreateMaterializeInClientChild(value_pb);
  }
  ExpectCreateMaterialize(ClientsV(values));
}

TEST_F(FederatingExecutorTest, CreateValueFailsWrongNumberClients) {
  EXPECT_THAT(test_executor_->CreateValue(ClientsV({})),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtClientsAllEqual) {
  federated_language::Array array_in_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  federated_language_executor::Value value_in_pb;
  value_in_pb.mutable_array()->Swap(&array_in_pb);
  ValueId child_id = ExpectCreateInClientChild(value_in_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto id, test_executor_->CreateValue(ClientsV({value_in_pb}, true)));
  federated_language::Array array_out_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  federated_language_executor::Value value_out_pb;
  value_out_pb.mutable_array()->Swap(&array_out_pb);
  ExpectMaterializeInClientChild(child_id, value_out_pb, ONCE_PER_CLIENT);
  federated_language_executor::Value clients_out =
      ClientsV(std::vector<federated_language_executor::Value>(NUM_CLIENTS,
                                                               value_out_pb),
               false);
  ExpectMaterialize(id, clients_out);
}

TEST_F(FederatingExecutorTest, CreateValueFailsMultipleAllEqualValues) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  std::vector<federated_language_executor::Value> values(NUM_CLIENTS, value_pb);
  EXPECT_THAT(test_executor_->CreateValue(ClientsV(values, true)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateMaterializeStructOfFederatedValues) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language_executor::Value value3_pb =
      StructV({ServerV(value1_pb), ServerV(value2_pb)});
  ExpectCreateMaterializeInServerChild(value1_pb);
  ExpectCreateMaterializeInServerChild(value2_pb);
  ExpectCreateMaterialize(value3_pb);
}

TEST_F(FederatingExecutorTest, CreateMaterializeStructOfMixedValues) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language_executor::Value value3_pb =
      StructV({value1_pb, ServerV(value2_pb)});
  ExpectCreateMaterializeInServerChild(value2_pb);
  ExpectCreateMaterialize(value3_pb);
}

TEST_F(FederatingExecutorTest, CreateMaterializeFederatedStruct) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value struct_pb = StructV({value_pb});
  ExpectCreateMaterializeInServerChild(struct_pb);
  ExpectCreateMaterialize(ServerV(struct_pb));
}

TEST_F(FederatingExecutorTest, CreateStructOfTensors) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto v1_id, test_executor_->CreateValue(value1_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto v2_id, test_executor_->CreateValue(value2_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateStruct({v1_id, v2_id}));
  ExpectMaterialize(id, StructV({value1_pb, value2_pb}));
}

TEST_F(FederatingExecutorTest, CreateSelection) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto s,
                           test_executor_->CreateValue(StructV({value_pb})));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, value_pb);
}

TEST_F(FederatingExecutorTest, CreateSelectionFromEmbeddedValue) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ValueId child_id = ExpectCreateInServerChild(value1_pb);
  ValueId child_selected_id =
      mock_server_executor_->ExpectCreateSelection(child_id, 0);
  ExpectMaterializeInServerChild(child_selected_id, value2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto s, test_executor_->CreateValue(value1_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, value2_pb);
}

TEST_F(FederatingExecutorTest, CreateSelectionFromFederatedValueFails) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  federated_language_executor::Value struct_pb = StructV({value_pb});
  federated_language_executor::Value fed = ServerV(struct_pb);
  ExpectCreateInServerChild(struct_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fed_id, test_executor_->CreateValue(fed));
  EXPECT_THAT(test_executor_->CreateSelection(fed_id, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateSelectionFromIntrinsicFails) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateValue(FederatedMapV()));
  EXPECT_THAT(test_executor_->CreateSelection(id, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateSelectionOutOfBoundsFails) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(StructV({})));
  EXPECT_THAT(test_executor_->CreateSelection(id, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateCallEmbeddedNoArg) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(value1_pb));
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, std::nullopt);
  ExpectMaterializeInServerChild(fn_result_child_id, value2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, std::nullopt));
  ExpectMaterialize(fn_result_id, value2_pb);
}

TEST_F(FederatingExecutorTest, CreateCallEmbeddedSingleArg) {
  federated_language::Array array_fn_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  federated_language::Array array_arg_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value_arg_pb;
  value_arg_pb.mutable_array()->Swap(&array_arg_pb);
  federated_language::Array array_result_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {3}));
  federated_language_executor::Value value_result_pb;
  value_result_pb.mutable_array()->Swap(&array_result_pb);
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(value_fn_pb));
  IdPair arg = TFF_ASSERT_OK(CreatePassthroughValue(value_arg_pb));
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, arg.child_id);
  ExpectMaterializeInServerChild(fn_result_child_id, value_result_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, arg.id));
  ExpectMaterialize(fn_result_id, value_result_pb);
}

TEST_F(FederatingExecutorTest, CreateCallEmbedsStructArg) {
  federated_language::Array array_fn_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(value_fn_pb));
  federated_language::Array array_arg_1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value_arg_1_pb;
  value_arg_1_pb.mutable_array()->Swap(&array_arg_1_pb);
  federated_language::Array array_arg_2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {3}));
  federated_language_executor::Value value_arg_2_pb;
  value_arg_2_pb.mutable_array()->Swap(&array_arg_2_pb);
  federated_language::Array array_result_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {4}));
  federated_language_executor::Value value_result_pb;
  value_result_pb.mutable_array()->Swap(&array_result_pb);
  ValueId arg_1_child_id = ExpectCreateInServerChild(value_arg_1_pb);
  ValueId arg_2_child_id = ExpectCreateInServerChild(value_arg_2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id,
      test_executor_->CreateValue(StructV({value_arg_1_pb, value_arg_2_pb})));
  ValueId arg_child_id =
      ExpectCreateStructInServerChild({arg_1_child_id, arg_2_child_id});
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, arg_child_id);

  ExpectMaterializeInServerChild(fn_result_child_id, value_result_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, arg_id));
  ExpectMaterialize(fn_result_id, value_result_pb);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueFails) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ExpectCreateInServerChild(value_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fed_id,
                           test_executor_->CreateValue(ServerV(value_pb)));
  EXPECT_THAT(test_executor_->CreateCall(fed_id, std::nullopt),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateCallStructFails) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateValue(StructV({value_pb})));
  EXPECT_THAT(test_executor_->CreateCall(struct_id, std::nullopt),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedAggregate) {
  std::vector<federated_language_executor::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    client_vals.emplace_back(value_pb);
    client_vals_child_ids.emplace_back(ExpectCreateInClientChild(value_pb));
  }
  federated_language_executor::Value value = ClientsV(client_vals);
  federated_language::Array array_zero_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"zero"}));
  federated_language_executor::Value value_zero_pb;
  value_zero_pb.mutable_array()->Swap(&array_zero_pb);
  ValueId zero_child_id = ExpectCreateInClientChild(value_zero_pb);
  federated_language::Array array_accumulate_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"accumulate"}));
  federated_language_executor::Value value_accumulate_pb;
  value_accumulate_pb.mutable_array()->Swap(&array_accumulate_pb);
  ValueId accumulate_child_id = ExpectCreateInClientChild(value_accumulate_pb);
  federated_language::Array array_merge_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"merge"}));
  federated_language_executor::Value value_merge_pb;
  value_merge_pb.mutable_array()->Swap(&array_merge_pb);
  federated_language::Array array_report_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"report"}));
  federated_language_executor::Value value_report_pb;
  value_report_pb.mutable_array()->Swap(&array_report_pb);
  ValueId report_child_id = ExpectCreateInServerChild(value_report_pb);
  federated_language_executor::Value arg =
      StructV({value, value_zero_pb, value_accumulate_pb, value_merge_pb,
               value_report_pb});
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id, test_executor_->CreateValue(arg));
  TFF_ASSERT_OK_AND_ASSIGN(auto intrinsic_id,
                           test_executor_->CreateValue(FederatedAggregateV()));
  ValueId current_child_id = zero_child_id;
  for (auto client_val_child_id : client_vals_child_ids) {
    ValueId call_arg_child_id = ExpectCreateStructInClientChild(
        {current_child_id, client_val_child_id});
    current_child_id =
        ExpectCreateCallInClientChild(accumulate_child_id, call_arg_child_id);
  }
  federated_language::Array array_client_child_result_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"result_value"}));
  federated_language_executor::Value value_client_child_result_pb;
  value_client_child_result_pb.mutable_array()->Swap(
      &array_client_child_result_pb);
  ExpectMaterializeInClientChild(current_child_id,
                                 value_client_child_result_pb);
  ValueId result_in_server_id =
      ExpectCreateInServerChild(value_client_child_result_pb);
  ValueId result_child_id =
      ExpectCreateCallInServerChild(report_child_id, result_in_server_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(intrinsic_id, arg_id));
  federated_language::Array array_child_result_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"result"}));
  federated_language_executor::Value value_child_result_pb;
  value_child_result_pb.mutable_array()->Swap(&array_child_result_pb);
  ExpectMaterializeInServerChild(result_child_id, value_child_result_pb);
  ExpectMaterialize(result_id, ServerV(value_child_result_pb));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedBroadcast) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  ValueId tensor_id = ExpectCreateInServerChild(value1_pb);
  ExpectMaterializeInServerChild(tensor_id, value1_pb);
  ValueId client_id = ExpectCreateInClientChild(value1_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(value1_pb)));
  TFF_ASSERT_OK_AND_ASSIGN(auto broadcast_id,
                           test_executor_->CreateValue(FederatedBroadcastV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto clients_id,
                           test_executor_->CreateCall(broadcast_id, server_id));
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ExpectMaterializeInClientChild(client_id, value2_pb, ONCE_PER_CLIENT);
  ExpectMaterialize(clients_id,
                    ClientsV(std::vector<federated_language_executor::Value>(
                        NUM_CLIENTS, value2_pb)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAtClients) {
  std::vector<federated_language_executor::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    client_vals.emplace_back(value_pb);
    client_vals_child_ids.emplace_back(ExpectCreateInClientChild(value_pb));
  }
  federated_language_executor::Value value = ClientsV(client_vals);
  TFF_ASSERT_OK_AND_ASSIGN(auto input_id,
                           test_executor_->CreateValue(ClientsV(client_vals)));
  federated_language::Array array_fn_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"fn"}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  auto fn_id = ExpectCreateInClientChild(value_fn_pb);
  for (int i = 0; i < NUM_CLIENTS; i++) {
    ValueId result_child_id =
        ExpectCreateCallInClientChild(fn_id, client_vals_child_ids[i]);
    ExpectMaterializeInClientChild(result_child_id, client_vals[i]);
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_at_fed_exec_id,
                           test_executor_->CreateValue(value_fn_pb));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id, test_executor_->CreateStruct({fn_at_fed_exec_id, input_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  ExpectMaterialize(result_id, value);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAllEqualAtClients) {
  std::vector<federated_language_executor::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    client_vals.emplace_back(value_pb);
    client_vals_child_ids.emplace_back(ExpectCreateInClientChild(value_pb));
  }
  federated_language_executor::Value value = ClientsV(client_vals);
  TFF_ASSERT_OK_AND_ASSIGN(auto input_id,
                           test_executor_->CreateValue(ClientsV(client_vals)));
  federated_language::Array array_fn_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"fn"}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  auto fn_id = ExpectCreateInClientChild(value_fn_pb);
  for (int i = 0; i < NUM_CLIENTS; i++) {
    ValueId result_child_id =
        ExpectCreateCallInClientChild(fn_id, client_vals_child_ids[i]);
    ExpectMaterializeInClientChild(result_child_id, client_vals[i]);
  }
  TFF_ASSERT_OK_AND_ASSIGN(
      auto map_id, test_executor_->CreateValue(FederatedMapAllEqualV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_at_fed_exec_id,
                           test_executor_->CreateValue(value_fn_pb));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id, test_executor_->CreateStruct({fn_at_fed_exec_id, input_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  ExpectMaterialize(result_id, value);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAtServer) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  ValueId tensor_child_id = ExpectCreateInServerChild(value1_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(value1_pb)));
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(value2_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateStruct({fn.id, server_id}));
  ValueId result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, tensor_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {3}));
  federated_language_executor::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  ExpectMaterializeInServerChild(result_child_id, value3_pb);
  ExpectMaterialize(result_id, ServerV(value3_pb));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedEvalAtClients) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn, test_executor_->CreateValue(value1_pb));
  auto fn_client_id = ExpectCreateInClientChild(value1_pb);
  ValueId result_child_id = ExpectCreateCallInClientChild(
      fn_client_id, std::nullopt, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtClientsV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn));
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {3}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ExpectMaterializeInClientChild(result_child_id, value2_pb, ONCE_PER_CLIENT);
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<federated_language_executor::Value>(
                        NUM_CLIENTS, value2_pb)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedEvalAtServer) {
  federated_language::Array array_fn_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  ValueId fn_child_id = ExpectCreateInServerChild(value_fn_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id,
                           test_executor_->CreateValue(value_fn_pb));
  ValueId result_child_id =
      ExpectCreateCallInServerChild(fn_child_id, std::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn_id));
  federated_language::Array array_result_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {3}));
  federated_language_executor::Value value_result_pb;
  value_result_pb.mutable_array()->Swap(&array_result_pb);
  ExpectMaterializeInServerChild(result_child_id, value_result_pb);
  ExpectMaterialize(result_id, ServerV(value_result_pb));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedSelectUniqueKeyPerClient) {
  OwnedValueId select_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(FederatedSelectV()));
  federated_language::Array array_fn_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"fn"}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto select_fn,
                           test_executor_->CreateValue(value_fn_pb));
  federated_language::Array array_max_key_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"max_key"}));
  federated_language_executor::Value value_max_key_pb;
  value_max_key_pb.mutable_array()->Swap(&array_max_key_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto max_key,
                           test_executor_->CreateValue(value_max_key_pb));
  federated_language::Array array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"value"}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ValueId server_value_child_id = ExpectCreateInServerChild(value_pb);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(value_pb)));
  ValueId args_into_sequence_id =
      ExpectCreateInServerChild(ArgsIntoSequenceV());
  std::vector<federated_language_executor::Value> keys_pbs;
  std::vector<federated_language_executor::Value> dataset_pbs;
  std::vector<ValueId> dataset_ids;

  ValueId select_fn_server_id = ExpectCreateInServerChild(value_fn_pb);
  for (int32_t i = 0; i < NUM_CLIENTS; i++) {
    federated_language::Array array_keys_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({1}), {i}));
    federated_language_executor::Value keys_for_client_pb;
    keys_for_client_pb.mutable_array()->Swap(&array_keys_pb);
    keys_pbs.push_back(keys_for_client_pb);
    ExpectCreateMaterializeInClientChild(keys_for_client_pb);
    federated_language::Array array_key_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i}));
    federated_language_executor::Value value_key_pb;
    value_key_pb.mutable_array()->Swap(&array_key_pb);
    ValueId key_id = ExpectCreateInServerChild(value_key_pb);

    ValueId select_fn_args_id =
        ExpectCreateStructInServerChild({server_value_child_id, key_id});
    ValueId slice_id =
        ExpectCreateCallInServerChild(select_fn_server_id, select_fn_args_id);
    ValueId slices_id = ExpectCreateStructInServerChild({slice_id});
    ValueId dataset_id =
        ExpectCreateCallInServerChild(args_into_sequence_id, slices_id);

    federated_language_executor::Value dataset_pb = SequenceV(i, i + 1, 1);
    dataset_ids.push_back(dataset_id);
    dataset_pbs.push_back(dataset_pb);
    ExpectMaterializeInServerChild(dataset_id, dataset_pb);
    ExpectCreateMaterializeInClientChild(dataset_pb);
  }
  OwnedValueId keys_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ClientsV(keys_pbs)));
  OwnedValueId select_args_id = TFF_ASSERT_OK(test_executor_->CreateStruct(
      {keys_id, max_key, server_value_id, select_fn}));
  OwnedValueId result_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(select_id, select_args_id));
  ExpectMaterialize(result_id, ClientsV(dataset_pbs));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedSelectAllClientsSameKeys) {
  OwnedValueId select_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(FederatedSelectV()));
  federated_language::Array array_fn_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"fn"}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  IdPair select_fn = TFF_ASSERT_OK(CreatePassthroughValue(value_fn_pb));
  federated_language::Array array_max_key_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"max_key"}));
  federated_language_executor::Value value_max_key_pb;
  value_max_key_pb.mutable_array()->Swap(&array_max_key_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto max_key,
                           test_executor_->CreateValue(value_max_key_pb));
  federated_language::Array array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"value"}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ValueId server_value_child_id = ExpectCreateInServerChild(value_pb);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(value_pb)));
  ValueId args_into_sequence_id =
      ExpectCreateInServerChild(ArgsIntoSequenceV());
  auto keys = {1, 2, 3};
  federated_language::Array array_keys_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({3}), keys));
  federated_language_executor::Value keys_pb;
  keys_pb.mutable_array()->Swap(&array_keys_pb);
  std::vector<ValueId> slice_child_ids;
  // Every unique key should only have its slice created once (not once per
  // client).
  for (int32_t key : keys) {
    federated_language::Array array_key_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {key}));
    federated_language_executor::Value value_key_pb;
    value_key_pb.mutable_array()->Swap(&array_key_pb);
    ValueId key_id = ExpectCreateInServerChild(value_key_pb);
    ValueId select_fn_args_id =
        ExpectCreateStructInServerChild({server_value_child_id, key_id});
    ValueId slice_id =
        ExpectCreateCallInServerChild(select_fn.child_id, select_fn_args_id);
    slice_child_ids.push_back(slice_id);
  }
  // However, each client should still create its own dataset from the slices:
  // we don't yet bother to optimize for the case where clients have the exact
  // same list of keys, as that should be less frequent in practice.
  ExpectCreateMaterializeInClientChild(keys_pb, ONCE_PER_CLIENT);
  ValueId slices_id =
      ExpectCreateStructInServerChild(slice_child_ids, ONCE_PER_CLIENT);
  ValueId dataset_id = ExpectCreateCallInServerChild(
      args_into_sequence_id, slices_id, ONCE_PER_CLIENT);
  federated_language_executor::Value dataset_pb = SequenceV(0, 10, 2);
  ExpectMaterializeInServerChild(dataset_id, dataset_pb, ONCE_PER_CLIENT);
  ExpectCreateMaterializeInClientChild(dataset_pb, ONCE_PER_CLIENT);
  std::vector<federated_language_executor::Value> keys_pbs;
  keys_pbs.resize(NUM_CLIENTS, keys_pb);
  std::vector<federated_language_executor::Value> dataset_pbs;
  dataset_pbs.resize(NUM_CLIENTS, dataset_pb);
  OwnedValueId keys_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ClientsV(keys_pbs)));
  OwnedValueId select_args_id = TFF_ASSERT_OK(test_executor_->CreateStruct(
      {keys_id, max_key, server_value_id, select_fn.id}));
  OwnedValueId result_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(select_id, select_args_id));

  ExpectMaterialize(result_id, ClientsV(dataset_pbs));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedSelectNonInt32KeysFails) {
  OwnedValueId select_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(FederatedSelectV()));
  federated_language::Array array_fn_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"fn"}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  IdPair select_fn = TFF_ASSERT_OK(CreatePassthroughValue(value_fn_pb));
  federated_language::Array array_max_key_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"max_key"}));
  federated_language_executor::Value value_max_key_pb;
  value_max_key_pb.mutable_array()->Swap(&array_max_key_pb);
  OwnedValueId max_key =
      TFF_ASSERT_OK(test_executor_->CreateValue(value_max_key_pb));
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ExpectCreateInServerChild(value_pb);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(value_pb)));

  federated_language::Array array_keys_pb;
  array_keys_pb.set_dtype(federated_language::DataType::DT_UINT8);
  array_keys_pb.mutable_shape()->mutable_dim()->Add(1);
  federated_language_executor::Value keys_pb;
  *keys_pb.mutable_array() = array_keys_pb;
  // The child `keys_pb` value is only created once due to the ALL_EQUALS bit,
  // and then is only materialized once after which the operation fails.
  ExpectCreateMaterializeInClientChild(keys_pb, ONCE);
  OwnedValueId keys_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ClientsV({keys_pb}, true)));
  OwnedValueId select_args_id = TFF_ASSERT_OK(test_executor_->CreateStruct(
      {keys_id, max_key, server_value_id, select_fn.id}));
  ASSERT_THAT(test_executor_->CreateCall(select_id, select_args_id),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Expected int32_t key")));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedSelectNonRankOneKeysFails) {
  OwnedValueId select_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(FederatedSelectV()));
  federated_language::Array array_fn_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"fn"}));
  federated_language_executor::Value value_fn_pb;
  value_fn_pb.mutable_array()->Swap(&array_fn_pb);
  IdPair select_fn = TFF_ASSERT_OK(CreatePassthroughValue(value_fn_pb));
  federated_language::Array array_max_key_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"max_key"}));
  federated_language_executor::Value value_max_key_pb;
  value_max_key_pb.mutable_array()->Swap(&array_max_key_pb);
  OwnedValueId max_key =
      TFF_ASSERT_OK(test_executor_->CreateValue(value_max_key_pb));
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ExpectCreateInServerChild(value_pb);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(value_pb)));

  federated_language::Array array_keys_pb;
  array_keys_pb.set_dtype(federated_language::DataType::DT_INT32);
  array_keys_pb.mutable_shape()->mutable_dim()->Add(2);
  array_keys_pb.mutable_shape()->mutable_dim()->Add(2);
  federated_language_executor::Value keys_pb;
  *keys_pb.mutable_array() = array_keys_pb;
  // The child `keys_pb` value is only created once due to the ALL_EQUALS bit,
  // and then is only materialized once after which the operation fails.
  ExpectCreateMaterializeInClientChild(keys_pb, ONCE);
  OwnedValueId keys_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ClientsV({keys_pb}, true)));
  OwnedValueId select_args_id = TFF_ASSERT_OK(test_executor_->CreateStruct(
      {keys_id, max_key, server_value_id, select_fn.id}));
  ASSERT_THAT(test_executor_->CreateCall(select_id, select_args_id),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Expected key tensor to be rank one")));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueAtClients) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  auto tensor = TFF_ASSERT_OK(test_executor_->CreateValue(value_pb));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtClientsV()));
  auto tensor_id = ExpectCreateInClientChild(value_pb);
  ExpectMaterializeInClientChild(tensor_id, value_pb, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor));
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<federated_language_executor::Value>(
                        NUM_CLIENTS, value_pb)));
}

TEST_F(FederatingExecutorTest,
       CreateCallFederatedValueAtClientsFromEmbeddedValue) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(value_pb));
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, std::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, std::nullopt));

  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtClientsV()));
  auto result_tensor_id = ExpectCreateInClientChild(value_pb);
  ExpectMaterializeInServerChild(fn_result_child_id, value_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto result_id, test_executor_->CreateCall(fed_val_id, fn_result_id));
  ExpectMaterializeInClientChild(result_tensor_id, value_pb, ONCE_PER_CLIENT);
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<federated_language_executor::Value>(
                        NUM_CLIENTS, value_pb)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueAtServer) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ValueId tensor_child_id = ExpectCreateInServerChild(value_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto tensor_id,
                           test_executor_->CreateValue(value_pb));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor_id));
  ExpectMaterializeInServerChild(tensor_child_id, value_pb);
  ExpectMaterialize(result_id, ServerV(value_pb));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtClientsFlat) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ValueId v1_child_id = ExpectCreateInClientChild(value1_pb);
  ValueId v2_child_id = ExpectCreateInClientChild(value2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id,
      test_executor_->CreateValue(
          StructV({ClientsV({value1_pb}, true), ClientsV({value2_pb}, true)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtClientsV()));
  ValueId struct_child_id = ExpectCreateStructInClientChild(
      {v1_child_id, v2_child_id}, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInClientChild(
      struct_child_id, StructV({value1_pb, value2_pb}), ONCE_PER_CLIENT);
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<federated_language_executor::Value>(
                        NUM_CLIENTS, StructV({value1_pb, value2_pb}))));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtClientsNested) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ValueId v1_child_id = ExpectCreateInClientChild(value1_pb);
  ValueId v2_child_id = ExpectCreateInClientChild(value2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id, test_executor_->CreateValue(
                     StructV({ClientsV({value1_pb}, true),
                              StructV({ClientsV({value2_pb}, true)})})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtClientsV()));
  ValueId v2_struct_child_id =
      ExpectCreateStructInClientChild({v2_child_id}, ONCE_PER_CLIENT);
  ValueId struct_child_id = ExpectCreateStructInClientChild(
      {v1_child_id, v2_struct_child_id}, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInClientChild(struct_child_id,
                                 StructV({value1_pb, StructV({value2_pb})}),
                                 ONCE_PER_CLIENT);
  ExpectMaterialize(
      result_id, ClientsV(std::vector<federated_language_executor::Value>(
                     NUM_CLIENTS, StructV({value1_pb, StructV({value2_pb})}))));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtServerFlat) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ValueId v1_child_id = ExpectCreateInServerChild(value1_pb);
  ValueId v2_child_id = ExpectCreateInServerChild(value2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id, test_executor_->CreateValue(
                     StructV({ServerV(value1_pb), ServerV(value2_pb)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  ValueId struct_child_id =
      ExpectCreateStructInServerChild({v1_child_id, v2_child_id});
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInServerChild(struct_child_id,
                                 StructV({value1_pb, value2_pb}));
  ExpectMaterialize(result_id, ServerV(StructV({value1_pb, value2_pb})));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtServerNested) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ValueId v1_child_id = ExpectCreateInServerChild(value1_pb);
  ValueId v2_child_id = ExpectCreateInServerChild(value2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id, test_executor_->CreateValue(StructV(
                     {ServerV(value1_pb), StructV({ServerV(value2_pb)})})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  ValueId v2_struct_child_id = ExpectCreateStructInServerChild({v2_child_id});
  ValueId struct_child_id =
      ExpectCreateStructInServerChild({v1_child_id, v2_struct_child_id});
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInServerChild(struct_child_id,
                                 StructV({value1_pb, StructV({value2_pb})}));
  ExpectMaterialize(result_id,
                    ServerV(StructV({value1_pb, StructV({value2_pb})})));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipMixedPlacementsFails) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  federated_language_executor::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  ExpectCreateInServerChild(value1_pb);
  ExpectCreateInClientChild(value2_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id, test_executor_->CreateValue(StructV(
                     {ServerV(value1_pb), ClientsV({value2_pb}, true)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  EXPECT_THAT(test_executor_->CreateCall(zip_id, v_id),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace

}  // namespace tensorflow_federated
