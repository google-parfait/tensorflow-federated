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
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_test_base.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::tensorflow_federated::testing::ClientsV;
using ::tensorflow_federated::testing::IntrinsicV;
using ::tensorflow_federated::testing::SequenceV;
using ::tensorflow_federated::testing::ServerV;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::TensorV;
using ::tensorflow_federated::testing::TensorVFromIntList;
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

  ValueId ExpectCreateInServerChild(const v0::Value& expected,
                                    Cardinality repeatedly = ONCE) {
    return mock_server_executor_->ExpectCreateValue(expected, repeatedly);
  }
  ValueId ExpectCreateInClientChild(const v0::Value& expected,
                                    Cardinality repeatedly = ONCE) {
    return mock_client_executor_->ExpectCreateValue(expected, repeatedly);
  }

  void ExpectMaterializeInServerChild(ValueId id, v0::Value to_return,
                                      Cardinality repeatedly = ONCE) {
    mock_server_executor_->ExpectMaterialize(id, std::move(to_return),
                                             repeatedly);
  }

  void ExpectMaterializeInClientChild(ValueId id, v0::Value to_return,
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
  void ExpectCreateMaterializeInServerChild(v0::Value value,
                                            Cardinality repeatedly = ONCE) {
    ValueId id = ExpectCreateInServerChild(value, repeatedly);
    ExpectMaterializeInServerChild(id, value, repeatedly);
  }
  void ExpectCreateMaterializeInClientChild(v0::Value value,
                                            Cardinality repeatedly = ONCE) {
    ValueId id = ExpectCreateInClientChild(value, repeatedly);
    ExpectMaterializeInClientChild(id, value, repeatedly);
  }

  absl::StatusOr<IdPair> CreatePassthroughValue(const v0::Value& value) {
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
  v0::Value value = TensorV(1.0);
  ExpectCreateMaterialize(value);
}

TEST_F(FederatingExecutorTest, CreateValueIntrinsic) {
  EXPECT_THAT(test_executor_->CreateValue(FederatedMapV()), IsOk());
}

TEST_F(FederatingExecutorTest,
       CreateValueNonFederatedIntrinsicForwardedToChild) {
  const v0::Value intrinsic_pb = IntrinsicV("sequence_reduce");
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
  auto elements = {TensorV(1.0), TensorV(3.5), TensorV(2.0)};
  ExpectCreateMaterialize(StructV(elements));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtServer) {
  v0::Value tensor_pb = TensorV(1.0);
  ExpectCreateMaterializeInServerChild(tensor_pb);
  ExpectCreateMaterialize(ServerV(tensor_pb));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtClients) {
  std::vector<v0::Value> values;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    v0::Value tensor_pb = TensorV(i);
    values.emplace_back(tensor_pb);
    ExpectCreateMaterializeInClientChild(tensor_pb);
  }
  ExpectCreateMaterialize(ClientsV(values));
}

TEST_F(FederatingExecutorTest, CreateValueFailsWrongNumberClients) {
  EXPECT_THAT(test_executor_->CreateValue(ClientsV({})),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtClientsAllEqual) {
  v0::Value tensor_in = TensorV(1.0);
  ValueId child_id = ExpectCreateInClientChild(tensor_in);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto id, test_executor_->CreateValue(ClientsV({tensor_in}, true)));
  v0::Value tensor_out = TensorV(2.0);
  ExpectMaterializeInClientChild(child_id, tensor_out, ONCE_PER_CLIENT);
  v0::Value clients_out =
      ClientsV(std::vector<v0::Value>(NUM_CLIENTS, tensor_out), false);
  ExpectMaterialize(id, clients_out);
}

TEST_F(FederatingExecutorTest, CreateValueFailsMultipleAllEqualValues) {
  std::vector<v0::Value> values(NUM_CLIENTS, TensorV(1.0));
  EXPECT_THAT(test_executor_->CreateValue(ClientsV(values, true)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateMaterializeStructOfFederatedValues) {
  v0::Value server_v1 = TensorV(6);
  v0::Value server_v2 = TensorV(7);
  v0::Value value_pb = StructV({ServerV(server_v1), ServerV(server_v2)});
  ExpectCreateMaterializeInServerChild(server_v1);
  ExpectCreateMaterializeInServerChild(server_v2);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(FederatingExecutorTest, CreateMaterializeStructOfMixedValues) {
  v0::Value unplaced = TensorV(6);
  v0::Value server = TensorV(7);
  v0::Value value_pb = StructV({unplaced, ServerV(server)});
  ExpectCreateMaterializeInServerChild(server);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(FederatingExecutorTest, CreateMaterializeFederatedStruct) {
  v0::Value struct_pb = StructV({TensorV(5.0)});
  ExpectCreateMaterializeInServerChild(struct_pb);
  ExpectCreateMaterialize(ServerV(struct_pb));
}

TEST_F(FederatingExecutorTest, CreateStructOfTensors) {
  v0::Value tensor_v1 = TensorV(5);
  v0::Value tensor_v2 = TensorV(6);
  TFF_ASSERT_OK_AND_ASSIGN(auto v1_id, test_executor_->CreateValue(tensor_v1));
  TFF_ASSERT_OK_AND_ASSIGN(auto v2_id, test_executor_->CreateValue(tensor_v2));
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateStruct({v1_id, v2_id}));
  ExpectMaterialize(id, StructV({tensor_v1, tensor_v2}));
}

TEST_F(FederatingExecutorTest, CreateSelection) {
  v0::Value tensor = TensorV(5);
  TFF_ASSERT_OK_AND_ASSIGN(auto s,
                           test_executor_->CreateValue(StructV({tensor})));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, tensor);
}

TEST_F(FederatingExecutorTest, CreateSelectionFromEmbeddedValue) {
  v0::Value pretend_struct = TensorV(5);
  v0::Value selected_tensor = TensorV(24);
  ValueId child_id = ExpectCreateInServerChild(pretend_struct);
  ValueId child_selected_id =
      mock_server_executor_->ExpectCreateSelection(child_id, 0);
  ExpectMaterializeInServerChild(child_selected_id, selected_tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto s, test_executor_->CreateValue(pretend_struct));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, selected_tensor);
}

TEST_F(FederatingExecutorTest, CreateSelectionFromFederatedValueFails) {
  v0::Value struct_pb = StructV({TensorV(6)});
  v0::Value fed = ServerV(struct_pb);
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
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(TensorV(5)));
  v0::Value result = TensorV(22);
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, std::nullopt);
  ExpectMaterializeInServerChild(fn_result_child_id, result);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, std::nullopt));

  ExpectMaterialize(fn_result_id, result);
}

TEST_F(FederatingExecutorTest, CreateCallEmbeddedSingleArg) {
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(TensorV(5)));
  IdPair arg = TFF_ASSERT_OK(CreatePassthroughValue(TensorV(6)));
  v0::Value result = TensorV(22);
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, arg.child_id);
  ExpectMaterializeInServerChild(fn_result_child_id, result);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, arg.id));
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(FederatingExecutorTest, CreateCallEmbedsStructArg) {
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(TensorV(5)));
  v0::Value arg_1 = TensorV(6);
  v0::Value arg_2 = TensorV(7);
  v0::Value result = TensorV(22);
  ValueId arg_1_child_id = ExpectCreateInServerChild(arg_1);
  ValueId arg_2_child_id = ExpectCreateInServerChild(arg_2);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id, test_executor_->CreateValue(StructV({arg_1, arg_2})));
  ValueId arg_child_id =
      ExpectCreateStructInServerChild({arg_1_child_id, arg_2_child_id});
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, arg_child_id);

  ExpectMaterializeInServerChild(fn_result_child_id, result);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, arg_id));
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueFails) {
  v0::Value tensor = TensorV(1);
  ExpectCreateInServerChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto fed_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  EXPECT_THAT(test_executor_->CreateCall(fed_id, std::nullopt),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateCallStructFails) {
  v0::Value tensor = TensorV(1);
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateValue(StructV({tensor})));
  EXPECT_THAT(test_executor_->CreateCall(struct_id, std::nullopt),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedAggregate) {
  std::vector<v0::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    client_vals.emplace_back(TensorV(i));
    client_vals_child_ids.emplace_back(ExpectCreateInClientChild(TensorV(i)));
  }
  v0::Value value = ClientsV(client_vals);
  v0::Value zero = TensorV("zero");
  ValueId zero_child_id = ExpectCreateInClientChild(zero);
  v0::Value accumulate = TensorV("accumulate");
  ValueId accumulate_child_id = ExpectCreateInClientChild(accumulate);
  v0::Value merge = TensorV("merge");
  // /* unused ValueId merge_child_id =*/ExpectCreateInServerChild(merge);
  v0::Value report = TensorV("report");
  ValueId report_child_id = ExpectCreateInServerChild(report);
  v0::Value arg = StructV({value, zero, accumulate, merge, report});
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
  v0::Value client_child_result = TensorV("result_val");
  ExpectMaterializeInClientChild(current_child_id, client_child_result);
  ValueId result_in_server_id = ExpectCreateInServerChild(client_child_result);
  ValueId result_child_id =
      ExpectCreateCallInServerChild(report_child_id, result_in_server_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(intrinsic_id, arg_id));
  v0::Value child_result = TensorV("result");
  ExpectMaterializeInServerChild(result_child_id, child_result);
  ExpectMaterialize(result_id, ServerV(child_result));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedBroadcast) {
  v0::Value tensor = TensorV(1);
  ValueId tensor_id = ExpectCreateInServerChild(tensor);
  ExpectMaterializeInServerChild(tensor_id, tensor);
  ValueId client_id = ExpectCreateInClientChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  TFF_ASSERT_OK_AND_ASSIGN(auto broadcast_id,
                           test_executor_->CreateValue(FederatedBroadcastV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto clients_id,
                           test_executor_->CreateCall(broadcast_id, server_id));
  v0::Value tensor_out = TensorV(4);
  ExpectMaterializeInClientChild(client_id, tensor_out, ONCE_PER_CLIENT);
  ExpectMaterialize(clients_id,
                    ClientsV(std::vector<v0::Value>(NUM_CLIENTS, tensor_out)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAtClients) {
  std::vector<v0::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    client_vals.emplace_back(TensorV(i));
    client_vals_child_ids.emplace_back(ExpectCreateInClientChild(TensorV(i)));
  }
  v0::Value value = ClientsV(client_vals);
  TFF_ASSERT_OK_AND_ASSIGN(auto input_id,
                           test_executor_->CreateValue(ClientsV(client_vals)));
  v0::Value function = TensorV(2);
  auto fn_id = ExpectCreateInClientChild(function);
  for (int i = 0; i < NUM_CLIENTS; i++) {
    ValueId result_child_id =
        ExpectCreateCallInClientChild(fn_id, client_vals_child_ids[i]);
    ExpectMaterializeInClientChild(result_child_id, client_vals[i]);
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_at_fed_exec_id,
                           test_executor_->CreateValue(function));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id, test_executor_->CreateStruct({fn_at_fed_exec_id, input_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  ExpectMaterialize(result_id, value);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAllEqualAtClients) {
  std::vector<v0::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    client_vals.emplace_back(TensorV(i));
    client_vals_child_ids.emplace_back(ExpectCreateInClientChild(TensorV(i)));
  }
  v0::Value value = ClientsV(client_vals);
  TFF_ASSERT_OK_AND_ASSIGN(auto input_id,
                           test_executor_->CreateValue(ClientsV(client_vals)));
  v0::Value function = TensorV(2);
  auto fn_id = ExpectCreateInClientChild(function);
  for (int i = 0; i < NUM_CLIENTS; i++) {
    ValueId result_child_id =
        ExpectCreateCallInClientChild(fn_id, client_vals_child_ids[i]);
    ExpectMaterializeInClientChild(result_child_id, client_vals[i]);
  }
  TFF_ASSERT_OK_AND_ASSIGN(
      auto map_id, test_executor_->CreateValue(FederatedMapAllEqualV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_at_fed_exec_id,
                           test_executor_->CreateValue(function));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id, test_executor_->CreateStruct({fn_at_fed_exec_id, input_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  ExpectMaterialize(result_id, value);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAtServer) {
  v0::Value tensor = TensorV(23);
  ValueId tensor_child_id = ExpectCreateInServerChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(TensorV(44)));
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateStruct({fn.id, server_id}));
  ValueId result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, tensor_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  v0::Value output_tensor = TensorV(89);
  ExpectMaterializeInServerChild(result_child_id, output_tensor);
  ExpectMaterialize(result_id, ServerV(output_tensor));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedEvalAtClients) {
  TFF_ASSERT_OK_AND_ASSIGN(auto fn, test_executor_->CreateValue(TensorV(1)));
  auto fn_client_id = ExpectCreateInClientChild(TensorV(1));
  ValueId result_child_id = ExpectCreateCallInClientChild(
      fn_client_id, std::nullopt, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtClientsV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn));
  v0::Value result_tensor = TensorV(3);
  ExpectMaterializeInClientChild(result_child_id, result_tensor,
                                 ONCE_PER_CLIENT);
  ExpectMaterialize(
      result_id, ClientsV(std::vector<v0::Value>(NUM_CLIENTS, result_tensor)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedEvalAtServer) {
  v0::Value fn = TensorV(1);
  ValueId fn_child_id = ExpectCreateInServerChild(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  ValueId result_child_id =
      ExpectCreateCallInServerChild(fn_child_id, std::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn_id));
  v0::Value result_tensor = TensorV(3);
  ExpectMaterializeInServerChild(result_child_id, result_tensor);
  ExpectMaterialize(result_id, ServerV(result_tensor));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedSelectUniqueKeyPerClient) {
  OwnedValueId select_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(FederatedSelectV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto select_fn,
                           test_executor_->CreateValue(TensorV("select_fn")));
  TFF_ASSERT_OK_AND_ASSIGN(auto max_key,
                           test_executor_->CreateValue(TensorV("max_key")));
  v0::Value server_value = TensorV("server_value");
  ValueId server_value_child_id = ExpectCreateInServerChild(server_value);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(server_value)));
  ValueId args_into_sequence_id =
      ExpectCreateInServerChild(ArgsIntoSequenceV());
  std::vector<v0::Value> keys_pbs;
  std::vector<v0::Value> dataset_pbs;
  std::vector<ValueId> dataset_ids;

  ValueId select_fn_server_id = ExpectCreateInServerChild(TensorV("select_fn"));
  for (int32_t i = 0; i < NUM_CLIENTS; i++) {
    v0::Value keys_for_client_pb = TensorVFromIntList({i});
    keys_pbs.push_back(keys_for_client_pb);
    ExpectCreateMaterializeInClientChild(keys_for_client_pb);
    ValueId key_id = ExpectCreateInServerChild(TensorV(i));
    ValueId select_fn_args_id =
        ExpectCreateStructInServerChild({server_value_child_id, key_id});
    ValueId slice_id =
        ExpectCreateCallInServerChild(select_fn_server_id, select_fn_args_id);
    ValueId slices_id = ExpectCreateStructInServerChild({slice_id});
    ValueId dataset_id =
        ExpectCreateCallInServerChild(args_into_sequence_id, slices_id);

    v0::Value dataset_pb = SequenceV(i, i + 1, 1);
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
  IdPair select_fn =
      TFF_ASSERT_OK(CreatePassthroughValue(TensorV("select_fn")));
  TFF_ASSERT_OK_AND_ASSIGN(auto max_key,
                           test_executor_->CreateValue(TensorV("max_key")));
  v0::Value server_value = TensorV("server_val");
  ValueId server_value_child_id = ExpectCreateInServerChild(server_value);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(server_value)));
  ValueId args_into_sequence_id =
      ExpectCreateInServerChild(ArgsIntoSequenceV());
  std::vector<int32_t> keys({1, 2, 3});
  v0::Value keys_pb = TensorVFromIntList(keys);
  std::vector<ValueId> slice_child_ids;
  // Every unique key should only have its slice created once (not once per
  // client).
  for (int32_t key : keys) {
    ValueId key_id = ExpectCreateInServerChild(TensorV(key));
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
  v0::Value dataset_pb = SequenceV(0, 10, 2);
  ExpectMaterializeInServerChild(dataset_id, dataset_pb, ONCE_PER_CLIENT);
  ExpectCreateMaterializeInClientChild(dataset_pb, ONCE_PER_CLIENT);
  std::vector<v0::Value> keys_pbs;
  keys_pbs.resize(NUM_CLIENTS, keys_pb);
  std::vector<v0::Value> dataset_pbs;
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
  IdPair select_fn =
      TFF_ASSERT_OK(CreatePassthroughValue(TensorV("select_fn")));
  OwnedValueId max_key =
      TFF_ASSERT_OK(test_executor_->CreateValue(TensorV("max_key")));
  v0::Value server_value = TensorV("server_val");
  ExpectCreateInServerChild(server_value);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(server_value)));

  tensorflow::TensorShape keys_shape({1});
  tensorflow::Tensor keys_tensor(tensorflow::DT_UINT8, keys_shape);
  keys_tensor.flat<uint8_t>()(0) = 0;
  v0::Value keys_pb = TensorV(keys_tensor);
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
  IdPair select_fn =
      TFF_ASSERT_OK(CreatePassthroughValue(TensorV("select_fn")));
  OwnedValueId max_key =
      TFF_ASSERT_OK(test_executor_->CreateValue(TensorV("max_key")));
  v0::Value server_value = TensorV("server_val");
  ExpectCreateInServerChild(server_value);
  OwnedValueId server_value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(ServerV(server_value)));

  tensorflow::TensorShape keys_shape({2, 2});
  const size_t num_keys = 2 * 2;
  tensorflow::Tensor keys_tensor(tensorflow::DT_INT32, keys_shape);
  auto flat_keys = keys_tensor.flat<int32_t>();
  for (size_t i = 0; i < num_keys; i++) {
    flat_keys(i) = 0;
  }
  v0::Value keys_pb = TensorV(keys_tensor);
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
  v0::Value tensor_pb = TensorV(1);
  auto tensor = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_pb));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtClientsV()));
  auto tensor_id = ExpectCreateInClientChild(tensor_pb);
  ExpectMaterializeInClientChild(tensor_id, tensor_pb, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor));
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<v0::Value>(NUM_CLIENTS, tensor_pb)));
}

TEST_F(FederatingExecutorTest,
       CreateCallFederatedValueAtClientsFromEmbeddedValue) {
  v0::Value tensor_pb = TensorV(5);
  IdPair fn = TFF_ASSERT_OK(CreatePassthroughValue(tensor_pb));
  ValueId fn_result_child_id =
      ExpectCreateCallInServerChild(fn.child_id, std::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn.id, std::nullopt));

  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtClientsV()));
  auto result_tensor_id = ExpectCreateInClientChild(tensor_pb);
  ExpectMaterializeInServerChild(fn_result_child_id, tensor_pb);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto result_id, test_executor_->CreateCall(fed_val_id, fn_result_id));
  ExpectMaterializeInClientChild(result_tensor_id, tensor_pb, ONCE_PER_CLIENT);
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<v0::Value>(NUM_CLIENTS, tensor_pb)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueAtServer) {
  v0::Value tensor = TensorV(1);
  ValueId tensor_child_id = ExpectCreateInServerChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto tensor_id, test_executor_->CreateValue(tensor));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor_id));
  ExpectMaterializeInServerChild(tensor_child_id, tensor);
  ExpectMaterialize(result_id, ServerV(tensor));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtClientsFlat) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ValueId v1_child_id = ExpectCreateInClientChild(v1);
  ValueId v2_child_id = ExpectCreateInClientChild(v2);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id, test_executor_->CreateValue(
                     StructV({ClientsV({v1}, true), ClientsV({v2}, true)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtClientsV()));
  ValueId struct_child_id = ExpectCreateStructInClientChild(
      {v1_child_id, v2_child_id}, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInClientChild(struct_child_id, StructV({v1, v2}),
                                 ONCE_PER_CLIENT);
  ExpectMaterialize(result_id, ClientsV(std::vector<v0::Value>(
                                   NUM_CLIENTS, StructV({v1, v2}))));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtClientsNested) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ValueId v1_child_id = ExpectCreateInClientChild(v1);
  ValueId v2_child_id = ExpectCreateInClientChild(v2);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id, test_executor_->CreateValue(StructV(
                     {ClientsV({v1}, true), StructV({ClientsV({v2}, true)})})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtClientsV()));
  ValueId v2_struct_child_id =
      ExpectCreateStructInClientChild({v2_child_id}, ONCE_PER_CLIENT);
  ValueId struct_child_id = ExpectCreateStructInClientChild(
      {v1_child_id, v2_struct_child_id}, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInClientChild(struct_child_id, StructV({v1, StructV({v2})}),
                                 ONCE_PER_CLIENT);
  ExpectMaterialize(result_id, ClientsV(std::vector<v0::Value>(
                                   NUM_CLIENTS, StructV({v1, StructV({v2})}))));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtServerFlat) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ValueId v1_child_id = ExpectCreateInServerChild(v1);
  ValueId v2_child_id = ExpectCreateInServerChild(v2);
  TFF_ASSERT_OK_AND_ASSIGN(auto v_id, test_executor_->CreateValue(
                                          StructV({ServerV(v1), ServerV(v2)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  ValueId struct_child_id =
      ExpectCreateStructInServerChild({v1_child_id, v2_child_id});
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInServerChild(struct_child_id, StructV({v1, v2}));
  ExpectMaterialize(result_id, ServerV(StructV({v1, v2})));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtServerNested) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ValueId v1_child_id = ExpectCreateInServerChild(v1);
  ValueId v2_child_id = ExpectCreateInServerChild(v2);
  TFF_ASSERT_OK_AND_ASSIGN(auto v_id,
                           test_executor_->CreateValue(
                               StructV({ServerV(v1), StructV({ServerV(v2)})})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  ValueId v2_struct_child_id = ExpectCreateStructInServerChild({v2_child_id});
  ValueId struct_child_id =
      ExpectCreateStructInServerChild({v1_child_id, v2_struct_child_id});
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInServerChild(struct_child_id, StructV({v1, StructV({v2})}));
  ExpectMaterialize(result_id, ServerV(StructV({v1, StructV({v2})})));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipMixedPlacementsFails) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ExpectCreateInServerChild(v1);
  ExpectCreateInClientChild(v2);
  TFF_ASSERT_OK_AND_ASSIGN(auto v_id,
                           test_executor_->CreateValue(
                               StructV({ServerV(v1), ClientsV({v2}, true)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  EXPECT_THAT(test_executor_->CreateCall(zip_id, v_id),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace

}  // namespace tensorflow_federated
