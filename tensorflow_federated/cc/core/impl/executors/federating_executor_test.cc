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
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
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
using ::testing::Cardinality;
using testing::ClientsV;
using testing::IntrinsicV;
using testing::ServerV;
using testing::StructV;
using testing::TensorV;
using testing::intrinsic::FederatedAggregateV;
using testing::intrinsic::FederatedBroadcastV;
using testing::intrinsic::FederatedEvalAtClientsV;
using testing::intrinsic::FederatedEvalAtServerV;
using testing::intrinsic::FederatedMapV;
using testing::intrinsic::FederatedValueAtClientsV;
using testing::intrinsic::FederatedValueAtServerV;
using testing::intrinsic::FederatedZipAtClientsV;
using testing::intrinsic::FederatedZipAtServerV;

const uint16_t NUM_CLIENTS = 10;

const Cardinality ONCE = ::testing::Exactly(1);
const Cardinality ONCE_PER_CLIENT = ::testing::Exactly(NUM_CLIENTS);

class FederatingExecutorTest : public ExecutorTestBase {
 public:
  FederatingExecutorTest() {
    // Extra method required in order to use `TFF_ASSERT_OK_AND_ASSIGN`.
    Initialize();
  }

  void Initialize() {
    TFF_ASSERT_OK_AND_ASSIGN(test_executor_,
                             tensorflow_federated::CreateFederatingExecutor(
                                 mock_executor_, {{"clients", NUM_CLIENTS}}));
  }

 protected:
  std::shared_ptr<::testing::StrictMock<MockExecutor>> mock_executor_ =
      std::make_shared<::testing::StrictMock<MockExecutor>>();

  ValueId ExpectCreateInChild(const v0::Value& expected,
                              Cardinality repeatedly = ONCE) {
    return mock_executor_->ExpectCreateValue(expected, repeatedly);
  }

  void ExpectMaterializeInChild(ValueId id, v0::Value to_return,
                                Cardinality repeatedly = ONCE) {
    mock_executor_->ExpectMaterialize(id, std::move(to_return), repeatedly);
  }

  ValueId ExpectCreateCallInChild(ValueId fn_id,
                                  absl::optional<const ValueId> arg_id,
                                  Cardinality repeatedly = ONCE) {
    return mock_executor_->ExpectCreateCall(fn_id, arg_id, repeatedly);
  }

  ValueId ExpectCreateStructInChild(const absl::Span<const ValueId> elements,
                                    Cardinality repeatedly = ONCE) {
    return mock_executor_->ExpectCreateStruct(elements, repeatedly);
  }

  void ExpectCreateMaterializeInChild(v0::Value value,
                                      Cardinality repeatedly = ONCE) {
    ValueId id = ExpectCreateInChild(value, repeatedly);
    ExpectMaterializeInChild(id, value, repeatedly);
  }
};

TEST_F(FederatingExecutorTest, ConstructsExecutorWithEmptyCardinalities) {
  EXPECT_THAT(
      tensorflow_federated::CreateFederatingExecutor(mock_executor_, {}),
      StatusIs(StatusCode::kNotFound));
}

TEST_F(FederatingExecutorTest, CreateMaterializeTensor) {
  v0::Value value = TensorV(1.0);
  ExpectCreateMaterializeInChild(value);
  ExpectCreateMaterialize(value);
}

TEST_F(FederatingExecutorTest, CreateValueIntrinsic) {
  EXPECT_THAT(test_executor_->CreateValue(FederatedMapV()), IsOk());
}

TEST_F(FederatingExecutorTest, CreateValueBadIntrinsic) {
  EXPECT_THAT(test_executor_->CreateValue(IntrinsicV("blech")),
              StatusIs(StatusCode::kUnimplemented));
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
  for (const auto& element : elements) {
    ExpectCreateMaterializeInChild(element);
  }
  ExpectCreateMaterialize(StructV(elements));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtServer) {
  v0::Value tensor_pb = TensorV(1.0);
  ExpectCreateMaterializeInChild(tensor_pb);
  ExpectCreateMaterialize(ServerV(tensor_pb));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtClients) {
  std::vector<v0::Value> values;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    v0::Value tensor_pb = TensorV(i);
    values.emplace_back(tensor_pb);
    ExpectCreateMaterializeInChild(tensor_pb);
  }
  ExpectCreateMaterialize(ClientsV(values));
}

TEST_F(FederatingExecutorTest, CreateValueFailsWrongNumberClients) {
  EXPECT_THAT(test_executor_->CreateValue(ClientsV({})),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateMaterializeAtClientsAllEqual) {
  v0::Value tensor_in = TensorV(1.0);
  ValueId child_id = ExpectCreateInChild(tensor_in);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto id, test_executor_->CreateValue(ClientsV({tensor_in}, true)));
  v0::Value tensor_out = TensorV(2.0);
  ExpectMaterializeInChild(child_id, tensor_out, ONCE_PER_CLIENT);
  v0::Value clients_out = ClientsV(std::vector(NUM_CLIENTS, tensor_out), false);
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
  ExpectCreateMaterializeInChild(server_v1);
  ExpectCreateMaterializeInChild(server_v2);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(FederatingExecutorTest, CreateMaterializeStructOfMixedValues) {
  v0::Value unplaced = TensorV(6);
  v0::Value server = TensorV(7);
  v0::Value value_pb = StructV({unplaced, ServerV(server)});
  ExpectCreateMaterializeInChild(unplaced);
  ExpectCreateMaterializeInChild(server);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(FederatingExecutorTest, CreateMaterializeFederatedStruct) {
  v0::Value struct_pb = StructV({TensorV(5.0)});
  ExpectCreateMaterializeInChild(struct_pb);
  ExpectCreateMaterialize(ServerV(struct_pb));
}

TEST_F(FederatingExecutorTest, CreateStructOfTensors) {
  v0::Value tensor_v1 = TensorV(5);
  v0::Value tensor_v2 = TensorV(6);
  ExpectCreateMaterializeInChild(tensor_v1);
  ExpectCreateMaterializeInChild(tensor_v2);
  TFF_ASSERT_OK_AND_ASSIGN(auto v1_id, test_executor_->CreateValue(tensor_v1));
  TFF_ASSERT_OK_AND_ASSIGN(auto v2_id, test_executor_->CreateValue(tensor_v2));
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateStruct({v1_id, v2_id}));
  ExpectMaterialize(id, StructV({tensor_v1, tensor_v2}));
}

TEST_F(FederatingExecutorTest, CreateSelection) {
  v0::Value tensor = TensorV(5);
  ExpectCreateMaterializeInChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto s,
                           test_executor_->CreateValue(StructV({tensor})));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, tensor);
}

TEST_F(FederatingExecutorTest, CreateSelectionFromEmbeddedValue) {
  v0::Value pretend_struct = TensorV(5);
  v0::Value selected_tensor = TensorV(24);
  ValueId child_id = ExpectCreateInChild(pretend_struct);
  ValueId child_selected_id =
      mock_executor_->ExpectCreateSelection(child_id, 0);
  ExpectMaterializeInChild(child_selected_id, selected_tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto s, test_executor_->CreateValue(pretend_struct));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, selected_tensor);
}

TEST_F(FederatingExecutorTest, CreateSelectionFromFederatedValueFails) {
  v0::Value struct_pb = StructV({TensorV(6)});
  v0::Value fed = ServerV(struct_pb);
  ExpectCreateInChild(struct_pb);
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
  v0::Value fn = TensorV(5);
  v0::Value result = TensorV(22);
  ValueId fn_child_id = ExpectCreateInChild(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  ValueId fn_result_child_id =
      ExpectCreateCallInChild(fn_child_id, absl::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn_id, absl::nullopt));
  ExpectMaterializeInChild(fn_result_child_id, result);
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(FederatingExecutorTest, CreateCallEmbeddedSingleArg) {
  v0::Value fn = TensorV(5);
  v0::Value arg = TensorV(6);
  v0::Value result = TensorV(22);
  ValueId fn_child_id = ExpectCreateInChild(fn);
  ValueId arg_child_id = ExpectCreateInChild(arg);
  ValueId fn_result_child_id =
      ExpectCreateCallInChild(fn_child_id, arg_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id, test_executor_->CreateValue(arg));
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn_id, arg_id));
  ExpectMaterializeInChild(fn_result_child_id, result);
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(FederatingExecutorTest, CreateCallEmbedsStructArg) {
  v0::Value fn = TensorV(5);
  v0::Value arg_1 = TensorV(6);
  v0::Value arg_2 = TensorV(7);
  v0::Value result = TensorV(22);
  ValueId fn_child_id = ExpectCreateInChild(fn);
  ValueId arg_1_child_id = ExpectCreateInChild(arg_1);
  ValueId arg_2_child_id = ExpectCreateInChild(arg_2);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id, test_executor_->CreateValue(StructV({arg_1, arg_2})));
  ValueId arg_child_id =
      ExpectCreateStructInChild({arg_1_child_id, arg_2_child_id});
  ValueId fn_result_child_id =
      ExpectCreateCallInChild(fn_child_id, arg_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn_id, arg_id));
  ExpectMaterializeInChild(fn_result_child_id, result);
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueFails) {
  v0::Value tensor = TensorV(1);
  ExpectCreateInChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto fed_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  EXPECT_THAT(test_executor_->CreateCall(fed_id, absl::nullopt),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateCallStructFails) {
  v0::Value tensor = TensorV(1);
  ExpectCreateInChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateValue(StructV({tensor})));
  EXPECT_THAT(test_executor_->CreateCall(struct_id, absl::nullopt),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedAggregate) {
  std::vector<v0::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    client_vals.emplace_back(TensorV(i));
    client_vals_child_ids.emplace_back(ExpectCreateInChild(TensorV(i)));
  }
  v0::Value value = ClientsV(client_vals);
  v0::Value zero = TensorV("zero");
  ValueId zero_child_id = ExpectCreateInChild(zero);
  v0::Value accumulate = TensorV("accumulate");
  ValueId accumulate_child_id = ExpectCreateInChild(accumulate);
  v0::Value merge = TensorV("merge");
  /* unused ValueId merge_child_id =*/ExpectCreateInChild(merge);
  v0::Value report = TensorV("report");
  ValueId report_child_id = ExpectCreateInChild(report);
  v0::Value arg = StructV({value, zero, accumulate, merge, report});
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id, test_executor_->CreateValue(arg));
  TFF_ASSERT_OK_AND_ASSIGN(auto intrinsic_id,
                           test_executor_->CreateValue(FederatedAggregateV()));
  ValueId current_child_id = zero_child_id;
  for (auto client_val_child_id : client_vals_child_ids) {
    ValueId call_arg_child_id =
        ExpectCreateStructInChild({current_child_id, client_val_child_id});
    current_child_id =
        ExpectCreateCallInChild(accumulate_child_id, call_arg_child_id);
  }
  ValueId result_child_id =
      ExpectCreateCallInChild(report_child_id, current_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(intrinsic_id, arg_id));
  v0::Value child_result = TensorV("result");
  ExpectMaterializeInChild(result_child_id, child_result);
  ExpectMaterialize(result_id, ServerV(child_result));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedBroadcast) {
  v0::Value tensor = TensorV(1);
  ValueId tensor_id = ExpectCreateInChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  TFF_ASSERT_OK_AND_ASSIGN(auto broadcast_id,
                           test_executor_->CreateValue(FederatedBroadcastV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto clients_id,
                           test_executor_->CreateCall(broadcast_id, server_id));
  v0::Value tensor_out = TensorV(4);
  ExpectMaterializeInChild(tensor_id, tensor_out, ONCE_PER_CLIENT);
  ExpectMaterialize(clients_id,
                    ClientsV(std::vector<v0::Value>(NUM_CLIENTS, tensor_out)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAtClients) {
  std::vector<v0::Value> client_vals;
  std::vector<ValueId> client_vals_child_ids;
  for (int i = 0; i < NUM_CLIENTS; i++) {
    client_vals.emplace_back(TensorV(i));
    client_vals_child_ids.emplace_back(ExpectCreateInChild(TensorV(i)));
  }
  v0::Value value = ClientsV(client_vals);
  TFF_ASSERT_OK_AND_ASSIGN(auto input_id,
                           test_executor_->CreateValue(ClientsV(client_vals)));
  v0::Value fn = TensorV(2);
  ValueId fn_child_id = ExpectCreateInChild(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  for (int i = 0; i < NUM_CLIENTS; i++) {
    ValueId result_child_id =
        ExpectCreateCallInChild(fn_child_id, client_vals_child_ids[i]);
    ExpectMaterializeInChild(result_child_id, client_vals[i]);
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateStruct({fn_id, input_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  ExpectMaterialize(result_id, value);
}

TEST_F(FederatingExecutorTest, CreateCallFederatedMapAtServer) {
  v0::Value tensor = TensorV(23);
  ValueId tensor_child_id = ExpectCreateInChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  v0::Value fn = TensorV(44);
  ValueId fn_child_id = ExpectCreateInChild(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateStruct({fn_id, server_id}));
  ValueId result_child_id =
      ExpectCreateCallInChild(fn_child_id, tensor_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  v0::Value output_tensor = TensorV(89);
  ExpectMaterializeInChild(result_child_id, output_tensor);
  ExpectMaterialize(result_id, ServerV(output_tensor));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedEvalAtClients) {
  v0::Value fn = TensorV(1);
  ValueId fn_child_id = ExpectCreateInChild(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  ValueId result_child_id =
      ExpectCreateCallInChild(fn_child_id, absl::nullopt, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtClientsV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn_id));
  v0::Value result_tensor = TensorV(3);
  ExpectMaterializeInChild(result_child_id, result_tensor, ONCE_PER_CLIENT);
  ExpectMaterialize(
      result_id, ClientsV(std::vector<v0::Value>(NUM_CLIENTS, result_tensor)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedEvalAtServer) {
  v0::Value fn = TensorV(1);
  ValueId fn_child_id = ExpectCreateInChild(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  ValueId result_child_id = ExpectCreateCallInChild(fn_child_id, absl::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn_id));
  v0::Value result_tensor = TensorV(3);
  ExpectMaterializeInChild(result_child_id, result_tensor);
  ExpectMaterialize(result_id, ServerV(result_tensor));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueAtClients) {
  v0::Value tensor = TensorV(1);
  ValueId tensor_child_id = ExpectCreateInChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto tensor_id, test_executor_->CreateValue(tensor));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtClientsV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor_id));
  ExpectMaterializeInChild(tensor_child_id, tensor, ONCE_PER_CLIENT);
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<v0::Value>(NUM_CLIENTS, tensor)));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedValueAtServer) {
  v0::Value tensor = TensorV(1);
  ValueId tensor_child_id = ExpectCreateInChild(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto tensor_id, test_executor_->CreateValue(tensor));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor_id));
  ExpectMaterializeInChild(tensor_child_id, tensor);
  ExpectMaterialize(result_id, ServerV(tensor));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtClients) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ValueId v1_child_id = ExpectCreateInChild(v1);
  ValueId v2_child_id = ExpectCreateInChild(v2);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto v_id, test_executor_->CreateValue(
                     StructV({ClientsV({v1}, true), ClientsV({v2}, true)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtClientsV()));
  ValueId struct_child_id =
      ExpectCreateStructInChild({v1_child_id, v2_child_id}, ONCE_PER_CLIENT);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInChild(struct_child_id, StructV({v1, v2}), ONCE_PER_CLIENT);
  ExpectMaterialize(result_id, ClientsV(std::vector<v0::Value>(
                                   NUM_CLIENTS, StructV({v1, v2}))));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipAtServer) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ValueId v1_child_id = ExpectCreateInChild(v1);
  ValueId v2_child_id = ExpectCreateInChild(v2);
  TFF_ASSERT_OK_AND_ASSIGN(auto v_id, test_executor_->CreateValue(
                                          StructV({ServerV(v1), ServerV(v2)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  ValueId struct_child_id =
      ExpectCreateStructInChild({v1_child_id, v2_child_id});
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  ExpectMaterializeInChild(struct_child_id, StructV({v1, v2}));
  ExpectMaterialize(result_id, ServerV(StructV({v1, v2})));
}

TEST_F(FederatingExecutorTest, CreateCallFederatedZipMixedPlacementsFails) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ExpectCreateInChild(v1);
  ExpectCreateInChild(v2);
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
