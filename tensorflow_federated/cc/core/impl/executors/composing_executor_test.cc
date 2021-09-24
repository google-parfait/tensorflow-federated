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

#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"

#include <cstddef>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/executors/computations.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_test_base.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
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

class ComposingExecutorTest : public ExecutorTestBase {
 public:
  ComposingExecutorTest() {
    // Extra method required in order to use `TFF_ASSERT_OK_AND_ASSIGN`.
    Initialize();
  }

  void Initialize() {
    // Test with a few different sizes of client to ensure they're all handled
    // appropriately.
    clients_per_child_ = {0, 1, 2, 3};
    total_clients_ = 0;
    std::vector<ComposingChild> composing_children;
    for (uint32_t num_clients : clients_per_child_) {
      total_clients_ += num_clients;
      auto exec = std::make_shared<::testing::StrictMock<MockExecutor>>();
      TFF_ASSERT_OK_AND_ASSIGN(
          auto child, ComposingChild::Make(exec, {{"clients", num_clients}}));
      composing_children.push_back(child);
      mock_children_.push_back(std::move(exec));
    }
    mock_server_ = std::make_shared<::testing::StrictMock<MockExecutor>>();
    test_executor_ =
        CreateComposingExecutor(mock_server_, std::move(composing_children));
  }

 protected:
  std::vector<uint32_t> clients_per_child_;
  uint32_t total_clients_;
  std::shared_ptr<::testing::StrictMock<MockExecutor>> mock_server_;
  std::vector<std::shared_ptr<::testing::StrictMock<MockExecutor>>>
      mock_children_;
};

TEST_F(ComposingExecutorTest, ChildConstructionWithNoClientCardinalitiesFails) {
  EXPECT_THAT(ComposingChild::Make(mock_server_, {}),
              StatusIs(StatusCode::kNotFound));
}

TEST_F(ComposingExecutorTest, CreateMaterializeTensor) {
  v0::Value value = TensorV(1.0);
  ExpectCreateMaterialize(value);
}

TEST_F(ComposingExecutorTest, CreateValueIntrinsic) {
  EXPECT_THAT(test_executor_->CreateValue(FederatedMapV()), IsOk());
}

TEST_F(ComposingExecutorTest, CreateValueBadIntrinsic) {
  EXPECT_THAT(test_executor_->CreateValue(IntrinsicV("blech")),
              StatusIs(StatusCode::kUnimplemented));
}

TEST_F(ComposingExecutorTest, MaterializeIntrinsicFails) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateValue(FederatedMapV()));
  EXPECT_THAT(test_executor_->Materialize(id),
              StatusIs(StatusCode::kUnimplemented));
}

TEST_F(ComposingExecutorTest, CreateMaterializeEmptyStruct) {
  ExpectCreateMaterialize(StructV({}));
}

TEST_F(ComposingExecutorTest, CreateMaterializeFlatStruct) {
  auto elements = {TensorV(1.0), TensorV(3.5), TensorV(2.0)};
  ExpectCreateMaterialize(StructV(elements));
}

TEST_F(ComposingExecutorTest, CreateMaterializeAtServer) {
  v0::Value tensor_pb = TensorV(1.0);
  mock_server_->ExpectCreateMaterialize(tensor_pb);
  ExpectCreateMaterialize(ServerV(tensor_pb));
}

TEST_F(ComposingExecutorTest, CreateMaterializeAtClients) {
  std::vector<v0::Value> values;
  uint32_t next_value = 0;
  for (uint32_t i = 0; i < mock_children_.size(); i++) {
    const auto& child = mock_children_[i];
    uint32_t num_clients = clients_per_child_[i];
    std::vector<v0::Value> child_values;
    for (uint32_t j = 0; j < num_clients; j++) {
      v0::Value tensor_pb = TensorV(next_value);
      next_value++;
      child_values.push_back(tensor_pb);
      values.push_back(tensor_pb);
    }
    child->ExpectCreateMaterialize(ClientsV(child_values));
  }
  ASSERT_EQ(next_value, total_clients_);
  ASSERT_EQ(values.size(), total_clients_);
  ExpectCreateMaterialize(ClientsV(values));
}

TEST_F(ComposingExecutorTest, CreateValueFailsWrongNumberClients) {
  EXPECT_THAT(test_executor_->CreateValue(ClientsV({})),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateMaterializeAtClientsAllEqual) {
  v0::Value tensor_pb = TensorV(2);
  v0::Value all_equal_value = ClientsV({tensor_pb}, true);
  for (uint32_t i = 0; i < mock_children_.size(); i++) {
    auto child = mock_children_[i];
    auto id = child->ExpectCreateValue(all_equal_value);
    child->ExpectMaterialize(
        id, ClientsV(std::vector(clients_per_child_[i], tensor_pb)));
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateValue(all_equal_value));
  ExpectMaterialize(id, ClientsV(std::vector(total_clients_, tensor_pb)));
}

TEST_F(ComposingExecutorTest, CreateFederatedValueInsideStruct) {
  v0::Value fed_pb = ClientsV({TensorV(5)}, true);
  v0::Value struct_pb = StructV({fed_pb});
  for (const auto& child : mock_children_) {
    child->ExpectCreateValue(fed_pb);
  }
  EXPECT_THAT(test_executor_->CreateValue(struct_pb), IsOk());
}

TEST_F(ComposingExecutorTest, CreateMaterializeStructOfFederatedValues) {
  v0::Value server_v1 = TensorV(6);
  v0::Value server_v2 = TensorV(7);
  v0::Value value_pb = StructV({ServerV(server_v1), ServerV(server_v2)});
  mock_server_->ExpectCreateMaterialize(server_v1);
  mock_server_->ExpectCreateMaterialize(server_v2);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(ComposingExecutorTest, CreateMaterializeStructOfMixedValues) {
  v0::Value unplaced = TensorV(6);
  v0::Value server = TensorV(7);
  v0::Value value_pb = StructV({unplaced, ServerV(server)});
  mock_server_->ExpectCreateMaterialize(server);
  ExpectCreateMaterialize(value_pb);
}

TEST_F(ComposingExecutorTest, CreateStructOfTensors) {
  v0::Value tensor_v1 = TensorV(5);
  v0::Value tensor_v2 = TensorV(6);
  TFF_ASSERT_OK_AND_ASSIGN(auto v1_id, test_executor_->CreateValue(tensor_v1));
  TFF_ASSERT_OK_AND_ASSIGN(auto v2_id, test_executor_->CreateValue(tensor_v2));
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateStruct({v1_id, v2_id}));
  ExpectMaterialize(id, StructV({tensor_v1, tensor_v2}));
}

TEST_F(ComposingExecutorTest, CreateSelection) {
  v0::Value tensor = TensorV(5);
  TFF_ASSERT_OK_AND_ASSIGN(auto s,
                           test_executor_->CreateValue(StructV({tensor})));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, tensor);
}

TEST_F(ComposingExecutorTest, CreateSelectionFromEmbeddedValue) {
  v0::Value pretend_struct = TensorV(5);
  v0::Value selected_tensor = TensorV(24);
  ValueId child_id = mock_server_->ExpectCreateValue(pretend_struct);
  ValueId child_selected_id = mock_server_->ExpectCreateSelection(child_id, 0);
  mock_server_->ExpectMaterialize(child_selected_id, selected_tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto s, test_executor_->CreateValue(pretend_struct));
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateSelection(s, 0));
  ExpectMaterialize(id, selected_tensor);
}

TEST_F(ComposingExecutorTest, CreateSelectionFromFederatedValueFails) {
  v0::Value struct_pb = StructV({TensorV(6)});
  v0::Value fed = ServerV(struct_pb);
  mock_server_->ExpectCreateValue(struct_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto fed_id, test_executor_->CreateValue(fed));
  EXPECT_THAT(test_executor_->CreateSelection(fed_id, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateSelectionFromIntrinsicFails) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id,
                           test_executor_->CreateValue(FederatedMapV()));
  EXPECT_THAT(test_executor_->CreateSelection(id, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateSelectionOutOfBoundsFails) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(StructV({})));
  EXPECT_THAT(test_executor_->CreateSelection(id, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateCallEmbeddedNoArg) {
  v0::Value fn = TensorV(5);
  v0::Value result = TensorV(22);
  ValueId fn_server_id = mock_server_->ExpectCreateValue(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  ValueId fn_result_child_id =
      mock_server_->ExpectCreateCall(fn_server_id, std::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn_id, std::nullopt));
  mock_server_->ExpectMaterialize(fn_result_child_id, result);
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(ComposingExecutorTest, CreateCallEmbeddedSingleArg) {
  v0::Value fn = TensorV(5);
  v0::Value arg = TensorV(6);
  v0::Value result = TensorV(22);
  ValueId fn_child_id = mock_server_->ExpectCreateValue(fn);
  ValueId arg_child_id = mock_server_->ExpectCreateValue(arg);
  ValueId fn_result_child_id =
      mock_server_->ExpectCreateCall(fn_child_id, arg_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id, test_executor_->CreateValue(arg));
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn_id, arg_id));
  mock_server_->ExpectMaterialize(fn_result_child_id, result);
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(ComposingExecutorTest, CreateCallEmbedsStructArg) {
  v0::Value fn = TensorV(5);
  v0::Value arg_1 = TensorV(6);
  v0::Value arg_2 = TensorV(7);
  v0::Value result = TensorV(22);
  ValueId fn_child_id = mock_server_->ExpectCreateValue(fn);
  ValueId arg_1_child_id = mock_server_->ExpectCreateValue(arg_1);
  ValueId arg_2_child_id = mock_server_->ExpectCreateValue(arg_2);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto arg_id, test_executor_->CreateValue(StructV({arg_1, arg_2})));
  ValueId arg_child_id =
      mock_server_->ExpectCreateStruct({arg_1_child_id, arg_2_child_id});
  ValueId fn_result_child_id =
      mock_server_->ExpectCreateCall(fn_child_id, arg_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_result_id,
                           test_executor_->CreateCall(fn_id, arg_id));
  mock_server_->ExpectMaterialize(fn_result_child_id, result);
  ExpectMaterialize(fn_result_id, result);
}

TEST_F(ComposingExecutorTest, CreateCallFederatedValueFails) {
  v0::Value tensor = TensorV(1);
  mock_server_->ExpectCreateValue(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto fed_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  TFF_ASSERT_OK_AND_ASSIGN(auto res_id,
                           test_executor_->CreateCall(fed_id, std::nullopt));
  // We don't see the error until materializing due to asynchrony caused by
  // evaluating the call in another thread.
  EXPECT_THAT(test_executor_->Materialize(res_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateCallStructFails) {
  v0::Value tensor = TensorV(1);
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_id,
                           test_executor_->CreateValue(StructV({tensor})));
  TFF_ASSERT_OK_AND_ASSIGN(auto res_id,
                           test_executor_->CreateCall(struct_id, std::nullopt));
  // We don't see the error until materializing due to asynchrony caused by
  // evaluating the call in another thread.
  EXPECT_THAT(test_executor_->Materialize(res_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedAggregate) {
  v0::Value value = ClientsV({TensorV("value")}, true);
  v0::Value zero = TensorV("zero");
  v0::Value accumulate = TensorV("accumulate");
  v0::Value merge = TensorV("merge");
  v0::Value report = TensorV("report");
  v0::Value result_from_child = ServerV(TensorV("result from child"));
  v0::Value final_result_unfed = TensorV("final result");
  for (const auto& child : mock_children_) {
    auto child_value = child->ExpectCreateValue(value);
    auto child_zero = child->ExpectCreateValue(zero);
    auto child_accumulate = child->ExpectCreateValue(accumulate);
    auto child_merge = child->ExpectCreateValue(merge);
    v0::Value report_val;
    *report_val.mutable_computation() = IdentityComp();
    auto child_report = child->ExpectCreateValue(report_val);
    auto child_agg = child->ExpectCreateValue(FederatedAggregateV());
    auto arg = child->ExpectCreateStruct(
        {child_value, child_zero, child_accumulate, child_merge, child_report});
    auto res = child->ExpectCreateCall(child_agg, arg);
    child->ExpectMaterialize(res, result_from_child);
  }
  // `merge` is used both on the server/controller and in the children.
  auto server_merge = mock_server_->ExpectCreateValue(merge);
  auto server_report = mock_server_->ExpectCreateValue(report);
  auto result_from_child_on_server = mock_server_->ExpectCreateValue(
      result_from_child.federated().value(0),
      ::testing::Exactly(mock_children_.size()));
  size_t num_total_merges = mock_children_.size() - 1;
  auto prev_merge_result = result_from_child_on_server;
  for (uint32_t i = 0; i < num_total_merges; i++) {
    auto merge_arg = mock_server_->ExpectCreateStruct(
        {prev_merge_result, result_from_child_on_server});
    prev_merge_result = mock_server_->ExpectCreateCall(server_merge, merge_arg);
  }
  auto post_report =
      mock_server_->ExpectCreateCall(server_report, prev_merge_result);
  mock_server_->ExpectMaterialize(post_report, final_result_unfed);
  TFF_ASSERT_OK_AND_ASSIGN(auto controller_value,
                           test_executor_->CreateValue(value));
  TFF_ASSERT_OK_AND_ASSIGN(auto controller_zero,
                           test_executor_->CreateValue(zero));
  TFF_ASSERT_OK_AND_ASSIGN(auto controller_accumulate,
                           test_executor_->CreateValue(accumulate));
  TFF_ASSERT_OK_AND_ASSIGN(auto controller_merge,
                           test_executor_->CreateValue(merge));
  TFF_ASSERT_OK_AND_ASSIGN(auto controller_report,
                           test_executor_->CreateValue(report));
  TFF_ASSERT_OK_AND_ASSIGN(auto controller_agg,
                           test_executor_->CreateValue(FederatedAggregateV()));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto agg_arg,
      test_executor_->CreateStruct({controller_value, controller_zero,
                                    controller_accumulate, controller_merge,
                                    controller_report}));
  TFF_ASSERT_OK_AND_ASSIGN(auto res,
                           test_executor_->CreateCall(controller_agg, agg_arg));
  ExpectMaterialize(res, ServerV(final_result_unfed));
}

TEST_F(ComposingExecutorTest,
       CreateCallFederatedAggregateFailsWithNonClientsPlacedValue) {
  v0::Value unplaced_value = TensorV(1);
  TFF_ASSERT_OK_AND_ASSIGN(auto unplaced_id,
                           test_executor_->CreateValue(unplaced_value));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg, test_executor_->CreateStruct({
                                         unplaced_id,  // value
                                         unplaced_id,  // zero
                                         unplaced_id,  // accumulate
                                         unplaced_id,  // merge
                                         unplaced_id,  // report
                                     }));
  TFF_ASSERT_OK_AND_ASSIGN(auto agg,
                           test_executor_->CreateValue(FederatedAggregateV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto res, test_executor_->CreateCall(agg, arg));
  EXPECT_THAT(test_executor_->Materialize(res),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedBroadcast) {
  v0::Value tensor = TensorV(1);
  mock_server_->ExpectCreateMaterialize(tensor);
  for (const auto& child : mock_children_) {
    child->ExpectCreateMaterialize(ClientsV({tensor}, true));
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  TFF_ASSERT_OK_AND_ASSIGN(auto broadcast_id,
                           test_executor_->CreateValue(FederatedBroadcastV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto clients_id,
                           test_executor_->CreateCall(broadcast_id, server_id));
  ExpectMaterialize(clients_id,
                    ClientsV(std::vector<v0::Value>(total_clients_, tensor)));
}

TEST_F(ComposingExecutorTest,
       CreateCallFederatedBroadcastFailsOnNonServerPlacedValue) {
  v0::Value tensor = TensorV(1);
  TFF_ASSERT_OK_AND_ASSIGN(auto unplaced_id,
                           test_executor_->CreateValue(tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto broadcast_id,
                           test_executor_->CreateValue(FederatedBroadcastV()));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto clients_id, test_executor_->CreateCall(broadcast_id, unplaced_id));
  EXPECT_THAT(test_executor_->Materialize(clients_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedMapAtClients) {
  std::vector<v0::Value> client_vals_in;
  std::vector<v0::Value> client_vals_out;
  v0::Value fn = TensorV(24601);
  for (uint32_t i = 0; i < mock_children_.size(); i++) {
    const auto& child = mock_children_[i];
    std::vector<v0::Value> in_vec;
    std::vector<v0::Value> out_vec;
    for (uint32_t j = 0; j < clients_per_child_[i]; j++) {
      v0::Value in = TensorV(i * 10000 + j * 100);
      v0::Value out = TensorV(i * 10000 + j * 100 + 1);
      client_vals_in.push_back(in);
      in_vec.push_back(in);
      client_vals_out.push_back(out);
      out_vec.push_back(out);
    }
    auto in_id = child->ExpectCreateValue(ClientsV(in_vec));
    auto map_id = child->ExpectCreateValue(FederatedMapV());
    auto fn_id = child->ExpectCreateValue(fn);
    auto args_id = child->ExpectCreateStruct({fn_id, in_id});
    auto res_id = child->ExpectCreateCall(map_id, args_id);
    child->ExpectMaterialize(res_id, ClientsV(out_vec));
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto input_id, test_executor_->CreateValue(ClientsV(client_vals_in)));
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateStruct({fn_id, input_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto res_id,
                           test_executor_->CreateCall(map_id, arg_id));
  ExpectMaterialize(res_id, ClientsV(client_vals_out));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedMapAtServer) {
  v0::Value tensor = TensorV(23);
  ValueId tensor_child_id = mock_server_->ExpectCreateValue(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto server_id,
                           test_executor_->CreateValue(ServerV(tensor)));
  v0::Value fn = TensorV(44);
  ValueId fn_child_id = mock_server_->ExpectCreateValue(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateStruct({fn_id, server_id}));
  ValueId result_child_id =
      mock_server_->ExpectCreateCall(fn_child_id, tensor_child_id);
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(map_id, arg_id));
  v0::Value output_tensor = TensorV(89);
  mock_server_->ExpectMaterialize(result_child_id, output_tensor);
  ExpectMaterialize(result_id, ServerV(output_tensor));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedMapOnUnplacedFails) {
  v0::Value tensor = TensorV(23);
  TFF_ASSERT_OK_AND_ASSIGN(auto unplaced_id,
                           test_executor_->CreateValue(tensor));
  v0::Value fn = TensorV(44);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(auto map_id,
                           test_executor_->CreateValue(FederatedMapV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateStruct({fn_id, unplaced_id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto res_id,
                           test_executor_->CreateCall(map_id, arg_id));
  EXPECT_THAT(test_executor_->Materialize(res_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedEvalAtClients) {
  v0::Value fn = TensorV(22);
  std::vector<v0::Value> client_results;
  for (uint32_t i = 0; i < mock_children_.size(); i++) {
    const auto& child = mock_children_[i];
    std::vector<v0::Value> child_results;
    for (uint32_t j = 0; j < clients_per_child_[i]; j++) {
      v0::Value result = TensorV(client_results.size());
      child_results.push_back(result);
      client_results.push_back(result);
    }
    ValueId child_eval = child->ExpectCreateValue(FederatedEvalAtClientsV());
    ValueId child_fn = child->ExpectCreateValue(fn);
    ValueId child_res = child->ExpectCreateCall(child_eval, child_fn);
    child->ExpectMaterialize(child_res, ClientsV(child_results));
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtClientsV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn_id));
  ExpectMaterialize(result_id, ClientsV(client_results));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedEvalAtServer) {
  v0::Value fn = TensorV(1);
  ValueId fn_child_id = mock_server_->ExpectCreateValue(fn);
  TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
  ValueId result_child_id =
      mock_server_->ExpectCreateCall(fn_child_id, std::nullopt);
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_eval_id, test_executor_->CreateValue(FederatedEvalAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_eval_id, fn_id));
  v0::Value result_tensor = TensorV(3);
  mock_server_->ExpectMaterialize(result_child_id, result_tensor);
  ExpectMaterialize(result_id, ServerV(result_tensor));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedValueAtClients) {
  v0::Value tensor = TensorV(1);
  for (const auto& child : mock_children_) {
    child->ExpectCreateMaterialize(ClientsV({tensor}, true));
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto tensor_id, test_executor_->CreateValue(tensor));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtClientsV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor_id));
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<v0::Value>(total_clients_, tensor)));
}

TEST_F(ComposingExecutorTest,
       CreateCallFederatedValueAtClientsWithComputedValue) {
  v0::Value tensor = TensorV(1);
  v0::Value identity_lambda;
  *identity_lambda.mutable_computation() = IdentityComp();
  auto server_fn_id = mock_server_->ExpectCreateValue(identity_lambda);
  auto server_tensor_id = mock_server_->ExpectCreateValue(tensor);
  auto server_computed_tensor_id =
      mock_server_->ExpectCreateCall(server_fn_id, server_tensor_id);
  mock_server_->ExpectMaterialize(server_computed_tensor_id, tensor);

  for (const auto& child : mock_children_) {
    child->ExpectCreateMaterialize(ClientsV({tensor}, true));
  }

  TFF_ASSERT_OK_AND_ASSIGN(auto tensor_id, test_executor_->CreateValue(tensor));
  TFF_ASSERT_OK_AND_ASSIGN(auto identity_id,
                           test_executor_->CreateValue(identity_lambda));
  TFF_ASSERT_OK_AND_ASSIGN(auto computed_tensor_id,
                           test_executor_->CreateCall(identity_id, tensor_id));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtClientsV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id, test_executor_->CreateCall(
                                               fed_val_id, computed_tensor_id));
  ExpectMaterialize(result_id,
                    ClientsV(std::vector<v0::Value>(total_clients_, tensor)));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedValueAtServer) {
  v0::Value tensor = TensorV(1);
  mock_server_->ExpectCreateMaterialize(tensor);
  TFF_ASSERT_OK_AND_ASSIGN(auto tensor_id, test_executor_->CreateValue(tensor));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto fed_val_id, test_executor_->CreateValue(FederatedValueAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(fed_val_id, tensor_id));
  ExpectMaterialize(result_id, ServerV(tensor));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedZipAtClients) {
  v0::Value v1 = ClientsV({TensorV(1)}, true);
  v0::Value v2 = ClientsV({TensorV(2)}, true);
  auto merged_struct = StructV({TensorV(1), TensorV(2)});
  v0::Value merged = ClientsV({merged_struct}, true);
  for (const auto& child : mock_children_) {
    auto child_v1 = child->ExpectCreateValue(v1);
    auto child_v2 = child->ExpectCreateValue(v2);
    auto child_zip = child->ExpectCreateValue(FederatedZipAtClientsV());
    auto child_zip_arg = child->ExpectCreateStruct({child_v1, child_v2});
    auto child_res = child->ExpectCreateCall(child_zip, child_zip_arg);
    child->ExpectMaterialize(child_res, merged);
  }
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateValue(StructV({v1, v2})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto res_id,
                           test_executor_->CreateCall(zip_id, arg_id));
  ExpectMaterialize(
      res_id, ClientsV(std::vector<v0::Value>(total_clients_, merged_struct)));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedZipAtServer) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  ValueId v1_child_id = mock_server_->ExpectCreateValue(v1);
  ValueId v2_child_id = mock_server_->ExpectCreateValue(v2);
  TFF_ASSERT_OK_AND_ASSIGN(auto v_id, test_executor_->CreateValue(
                                          StructV({ServerV(v1), ServerV(v2)})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  ValueId struct_child_id =
      mock_server_->ExpectCreateStruct({v1_child_id, v2_child_id});
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  mock_server_->ExpectMaterialize(struct_child_id, StructV({v1, v2}));
  ExpectMaterialize(result_id, ServerV(StructV({v1, v2})));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedZipDifferentPlacementsFails) {
  v0::Value v1 = ClientsV({TensorV(1)}, true);
  v0::Value v2_inner = TensorV(2);
  v0::Value v2 = ServerV(v2_inner);
  for (const auto& child : mock_children_) {
    child->ExpectCreateValue(v1);
  }
  mock_server_->ExpectCreateValue(v2_inner);
  TFF_ASSERT_OK_AND_ASSIGN(auto arg_id,
                           test_executor_->CreateValue(StructV({v1, v2})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto res_id,
                           test_executor_->CreateCall(zip_id, arg_id));
  EXPECT_THAT(test_executor_->Materialize(res_id),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ComposingExecutorTest, CreateCallFederatedZipUnplacedFails) {
  v0::Value v1 = TensorV(1);
  v0::Value v2 = TensorV(2);
  TFF_ASSERT_OK_AND_ASSIGN(auto v_id,
                           test_executor_->CreateValue(StructV({v1, v2})));
  TFF_ASSERT_OK_AND_ASSIGN(
      auto zip_id, test_executor_->CreateValue(FederatedZipAtServerV()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result_id,
                           test_executor_->CreateCall(zip_id, v_id));
  EXPECT_THAT(test_executor_->Materialize(result_id),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace

}  // namespace tensorflow_federated
