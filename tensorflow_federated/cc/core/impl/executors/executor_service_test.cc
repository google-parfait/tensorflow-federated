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

#include "tensorflow_federated/cc/core/impl/executors/executor_service.h"

#include <memory>
#include <optional>

#include "net/grpc/public/include/grpcpp/impl/codegen/server_context.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.proto.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.proto.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

static constexpr char kClients[] = "clients";

v0::SetCardinalitiesRequest CreateSetCardinalitiesRequest(
    int client_cardinalities) {
  v0::SetCardinalitiesRequest request_pb;

  v0::SetCardinalitiesRequest::Cardinality* cardinalities =
      request_pb.mutable_cardinalities()->Add();
  cardinalities->mutable_placement()->mutable_uri()->assign(kClients);
  cardinalities->set_cardinality(client_cardinalities);
  return request_pb;
}

v0::CreateValueRequest CreateValueFloatRequest(float float_value) {
  v0::CreateValueRequest request_pb;
  request_pb.mutable_value()->MergeFrom(testing::TensorV(float_value));
  return request_pb;
}

v0::DisposeRequest DisposeRequestForIds(absl::Span<const std::string> ids) {
  v0::DisposeRequest request_pb;
  for (const std::string& id : ids) {
    v0::ValueRef value_ref;
    value_ref.mutable_id()->assign(id);
    request_pb.mutable_value_ref()->Add(std::move(value_ref));
  }
  return request_pb;
}

v0::ComputeRequest ComputeRequestForId(std::string id) {
  v0::ComputeRequest compute_request_pb;
  compute_request_pb.mutable_value_ref()->mutable_id()->assign(id);
  return compute_request_pb;
}

absl::Status ReturnOk() { return absl::OkStatus(); }

v0::CreateCallRequest CreateCallRequestForIds(
    std::string function_id, std::optional<std::string> argument_id) {
  v0::CreateCallRequest create_call_request_pb;
  create_call_request_pb.mutable_function_ref()->mutable_id()->assign(
      function_id);
  if (argument_id != std::nullopt) {
    create_call_request_pb.mutable_argument_ref()->mutable_id()->assign(
        *argument_id);
  }
  return create_call_request_pb;
}

v0::CreateStructRequest CreateStructForIds(
    const absl::Span<const absl::string_view> ids_for_struct) {
  v0::CreateStructRequest create_struct_request_pb;
  for (const std::string_view& id : ids_for_struct) {
    v0::CreateStructRequest::Element elem;
    elem.mutable_value_ref()->mutable_id()->append(id);
    create_struct_request_pb.mutable_element()->Add(std::move(elem));
  }
  return create_struct_request_pb;
}

v0::CreateStructRequest CreateNamedStructForIds(
    const absl::Span<const absl::string_view> ids_for_struct) {
  v0::CreateStructRequest create_struct_request_pb;
  // Assign an integer index as name internally. Names are dropped on the C++
  // side, but a caller may supply them.
  int idx = 0;
  for (const std::string_view& id : ids_for_struct) {
    v0::CreateStructRequest::Element elem;
    elem.mutable_value_ref()->mutable_id()->assign(id);
    elem.mutable_name()->assign(std::to_string(idx));
    idx++;
    create_struct_request_pb.mutable_element()->Add(std::move(elem));
  }
  return create_struct_request_pb;
}

v0::CreateSelectionRequest CreateSelectionRequestForIndex(
    std::string source_ref_id, int index) {
  v0::CreateSelectionRequest create_selection_request_pb;
  create_selection_request_pb.mutable_source_ref()->mutable_id()->assign(
      source_ref_id);
  create_selection_request_pb.set_index(index);
  return create_selection_request_pb;
}

TEST(ExecutorServiceFailureTest, CreateValueBeforeSetCardinalities) {
  auto executor_ptr = std::make_shared<::testing::StrictMock<MockExecutor>>();
  ExecutorService executor_service_ =
      ExecutorService([&](auto cardinalities) { return executor_ptr; });

  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  auto response_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);

  ASSERT_THAT(
      response_status,
      ::testing::status::StatusIs(
          grpc::StatusCode::UNAVAILABLE,
          ::testing::HasSubstr("CreateValue before setting cardinalities")));
}

class ExecutorServiceTest : public ::testing::Test {
 public:
  absl::StatusOr<OwnedValueId> TestId(uint64_t id) {
    return OwnedValueId(executor_ptr_, id);
  }

 private:
  void SetUp() override {
    const v0::SetCardinalitiesRequest request_pb =
        CreateSetCardinalitiesRequest(1);
    v0::SetCardinalitiesResponse response_pb;
    grpc::ServerContext server_context;
    auto ok_status = executor_service_.SetCardinalities(
        &server_context, &request_pb, &response_pb);
    ASSERT_OK(ok_status);
  }

 protected:
  ExecutorService executor_service_ = ExecutorService(
      [&](auto cardinalities) { return *(&this->executor_ptr_); });
  std::shared_ptr<MockExecutor> executor_ptr_ =
      std::make_shared<::testing::StrictMock<MockExecutor>>();
};

TEST_F(ExecutorServiceTest, SetCardinalitiesReturnsOK) {
  int client_cards = 5;
  auto request_pb = CreateSetCardinalitiesRequest(client_cards);
  v0::SetCardinalitiesResponse response_pb;
  grpc::ServerContext server_context;

  auto response_status = executor_service_.SetCardinalities(
      &server_context, &request_pb, &response_pb);

  ASSERT_THAT(response_status, ::testing::status::IsOk());
}

TEST_F(ExecutorServiceTest, CreateValueReturnsZeroRef) {
  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateValue(::testing::_)).WillOnce([this] {
    return TestId(0);
  });

  auto response_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);

  ASSERT_THAT(response_status, ::testing::status::IsOk());
  // First element in the id is the id in the mock executor; the second is the
  // executor's generation.
  EXPECT_THAT(response_pb, ::testing::EqualsProto("value_ref { id: '0-0' }"));
}

TEST_F(ExecutorServiceTest, SetCardinalitiesIncrementsExecutorGeneration) {
  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse first_response_pb;
  v0::CreateValueResponse second_response_pb;
  grpc::ServerContext server_context;

  v0::SetCardinalitiesRequest set_cardinalities_request_pb =
      CreateSetCardinalitiesRequest(1);
  v0::SetCardinalitiesResponse set_cardinalities_response_pb;

  EXPECT_CALL(*executor_ptr_, CreateValue(::testing::_)).WillRepeatedly([this] {
    return TestId(0);
  });

  // A second set cardinalities call increments the executor generation in the
  // service.
  auto first_response_status = executor_service_.CreateValue(
      &server_context, &request_pb, &first_response_pb);
  auto ok_status = executor_service_.SetCardinalities(
      &server_context, &set_cardinalities_request_pb,
      &set_cardinalities_response_pb);
  auto second_response_status = executor_service_.CreateValue(
      &server_context, &request_pb, &second_response_pb);

  ASSERT_THAT(first_response_status, ::testing::status::IsOk());
  EXPECT_THAT(first_response_pb,
              ::testing::EqualsProto("value_ref { id: '0-0' }"));
  ASSERT_THAT(second_response_status, ::testing::status::IsOk());
  EXPECT_THAT(second_response_pb,
              ::testing::EqualsProto("value_ref { id: '0-1' }"));
}

TEST_F(ExecutorServiceTest, ComputeWithMalformedRefFails) {
  v0::ComputeResponse compute_response_pb;
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  v0::ComputeRequest compute_request_pb = ComputeRequestForId("malformed_id");

  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(
      compute_response_status,
      ::testing::status::StatusIs(
          grpc::StatusCode::INVALID_ARGUMENT,
          ::testing::HasSubstr("Remote value ID malformed_id malformed")));
}

TEST_F(ExecutorServiceTest, ComputeWithNoIntsInRefFails) {
  v0::ComputeResponse compute_response_pb;
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  v0::ComputeRequest compute_request_pb = ComputeRequestForId("malformed-id");

  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(
      compute_response_status,
      ::testing::status::StatusIs(
          grpc::StatusCode::INVALID_ARGUMENT,
          ::testing::HasSubstr("Remote value ID malformed-id malformed")));
}

TEST_F(ExecutorServiceTest, ComputeWithDashOnlyFails) {
  v0::ComputeResponse compute_response_pb;
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  v0::ComputeRequest compute_request_pb = ComputeRequestForId("-");

  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(compute_response_status,
              ::testing::status::StatusIs(
                  grpc::StatusCode::INVALID_ARGUMENT,
                  ::testing::HasSubstr("Remote value ID - malformed")));
}

TEST_F(ExecutorServiceTest, ComputeUnknownRefForwardsFromMock) {
  v0::ComputeResponse compute_response_pb;
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  // This value does not exist in the lower-level executor, as it has not been
  // preceded by a create_value call.
  v0::ComputeRequest compute_request_pb = ComputeRequestForId("0-0");
  EXPECT_CALL(*executor_ptr_, Materialize(::testing::_, ::testing::_))
      .WillOnce([](ValueId id, v0::Value* val) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "Unknown value ref");
      });

  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(
      compute_response_status,
      ::testing::status::StatusIs(grpc::StatusCode::INVALID_ARGUMENT,
                                  ::testing::HasSubstr("Unknown value ref")));
}

TEST_F(ExecutorServiceTest, ComputeBadGenerationFails) {
  v0::ComputeResponse compute_response_pb;
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  // The 0th executor generation is the live one per the test fixture setup.
  v0::ComputeRequest compute_request_pb = ComputeRequestForId("0-1");

  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(compute_response_status,
              ::testing::status::StatusIs(
                  grpc::StatusCode::INVALID_ARGUMENT,
                  ::testing::HasSubstr("non-live executor generation.")));
}

TEST_F(ExecutorServiceTest, ComputeReturnsMockValue) {
  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse create_value_response_pb;
  v0::ComputeResponse compute_response_pb;
  // We will return this value from the mock's materialize and expect it to
  // come out of the service's compute.
  v0::Value expected_value = testing::TensorV(3.0f);
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateValue(::testing::_)).WillOnce([this] {
    return TestId(0);
  });
  EXPECT_CALL(*executor_ptr_, Materialize(::testing::_, ::testing::_))
      .WillOnce([&expected_value](ValueId id, v0::Value* val) {
        val->CopyFrom(expected_value);
        return grpc::Status::OK;
      });

  auto create_value_response_status = executor_service_.CreateValue(
      &server_context, &request_pb, &create_value_response_pb);
  ASSERT_THAT(create_value_response_status, ::testing::status::IsOk());

  v0::ComputeRequest compute_request_pb =
      ComputeRequestForId(create_value_response_pb.value_ref().id());
  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(compute_response_status, ::testing::status::IsOk());
  EXPECT_THAT(compute_response_pb.value(),
              ::testing::EqualsProto(expected_value));
}

TEST_F(ExecutorServiceTest, ComputeTwoValuesReturnsAppropriateValues) {
  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse create_value_response_pb;
  v0::ComputeResponse first_compute_response_pb;
  v0::ComputeResponse second_compute_response_pb;
  // We will return this value from the mock's materialize and expect it to
  // come out of the service's compute.
  v0::Value expected_three = testing::TensorV(3.0f);
  v0::Value expected_four = testing::TensorV(4.0f);
  v0::CreateValueResponse first_value_response_pb;
  v0::CreateValueResponse second_value_response_pb;
  grpc::ServerContext server_context;

  // We expect two create value calls, which should return different ids.
  EXPECT_CALL(*executor_ptr_, CreateValue(::testing::_))
      .WillOnce([this] { return TestId(0); })
      .WillOnce([this] { return TestId(1); });

  // We expect materializing the 0th id to retun 3, the 1st to return 4.
  EXPECT_CALL(*executor_ptr_, Materialize(::testing::_, ::testing::_))
      .WillRepeatedly([&expected_three, &expected_four](ValueId id,
                                                        v0::Value* val) {
        if (id == 0) {
          val->CopyFrom(expected_three);
          return grpc::Status::OK;
        } else if (id == 1) {
          val->CopyFrom(expected_four);
          return grpc::Status::OK;
        } else {
          return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Unknown id");
        }
      });

  auto first_create_value_response_status = executor_service_.CreateValue(
      &server_context, &request_pb, &first_value_response_pb);
  auto second_create_value_response_status = executor_service_.CreateValue(
      &server_context, &request_pb, &second_value_response_pb);

  ASSERT_THAT(first_create_value_response_status, ::testing::status::IsOk());
  ASSERT_THAT(second_create_value_response_status, ::testing::status::IsOk());

  v0::ComputeRequest first_compute_request_pb =
      ComputeRequestForId(first_value_response_pb.value_ref().id());
  v0::ComputeRequest second_compute_request_pb =
      ComputeRequestForId(second_value_response_pb.value_ref().id());
  auto first_compute_response_status = executor_service_.Compute(
      &server_context, &first_compute_request_pb, &first_compute_response_pb);
  auto second_compute_response_status = executor_service_.Compute(
      &server_context, &second_compute_request_pb, &second_compute_response_pb);
  ASSERT_THAT(first_compute_response_status, ::testing::status::IsOk());
  ASSERT_THAT(second_compute_response_status, ::testing::status::IsOk());

  // We expect materializing the 0th id to retun 3, the 1st to return 4.
  EXPECT_THAT(first_compute_response_pb.value(),
              ::testing::EqualsProto(expected_three));
  EXPECT_THAT(second_compute_response_pb.value(),
              ::testing::EqualsProto(expected_four));
}

TEST_F(ExecutorServiceTest, DisposePassesCallsDown) {
  auto dispose_request = DisposeRequestForIds({"0-0", "1-0"});
  v0::DisposeResponse dispose_response;
  grpc::ServerContext server_context;

  // We expect two forwarded dispose calls with appropriate IDs
  EXPECT_CALL(*executor_ptr_, Dispose(0)).WillOnce(ReturnOk);
  EXPECT_CALL(*executor_ptr_, Dispose(1)).WillOnce(ReturnOk);
  auto dispose_status = executor_service_.Dispose(
      &server_context, &dispose_request, &dispose_response);
  ASSERT_THAT(dispose_status, ::testing::status::IsOk());
}

TEST_F(ExecutorServiceTest, DisposeFiltersBadGeneration) {
  auto dispose_request =
      DisposeRequestForIds(std::vector<std::string>{"0-0", "1-1"});
  v0::DisposeResponse dispose_response;
  grpc::ServerContext server_context;

  // We expect one forwarded dispose call, as the generation of the second id is
  // not live.
  EXPECT_CALL(*executor_ptr_, Dispose(0)).WillOnce(ReturnOk);
  auto dispose_status = executor_service_.Dispose(
      &server_context, &dispose_request, &dispose_response);
  ASSERT_THAT(dispose_status, ::testing::status::IsOk());
}

TEST_F(ExecutorServiceTest, ClearExecutorThenCreateValueFails) {
  v0::ClearExecutorRequest clear_executor_request;
  v0::ClearExecutorResponse clear_executor_response;
  grpc::ServerContext server_context;

  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse response_pb;

  auto clear_executor_status = executor_service_.ClearExecutor(
      &server_context, &clear_executor_request, &clear_executor_response);
  ASSERT_THAT(clear_executor_status, ::testing::status::IsOk());

  auto create_value_response_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);

  ASSERT_THAT(
      create_value_response_status,
      ::testing::status::StatusIs(
          grpc::StatusCode::UNAVAILABLE,
          ::testing::HasSubstr("CreateValue before setting cardinalities")));
}

TEST_F(ExecutorServiceTest, ClearExecutorThenDisposeFails) {
  v0::ClearExecutorRequest clear_executor_request;
  v0::ClearExecutorResponse clear_executor_response;
  auto dispose_request = DisposeRequestForIds({"0-0"});
  v0::DisposeResponse dispose_response;
  grpc::ServerContext server_context;

  auto clear_executor_status = executor_service_.ClearExecutor(
      &server_context, &clear_executor_request, &clear_executor_response);
  ASSERT_THAT(clear_executor_status, ::testing::status::IsOk());

  auto dispose_response_status = executor_service_.Dispose(
      &server_context, &dispose_request, &dispose_response);

  ASSERT_THAT(
      dispose_response_status,
      ::testing::status::StatusIs(
          grpc::StatusCode::UNAVAILABLE,
          ::testing::HasSubstr("Dispose before setting cardinalities")));
}

TEST_F(ExecutorServiceTest, CreateCallNoArgFnArgumentSetToEmptyString) {
  // The argument ref in the associated create call request will be marked as
  // set, but to an empty string.
  v0::CreateCallRequest call_request = CreateCallRequestForIds("0-0", "");
  v0::CreateCallResponse create_call_response_pb;
  grpc::ServerContext server_context;

  auto create_call_response_status = executor_service_.CreateCall(
      &server_context, &call_request, &create_call_response_pb);

  ASSERT_THAT(create_call_response_status,
              ::testing::status::StatusIs(grpc::StatusCode::INVALID_ARGUMENT,
                                          ::testing::HasSubstr("malformed")));
}

TEST_F(ExecutorServiceTest, CreateCallNoArgFn) {
  v0::CreateCallRequest call_request =
      CreateCallRequestForIds("0-0", std::nullopt);
  v0::CreateCallResponse create_call_response_pb;
  grpc::ServerContext server_context;

  // We expect the ID returned from this call to be set reflected in the
  // returned value.
  EXPECT_CALL(*executor_ptr_, CreateCall(0, ::testing::Eq(std::nullopt)))
      .WillOnce([this] { return TestId(1); });

  auto create_call_response_status = executor_service_.CreateCall(
      &server_context, &call_request, &create_call_response_pb);

  ASSERT_THAT(create_call_response_status, ::testing::status::IsOk());
  EXPECT_THAT(create_call_response_pb,
              ::testing::EqualsProto("value_ref { id: '1-0' }"));
}

TEST_F(ExecutorServiceTest, CreateCallFunctionWithArgument) {
  v0::CreateCallRequest call_request = CreateCallRequestForIds("0-0", "1-0");
  v0::CreateCallResponse create_call_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateCall(0, ::testing::Optional(1)))
      .WillOnce([this] { return TestId(2); });

  auto create_call_response_status = executor_service_.CreateCall(
      &server_context, &call_request, &create_call_response_pb);

  ASSERT_THAT(create_call_response_status, ::testing::status::IsOk());
  EXPECT_THAT(create_call_response_pb,
              ::testing::EqualsProto("value_ref { id: '2-0' }"));
}

TEST_F(ExecutorServiceTest, CreateSelection) {
  v0::CreateSelectionRequest first_selection_request =
      CreateSelectionRequestForIndex("0-0", 1);
  v0::CreateSelectionRequest second_selection_request =
      CreateSelectionRequestForIndex("2-0", 2);
  v0::CreateSelectionResponse first_create_selection_response_pb;
  v0::CreateSelectionResponse second_create_selection_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateSelection(0, 1)).WillOnce([this] {
    return TestId(1);
  });
  EXPECT_CALL(*executor_ptr_, CreateSelection(2, 2)).WillOnce([this] {
    return TestId(3);
  });

  auto first_create_selection_response_status =
      executor_service_.CreateSelection(&server_context,
                                        &first_selection_request,
                                        &first_create_selection_response_pb);

  auto second_create_selection_response_status =
      executor_service_.CreateSelection(&server_context,
                                        &second_selection_request,
                                        &second_create_selection_response_pb);

  ASSERT_THAT(first_create_selection_response_status,
              ::testing::status::IsOk());
  ASSERT_THAT(second_create_selection_response_status,
              ::testing::status::IsOk());
  EXPECT_THAT(first_create_selection_response_pb,
              ::testing::EqualsProto("value_ref { id: '1-0' }"));
  EXPECT_THAT(second_create_selection_response_pb,
              ::testing::EqualsProto("value_ref { id: '3-0' }"));
}

TEST_F(ExecutorServiceTest, CreateStructFailsWithBadGeneration) {
  v0::CreateStructRequest struct_request = CreateStructForIds({"0-0", "0-1"});
  v0::CreateStructResponse struct_response_pb;
  grpc::ServerContext server_context;

  auto create_struct_response_status = executor_service_.CreateStruct(
      &server_context, &struct_request, &struct_response_pb);

  ASSERT_THAT(create_struct_response_status,
              ::testing::status::StatusIs(
                  grpc::StatusCode::INVALID_ARGUMENT,
                  ::testing::HasSubstr("non-live executor generation.")));
}

TEST_F(ExecutorServiceTest, CreateEmptyStruct) {
  v0::CreateStructRequest struct_request = CreateStructForIds({});
  v0::CreateStructResponse struct_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_,
              CreateStruct(::testing::Eq(std::vector<ValueId>{})))
      .WillOnce([this] { return TestId(0); });

  auto create_struct_response_status = executor_service_.CreateStruct(
      &server_context, &struct_request, &struct_response_pb);

  ASSERT_THAT(create_struct_response_status, ::testing::status::IsOk());
  EXPECT_THAT(struct_response_pb,
              ::testing::EqualsProto("value_ref { id: '0-0' }"));
}

TEST_F(ExecutorServiceTest, CreateNonemptyStruct) {
  v0::CreateStructRequest struct_request = CreateStructForIds({"0-0", "1-0"});
  v0::CreateStructResponse struct_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_,
              CreateStruct(::testing::Eq(std::vector<ValueId>{0, 1})))
      .WillOnce([this] { return TestId(0); });

  auto create_struct_response_status = executor_service_.CreateStruct(
      &server_context, &struct_request, &struct_response_pb);

  ASSERT_THAT(create_struct_response_status, ::testing::status::IsOk());
  EXPECT_THAT(struct_response_pb,
              ::testing::EqualsProto("value_ref { id: '0-0' }"));
}

TEST_F(ExecutorServiceTest, CreateNamedNonemptyStruct) {
  v0::CreateStructRequest struct_request =
      CreateNamedStructForIds({"0-0", "1-0"});
  v0::CreateStructResponse struct_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_,
              CreateStruct(::testing::Eq(std::vector<ValueId>{0, 1})))
      .WillOnce([this] { return TestId(0); });

  auto create_struct_response_status = executor_service_.CreateStruct(
      &server_context, &struct_request, &struct_response_pb);

  ASSERT_THAT(create_struct_response_status, ::testing::status::IsOk());
  EXPECT_THAT(struct_response_pb,
              ::testing::EqualsProto("value_ref { id: '0-0' }"));
}

}  // namespace tensorflow_federated
