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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

static constexpr char kClients[] = "clients";

v0::GetExecutorRequest CreateGetExecutorRequest(int client_cardinalities) {
  v0::GetExecutorRequest request_pb;

  v0::Cardinality* cardinalities = request_pb.mutable_cardinalities()->Add();
  cardinalities->mutable_placement()->mutable_uri()->assign(kClients);
  cardinalities->set_cardinality(client_cardinalities);
  return request_pb;
}

absl::Status ReturnOk() { return absl::OkStatus(); }

TEST(ExecutorServiceFailureTest, CreateValueWithoutExecutorFails) {
  auto executor_ptr = std::make_shared<::testing::StrictMock<MockExecutor>>();
  ExecutorService executor_service_(
      [&](auto cardinalities) { return executor_ptr; });

  v0::CreateValueRequest request_pb;
  request_pb.mutable_value()->MergeFrom(testing::TensorV(1.0));
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  auto response_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);

  ASSERT_THAT(response_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID: ''."));
}

class ExecutorServiceTest : public ::testing::Test {
 public:
  ExecutorServiceTest()
      : executor_ptr_(std::make_shared<::testing::StrictMock<MockExecutor>>()),
        executor_service_([&](const CardinalityMap& cardinalities)
                              -> std::shared_ptr<Executor> {
          return *(&this->executor_ptr_);
        }) {}
  absl::StatusOr<OwnedValueId> TestId(uint64_t id) {
    return OwnedValueId(executor_ptr_, id);
  }

 private:
  void SetUp() override {
    const v0::GetExecutorRequest request_pb = CreateGetExecutorRequest(1);
    v0::GetExecutorResponse response_pb;
    grpc::ServerContext server_context;
    auto ok_status = executor_service_.GetExecutor(&server_context, &request_pb,
                                                   &response_pb);
    TFF_ASSERT_OK(grpc_to_absl(ok_status));
    executor_pb_ = response_pb.executor();
  }

 protected:
  std::shared_ptr<MockExecutor> executor_ptr_;
  ExecutorService executor_service_;
  v0::ExecutorId executor_pb_;

  v0::CreateValueRequest CreateValueFloatRequest(float float_value) {
    v0::CreateValueRequest request_pb;
    *request_pb.mutable_executor() = executor_pb_;
    request_pb.mutable_value()->MergeFrom(testing::TensorV(float_value));
    return request_pb;
  }

  v0::DisposeRequest DisposeRequestForIds(absl::Span<const std::string> ids) {
    v0::DisposeRequest request_pb;
    *request_pb.mutable_executor() = executor_pb_;
    for (const std::string& id : ids) {
      v0::ValueRef value_ref;
      value_ref.mutable_id()->assign(id);
      request_pb.mutable_value_ref()->Add(std::move(value_ref));
    }
    return request_pb;
  }

  v0::ComputeRequest ComputeRequestForId(std::string id) {
    v0::ComputeRequest compute_request_pb;
    *compute_request_pb.mutable_executor() = executor_pb_;
    compute_request_pb.mutable_value_ref()->mutable_id()->assign(id);
    return compute_request_pb;
  }

  v0::CreateCallRequest CreateCallRequestForIds(
      std::string function_id, std::optional<std::string> argument_id) {
    v0::CreateCallRequest create_call_request_pb;
    *create_call_request_pb.mutable_executor() = executor_pb_;
    create_call_request_pb.mutable_function_ref()->mutable_id()->assign(
        function_id);
    if (argument_id != std::nullopt) {
      create_call_request_pb.mutable_argument_ref()->mutable_id()->assign(
          *argument_id);
    }
    return create_call_request_pb;
  }

  v0::CreateStructRequest CreateStructForIds(
      const absl::Span<const std::string_view> ids_for_struct) {
    v0::CreateStructRequest create_struct_request_pb;
    *create_struct_request_pb.mutable_executor() = executor_pb_;
    for (std::string_view id : ids_for_struct) {
      v0::CreateStructRequest::Element elem;
      elem.mutable_value_ref()->mutable_id()->append(id.data(), id.size());
      create_struct_request_pb.mutable_element()->Add(std::move(elem));
    }
    return create_struct_request_pb;
  }

  v0::CreateStructRequest CreateNamedStructForIds(
      const absl::Span<const std::string_view> ids_for_struct) {
    v0::CreateStructRequest create_struct_request_pb;
    *create_struct_request_pb.mutable_executor() = executor_pb_;
    // Assign an integer index as name internally. Names are dropped on the C++
    // side, but a caller may supply them.
    int idx = 0;
    for (const std::string_view& id : ids_for_struct) {
      v0::CreateStructRequest::Element elem;
      elem.mutable_value_ref()->mutable_id()->assign(id.data(), id.size());
      elem.mutable_name()->assign(std::to_string(idx));
      idx++;
      create_struct_request_pb.mutable_element()->Add(std::move(elem));
    }
    return create_struct_request_pb;
  }

  v0::CreateSelectionRequest CreateSelectionRequestForIndex(
      std::string source_ref_id, int index) {
    v0::CreateSelectionRequest create_selection_request_pb;
    *create_selection_request_pb.mutable_executor() = executor_pb_;
    create_selection_request_pb.mutable_source_ref()->mutable_id()->assign(
        source_ref_id);
    create_selection_request_pb.set_index(index);
    return create_selection_request_pb;
  }
};

TEST_F(ExecutorServiceTest, GetExecutorReturnsOk) {
  int client_cards = 5;
  auto request_pb = CreateGetExecutorRequest(client_cards);
  v0::GetExecutorResponse response_pb;
  grpc::ServerContext server_context;

  TFF_EXPECT_OK(grpc_to_absl(executor_service_.GetExecutor(
      &server_context, &request_pb, &response_pb)));
}

TEST_F(ExecutorServiceTest, CreateValueReturnsZeroRef) {
  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateValue(::testing::_)).WillOnce([this] {
    return TestId(0);
  });

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateValue(
      &server_context, &request_pb, &response_pb)));
  // First element in the id is the id in the mock executor; the second is the
  // executor's generation.
  EXPECT_THAT(response_pb, testing::EqualsProto("value_ref { id: '0' }"));
}

TEST_F(ExecutorServiceTest, CreateValueFailedPreconditionDestroysExecutor) {
  // If an executor returns FailedPrecondition, the service must invalidate any
  // outstanding references to this executor. This test asserts that once the
  // underlying executor indicates it needs configuring, the service will fail
  // to resolve requests for this executor.
  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateValue(::testing::_)).WillOnce([] {
    return absl::FailedPreconditionError("Needs setting");
  });

  auto value_response_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);
  ASSERT_THAT(
      value_response_status,
      GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION, "Needs setting"));
  auto no_executor_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);
  ASSERT_THAT(no_executor_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID"));
}

TEST_F(ExecutorServiceTest, CreateCallFailedPreconditionDestroysExecutor) {
  auto request_pb = CreateCallRequestForIds("0", std::nullopt);
  v0::CreateCallResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateCall(::testing::_, ::testing::_))
      .WillOnce([] { return absl::FailedPreconditionError("Needs setting"); });

  auto value_response_status =
      executor_service_.CreateCall(&server_context, &request_pb, &response_pb);
  ASSERT_THAT(
      value_response_status,
      GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION, "Needs setting"));
  auto no_executor_status =
      executor_service_.CreateCall(&server_context, &request_pb, &response_pb);
  ASSERT_THAT(no_executor_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID"));
}

TEST_F(ExecutorServiceTest, CreateSelectionFailedPreconditionDestroysExecutor) {
  auto request_pb = CreateSelectionRequestForIndex("0", 0);
  v0::CreateSelectionResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateSelection(::testing::_, ::testing::_))
      .WillOnce([] { return absl::FailedPreconditionError("Needs setting"); });

  auto value_response_status = executor_service_.CreateSelection(
      &server_context, &request_pb, &response_pb);
  ASSERT_THAT(
      value_response_status,
      GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION, "Needs setting"));
  auto no_executor_status = executor_service_.CreateSelection(
      &server_context, &request_pb, &response_pb);
  ASSERT_THAT(no_executor_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID"));
}

TEST_F(ExecutorServiceTest, CreateStructFailedPreconditionDestroysExecutor) {
  auto request_pb = CreateStructForIds({"0"});
  v0::CreateStructResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateStruct(::testing::_)).WillOnce([] {
    return absl::FailedPreconditionError("Needs setting");
  });

  auto value_response_status = executor_service_.CreateStruct(
      &server_context, &request_pb, &response_pb);
  ASSERT_THAT(
      value_response_status,
      GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION, "Needs setting"));
  auto no_executor_status = executor_service_.CreateStruct(
      &server_context, &request_pb, &response_pb);
  ASSERT_THAT(no_executor_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID"));
}

TEST_F(ExecutorServiceTest, ComputeFailedPreconditionDestroysExecutor) {
  auto request_pb = ComputeRequestForId("0");
  v0::ComputeResponse response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, Materialize(::testing::_, ::testing::_))
      .WillOnce([] { return absl::FailedPreconditionError("Needs setting"); });

  auto value_response_status =
      executor_service_.Compute(&server_context, &request_pb, &response_pb);
  ASSERT_THAT(
      value_response_status,
      GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION, "Needs setting"));
  auto no_executor_status =
      executor_service_.Compute(&server_context, &request_pb, &response_pb);
  ASSERT_THAT(no_executor_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID"));
}

TEST_F(ExecutorServiceTest, GetExecutorReturnsCardinalitySpecificIds) {
  grpc::ServerContext context;

  v0::GetExecutorRequest get_executor_request_1 = CreateGetExecutorRequest(1);
  v0::GetExecutorResponse get_executor_response_1;

  v0::GetExecutorRequest get_executor_request_2 = CreateGetExecutorRequest(2);
  v0::GetExecutorResponse get_executor_response_2;

  TFF_EXPECT_OK(grpc_to_absl(executor_service_.GetExecutor(
      &context, &get_executor_request_1, &get_executor_response_1)));
  TFF_EXPECT_OK(grpc_to_absl(executor_service_.GetExecutor(
      &context, &get_executor_request_2, &get_executor_response_2)));

  std::string first_ex_id = get_executor_response_1.executor().id();
  ASSERT_THAT(first_ex_id, ::testing::HasSubstr("clients=1"));

  std::string second_ex_id = get_executor_response_2.executor().id();
  ASSERT_THAT(second_ex_id, ::testing::HasSubstr("clients=2"));
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
      GrpcStatusIs(
          grpc::StatusCode::INVALID_ARGUMENT,
          "Expected value ref to be an integer id, found malformed_id"));
}

TEST_F(ExecutorServiceTest, ComputeUnknownRefForwardsFromMock) {
  v0::ComputeResponse compute_response_pb;
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  // This value does not exist in the lower-level executor, as it has not been
  // preceded by a create_value call.
  v0::ComputeRequest compute_request_pb = ComputeRequestForId("0");
  EXPECT_CALL(*executor_ptr_, Materialize(::testing::_, ::testing::_))
      .WillOnce([](ValueId id, v0::Value* val) {
        return absl::InvalidArgumentError("Unknown value ref");
      });

  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(
      compute_response_status,
      GrpcStatusIs(grpc::StatusCode::INVALID_ARGUMENT, "Unknown value ref"));
}

TEST_F(ExecutorServiceTest, ComputeInvalidExecutorFails) {
  v0::ComputeResponse compute_response_pb;
  v0::CreateValueResponse response_pb;
  grpc::ServerContext server_context;

  // The 0th executor generation is the live one per the test fixture setup.
  v0::ComputeRequest compute_request_pb;
  compute_request_pb.mutable_executor()->set_id("booyeah");

  auto compute_response_status = executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb);
  ASSERT_THAT(compute_response_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID: 'booyeah'."));
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
        *val = expected_value;
        return absl::OkStatus();
      });

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateValue(
      &server_context, &request_pb, &create_value_response_pb)));

  v0::ComputeRequest compute_request_pb =
      ComputeRequestForId(create_value_response_pb.value_ref().id());
  TFF_ASSERT_OK(grpc_to_absl(executor_service_.Compute(
      &server_context, &compute_request_pb, &compute_response_pb)));
  EXPECT_THAT(compute_response_pb.value(),
              testing::EqualsProto(expected_value));
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
      .WillRepeatedly(
          [&expected_three, &expected_four](ValueId id, v0::Value* val) {
            if (id == 0) {
              *val = expected_three;
            } else if (id == 1) {
              *val = expected_four;
            } else {
              return absl::InvalidArgumentError("Unknown id");
            }
            return absl::OkStatus();
          });

  auto first_create_value_response_status = executor_service_.CreateValue(
      &server_context, &request_pb, &first_value_response_pb);
  auto second_create_value_response_status = executor_service_.CreateValue(
      &server_context, &request_pb, &second_value_response_pb);

  TFF_ASSERT_OK(grpc_to_absl(first_create_value_response_status));
  TFF_ASSERT_OK(grpc_to_absl(second_create_value_response_status));

  v0::ComputeRequest first_compute_request_pb =
      ComputeRequestForId(first_value_response_pb.value_ref().id());
  v0::ComputeRequest second_compute_request_pb =
      ComputeRequestForId(second_value_response_pb.value_ref().id());
  auto first_compute_response_status = executor_service_.Compute(
      &server_context, &first_compute_request_pb, &first_compute_response_pb);
  auto second_compute_response_status = executor_service_.Compute(
      &server_context, &second_compute_request_pb, &second_compute_response_pb);
  TFF_ASSERT_OK(grpc_to_absl(first_compute_response_status));
  TFF_ASSERT_OK(grpc_to_absl(second_compute_response_status));

  // We expect materializing the 0th id to retun 3, the 1st to return 4.
  EXPECT_THAT(first_compute_response_pb.value(),
              testing::EqualsProto(expected_three));
  EXPECT_THAT(second_compute_response_pb.value(),
              testing::EqualsProto(expected_four));
}

TEST_F(ExecutorServiceTest, DisposePassesCallsDown) {
  auto dispose_request = DisposeRequestForIds({"0", "1"});
  v0::DisposeResponse dispose_response;
  grpc::ServerContext server_context;

  // We expect two forwarded dispose calls with appropriate IDs
  EXPECT_CALL(*executor_ptr_, Dispose(0)).WillOnce(ReturnOk);
  EXPECT_CALL(*executor_ptr_, Dispose(1)).WillOnce(ReturnOk);
  auto dispose_status = executor_service_.Dispose(
      &server_context, &dispose_request, &dispose_response);
  TFF_ASSERT_OK(grpc_to_absl(dispose_status));
}

TEST_F(ExecutorServiceTest, DisposeOnNonexistetExecutor) {
  auto dispose_request = DisposeRequestForIds({"0", "1"});
  *dispose_request.mutable_executor()->mutable_id() =
      "this_executor_does_not_exist";
  v0::DisposeResponse dispose_response;
  grpc::ServerContext server_context;

  // Nothing is passed down, but the call succeeds.
  auto dispose_status = executor_service_.Dispose(
      &server_context, &dispose_request, &dispose_response);
  TFF_ASSERT_OK(grpc_to_absl(dispose_status));
}

TEST_F(ExecutorServiceTest, DisposeExecutorThenCreateValueFails) {
  v0::DisposeExecutorRequest dispose_executor_request;
  *dispose_executor_request.mutable_executor() = executor_pb_;
  v0::DisposeExecutorResponse dispose_executor_response;
  grpc::ServerContext server_context;

  auto request_pb = CreateValueFloatRequest(2.0f);
  v0::CreateValueResponse response_pb;

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.DisposeExecutor(
      &server_context, &dispose_executor_request, &dispose_executor_response)));

  auto create_value_response_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);

  ASSERT_THAT(create_value_response_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID"));
}

TEST_F(ExecutorServiceTest, DisposeExecutorDoesntRemoveUnlessItsTheLastRef) {
  grpc::ServerContext server_context;

  // Create another ref by calling `GetExecutor` again.
  {
    auto request_pb = CreateGetExecutorRequest(1);
    v0::GetExecutorResponse response_pb;
    TFF_ASSERT_OK(grpc_to_absl(executor_service_.GetExecutor(
        &server_context, &request_pb, &response_pb)));
    EXPECT_THAT(response_pb.executor(), testing::EqualsProto(executor_pb_));
  }

  {
    v0::DisposeExecutorRequest request_pb;
    *request_pb.mutable_executor() = executor_pb_;
    v0::DisposeExecutorResponse response_pb;
    TFF_ASSERT_OK(grpc_to_absl(executor_service_.DisposeExecutor(
        &server_context, &request_pb, &response_pb)));
  }

  // Should still succeed-- one reference remains.
  auto request_pb = CreateValueFloatRequest(2.0f);
  EXPECT_CALL(*executor_ptr_, CreateValue(::testing::_)).WillOnce([this] {
    return TestId(0);
  });
  v0::CreateValueResponse response_pb;
  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateValue(
      &server_context, &request_pb, &response_pb)));
  {
    // A second DisposeEx, however, should remove the executor.
    v0::DisposeExecutorRequest request_pb;
    *request_pb.mutable_executor() = executor_pb_;
    v0::DisposeExecutorResponse response_pb;
    TFF_ASSERT_OK(grpc_to_absl(executor_service_.DisposeExecutor(
        &server_context, &request_pb, &response_pb)));
  }
  // So that this CreateValue call should fail.
  auto create_value_response_status =
      executor_service_.CreateValue(&server_context, &request_pb, &response_pb);
  ASSERT_THAT(create_value_response_status,
              GrpcStatusIs(grpc::StatusCode::FAILED_PRECONDITION,
                           "No executor found for ID"));
}

TEST_F(ExecutorServiceTest, DisposeExecutorThenDisposeSucceeds) {
  v0::DisposeExecutorRequest dispose_executor_request;
  *dispose_executor_request.mutable_executor() = executor_pb_;
  v0::DisposeExecutorResponse dispose_executor_response;
  auto dispose_request = DisposeRequestForIds({"whimsy_id"});
  v0::DisposeResponse dispose_response;
  grpc::ServerContext server_context;

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.DisposeExecutor(
      &server_context, &dispose_executor_request, &dispose_executor_response)));

  // Second disposal succeeds; TFF service declares its clients free to
  // call dispose on a nonexistent executor.
  TFF_ASSERT_OK(grpc_to_absl(executor_service_.DisposeExecutor(
      &server_context, &dispose_executor_request, &dispose_executor_response)));
}

TEST_F(ExecutorServiceTest, CreateCallNoArgFnArgumentSetToEmptyString) {
  // The argument ref in the associated create call request will be marked as
  // set, but to an empty string.
  v0::CreateCallRequest call_request = CreateCallRequestForIds("0", "");
  v0::CreateCallResponse create_call_response_pb;
  grpc::ServerContext server_context;

  auto create_call_response_status = executor_service_.CreateCall(
      &server_context, &call_request, &create_call_response_pb);

  ASSERT_THAT(create_call_response_status,
              GrpcStatusIs(grpc::StatusCode::INVALID_ARGUMENT,
                           "Expected value ref to be an integer id"));
}

TEST_F(ExecutorServiceTest, CreateCallNoArgFn) {
  v0::CreateCallRequest call_request =
      CreateCallRequestForIds("0", std::nullopt);
  v0::CreateCallResponse create_call_response_pb;
  grpc::ServerContext server_context;

  // We expect the ID returned from this call to be set reflected in the
  // returned value.
  EXPECT_CALL(*executor_ptr_, CreateCall(0, ::testing::Eq(std::nullopt)))
      .WillOnce([this] { return TestId(1); });

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateCall(
      &server_context, &call_request, &create_call_response_pb)));

  EXPECT_THAT(create_call_response_pb,
              testing::EqualsProto("value_ref { id: '1' }"));
}

TEST_F(ExecutorServiceTest, CreateCallFunctionWithArgument) {
  v0::CreateCallRequest call_request = CreateCallRequestForIds("0", "1");
  v0::CreateCallResponse create_call_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_, CreateCall(0, ::testing::Optional(1)))
      .WillOnce([this] { return TestId(2); });

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateCall(
      &server_context, &call_request, &create_call_response_pb)));

  EXPECT_THAT(create_call_response_pb,
              testing::EqualsProto("value_ref { id: '2' }"));
}

TEST_F(ExecutorServiceTest, CreateSelection) {
  v0::CreateSelectionRequest first_selection_request =
      CreateSelectionRequestForIndex("0", 1);
  v0::CreateSelectionRequest second_selection_request =
      CreateSelectionRequestForIndex("2", 2);
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

  TFF_ASSERT_OK(grpc_to_absl(first_create_selection_response_status));
  TFF_ASSERT_OK(grpc_to_absl(second_create_selection_response_status));
  EXPECT_THAT(first_create_selection_response_pb,
              testing::EqualsProto("value_ref { id: '1' }"));
  EXPECT_THAT(second_create_selection_response_pb,
              testing::EqualsProto("value_ref { id: '3' }"));
}

TEST_F(ExecutorServiceTest, CreateEmptyStruct) {
  v0::CreateStructRequest struct_request = CreateStructForIds({});
  v0::CreateStructResponse struct_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_,
              CreateStruct(::testing::Eq(std::vector<ValueId>{})))
      .WillOnce([this] { return TestId(0); });

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateStruct(
      &server_context, &struct_request, &struct_response_pb)));

  EXPECT_THAT(struct_response_pb,
              testing::EqualsProto("value_ref { id: '0' }"));
}

TEST_F(ExecutorServiceTest, CreateNonemptyStruct) {
  v0::CreateStructRequest struct_request = CreateStructForIds({"0", "1"});
  v0::CreateStructResponse struct_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_,
              CreateStruct(::testing::Eq(std::vector<ValueId>{0, 1})))
      .WillOnce([this] { return TestId(0); });

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateStruct(
      &server_context, &struct_request, &struct_response_pb)));

  EXPECT_THAT(struct_response_pb,
              testing::EqualsProto("value_ref { id: '0' }"));
}

TEST_F(ExecutorServiceTest, CreateNamedNonemptyStruct) {
  v0::CreateStructRequest struct_request = CreateNamedStructForIds({"0", "1"});
  v0::CreateStructResponse struct_response_pb;
  grpc::ServerContext server_context;

  EXPECT_CALL(*executor_ptr_,
              CreateStruct(::testing::Eq(std::vector<ValueId>{0, 1})))
      .WillOnce([this] { return TestId(0); });

  TFF_ASSERT_OK(grpc_to_absl(executor_service_.CreateStruct(
      &server_context, &struct_request, &struct_response_pb)));

  EXPECT_THAT(struct_response_pb,
              testing::EqualsProto("value_ref { id: '0' }"));
}

}  // namespace tensorflow_federated
