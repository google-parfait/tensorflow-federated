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

#include "tensorflow_federated/cc/core/impl/executors/remote_executor.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_grpc.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

using testing::EqualsProto;
using testing::proto::IgnoringRepeatedFieldOrdering;

constexpr char kExecutorId[] = "executor_id";

template <typename ResponseType>
auto CreateResponseForId(std::string id) {
  ResponseType response;
  response.mutable_value_ref()->mutable_id()->assign(std::move(id));
  return response;
}

template <typename ResponseType>
auto ReturnOkWithResponseId(std::string response_id) {
  return ::testing::DoAll(
      ::testing::SetArgPointee<2>(
          CreateResponseForId<ResponseType>(std::move(response_id))),
      ::testing::Return(grpc::Status::OK));
}

v0::ComputeResponse ComputeResponseForValue(const v0::Value& value_proto) {
  v0::ComputeResponse response;
  *response.mutable_value() = value_proto;
  return response;
}

auto ReturnOkWithComputeResponse(const v0::Value& value_proto) {
  return ::testing::DoAll(
      ::testing::SetArgPointee<2>(ComputeResponseForValue(value_proto)),
      ::testing::Return(grpc::Status::OK));
}

v0::CreateValueRequest CreateValueRequestForValue(
    const v0::Value& value_proto) {
  v0::CreateValueRequest request;
  request.mutable_executor()->set_id(kExecutorId);
  *request.mutable_value() = value_proto;
  return request;
}

v0::ComputeRequest ComputeRequestForId(std::string ref) {
  v0::ComputeRequest request;
  request.mutable_executor()->set_id(kExecutorId);
  request.mutable_value_ref()->mutable_id()->assign(std::move(ref));
  return request;
}

std::function<grpc::Status()> NotifyAndReturnOk(
    absl::Notification& done_notification) {
  return [&done_notification] {
    done_notification.Notify();
    return grpc::Status::OK;
  };
}

constexpr char kExpectedGetExecutorRequest[] = R"pb(
  cardinalities {
    placement { uri: "clients" }
    cardinality: 1
  }
  cardinalities {
    placement { uri: "server" }
    cardinality: 1
  }
)pb";

grpc::Status UnimplementedPlaceholder() {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Test");
}

::testing::Matcher<grpc::Status> IsUnimplementedPlaceholder() {
  return GrpcStatusIs(grpc::StatusCode::UNIMPLEMENTED, "Test");
}

class RemoteExecutorTest : public ::testing::Test {
 protected:
  RemoteExecutorTest() : mock_executor_service_(mock_executor_.service()) {
    std::unique_ptr<v0::ExecutorGroup::Stub> stub_ptr(mock_executor_.NewStub());
    CardinalityMap cardinalities = {{"server", 1}, {"clients", 1}};
    test_executor_ = CreateRemoteExecutor(std::move(stub_ptr), cardinalities);
  }
  ~RemoteExecutorTest() override { test_executor_ = nullptr; }

  // Adds expectations of calls to `GetExecutor` and `DisposeExecutor` and
  // returns a notification which notifies when `DisposeExecutor` is called.
  //
  // Tests which call this method should end with a call to
  // `WaitForDisposeExecutor`.
  void ExpectGetAndDisposeExecutor(
      absl::Notification& dispose_notification_out) {
    v0::GetExecutorResponse get_response;
    *get_response.mutable_executor()->mutable_id() = kExecutorId;
    EXPECT_CALL(*mock_executor_service_,
                GetExecutor(::testing::_,
                            IgnoringRepeatedFieldOrdering(
                                EqualsProto(kExpectedGetExecutorRequest)),
                            ::testing::_))
        .WillOnce(::testing::DoAll(::testing::SetArgPointee<2>(get_response),
                                   ::testing::Return(grpc::Status::OK)));

    absl::Notification done;
    v0::DisposeExecutorRequest dispose_request;
    *dispose_request.mutable_executor()->mutable_id() = kExecutorId;
    EXPECT_CALL(*mock_executor_service_,
                DisposeExecutor(::testing::_, EqualsProto(dispose_request),
                                ::testing::_))
        .WillOnce(NotifyAndReturnOk(dispose_notification_out));
  }

  // Discards `test_executor_` and waits for `notification`. This method is
  // intended to be used at the end of a testcase in combination with a call to
  // `ExpectGetAndDisposeExecutor` at the beginning of the testcase.
  void WaitForDisposeExecutor(absl::Notification& notification) {
    // Remove the `test_executor_` to allow the underlying stub to be deleted,
    // triggering the call to `DisposeExecutor`.
    test_executor_ = nullptr;
    while (!notification.WaitForNotificationWithTimeout(absl::Seconds(1))) {
      LOG(WARNING) << "Waiting for call to `DisposeExecutor`...";
    }
  }

  MockGrpcExecutorServer mock_executor_;
  MockGrpcExecutorService* mock_executor_service_;
  std::shared_ptr<Executor> test_executor_;
};

TEST_F(RemoteExecutorTest, ConstructRemoteExecutorFromChannel) {
  std::shared_ptr<grpc::ChannelCredentials> credentials =
      grpc::InsecureChannelCredentials();
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel("fake_target", credentials);
  CardinalityMap cardinalities = {{"server", 1}, {"clients", 1}};
  auto remote_executor = CreateRemoteExecutor(channel, cardinalities);
  static_assert(
      std::is_same<decltype(remote_executor), decltype(test_executor_)>::value);
}

TEST_F(RemoteExecutorTest, GetExecutorErrorSurfaces) {
  EXPECT_CALL(*mock_executor_service_,
              GetExecutor(::testing::_,
                          IgnoringRepeatedFieldOrdering(
                              EqualsProto(kExpectedGetExecutorRequest)),
                          ::testing::_))
      .WillOnce(::testing::Return(UnimplementedPlaceholder()));
  v0::Value tensor_two = testing::TensorV(2.0f);

  absl::StatusOr<OwnedValueId> value_ref =
      test_executor_->CreateValue(tensor_two);
  EXPECT_THAT(value_ref.status(),
              StatusIs(absl::StatusCode::kUnimplemented, "Test"));
}

TEST_F(RemoteExecutorTest, CreateValueTensor) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);

  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(*mock_executor_service_,
                CreateValue(::testing::_,
                            EqualsProto(CreateValueRequestForValue(tensor_two)),
                            ::testing::_))
        .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>("value_ref"));

    v0::DisposeRequest expected_dispose_request;
    expected_dispose_request.mutable_executor()->set_id(kExecutorId);
    expected_dispose_request.mutable_value_ref()->Add()->set_id("value_ref");
    EXPECT_CALL(*mock_executor_service_,
                Dispose(::testing::_, EqualsProto(expected_dispose_request),
                        ::testing::_))
        .WillOnce(::testing::Return(grpc::Status::OK));

    OwnedValueId value_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    EXPECT_CALL(*mock_executor_service_,
                Compute(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(value_ref, &materialized_value);
  }

  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateValueWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = testing::TensorV(2.0f);

  EXPECT_CALL(*mock_executor_service_,
              CreateValue(::testing::_, ::testing::_, ::testing::_))
      .WillOnce(::testing::Return(UnimplementedPlaceholder()));

  OwnedValueId value_ref =
      TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));
  v0::Value materialized_value;
  // If the CreateValue fails we dont expect a Compute call on the other side.
  // Nor do we expect a dispose, because no value has been created.
  absl::Status materialize_status =
      test_executor_->Materialize(value_ref, &materialized_value);
  EXPECT_THAT(materialize_status,
              StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, MaterializeWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = testing::TensorV(2.0f);

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    EXPECT_CALL(*mock_executor_service_,
                CreateValue(::testing::_,
                            EqualsProto(CreateValueRequestForValue(tensor_two)),
                            ::testing::_))
        .WillOnce(ReturnOkWithResponseId<v0::CreateValueResponse>("value_ref"));
    OwnedValueId value_ref =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    EXPECT_CALL(*mock_executor_service_,
                Compute(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(UnimplementedPlaceholder()));
    materialize_status =
        test_executor_->Materialize(value_ref, &materialized_value);
    EXPECT_THAT(materialize_status,
                StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  }
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateCallFnWithArg) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value tensor_three = testing::TensorV(3.0f);

  v0::Value materialized_value;
  absl::Status materialize_status;

  {
    EXPECT_CALL(*mock_executor_service_,
                CreateValue(::testing::_,
                            EqualsProto(CreateValueRequestForValue(tensor_two)),
                            ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("function_ref"));
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(::testing::_,
                    EqualsProto(CreateValueRequestForValue(tensor_three)),
                    ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("argument_ref"));

    // The literal argument itself is thrown away, but is used to identify which
    // CreateValue represents the function and which the argument.
    OwnedValueId fn = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));
    OwnedValueId arg = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_three));

    v0::CreateCallRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_function_ref()->set_id("function_ref");
    expected_request.mutable_argument_ref()->set_id("argument_ref");

    EXPECT_CALL(
        *mock_executor_service_,
        CreateCall(::testing::_, EqualsProto(expected_request), ::testing::_))
        .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>("call_ref"));

    OwnedValueId call_result =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn, arg));

    // We need to call Materialize to force synchronization of the calls above.
    EXPECT_CALL(
        *mock_executor_service_,
        Compute(::testing::_, EqualsProto(ComputeRequestForId("call_ref")),
                ::testing::_))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(call_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateCallNoArgFn) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  EXPECT_CALL(*mock_executor_service_,
              CreateValue(::testing::_, ::testing::_, ::testing::_))
      .WillOnce(
          ReturnOkWithResponseId<v0::CreateValueResponse>("function_ref"));

  // The literal argument itself is unused.
  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    OwnedValueId fn = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    v0::CreateCallRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_function_ref()->set_id("function_ref");

    EXPECT_CALL(
        *mock_executor_service_,
        CreateCall(::testing::_, EqualsProto(expected_request), ::testing::_))
        .WillOnce(ReturnOkWithResponseId<v0::CreateCallResponse>("call_ref"));

    OwnedValueId call_result =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn, absl::nullopt));

    // We need to call Materialize to force synchronization of the calls above.
    EXPECT_CALL(
        *mock_executor_service_,
        Compute(::testing::_, EqualsProto(ComputeRequestForId("call_ref")),
                ::testing::_))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(call_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateCallError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  EXPECT_CALL(*mock_executor_service_,
              CreateValue(::testing::_, ::testing::_, ::testing::_))
      .WillOnce(
          ReturnOkWithResponseId<v0::CreateValueResponse>("function_ref"));

  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    // The literal value itself is unused.
    OwnedValueId fn = TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    EXPECT_CALL(*mock_executor_service_,
                CreateCall(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(UnimplementedPlaceholder()));

    OwnedValueId call_result =
        TFF_ASSERT_OK(test_executor_->CreateCall(fn, absl::nullopt));

    // We expect the executor to shortcircuit and never call Compute if an
    // intermediate result errors out.
    materialize_status =
        test_executor_->Materialize(call_result, &materialized_value);
  }
  EXPECT_THAT(materialize_status,
              StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateStructWithTwoElements) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value tensor_three = testing::TensorV(3.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(*mock_executor_service_,
                CreateValue(::testing::_,
                            EqualsProto(CreateValueRequestForValue(tensor_two)),
                            ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("struct_elem1"));
    EXPECT_CALL(
        *mock_executor_service_,
        CreateValue(::testing::_,
                    EqualsProto(CreateValueRequestForValue(tensor_three)),
                    ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("struct_elem2"));
    OwnedValueId first_struct_element =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));
    OwnedValueId second_struct_element =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_three));

    v0::CreateStructRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.add_element()->mutable_value_ref()->set_id("struct_elem1");
    expected_request.add_element()->mutable_value_ref()->set_id("struct_elem2");

    EXPECT_CALL(
        *mock_executor_service_,
        CreateStruct(::testing::_, EqualsProto(expected_request), ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateStructResponse>("struct_ref"));

    std::vector<ValueId> struct_to_create = {std::move(first_struct_element),
                                             std::move(second_struct_element)};
    OwnedValueId struct_result =
        TFF_ASSERT_OK(test_executor_->CreateStruct(struct_to_create));

    // We need to call Materialize to force synchronization of the calls above.
    EXPECT_CALL(
        *mock_executor_service_,
        Compute(::testing::_, EqualsProto(ComputeRequestForId("struct_ref")),
                ::testing::_))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(struct_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateStructWithNoElements) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    v0::CreateStructRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    EXPECT_CALL(
        *mock_executor_service_,
        CreateStruct(::testing::_, EqualsProto(expected_request), ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateStructResponse>("struct_ref"));
    std::vector<ValueId> struct_to_create = {};
    OwnedValueId struct_result =
        TFF_ASSERT_OK(test_executor_->CreateStruct(struct_to_create));

    // We need to call Materialize to force synchronization of the calls above.
    EXPECT_CALL(
        *mock_executor_service_,
        Compute(::testing::_, EqualsProto(ComputeRequestForId("struct_ref")),
                ::testing::_))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(struct_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateStructWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::CreateStructRequest expected_request;
  expected_request.mutable_executor()->set_id(kExecutorId);
  EXPECT_CALL(
      *mock_executor_service_,
      CreateStruct(::testing::_, EqualsProto(expected_request), ::testing::_))
      .WillOnce(::testing::Return(UnimplementedPlaceholder()));

  std::vector<ValueId> struct_to_create = {};
  OwnedValueId struct_result =
      TFF_ASSERT_OK(test_executor_->CreateStruct(struct_to_create));

  // If the CreateStruct fails we dont expect a Compute call on the other side
  v0::Value materialized_value;
  absl::Status materialize_status =
      test_executor_->Materialize(struct_result, &materialized_value);
  EXPECT_THAT(materialize_status,
              StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateSelection) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(*mock_executor_service_,
                CreateValue(::testing::_,
                            EqualsProto(CreateValueRequestForValue(tensor_two)),
                            ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("source_ref"));
    OwnedValueId source_value =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));

    v0::CreateSelectionRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_source_ref()->set_id("source_ref");
    expected_request.set_index(2);

    EXPECT_CALL(*mock_executor_service_,
                CreateSelection(::testing::_, EqualsProto(expected_request),
                                ::testing::_))
        .WillOnce(ReturnOkWithResponseId<v0::CreateSelectionResponse>(
            "selection_ref"));

    OwnedValueId selection_result =
        TFF_ASSERT_OK(test_executor_->CreateSelection(source_value, 2));

    EXPECT_CALL(
        *mock_executor_service_,
        Compute(::testing::_, EqualsProto(ComputeRequestForId("selection_ref")),
                ::testing::_))
        .WillOnce(ReturnOkWithComputeResponse(tensor_two));
    materialize_status =
        test_executor_->Materialize(selection_result, &materialized_value);
  }
  TFF_EXPECT_OK(materialize_status);
  EXPECT_THAT(materialized_value, EqualsProto(tensor_two));
  WaitForDisposeExecutor(dispose_notification);
}

TEST_F(RemoteExecutorTest, CreateSelectionWithError) {
  absl::Notification dispose_notification;
  ExpectGetAndDisposeExecutor(dispose_notification);
  v0::Value tensor_two = testing::TensorV(2.0f);
  v0::Value materialized_value;
  absl::Status materialize_status;
  {
    EXPECT_CALL(*mock_executor_service_,
                CreateValue(::testing::_,
                            EqualsProto(CreateValueRequestForValue(tensor_two)),
                            ::testing::_))
        .WillOnce(
            ReturnOkWithResponseId<v0::CreateValueResponse>("source_ref"));

    v0::CreateSelectionRequest expected_request;
    expected_request.mutable_executor()->set_id(kExecutorId);
    expected_request.mutable_source_ref()->set_id("source_ref");
    expected_request.set_index(1);

    EXPECT_CALL(*mock_executor_service_,
                CreateSelection(::testing::_, EqualsProto(expected_request),
                                ::testing::_))
        .WillOnce(::testing::Return(UnimplementedPlaceholder()));
    OwnedValueId source_value =
        TFF_ASSERT_OK(test_executor_->CreateValue(tensor_two));
    OwnedValueId selection_result =
        TFF_ASSERT_OK(test_executor_->CreateSelection(source_value, 1));

    // If the CreateSelection fails we dont expect a Compute call on the other
    // side
    materialize_status =
        test_executor_->Materialize(selection_result, &materialized_value);
  }
  EXPECT_THAT(materialize_status,
              StatusIs(absl::StatusCode::kUnimplemented, "Test"));
  WaitForDisposeExecutor(dispose_notification);
}

}  // namespace tensorflow_federated
