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

#include "tensorflow_federated/cc/core/impl/executor_stacks/remote_stacks.h"

#include <memory>

#include "grpcpp/grpcpp.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

using absl::StatusCode;
using ::testing::AnyOfArray;
using ::testing::HasSubstr;
using ::testing::MockFunction;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::status::StatusIs;

namespace tensorflow_federated {

std::shared_ptr<StrictMock<MockExecutor>> get_mock_executor() {
  static auto mock_ex_ = std::make_shared<StrictMock<MockExecutor>>();
  return mock_ex_;
}

MockFunction<absl::StatusOr<std::shared_ptr<Executor>>(void)>
    mock_executor_factory_;

MockFunction<absl::StatusOr<ComposingChild>(
    std::shared_ptr<grpc::ChannelInterface>, const CardinalityMap&)>
    mock_composing_child_factory_;

// TODO(b/191092505): Follow up and investigate removing the need to mock the
// entire grpc ChannelInterface interface.
class MockGrpcChannelInterface : public grpc::ChannelInterface {
 public:
  MOCK_METHOD(grpc_connectivity_state, GetState, (bool), (override));

  MOCK_METHOD(grpc::internal::Call, CreateCall,
              (const grpc::internal::RpcMethod&, grpc::ClientContext*,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(void, PerformOpsOnCall,
              (grpc::internal::CallOpSetInterface*, grpc::internal::Call*),
              (override));

  // GRPC registers the methods of TFF's Executor service on this channel; these
  // are the calls we expect to the mock below.
  MOCK_METHOD(void*, RegisterMethod, (const char*), (override));

  MOCK_METHOD(void, NotifyOnStateChangeImpl,
              (grpc_connectivity_state, gpr_timespec, grpc::CompletionQueue*,
               void*),
              (override));

  MOCK_METHOD(bool, WaitForStateChangeImpl,
              (grpc_connectivity_state, gpr_timespec), (override));
};

class RemoteExecutorStackTest : public ::testing::Test {};

TEST_F(RemoteExecutorStackTest, UnspecifiedClientsReturnsInvalidArgError) {
  absl::StatusOr<std::shared_ptr<Executor>> status_or_executor =
      CreateRemoteExecutorStack(
          {grpc::CreateChannel("localhost:8000",
                               grpc::InsecureChannelCredentials())},
          {});
  EXPECT_THAT(status_or_executor.status(),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("clients not specified")));
}

TEST_F(RemoteExecutorStackTest, NoTargetsNonzeroClientsReturnsError) {
  absl::StatusOr<std::shared_ptr<Executor>> status_or_executor =
      CreateRemoteExecutorStack({}, {{std::string(kClientsUri), 1}});
  EXPECT_THAT(status_or_executor.status(),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Found 0 remote channels")));
}

// Our implementation of filtering to live workers uses WaitForConnected, which
// polls the mock methods GetState and WaitForStateChangeImpl.
void ExpectCallsToFailedChannel(
    std::shared_ptr<MockGrpcChannelInterface> channel,
    grpc_connectivity_state state) {
  EXPECT_CALL(*channel, GetState(::testing::IsTrue()))
      .WillRepeatedly(Return(state));
  EXPECT_CALL(*channel, WaitForStateChangeImpl(::testing::_, ::testing::_))
      .WillRepeatedly(Return(false));
}

TEST_F(RemoteExecutorStackTest, UnavailableChannelInterfacesReturnsError) {
  auto failed_channel =
      std::make_shared<StrictMock<MockGrpcChannelInterface>>();
  auto connecting_channel =
      std::make_shared<StrictMock<MockGrpcChannelInterface>>();
  auto shutdown_channel =
      std::make_shared<StrictMock<MockGrpcChannelInterface>>();
  auto idle_channel = std::make_shared<StrictMock<MockGrpcChannelInterface>>();

  ExpectCallsToFailedChannel(
      failed_channel, grpc_connectivity_state::GRPC_CHANNEL_TRANSIENT_FAILURE);
  ExpectCallsToFailedChannel(connecting_channel,
                             grpc_connectivity_state::GRPC_CHANNEL_CONNECTING);
  ExpectCallsToFailedChannel(shutdown_channel,
                             grpc_connectivity_state::GRPC_CHANNEL_SHUTDOWN);
  ExpectCallsToFailedChannel(idle_channel,
                             grpc_connectivity_state::GRPC_CHANNEL_IDLE);
  EXPECT_CALL(mock_executor_factory_, Call())
      .WillOnce(Return(get_mock_executor()));
  absl::StatusOr<std::shared_ptr<Executor>> status_or_executor =
      CreateRemoteExecutorStack(
          {failed_channel, connecting_channel, shutdown_channel},
          {{std::string(kClientsUri), 100}},
          mock_executor_factory_.AsStdFunction(),
          mock_composing_child_factory_.AsStdFunction());
  EXPECT_THAT(status_or_executor.status(),
              testing::status::StatusIs(StatusCode::kUnavailable,
                                        HasSubstr("No TFF workers are ready")));
}

TEST_F(RemoteExecutorStackTest,
       ZeroClientsResultsInZeroChannelInterfaceReadyCalls) {
  std::vector<std::shared_ptr<grpc::ChannelInterface>> channel_args;
  for (int i = 0; i < 5; i++) {
    channel_args.emplace_back(
        std::make_shared<StrictMock<MockGrpcChannelInterface>>());
  }
  EXPECT_CALL(mock_executor_factory_, Call())
      .WillOnce(Return(get_mock_executor()));
  absl::StatusOr<std::shared_ptr<Executor>> status_or_executor =
      CreateRemoteExecutorStack(channel_args, {{std::string(kClientsUri), 0}},
                                mock_executor_factory_.AsStdFunction(),
                                mock_composing_child_factory_.AsStdFunction());
  EXPECT_OK(status_or_executor);
}

TEST_F(RemoteExecutorStackTest, FailedChannelInterfacesNotAddressed) {
  std::vector<std::shared_ptr<grpc::ChannelInterface>> channel_args;
  for (int i = 0; i < 5; i++) {
    auto mock_channel =
        std::make_shared<StrictMock<MockGrpcChannelInterface>>();
    if (i < 3) {
      EXPECT_CALL(*mock_channel, GetState(::testing::IsTrue()))
          .Times(2)
          .WillRepeatedly(Return(grpc_connectivity_state::GRPC_CHANNEL_READY));
      EXPECT_CALL(*mock_channel, RegisterMethod(::testing::_))
          .WillRepeatedly(Return(nullptr));
    } else {
      ExpectCallsToFailedChannel(
          mock_channel,
          grpc_connectivity_state::GRPC_CHANNEL_TRANSIENT_FAILURE);
      // Since these failed channels should not be passed to a stub, we dont
      // expect them to see any RegisterMethod calls.
    }
    channel_args.emplace_back(mock_channel);
  }

  EXPECT_CALL(mock_executor_factory_, Call())
      .WillOnce(Return(get_mock_executor()));
  CardinalityMap two_client_cards = {{std::string(kClientsUri), 2}};
  CardinalityMap one_client_cards = {{std::string(kClientsUri), 1}};
  ComposingChild child =
      ComposingChild::Make(get_mock_executor(), one_client_cards).ValueOrDie();
  EXPECT_CALL(mock_composing_child_factory_,
              Call(AnyOfArray(channel_args), two_client_cards))
      .Times(2)
      .WillRepeatedly(Return(child));
  EXPECT_CALL(mock_composing_child_factory_,
              Call(AnyOfArray(channel_args), one_client_cards))
      .Times(1)
      .WillRepeatedly(Return(child));
  absl::StatusOr<std::shared_ptr<Executor>> status_or_executor =
      CreateRemoteExecutorStack(channel_args, {{std::string(kClientsUri), 5}},
                                mock_executor_factory_.AsStdFunction(),
                                mock_composing_child_factory_.AsStdFunction());
  EXPECT_OK(status_or_executor);
}

TEST_F(RemoteExecutorStackTest, OneClientPassedToOnlyOneChild) {
  std::vector<std::shared_ptr<grpc::ChannelInterface>> channel_args;
  for (int i = 0; i < 5; i++) {
    auto mock_channel =
        std::make_shared<StrictMock<MockGrpcChannelInterface>>();
    // We expect every ready channel to have TFF's executor.proto service
    // messages registered when the gRPC stub is constructed.
    EXPECT_CALL(*mock_channel, GetState(::testing::IsTrue()))
        .Times(2)
        .WillRepeatedly(Return(grpc_connectivity_state::GRPC_CHANNEL_READY));
    EXPECT_CALL(*mock_channel, RegisterMethod(::testing::_))
        .WillRepeatedly(Return(nullptr));
    channel_args.emplace_back(mock_channel);
  }

  CardinalityMap one_client_cards = {{std::string(kClientsUri), 1}};
  CardinalityMap zero_client_cards = {{std::string(kClientsUri), 0}};
  ComposingChild child =
      ComposingChild::Make(get_mock_executor(), one_client_cards).ValueOrDie();

  // One call to the mock executor factory for creation of the server.
  EXPECT_CALL(mock_executor_factory_, Call())
      .WillOnce(Return(get_mock_executor()));

  // We create an executor stack with 1 client; therefore we expect one of the
  // composing child factory to have a single client passed to it upon
  // invocation, and the children corresponding to the other 4 channels to have
  // 0 clients.
  EXPECT_CALL(mock_composing_child_factory_,
              Call(AnyOfArray(channel_args), one_client_cards))
      .WillOnce(Return(child));
  EXPECT_CALL(mock_composing_child_factory_,
              Call(AnyOfArray(channel_args), zero_client_cards))
      .Times(4)
      .WillRepeatedly(Return(child));

  absl::StatusOr<std::shared_ptr<Executor>> status_or_executor =
      CreateRemoteExecutorStack(channel_args, one_client_cards,
                                mock_executor_factory_.AsStdFunction(),
                                mock_composing_child_factory_.AsStdFunction());
  EXPECT_OK(status_or_executor);
}

}  // namespace tensorflow_federated
