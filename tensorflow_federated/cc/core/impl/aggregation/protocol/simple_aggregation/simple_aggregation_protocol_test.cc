/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/simple_aggregation/simple_aggregation_protocol.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "tensorflow_federated/cc/core/impl/aggregation/testing/parse_text_proto.h"
// clang-format on
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/scheduler.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/simulated_clock.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/mocks.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated::aggregation {
namespace {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::StrEq;

class SimpleAggregationProtocolTest : public ::testing::Test {
 protected:
  // Returns default configuration.
  Configuration default_configuration() const;

  // Returns the default instance of checkpoint bilder;
  MockCheckpointBuilder& ExpectCheckpointBuilder() {
    MockCheckpointBuilder& checkpoint_builder = *wrapped_checkpoint_builder_;
    EXPECT_CALL(checkpoint_builder_factory_, Create())
        .WillOnce(Return(ByMove(std::move(wrapped_checkpoint_builder_))));
    EXPECT_CALL(checkpoint_builder, Build()).WillOnce(Return(absl::Cord{}));
    return checkpoint_builder;
  }

  // Creates an instance of SimpleAggregationProtocol with the specified config.
  std::unique_ptr<SimpleAggregationProtocol> CreateProtocol(
      Configuration config,
      std::optional<SimpleAggregationProtocol::OutlierDetectionParameters>
          params = std::nullopt);

  // Creates an instance of SimpleAggregationProtocol with the default config.
  std::unique_ptr<SimpleAggregationProtocol> CreateProtocolWithDefaultConfig() {
    return CreateProtocol(default_configuration());
  }

  std::unique_ptr<SimpleAggregationProtocol> CreateProtocolWithDefaultConfig(
      SimpleAggregationProtocol::OutlierDetectionParameters params) {
    return CreateProtocol(default_configuration(), params);
  }

  MockCheckpointParserFactory checkpoint_parser_factory_;
  MockCheckpointBuilderFactory checkpoint_builder_factory_;
  MockResourceResolver resource_resolver_;
  SimulatedClock clock_;

 private:
  std::unique_ptr<MockCheckpointBuilder> wrapped_checkpoint_builder_ =
      std::make_unique<MockCheckpointBuilder>();
};

Configuration SimpleAggregationProtocolTest::default_configuration() const {
  // One "federated_sum" intrinsic with a single scalar int32 tensor.
  return PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
}

std::unique_ptr<SimpleAggregationProtocol>
SimpleAggregationProtocolTest::CreateProtocol(
    Configuration config,
    std::optional<SimpleAggregationProtocol::OutlierDetectionParameters>
        params) {
  // Verify that the protocol can be created successfully.
  absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
      protocol_or_status = SimpleAggregationProtocol::Create(
          config, &checkpoint_parser_factory_, &checkpoint_builder_factory_,
          &resource_resolver_, &clock_, std::move(params));
  EXPECT_THAT(protocol_or_status, IsOk());
  return std::move(protocol_or_status).value();
}

ClientMessage MakeClientMessage() {
  ClientMessage message;
  message.mutable_simple_aggregation()->mutable_input()->set_inline_bytes("");
  return message;
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_Success) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(3), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_pending: 3"));
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_MultipleCalls) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());
  // The second Start call must fail.
  EXPECT_THAT(protocol->Start(1), StatusIs(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_ZeroClients) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(0), IsOk());
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_NegativeClients) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(-1), StatusIs(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, AddClients_Success) {
  auto protocol = CreateProtocolWithDefaultConfig();

  EXPECT_THAT(protocol->Start(1), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_pending: 1"));

  auto status_or_result = protocol->AddClients(3);
  EXPECT_THAT(status_or_result, IsOk());
  EXPECT_EQ(status_or_result.value(), 1);
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_pending: 4"));
}

TEST_F(SimpleAggregationProtocolTest, AddClients_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->AddClients(1), StatusIs(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, PollServerMessage_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  // Must fail because the protocol isn't started.
  EXPECT_THAT(protocol->PollServerMessage(0), StatusIs(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, PollServerMessage_InvalidClientId) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());
  // Must fail for the client_id -1 and 2.
  EXPECT_THAT(protocol->PollServerMessage(-1), StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(protocol->PollServerMessage(2), StatusIs(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, IsClientClosed_InvalidClient) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());
  // Must fail for the client_id -1 and 2.
  EXPECT_THAT(protocol->IsClientClosed(-1), StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(protocol->IsClientClosed(2), StatusIs(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  // Must fail because the protocol isn't started.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()),
              StatusIs(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_InvalidMessage) {
  auto protocol = CreateProtocolWithDefaultConfig();
  ClientMessage message;
  // Empty message without SimpleAggregation.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message),
              StatusIs(INVALID_ARGUMENT));
  // Message with SimpleAggregation but without the input.
  message.mutable_simple_aggregation();
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message),
              StatusIs(INVALID_ARGUMENT));
  // Message with empty input.
  message.mutable_simple_aggregation()->mutable_input();
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message),
              StatusIs(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_InvalidClientId) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());
  // Must fail for the client_id -1 and 2.
  EXPECT_THAT(protocol->ReceiveClientMessage(-1, MakeClientMessage()),
              StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(protocol->ReceiveClientMessage(2, MakeClientMessage()),
              StatusIs(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest,
       ReceiveClientMessage_DuplicateClientIdInputs) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(2), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  // The second input for the same client must succeed to without changing the
  // aggregated state.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto(
          "protocol_state: PROTOCOL_STARTED num_clients_pending: 1 "
          "num_clients_completed: 1 num_inputs_aggregated_and_included: 1"));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_AfterClosingClient) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());

  EXPECT_THAT(protocol->CloseClient(0, absl::OkStatus()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_completed: 1 "
                  "num_inputs_discarded: 1"));
  // This must succeed to without changing the aggregated state.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_completed: 1 "
                  "num_inputs_discarded: 1"));
}

TEST_F(SimpleAggregationProtocolTest,
       ReceiveClientMessage_FailureToParseInput) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(
          Return(ByMove(absl::InvalidArgumentError("Invalid checkpoint"))));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(INVALID_ARGUMENT));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_failed: 1"));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_MissingTensor) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo")))
      .WillOnce(Return(ByMove(absl::NotFoundError("Missing tensor foo"))));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(NOT_FOUND));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_failed: 1"));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_MismatchingTensor) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {}, CreateTestData({2.f}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(INVALID_ARGUMENT));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_failed: 1"));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_UriType_Success) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());
  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));
  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  // Receive input for the client #0
  EXPECT_CALL(resource_resolver_, RetrieveResource(0, StrEq("foo_uri")))
      .WillOnce(Return(absl::Cord{}));
  ClientMessage message;
  message.mutable_simple_aggregation()->mutable_input()->set_uri("foo_uri");
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message), IsOk());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_completed: 1 "
                  "num_inputs_aggregated_and_included: 1"));
}

TEST_F(SimpleAggregationProtocolTest,
       ReceiveClientMessage_UriType_FailToParse) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(1), IsOk());

  // Receive invalid input for the client #0
  EXPECT_CALL(resource_resolver_, RetrieveResource(0, _))
      .WillOnce(Return(absl::InvalidArgumentError("Invalid uri")));
  ClientMessage message;
  message.mutable_simple_aggregation()->mutable_input()->set_uri("foo_uri");

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message), IsOk());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(INVALID_ARGUMENT));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_failed: 1"));
}

TEST_F(SimpleAggregationProtocolTest, Complete_NoInputsReceived) {
  // Two intrinsics:
  // 1) federated_sum "foo" that takes int32 {2,3} tensors.
  // 2) federated_sum "bar" that takes scalar float tensors.
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: 2 dim_sizes: 3 }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: 2 dim_sizes: 3 }
      }
    }
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_FLOAT
          shape {}
        }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_FLOAT
        shape {}
      }
    }
  )pb");
  auto protocol = CreateProtocol(config_message);

  EXPECT_THAT(protocol->Start(1), IsOk());

  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that foo_out and bar_out tensors are added to the result checkpoint
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("foo_out"), IsTensor({2, 3}, {0, 0, 0, 0, 0, 0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder, Add(StrEq("bar_out"), IsTensor({}, {0.f})))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_pending: 1"));

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_COMPLETED num_clients_aborted: 1"));
  EXPECT_OK(protocol->GetResult());
  // Verify that the pending client is closed.
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
}

TEST_F(SimpleAggregationProtocolTest, Complete_TwoInputsReceived) {
  // Two intrinsics:
  // 1) federated_sum "foo" that takes int32 {2,3} tensors.
  // 2) fedsql_group_by with two grouping keys key1 and key2, only the first one
  //    of which should be output, and two inner GoogleSQL:sum intrinsics bar
  //    and baz operating on float tensors.
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: 2 dim_sizes: 3 }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: 2 dim_sizes: 3 }
      }
    }

    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key1"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
      }
      intrinsic_args {
        input_tensor {
          name: "key2"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "key1_out"
        dtype: DT_STRING
        shape { dim_sizes: -1 }
      }
      output_tensors {
        name: ""
        dtype: DT_STRING
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
    }
  )pb");
  auto protocol = CreateProtocol(config_message);
  EXPECT_THAT(protocol->Start(2), IsOk());

  // Expect five inputs.
  auto parser1 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser1, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {2, 3},
                          CreateTestData({4, 3, 11, 7, 1, 6}));
  }));
  EXPECT_CALL(*parser1, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"large", "small"}));
  }));
  EXPECT_CALL(*parser1, GetTensor(StrEq("key2"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"cat", "dog"}));
  }));
  EXPECT_CALL(*parser1, GetTensor(StrEq("bar"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({1.f, 2.f}));
  }));
  EXPECT_CALL(*parser1, GetTensor(StrEq("baz"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({3.f, 4.f}));
  }));

  auto parser2 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser2, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {2, 3},
                          CreateTestData({1, 8, 2, 10, 13, 2}));
  }));
  EXPECT_CALL(*parser2, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"small", "small"}));
  }));
  EXPECT_CALL(*parser2, GetTensor(StrEq("key2"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"dog", "cat"}));
  }));
  EXPECT_CALL(*parser2, GetTensor(StrEq("bar"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({1.f, 2.f}));
  }));
  EXPECT_CALL(*parser2, GetTensor(StrEq("baz"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({3.f, 5.f}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser1))))
      .WillOnce(Return(ByMove(std::move(parser2))));

  // Handle the inputs.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_pending: 1 "
                  "num_clients_completed: 1 "
                  "num_inputs_aggregated_and_included: 1"));

  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(1).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_completed: 2 "
                  "num_inputs_aggregated_and_included: 2"));

  // Complete the protocol.
  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that expected output tensors are added to the result checkpoint.
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("foo_out"), IsTensor({2, 3}, {5, 11, 13, 17, 14, 8})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("key1_out"),
                  IsTensor<string_view>({3}, {"large", "small", "small"})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("bar_out"), IsTensor({3}, {1.f, 3.f, 2.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("baz_out"), IsTensor({3}, {3.f, 7.f, 5.f})))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_COMPLETED num_clients_completed: 2 "
                  "num_inputs_aggregated_and_included: 2"));
  EXPECT_OK(protocol->GetResult());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_EQ(protocol->PollServerMessage(1)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_TRUE(protocol->IsClientClosed(1).value());
}

TEST_F(SimpleAggregationProtocolTest, CompleteNoInputsReceivedFedSqlGroupBy) {
  // One intrinsic:
  //    fedsql_group_by with two grouping keys key1 and key2, only the first one
  //    of which should be output, and two inner GoogleSQL:sum intrinsics bar
  //    and baz operating on float tensors.
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key1"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
      }
      intrinsic_args {
        input_tensor {
          name: "key2"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "key1_out"
        dtype: DT_STRING
        shape { dim_sizes: -1 }
      }
      output_tensors {
        name: ""
        dtype: DT_STRING
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
    }
  )pb");
  auto protocol = CreateProtocol(config_message);
  EXPECT_THAT(protocol->Start(1), IsOk());

  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();
  // Verify that empty tensors are added to the result checkpoint.
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("key1_out"), IsTensor<string_view>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("bar_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("baz_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_pending: 1"));

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_COMPLETED num_clients_aborted: 1"));
  EXPECT_OK(protocol->GetResult());
  // Verify that the pending client is closed.
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
}

TEST_F(SimpleAggregationProtocolTest,
       CompleteOnlyEmptyInputsReceivedFedSqlGroupBy) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key1"
          dtype: DT_FLOAT
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "key1_out"
        dtype: DT_FLOAT
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "val1"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "val1_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
    }
  )pb");
  auto protocol = CreateProtocol(config_message);
  EXPECT_THAT(protocol->Start(2), IsOk());

  // Expect two empty client inputs.
  auto parser1 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser1, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  }));
  EXPECT_CALL(*parser1, GetTensor(StrEq("val1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  }));

  auto parser2 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser2, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  }));
  EXPECT_CALL(*parser2, GetTensor(StrEq("val1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser1))))
      .WillOnce(Return(ByMove(std::move(parser2))));

  // Handle the inputs.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto(
          "protocol_state: PROTOCOL_STARTED num_clients_pending: 1 "
          "num_clients_completed: 1 num_inputs_aggregated_and_included: 1"));

  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_STARTED num_clients_completed: 2 "
                  "num_inputs_aggregated_and_included: 2"));

  // Complete the protocol.
  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that expected empty output tensors are added to the result
  // checkpoint.
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("key1_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("val1_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_COMPLETED num_clients_completed: 2 "
                  "num_inputs_aggregated_and_included: 2"));

  EXPECT_OK(protocol->GetResult());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_EQ(protocol->PollServerMessage(1)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_TRUE(protocol->IsClientClosed(1).value());
}

TEST_F(SimpleAggregationProtocolTest, Complete_TensorUsedInMultipleIntrinsics) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key1"
          dtype: DT_FLOAT
          shape { dim_sizes: -1 }
        }
      }
      intrinsic_args {
        input_tensor {
          name: "key2"
          dtype: DT_FLOAT
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: ""
        dtype: DT_FLOAT
        shape { dim_sizes: -1 }
      }
      output_tensors {
        name: ""
        dtype: DT_FLOAT
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "key1"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "key1_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "key2"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "key2_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
    }
  )pb");
  auto protocol = CreateProtocol(config_message);
  EXPECT_THAT(protocol->Start(2), IsOk());

  // Expect five inputs.
  auto parser1 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser1, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({1.f, 2.f}));
  }));
  EXPECT_CALL(*parser1, GetTensor(StrEq("key2"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({3.f, 4.f}));
  }));

  auto parser2 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser2, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({1.f, 2.f}));
  }));
  EXPECT_CALL(*parser2, GetTensor(StrEq("key2"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({3.f, 5.f}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser1))))
      .WillOnce(Return(ByMove(std::move(parser2))));

  // Handle the inputs.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto("protocol_state: PROTOCOL_STARTED "
                           "num_clients_pending: 1 num_clients_completed: 1 "
                           "num_inputs_aggregated_and_included: 1"));

  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto(
          "protocol_state: PROTOCOL_STARTED "
          "num_clients_completed: 2 num_inputs_aggregated_and_included: 2"));

  // Complete the protocol.
  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that expected output tensors are added to the result checkpoint.
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("key1_out"), IsTensor({3}, {2.f, 2.f, 2.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("key2_out"), IsTensor({3}, {6.f, 4.f, 5.f})))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto(
          "protocol_state: PROTOCOL_COMPLETED "
          "num_clients_completed: 2 num_inputs_aggregated_and_included: 2"));

  EXPECT_OK(protocol->GetResult());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_EQ(protocol->PollServerMessage(1)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_TRUE(protocol->IsClientClosed(1).value());
}

TEST_F(SimpleAggregationProtocolTest, Complete_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Complete(), StatusIs(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, Abort_NoInputsReceived) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(2), IsOk());

  EXPECT_THAT(protocol->Abort(), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_ABORTED num_clients_aborted: 2"));
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_EQ(protocol->PollServerMessage(1)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_TRUE(protocol->IsClientClosed(1).value());
}

TEST_F(SimpleAggregationProtocolTest, Abort_OneInputReceived) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(2), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  // Receive input for the client #1
  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());
  EXPECT_EQ(protocol->PollServerMessage(1)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(OK));
  EXPECT_TRUE(protocol->IsClientClosed(1).value());

  // The client #0 should be aborted on Abort().
  EXPECT_THAT(protocol->Abort(), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_ABORTED num_clients_aborted: 1 "
                  "num_clients_completed:1 num_inputs_discarded: 1"));
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
}

TEST_F(SimpleAggregationProtocolTest, Abort_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Abort(), StatusIs(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, ConcurrentAggregation_Success) {
  const int64_t kNumClients = 10;
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(kNumClients), IsOk());

  // The following block will repeatedly create CheckpointParser instances
  // which will be creating scalar int tensors with repeatedly incrementing
  // values.
  std::atomic<int> tensor_value = 0;
  EXPECT_CALL(checkpoint_parser_factory_, Create(_)).WillRepeatedly(Invoke([&] {
    auto parser = std::make_unique<MockCheckpointParser>();
    EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([&] {
      return Tensor::Create(DT_INT32, {}, CreateTestData({++tensor_value}));
    }));
    return parser;
  }));

  // Schedule receiving inputs on 4 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(4);
  for (int64_t i = 0; i < kNumClients; ++i) {
    scheduler->Schedule([&, i]() {
      EXPECT_THAT(protocol->ReceiveClientMessage(i, MakeClientMessage()),
                  IsOk());
    });
  }
  scheduler->WaitUntilIdle();

  // Complete the protocol.
  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();
  // Verify that foo_out tensor is added to the result checkpoint
  EXPECT_CALL(checkpoint_builder, Add(StrEq("foo_out"), IsTensor({}, {55})))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_OK(protocol->GetResult());
}

// A trivial test aggregator that delegates aggregation to a function.
class FunctionAggregator final : public AggVectorAggregator<int> {
 public:
  using Func = std::function<int(int, int)>;

  FunctionAggregator(DataType dtype, TensorShape shape, Func agg_function)
      : AggVectorAggregator<int>(dtype, shape), agg_function_(agg_function) {}

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {
    for (auto [i, v] : agg_vector) {
      data()[i] = agg_function_(data()[i], v);
    }
  }

  const Func agg_function_;
};

// Factory for the FunctionAggregator.
class FunctionAggregatorFactory final : public TensorAggregatorFactory {
 public:
  explicit FunctionAggregatorFactory(FunctionAggregator::Func agg_function)
      : agg_function_(agg_function) {}

 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (intrinsic.inputs[0].dtype() != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<FunctionAggregator>(intrinsic.inputs[0].dtype(),
                                                intrinsic.inputs[0].shape(),
                                                agg_function_);
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return TFF_STATUS(UNIMPLEMENTED);
  }

  const FunctionAggregator::Func agg_function_;
};

TEST_F(SimpleAggregationProtocolTest, ConcurrentAggregation_AbortWhileQueued) {
  const int64_t kNumClients = 10;
  const int64_t kNumClientBeforeBlocking = 3;

  // Notifies the aggregation to unblock;
  absl::Notification resume_aggregation_notification;
  absl::Notification aggregation_blocked_notification;
  std::atomic<int> agg_counter = 0;
  FunctionAggregatorFactory agg_factory([&](int a, int b) {
    if (++agg_counter > kNumClientBeforeBlocking &&
        !aggregation_blocked_notification.HasBeenNotified()) {
      aggregation_blocked_notification.Notify();
      resume_aggregation_notification.WaitForNotification();
    }
    return a + b;
  });
  RegisterAggregatorFactory("foo1_aggregation", &agg_factory);

  // The configuration below refers to the custom aggregation registered
  // above.
  auto protocol = CreateProtocol(PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "foo1_aggregation"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb"));
  EXPECT_THAT(protocol->Start(kNumClients), IsOk());

  EXPECT_CALL(checkpoint_parser_factory_, Create(_)).WillRepeatedly(Invoke([&] {
    auto parser = std::make_unique<MockCheckpointParser>();
    EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([&] {
      return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
    }));
    return parser;
  }));

  // Schedule receiving inputs on 10 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(10);
  for (int64_t i = 0; i < kNumClients; ++i) {
    scheduler->Schedule([&, i]() {
      EXPECT_THAT(protocol->ReceiveClientMessage(i, MakeClientMessage()),
                  IsOk());
    });
  }

  aggregation_blocked_notification.WaitForNotification();

  // Poll until 3 inputs are reported as aggregated and there are no pending
  // clients. The polling is needed to work around the concurrent execution from
  // all clients.
  StatusMessage status_message;
  do {
    status_message = protocol->GetStatus();
  } while (status_message.num_clients_pending() > 0 ||
           status_message.num_inputs_aggregated_and_included() < 3);

  // At this point one input must be blocked inside the aggregation waiting for
  // the notification, 3 inputs should already be gone through the aggregation,
  // and the remaining 6 inputs should be blocked waiting to enter the
  // aggregation.

  // TODO: b/260946510 - Need to revise the status implementation because it
  // treats received and pending (queued) inputs "as aggregated and pending".
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto("protocol_state: PROTOCOL_STARTED "
                                   "num_clients_completed: 10 "
                                   "num_inputs_aggregated_and_pending: 7 "
                                   "num_inputs_aggregated_and_included: 3"));

  resume_aggregation_notification.Notify();

  // Abort and let all blocked aggregations continue.
  EXPECT_THAT(protocol->Abort(), IsOk());
  scheduler->WaitUntilIdle();

  // All 10 inputs should now be discarded.
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_ABORTED num_clients_completed: 10 "
                  "num_inputs_discarded: 10"));
}

TEST_F(SimpleAggregationProtocolTest,
       ConcurrentAggregation_CompleteWhileQueued) {
  const int64_t kNumClients = 10;
  const int64_t kNumClientBeforeBlocking = 3;

  // Notifies the aggregation to unblock;
  absl::Notification resume_aggregation_notification;
  absl::Notification aggregation_blocked_notification;
  std::atomic<int> agg_counter = 0;
  FunctionAggregatorFactory agg_factory([&](int a, int b) {
    if (++agg_counter > kNumClientBeforeBlocking &&
        !aggregation_blocked_notification.HasBeenNotified()) {
      aggregation_blocked_notification.Notify();
      resume_aggregation_notification.WaitForNotification();
    }
    return a + b;
  });
  RegisterAggregatorFactory("foo2_aggregation", &agg_factory);

  // The configuration below refers to the custom aggregation registered
  // above.
  auto protocol = CreateProtocol(PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "foo2_aggregation"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb"));
  EXPECT_THAT(protocol->Start(kNumClients), IsOk());

  EXPECT_CALL(checkpoint_parser_factory_, Create(_)).WillRepeatedly(Invoke([&] {
    auto parser = std::make_unique<MockCheckpointParser>();
    EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([&] {
      return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
    }));
    return parser;
  }));

  // Schedule receiving inputs on 10 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(10);
  for (int64_t i = 0; i < kNumClients; ++i) {
    scheduler->Schedule([&, i]() {
      EXPECT_THAT(protocol->ReceiveClientMessage(i, MakeClientMessage()),
                  IsOk());
    });
  }

  aggregation_blocked_notification.WaitForNotification();

  // Poll until 3 inputs are reported as aggregated and there are no pending
  // clients. The polling is needed to work around the concurrent execution from
  // all clients.
  StatusMessage status_message;
  do {
    status_message = protocol->GetStatus();
  } while (status_message.num_clients_pending() > 0 ||
           status_message.num_inputs_aggregated_and_included() < 3);

  // At this point one input must be blocked inside the aggregation waiting for
  // the notification, 3 inputs should already be gone through the aggregation,
  // and the remaining 6 inputs should be blocked waiting to enter the
  // aggregation.

  // TODO: b/260946510 - Need to revise the status implementation because it
  // treats received and pending (queued) inputs "as aggregated and pending".
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto("protocol_state: PROTOCOL_STARTED "
                                   "num_clients_completed: 10 "
                                   "num_inputs_aggregated_and_pending: 7 "
                                   "num_inputs_aggregated_and_included: 3"));

  resume_aggregation_notification.Notify();

  // Complete and let all blocked aggregations continue.
  auto& checkpoint_builder = ExpectCheckpointBuilder();
  // Verify that foo_out tensor is added to the result checkpoint
  EXPECT_CALL(checkpoint_builder, Add(StrEq("foo_out"), _))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(protocol->Complete(), IsOk());
  scheduler->WaitUntilIdle();

  status_message = protocol->GetStatus();
  ASSERT_EQ(status_message.protocol_state(), PROTOCOL_COMPLETED);
  ASSERT_EQ(status_message.num_clients_completed(), 10);
  // At least 4 clients must be aggregated
  ASSERT_GE(status_message.num_inputs_aggregated_and_included(), 4);
}

TEST_F(SimpleAggregationProtocolTest, OutlierDetection_ClosePendingClients) {
  constexpr absl::Duration kInterval = absl::Seconds(1);
  constexpr absl::Duration kGracePeriod = absl::Seconds(1);

  // Start by creating the protocol with 5 clients.
  auto protocol = CreateProtocolWithDefaultConfig({kInterval, kGracePeriod});
  EXPECT_THAT(protocol->Start(5), IsOk());

  // The following block will repeatedly create CheckpointParser instances
  // which will be creating scalar int tensors with integer value 1.
  EXPECT_CALL(checkpoint_parser_factory_, Create(_)).WillRepeatedly(Invoke([&] {
    auto parser = std::make_unique<MockCheckpointParser>();
    EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([&] {
      return Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({1}));
    }));
    return parser;
  }));

  // Validate that no clients have any pending messages to poll at the start.
  for (int i = 0; i < 5; ++i) {
    absl::StatusOr<std::optional<ServerMessage>> polled_message =
        protocol->PollServerMessage(i);
    EXPECT_OK(polled_message.status());
    EXPECT_FALSE(polled_message.value().has_value());
    EXPECT_FALSE(protocol->IsClientClosed(i).value());
  }

  // 0 seconds into the protocol.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(0).value());

  // 1 second
  clock_.AdvanceTime(kInterval);
  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(1).value());

  // 2 seconds
  // num_latency_samples = 2, mean_latency = 500ms,
  // six_sigma_threshold = 4.7426406875s, num_pending_clients = 3
  clock_.AdvanceTime(kInterval);
  EXPECT_THAT(protocol->ReceiveClientMessage(3, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->ReceiveClientMessage(4, MakeClientMessage()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(3).value());
  EXPECT_TRUE(protocol->IsClientClosed(4).value());

  // 3, 4, 5 seconds - iterate one seconds at a time to give the outlier
  // detection more chances to run.
  // num_latency_samples = 4, mean_latency = 1.25s,
  // six_sigma_threshold = 6.9945626465s, num_pending_clients = 1
  clock_.AdvanceTime(kInterval);
  clock_.AdvanceTime(kInterval);
  clock_.AdvanceTime(kInterval);

  // Add 5 additional clients 5 seconds into the protocol.
  auto status_or_result = protocol->AddClients(5);
  EXPECT_THAT(status_or_result, IsOk());
  EXPECT_EQ(status_or_result.value(), 5);

  // 6 seconds.
  // num_latency_samples = 4, mean_latency = 1.25s,
  // six_sigma_threshold = 6.9945626465s, num_pending_clients = 6
  clock_.AdvanceTime(kInterval);
  EXPECT_THAT(protocol->ReceiveClientMessage(5, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->ReceiveClientMessage(7, MakeClientMessage()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(5).value());
  EXPECT_TRUE(protocol->IsClientClosed(7).value());

  // 7 seconds.
  // num_latency_samples = 6, mean_latency = 1.16666666675s,
  // six_sigma_threshold = 5.68330258325s, num_pending_clients = 4
  clock_.AdvanceTime(kInterval);
  // Client #2 should be closed at this time. It wasn't closed earlier because
  // the grace period (1 second) is added to the six_sigma_threshold (5.7s)
  // So the first chance to close the client #2 is at 7 seconds.
  EXPECT_EQ(protocol->PollServerMessage(2)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_TRUE(protocol->IsClientClosed(2).value());

  EXPECT_THAT(protocol->ReceiveClientMessage(9, MakeClientMessage()), IsOk());
  EXPECT_TRUE(protocol->IsClientClosed(9).value());

  // 8, 9, 10, 11, 12 seconds.
  // num_latency_samples = 7, mean_latency = 1.28571428575s,
  // six_sigma_threshold = 5.82128796175s, num_pending_clients = 2
  clock_.AdvanceTime(kInterval);
  clock_.AdvanceTime(kInterval);
  clock_.AdvanceTime(kInterval);
  clock_.AdvanceTime(kInterval);
  clock_.AdvanceTime(kInterval);

  // Client 2 closed at 7 seconds, clients 6 and 8 - at 12 seconds.
  // Clients #6 and #8 are added at 5 seconds into the protocol.
  // The first chance to abort them is at 12 seconds because the grace
  // period (1s) is added to the six_sigma_threshold (5.8s). So the first chance
  // to close those clients is at 12 seconds.
  EXPECT_EQ(protocol->PollServerMessage(6)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_EQ(protocol->PollServerMessage(8)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_TRUE(protocol->IsClientClosed(6).value());
  EXPECT_TRUE(protocol->IsClientClosed(8).value());

  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto(
          "protocol_state: PROTOCOL_STARTED num_clients_completed: 7 "
          "num_inputs_aggregated_and_included: 7 num_clients_aborted: 3"));
}

TEST_F(SimpleAggregationProtocolTest, OutlierDetection_NoPendingClients) {
  constexpr absl::Duration kInterval = absl::Seconds(1);
  constexpr absl::Duration kGracePeriod = absl::Seconds(1);

  // Start by creating the protocol with 5 clients.
  auto protocol = CreateProtocolWithDefaultConfig({kInterval, kGracePeriod});
  EXPECT_THAT(protocol->Start(5), IsOk());

  // The following block will repeatedly create CheckpointParser instances
  // which will be creating scalar int tensors with integer value 1.
  EXPECT_CALL(checkpoint_parser_factory_, Create(_)).WillRepeatedly(Invoke([&] {
    auto parser = std::make_unique<MockCheckpointParser>();
    EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([&] {
      return Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({1}));
    }));
    return parser;
  }));

  // Receive messages from 4 clients and close one client from the client side.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->ReceiveClientMessage(2, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->ReceiveClientMessage(3, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->CloseClient(4, absl::InternalError("foo")), IsOk());

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(protocol->IsClientClosed(i).value());
    EXPECT_EQ(protocol->PollServerMessage(i)
                  .value()
                  ->simple_aggregation()
                  .close_message()
                  .code(),
              static_cast<int>(OK));
  }

  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto(
          "protocol_state: PROTOCOL_STARTED num_clients_completed: 4 "
          "num_inputs_aggregated_and_included: 4 num_clients_failed: 1"));

  // Advance the time by 10 seconds and verify that the outlier detection
  // doesn't change outcome of any of the above clients.
  clock_.AdvanceTime(10 * kInterval);
  EXPECT_THAT(
      protocol->GetStatus(),
      testing::EqualsProto(
          "protocol_state: PROTOCOL_STARTED num_clients_completed: 4 "
          "num_inputs_aggregated_and_included: 4 num_clients_failed: 1"));
}

TEST_F(SimpleAggregationProtocolTest, OutlierDetection_AfterAbort) {
  constexpr absl::Duration kInterval = absl::Seconds(1);
  constexpr absl::Duration kGracePeriod = absl::Seconds(1);
  auto protocol = CreateProtocolWithDefaultConfig({kInterval, kGracePeriod});
  EXPECT_THAT(protocol->Start(2), IsOk());

  EXPECT_THAT(protocol->Abort(), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_ABORTED num_clients_aborted: 2"));
  EXPECT_TRUE(protocol->IsClientClosed(0).value());
  EXPECT_TRUE(protocol->IsClientClosed(1).value());
  EXPECT_EQ(protocol->PollServerMessage(0)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));
  EXPECT_EQ(protocol->PollServerMessage(1)
                .value()
                ->simple_aggregation()
                .close_message()
                .code(),
            static_cast<int>(ABORTED));

  // Advance the time by 10 seconds and verify num_clients_aborted.
  clock_.AdvanceTime(10 * kInterval);
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_ABORTED num_clients_aborted: 2"));
}

// Fake aggregator that accepts inputs but always produces a predefined output.
class FakeAggregator final : public TensorAggregator {
 public:
  explicit FakeAggregator(Tensor output) : output_(std::move(output)) {}

  absl::Status AggregateTensors(
      InputTensorList tensors,
      std::optional<google::protobuf::Any> metadata) override {
    return absl::OkStatus();
  }

  absl::Status MergeWith(TensorAggregator&& other) override {
    return absl::OkStatus();
  }

  absl::Status CheckValid() const override { return absl::OkStatus(); }

  OutputTensorList TakeOutputs() && override {
    OutputTensorList outputs(1);
    outputs.push_back(std::move(output_));
    return outputs;
  }

  int GetNumInputs() const override { return 0; }

  StatusOr<std::string> Serialize() && override {
    return TFF_STATUS(UNIMPLEMENTED);
  };

 private:
  Tensor output_;
};

// Factory for the FakeAggregator.
class FakeAggregatorFactory final : public TensorAggregatorFactory {
 public:
  explicit FakeAggregatorFactory(Tensor output) : output_(std::move(output)) {}

 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    return std::make_unique<FakeAggregator>(std::move(output_));
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return TFF_STATUS(UNIMPLEMENTED);
  }

  mutable Tensor output_;
};

TEST_F(SimpleAggregationProtocolTest,
       Complete_ErrorToProduceOutputFollowedByClientInput) {
  // This is a special case where an error is triggered by a mismatched output,
  // which transitions the protocol into an invalid state where the protocol
  // must be aborted. In this state any subsequent client inputs must be
  // handled gracefully.

  // Register a factory for the fake aggregator that returns a scalar string
  // tensor with string value "foo".
  auto string_tensor =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}))
          .value();
  FakeAggregatorFactory agg_factory(std::move(string_tensor));
  RegisterAggregatorFactory("fake1_aggregation", &agg_factory);

  // The configuration below refers to fake1_aggregation registered above,
  // but the expected output is DT_INT32.
  auto protocol = CreateProtocol(PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "fake1_aggregation"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb"));

  EXPECT_THAT(protocol->Start(1), IsOk());

  EXPECT_CALL(checkpoint_builder_factory_, Create())
      .WillOnce(Return(ByMove(std::make_unique<MockCheckpointBuilder>())));
  EXPECT_THAT(protocol->Complete(), StatusIs(INTERNAL));
  EXPECT_THAT(protocol->GetStatus(),
              testing::EqualsProto(
                  "protocol_state: PROTOCOL_ABORTED num_clients_aborted: 1"));

  EXPECT_OK(protocol->ReceiveClientMessage(0, MakeClientMessage()));
}

}  // namespace
}  // namespace tensorflow_federated::aggregation
