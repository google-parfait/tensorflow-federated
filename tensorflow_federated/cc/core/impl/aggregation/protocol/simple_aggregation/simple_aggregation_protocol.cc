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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/clock.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/resource_resolver.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/simple_aggregation/cancelable_callback.h"

namespace tensorflow_federated::aggregation {

namespace {

// Timeout for waiting for the aggregation queue to empty when the aggregation
// is completed or aborted. Please note that there is no actual queue but
// there may potentially be multiple concurrent calls to accumulate client
// inputs that all block inside ReceiveClientMessage, which is indicated by
// clients being in CLIENT_RECEIVED_INPUT_AND_PENDING state and the
// num_clients_received_and_pending_ count being greater than zero.
// This timeout specifies how long Complete() or Abort() method will wait for
// those blocked calls to go through and the num_clients_received_and_pending_
// count to drop back to zero.
constexpr absl::Duration kAggregationQueueWaitTimeout = absl::Seconds(5);

ServerMessage MakeCloseClientMessage(absl::Status status) {
  ServerMessage message;
  message.mutable_simple_aggregation()->mutable_close_message()->set_code(
      static_cast<int>(status.code()));
  message.mutable_simple_aggregation()->mutable_close_message()->set_message(
      std::string(std::move(status).message()));
  return message;
}
}  // namespace

absl::Status SimpleAggregationProtocol::ValidateConfig(
    const Configuration& configuration) {
  return CheckpointAggregator::ValidateConfig(configuration);
}

absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
SimpleAggregationProtocol::Create(
    const Configuration& configuration,
    const CheckpointParserFactory* checkpoint_parser_factory,
    const CheckpointBuilderFactory* checkpoint_builder_factory,
    ResourceResolver* resource_resolver, Clock* clock,
    std::optional<OutlierDetectionParameters> outlier_detection_parameters) {
  TFF_CHECK(checkpoint_parser_factory != nullptr);
  TFF_CHECK(checkpoint_builder_factory != nullptr);
  TFF_CHECK(resource_resolver != nullptr);
  TFF_CHECK(clock != nullptr);

  TFF_ASSIGN_OR_RETURN(auto checkpoint_aggregator,
                       CheckpointAggregator::Create(configuration));

  return absl::WrapUnique(new SimpleAggregationProtocol(
      std::move(checkpoint_aggregator), checkpoint_parser_factory,
      checkpoint_builder_factory, resource_resolver, clock,
      std::move(outlier_detection_parameters)));
}

SimpleAggregationProtocol::SimpleAggregationProtocol(
    std::unique_ptr<CheckpointAggregator> checkpoint_aggregator,
    const CheckpointParserFactory* checkpoint_parser_factory,
    const CheckpointBuilderFactory* checkpoint_builder_factory,
    ResourceResolver* resource_resolver, Clock* clock,
    std::optional<OutlierDetectionParameters> outlier_detection_parameters)
    : protocol_state_(PROTOCOL_CREATED),
      checkpoint_aggregator_(std::move(checkpoint_aggregator)),
      checkpoint_parser_factory_(checkpoint_parser_factory),
      checkpoint_builder_factory_(checkpoint_builder_factory),
      resource_resolver_(resource_resolver),
      clock_(clock),
      outlier_detection_parameters_(std::move(outlier_detection_parameters)) {}

SimpleAggregationProtocol::~SimpleAggregationProtocol() {
  // Stop outlier detection in case it wasn't stopped before.
  StopOutlierDetection();
}

absl::string_view SimpleAggregationProtocol::ClientStateDebugString(
    ClientState state) {
  switch (state) {
    case CLIENT_PENDING:
      return "CLIENT_PENDING";
    case CLIENT_RECEIVED_INPUT_AND_PENDING:
      return "CLIENT_RECEIVED_INPUT_AND_PENDING";
    case CLIENT_COMPLETED:
      return "CLIENT_COMPLETED";
    case CLIENT_FAILED:
      return "CLIENT_FAILED";
    case CLIENT_ABORTED:
      return "CLIENT_ABORTED";
    case CLIENT_DISCARDED:
      return "CLIENT_DISCARDED";
  }
}

absl::Status SimpleAggregationProtocol::CheckProtocolState(
    ProtocolState state) const {
  if (protocol_state_ != state) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "The current protocol state is %s, expected %s.",
        ProtocolState_Name(protocol_state_), ProtocolState_Name(state)));
  }
  return absl::OkStatus();
}

void SimpleAggregationProtocol::SetProtocolState(ProtocolState state) {
  TFF_CHECK(
      (protocol_state_ == PROTOCOL_CREATED && state == PROTOCOL_STARTED) ||
      (protocol_state_ == PROTOCOL_STARTED &&
       (state == PROTOCOL_COMPLETED || state == PROTOCOL_ABORTED)))
      << "Invalid protocol state transition from "
      << ProtocolState_Name(protocol_state_) << " to "
      << ProtocolState_Name(state) << ".";
  protocol_state_ = state;
}

absl::StatusOr<SimpleAggregationProtocol::ClientState>
SimpleAggregationProtocol::GetClientState(int64_t client_id) const {
  if (client_id < 0 || client_id >= all_clients_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("client_id %ld is outside the valid range", client_id));
  }
  return all_clients_[client_id].state;
}

int64_t SimpleAggregationProtocol::AddPendingClients(size_t num_clients,
                                                     absl::Time now) {
  int64_t start_index = all_clients_.size();
  int64_t end_index = start_index + num_clients;
  all_clients_.resize(end_index, {CLIENT_PENDING, std::nullopt});
  for (int64_t client_id = start_index; client_id < end_index; ++client_id) {
    pending_clients_.emplace(client_id, now);
  }
  return start_index;
}

void SimpleAggregationProtocol::SetClientState(int64_t client_id,
                                               ClientState to_state) {
  TFF_CHECK(client_id >= 0 && client_id < all_clients_.size());
  ClientState from_state = all_clients_[client_id].state;
  TFF_CHECK(from_state != to_state);
  if (from_state == CLIENT_RECEIVED_INPUT_AND_PENDING) {
    num_clients_received_and_pending_--;
  } else if (from_state == CLIENT_COMPLETED) {
    TFF_CHECK(to_state == CLIENT_DISCARDED)
        << "Client state can't be changed from CLIENT_COMPLETED to "
        << ClientStateDebugString(to_state);
    num_clients_aggregated_--;
  } else {
    TFF_CHECK(from_state == CLIENT_PENDING)
        << "Client state can't be changed from "
        << ClientStateDebugString(from_state);
    // Remove the client from the pending set.
    pending_clients_.erase(client_id);
  }
  all_clients_[client_id].state = to_state;
  switch (to_state) {
    case CLIENT_PENDING:
      TFF_LOG(FATAL) << "Client state can't be changed to CLIENT_PENDING";
      break;
    case CLIENT_RECEIVED_INPUT_AND_PENDING:
      num_clients_received_and_pending_++;
      break;
    case CLIENT_COMPLETED:
      num_clients_aggregated_++;
      break;
    case CLIENT_FAILED:
      num_clients_failed_++;
      break;
    case CLIENT_ABORTED:
      num_clients_aborted_++;
      break;
    case CLIENT_DISCARDED:
      num_clients_discarded_++;
      break;
  }
}

absl::StatusOr<absl::Cord> SimpleAggregationProtocol::CreateReport() {
  if (!checkpoint_aggregator_->CanReport()) {
    return absl::FailedPreconditionError(
        "The aggregation can't be completed due to failed preconditions.");
  }

  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      checkpoint_builder_factory_->Create();

  // Build the resulting checkpoint.
  TFF_RETURN_IF_ERROR(checkpoint_aggregator_->Report(*checkpoint_builder));
  return checkpoint_builder->Build();
}

absl::Status SimpleAggregationProtocol::Start(int64_t num_clients) {
  if (num_clients < 0) {
    return absl::InvalidArgumentError("Number of clients cannot be negative.");
  }
  absl::Time now = clock_->Now();
  {
    absl::MutexLock lock(&state_mu_);
    TFF_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_CREATED));
    SetProtocolState(PROTOCOL_STARTED);
    TFF_CHECK(all_clients_.empty());
    AddPendingClients(num_clients, now);
  }
  ScheduleOutlierDetection();
  return absl::OkStatus();
}

absl::StatusOr<int64_t> SimpleAggregationProtocol::AddClients(
    int64_t num_clients) {
  absl::Time now = clock_->Now();
  int64_t start_index;
  {
    absl::MutexLock lock(&state_mu_);
    TFF_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_STARTED));
    if (num_clients <= 0) {
      return absl::InvalidArgumentError("Non-zero number of clients required");
    }
    start_index = AddPendingClients(num_clients, now);
  }
  return start_index;
}

absl::Status SimpleAggregationProtocol::ReceiveClientMessage(
    int64_t client_id, const ClientMessage& message) {
  if (!message.has_simple_aggregation() ||
      !message.simple_aggregation().has_input()) {
    return absl::InvalidArgumentError("Unexpected message");
  }

  if (!message.simple_aggregation().input().has_inline_bytes() &&
      !message.simple_aggregation().input().has_uri()) {
    return absl::InvalidArgumentError(
        "Only inline_bytes or uri type of input is supported");
  }

  // Verify the state.
  absl::Time client_start_time;
  {
    absl::ReleasableMutexLock lock(&state_mu_);
    if (protocol_state_ == PROTOCOL_CREATED) {
      return absl::FailedPreconditionError("The protocol hasn't been started");
    }
    TFF_ASSIGN_OR_RETURN(auto client_state, GetClientState(client_id));
    if (client_state != CLIENT_PENDING) {
      lock.Release();
      // TODO: b/252825568 - Decide whether the logging level should be INFO or
      // WARNING, or perhaps it should depend on the client state (e.g. WARNING
      // for COMPLETED and INFO for other states).
      TFF_LOG(INFO) << "ReceiveClientMessage: client " << client_id
                    << " message ignored, the state is already "
                    << ClientStateDebugString(client_state);
      return absl::OkStatus();
    }
    client_start_time = pending_clients_[client_id];
    SetClientState(client_id, CLIENT_RECEIVED_INPUT_AND_PENDING);
  }
  absl::Duration client_latency = clock_->Now() - client_start_time;

  absl::Status client_completion_status = absl::OkStatus();
  ClientState client_completion_state = CLIENT_COMPLETED;

  absl::Cord report;
  if (message.simple_aggregation().input().has_inline_bytes()) {
    // Parse the client input concurrently with other protocol calls.
    report =
        absl::Cord(message.simple_aggregation().input().inline_bytes());
  } else {
    absl::StatusOr<absl::Cord> report_or_status =
        resource_resolver_->RetrieveResource(
            client_id, message.simple_aggregation().input().uri());
    if (!report_or_status.ok()) {
      client_completion_status = report_or_status.status();
      client_completion_state = CLIENT_FAILED;
      TFF_LOG(WARNING) << "Report with resource uri "
                       << message.simple_aggregation().input().uri()
                       << " for client " << client_id << "is missing. "
                       << client_completion_status;
    } else {
      report = std::move(report_or_status.value());
    }
  }

  if (client_completion_state != CLIENT_FAILED) {
    absl::StatusOr<std::unique_ptr<CheckpointParser>> parser_or_status =
        checkpoint_parser_factory_->Create(report);
    if (!parser_or_status.ok()) {
      client_completion_status = parser_or_status.status();
      client_completion_state = CLIENT_FAILED;
      TFF_LOG(WARNING) << "Client " << client_id << " input can't be parsed: "
                       << client_completion_status;
    } else {
      client_completion_status =
          checkpoint_aggregator_->Accumulate(*parser_or_status.value());
      if (client_completion_status.code() == StatusCode::kAborted) {
        client_completion_state = CLIENT_DISCARDED;
        TFF_LOG(INFO) << "Client " << client_id
                      << " input is discarded: " << client_completion_status;
      } else if (!client_completion_status.ok()) {
        client_completion_state = CLIENT_FAILED;
        TFF_LOG(INFO) << "Client " << client_id
                      << " input can't be aggregated: "
                      << client_completion_status;
      }
    }
  }

  // Update the state post aggregation.
  ServerMessage close_message =
      MakeCloseClientMessage(client_completion_status);
  {
    absl::MutexLock lock(&state_mu_);
    latency_aggregator_.Add(client_latency);
    auto client_state = GetClientState(client_id);
    // Expect this to succeed because the client_id has already been validated.
    TFF_CHECK(client_state.ok());
    // Update the client state only if the client is still in the state
    // CLIENT_RECEIVED_INPUT_AND_PENDING that was set earlier in this method.
    // If the state has changed, that means another thread has updated the
    // client state to a terminal state (e.g. CLIENT_COMPLETED, CLIENT_FAILED,
    // CLIENT_ABORTED, CLIENT_DISCARDED), and we should not overwrite that
    // state with the current state.
    if (*client_state == CLIENT_RECEIVED_INPUT_AND_PENDING) {
      SetClientState(client_id, client_completion_state);
      all_clients_[client_id].server_message = std::move(close_message);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<ServerMessage>>
SimpleAggregationProtocol::PollServerMessage(int64_t client_id) {
  absl::MutexLock lock(&state_mu_);
  if (protocol_state_ == PROTOCOL_CREATED) {
    return absl::FailedPreconditionError("The protocol hasn't been started");
  }
  if (client_id < 0 || client_id >= all_clients_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("client_id %ld is outside the valid range", client_id));
  }
  std::optional<ServerMessage> output;
  std::swap(all_clients_[client_id].server_message, output);
  return output;
}

absl::Status SimpleAggregationProtocol::CloseClient(
    int64_t client_id, absl::Status client_status) {
  bool closed_client = false;
  {
    absl::MutexLock lock(&state_mu_);
    if (protocol_state_ == PROTOCOL_CREATED) {
      return absl::FailedPreconditionError("The protocol hasn't been started");
    }
    TFF_ASSIGN_OR_RETURN(auto client_state, GetClientState(client_id));
    // Close the client only if the client is currently pending.
    if (client_state == CLIENT_PENDING) {
      closed_client = true;
      SetClientState(client_id,
                     client_status.ok() ? CLIENT_DISCARDED : CLIENT_FAILED);
    }
  }
  TFF_LOG_IF(INFO, closed_client)
      << "Closing client " << client_id << " with the status " << client_status;

  return absl::OkStatus();
}

void SimpleAggregationProtocol::CloseAllClients() {
  TFF_CHECK(protocol_state_ == PROTOCOL_COMPLETED ||
            protocol_state_ == PROTOCOL_ABORTED);
  ServerMessage close_message = MakeCloseClientMessage(
      absl::AbortedError("The protocol has terminated before the "
                         "client input has been aggregated."));
  for (int64_t client_id = 0; client_id < all_clients_.size(); client_id++) {
    switch (all_clients_[client_id].state) {
      case CLIENT_PENDING:
        SetClientState(client_id, CLIENT_ABORTED);
        all_clients_[client_id].server_message = close_message;
        break;
      case CLIENT_RECEIVED_INPUT_AND_PENDING:
        // Please note that all clients in this state are supposed to finish
        // and change their state to either CLIENT_COMPLETED or CLIENT_DISCARDED
        // by the time this method is called. Hitting this case is most likely
        // an indication of some error.  Normally all concurrent calls to
        // ReceiveClientMessage() should get an opportunity to finish when the
        // AwaitAggregationQueueEmpty() method is called.  At that time the
        // the CheckpointAggregation should already be in the state where all
        // pending Accumulate() calls should finish instantly without doing any
        // actual work.
        TFF_LOG(WARNING) << "Client " << client_id
                         << " is in CLIENT_RECEIVED_INPUT_AND_PENDING state at "
                            "the termination of the protocol.";
        SetClientState(client_id, CLIENT_DISCARDED);
        all_clients_[client_id].server_message = close_message;
        break;
      case CLIENT_COMPLETED:
        if (protocol_state_ == PROTOCOL_ABORTED) {
          SetClientState(client_id, CLIENT_DISCARDED);
        }
        break;
      default:
        break;
    }
  }
}

bool SimpleAggregationProtocol::IsAggregationQueueEmpty() {
  return num_clients_received_and_pending_ == 0;
}

void SimpleAggregationProtocol::AwaitAggregationQueueEmpty() {
  TFF_CHECK(protocol_state_ == PROTOCOL_COMPLETED ||
            protocol_state_ == PROTOCOL_ABORTED);
  if (!state_mu_.AwaitWithTimeout(
          absl::Condition(this,
                          &SimpleAggregationProtocol::IsAggregationQueueEmpty),
          kAggregationQueueWaitTimeout)) {
    TFF_LOG(ERROR) << "Aggregation queue is not empty after "
                   << kAggregationQueueWaitTimeout;
  }
}

absl::Status SimpleAggregationProtocol::Complete() {
  StopOutlierDetection();
  {
    absl::MutexLock lock(&state_mu_);
    TFF_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_STARTED));
  }

  auto report = CreateReport();

  absl::MutexLock lock(&state_mu_);
  // Make sure the protocol wasn't aborted while creating the report.
  TFF_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_STARTED));
  if (report.ok()) {
    SetProtocolState(PROTOCOL_COMPLETED);
    result_ = std::move(report.value());
  } else {
    SetProtocolState(PROTOCOL_ABORTED);
  }
  AwaitAggregationQueueEmpty();
  CloseAllClients();
  return report.status();
}

absl::Status SimpleAggregationProtocol::Abort() {
  StopOutlierDetection();
  absl::MutexLock lock(&state_mu_);
  TFF_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_STARTED));
  checkpoint_aggregator_->Abort();
  SetProtocolState(PROTOCOL_ABORTED);
  AwaitAggregationQueueEmpty();
  CloseAllClients();
  return absl::OkStatus();
}

StatusMessage SimpleAggregationProtocol::GetStatus() {
  absl::MutexLock lock(&state_mu_);
  int64_t num_clients_completed = num_clients_received_and_pending_ +
                                  num_clients_aggregated_ +
                                  num_clients_discarded_;
  StatusMessage message;
  message.set_protocol_state(protocol_state_);
  message.set_num_clients_completed(num_clients_completed);
  message.set_num_clients_failed(num_clients_failed_);
  message.set_num_clients_pending(all_clients_.size() - num_clients_completed -
                                  num_clients_failed_ - num_clients_aborted_);
  message.set_num_inputs_aggregated_and_included(num_clients_aggregated_);
  message.set_num_inputs_aggregated_and_pending(
      num_clients_received_and_pending_);
  message.set_num_clients_aborted(num_clients_aborted_);
  message.set_num_inputs_discarded(num_clients_discarded_);
  return message;
}

absl::StatusOr<absl::Cord> SimpleAggregationProtocol::GetResult() {
  absl::MutexLock lock(&state_mu_);
  TFF_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_COMPLETED));
  return result_;
}

absl::StatusOr<bool> SimpleAggregationProtocol::IsClientClosed(
    int64_t client_id) {
  absl::MutexLock lock(&state_mu_);
  if (client_id < 0 || client_id >= all_clients_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("client_id %ld is outside the valid range", client_id));
  }
  ClientState client_state = all_clients_[client_id].state;
  return client_state == CLIENT_COMPLETED || client_state == CLIENT_ABORTED ||
         client_state == CLIENT_DISCARDED || client_state == CLIENT_FAILED;
}

void SimpleAggregationProtocol::ScheduleOutlierDetection() {
  absl::MutexLock lock(&outlier_detection_mu_);
  if (outlier_detection_parameters_.has_value()) {
    TFF_LOG(INFO) << "Scheduling outlier detection";
    outlier_detection_cancelation_token_ = ScheduleCallback(
        clock_, outlier_detection_parameters_->detection_interval, [this]() {
          PerformOutlierDetection();
          ScheduleOutlierDetection();
        });
  }
}

void SimpleAggregationProtocol::PerformOutlierDetection() {
  TFF_LOG(INFO) << "Performing outlier detection";
  absl::Duration grace_period;
  {
    absl::MutexLock lock(&outlier_detection_mu_);
    if (!outlier_detection_parameters_.has_value()) {
      // Normally this method shouldn't be called when
      // outlier_detection_parameters is empty, however there is a small chance
      // for a race condition between performing outlier detection and canceling
      // outlier detection which would result in this method being called with
      // empty outlier_detection_parameters_.
      return;
    }
    grace_period = outlier_detection_parameters_->grace_period;
  }

  std::vector<int64_t> client_ids_to_close;
  // Perform this part of the algorithm under the lock to ensure exclusive
  // access to the all_clients_ and pending_clients_
  absl::Time now = clock_->Now();
  {
    absl::MutexLock lock(&state_mu_);
    TFF_CHECK(CheckProtocolState(PROTOCOL_STARTED).ok())
        << "The protocol is not in PROTOCOL_STARTED state.";

    // Cannot perform analysis if there are no pending clients or too few
    // client latency samples.
    if (pending_clients_.empty() || latency_aggregator_.GetCount() <= 1) return;

    absl::StatusOr<absl::Duration> latency_standard_deviation =
        latency_aggregator_.GetStandardDeviation();
    // GetStandardDeviation can fail only if there are too few samples.
    TFF_CHECK(latency_standard_deviation.ok())
        << "GetStandardDeviation() has unexpectedly failed: "
        << latency_standard_deviation.status();

    absl::Duration six_sigma_threshold =
        latency_aggregator_.GetMean() + 6 * latency_standard_deviation.value();
    TFF_VLOG(1) << "SimpleAggregationProtocol: num_latency_samples = "
                << latency_aggregator_.GetCount()
                << ", mean_latency = " << latency_aggregator_.GetMean()
                << ", six_sigma_threshold = " << six_sigma_threshold
                << ", num_pending_clients = " << pending_clients_.size();

    absl::Duration outlier_threshold = six_sigma_threshold + grace_period;
    for (auto [client_id, start_time] : pending_clients_) {
      if (now - start_time > outlier_threshold) {
        TFF_VLOG(1) << "SimpleAggregationProtocol: client " << client_id
                    << " is outlier: elapsed time = " << now - start_time
                    << ", outlier_threshold = " << outlier_threshold;
        client_ids_to_close.push_back(client_id);
      }
    }

    ServerMessage close_message = MakeCloseClientMessage(absl::AbortedError(
        "Client aborted due to being detected as an outlier"));
    for (int64_t client_id : client_ids_to_close) {
      SetClientState(client_id, CLIENT_ABORTED);
      all_clients_[client_id].server_message = close_message;
    }
  }
}

void SimpleAggregationProtocol::StopOutlierDetection() {
  CancelationToken cancellation_token;
  {
    absl::MutexLock lock(&outlier_detection_mu_);
    cancellation_token = std::move(outlier_detection_cancelation_token_);
    outlier_detection_parameters_.reset();
  }
  if (cancellation_token) {
    TFF_LOG(INFO) << "Canceling outlier detection";
    cancellation_token->Cancel();
  }
}

}  // namespace tensorflow_federated::aggregation
