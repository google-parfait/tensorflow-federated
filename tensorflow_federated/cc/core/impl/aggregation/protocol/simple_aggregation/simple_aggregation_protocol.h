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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_SIMPLE_AGGREGATION_PROTOCOL_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_SIMPLE_AGGREGATION_PROTOCOL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/clock.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/aggregation_protocol.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/resource_resolver.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/simple_aggregation/cancelable_callback.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/simple_aggregation/latency_aggregator.h"

namespace tensorflow_federated::aggregation {

// Implementation of the simple aggregation protocol.
//
// This version of the protocol receives updates in the clear from clients in a
// TF checkpoint and aggregates them in memory. The aggregated updates are
// released only if the number of participants exceed configured threshold.
class SimpleAggregationProtocol final : public AggregationProtocol {
 public:
  struct OutlierDetectionParameters {
    // Specifies an interval at which outlier detection should be performed
    absl::Duration detection_interval;
    // Specifies how long the protocol implementation will wait after
    // determining that a pending client has become an outlier (e.g. the
    // client is not responding like other clients) before closing that client.
    absl::Duration grace_period;
  };

  // Validates the Configuration that will subsequently be used to create an
  // instance of this protocol.
  // Returns INVALID_ARGUMENT if the configuration is invalid.
  static absl::Status ValidateConfig(const Configuration& configuration);

  // Factory method to create an instance of the Simple Aggregation Protocol.
  //
  // Does not take ownership of the callback, which must refer to a valid object
  // that outlives the SimpleAggregationProtocol instance.
  //
  // Arguments:
  // - `configuration`: aggregation intrinsics configuration.
  // - `callback`: provided by the protocol host to receive the protocol
  //   callbacks.
  // - `checkpoint_parser_factory`: provides CheckpointParser instances for
  //   parsing input checkpoints.
  // - `checkpoint_builder_factory`: provides CheckpointBuilder instances for
  //   building output checkpoints.
  // - `resource_resolver`: if a client message references a resource, such as
  //   an input checkpoint, this resolver provides ability to retrieve it.
  // - `clock`: provides access to current time and ability to schedule delayed
  //   callbacks.
  // - `outlier_detection_parameters`: if provided, specifies parameters for the
  //    outliers (stale clients) detection algorithm, which is based on
  //    statistical analysis of client response times.  The purpose of the
  //    outlier detection is to close unresonsive clients.
  //    If not provided, the outlier detection is disabled.
  static absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>> Create(
      const Configuration& configuration,
      const CheckpointParserFactory* checkpoint_parser_factory,
      const CheckpointBuilderFactory* checkpoint_builder_factory,
      ResourceResolver* resource_resolver, Clock* clock = Clock::RealClock(),
      std::optional<OutlierDetectionParameters> outlier_detection_parameters =
          std::nullopt);

  // Implementation of the overridden Aggregation Protocol methods.
  absl::Status Start(int64_t num_clients) override;
  absl::StatusOr<int64_t> AddClients(int64_t num_clients) override;
  absl::Status ReceiveClientMessage(int64_t client_id,
                                    const ClientMessage& message) override;
  absl::StatusOr<std::optional<ServerMessage>> PollServerMessage(
      int64_t client_id) override;
  absl::Status CloseClient(int64_t client_id,
                           absl::Status client_status) override;
  absl::Status Complete() override;
  absl::Status Abort() override;
  StatusMessage GetStatus() override;
  absl::StatusOr<absl::Cord> GetResult() override;
  absl::StatusOr<bool> IsClientClosed(int64_t client_id) override;

  ~SimpleAggregationProtocol() override;

  // SimpleAggregationProtocol is neither copyable nor movable.
  SimpleAggregationProtocol(const SimpleAggregationProtocol&) = delete;
  SimpleAggregationProtocol& operator=(const SimpleAggregationProtocol&) =
      delete;

 private:
  // Private constructor.
  SimpleAggregationProtocol(
      std::unique_ptr<CheckpointAggregator> checkpoint_aggregator,
      const CheckpointParserFactory* checkpoint_parser_factory,
      const CheckpointBuilderFactory* checkpoint_builder_factory,
      ResourceResolver* resource_resolver, Clock* clock,
      std::optional<OutlierDetectionParameters> outlier_detection_parameters);

  // Creates an aggregator based on the intrinsic configuration.
  static absl::StatusOr<std::unique_ptr<TensorAggregator>> CreateAggregator(
      const Intrinsic& intrinsic);

  // Describes state of each client participating in the protocol.
  enum ClientState : uint8_t {
    // No input received from the client yet.
    CLIENT_PENDING,
    // Client input received but the aggregation still pending, which may
    // be the case when there are multiple concurrent ReceiveClientMessage
    // calls.
    CLIENT_RECEIVED_INPUT_AND_PENDING,
    // Client input has been successfully aggregated.
    CLIENT_COMPLETED,
    // Client failed either by being closed with an error or by submitting a
    // malformed input.
    CLIENT_FAILED,
    // Client which has been aborted by the server before its input has been
    // received.
    CLIENT_ABORTED,
    // Client input has been received but discarded, for example due to the
    // protocol Abort method being called.
    CLIENT_DISCARDED
  };

  struct ClientInfo {
    // State of the client.
    ClientState state;
    // Pending server message for a client that should be polled. In the case of
    // SimpleAggregation, the client (device) does not directly poll this
    // message, rather the server infrastructure does on behalf of the client.
    //
    // If empty, there is no message to poll.
    std::optional<ServerMessage> server_message;
  };

  // Returns string representation of the client state.
  static absl::string_view ClientStateDebugString(ClientState state);

  // Returns an error if the current protocol state isn't the expected one.
  absl::Status CheckProtocolState(ProtocolState state) const
      ABSL_SHARED_LOCKS_REQUIRED(state_mu_);

  // Changes the protocol state.
  void SetProtocolState(ProtocolState state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Add clients and initialize their state to CLIENT_PENDING.
  // Returns the first index of newly added clients.
  int64_t AddPendingClients(size_t num_clients, absl::Time now)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Gets the client state for the given client ID.
  absl::StatusOr<ClientState> GetClientState(int64_t client_id) const
      ABSL_SHARED_LOCKS_REQUIRED(state_mu_);

  // Sets the client state for the given client ID.
  void SetClientState(int64_t client_id, ClientState state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Close all pending clients when the protocol is terminated.
  void CloseAllClients() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // This two functions are used to wait for a condition where there are no
  // clients waiting for the their input to be aggregated.
  bool IsAggregationQueueEmpty() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void AwaitAggregationQueueEmpty() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Produces the report via the underlying aggregators.
  absl::StatusOr<absl::Cord> CreateReport();

  void ScheduleOutlierDetection()
      ABSL_LOCKS_EXCLUDED(state_mu_, outlier_detection_mu_);
  // Called periodically from a background thread to perform outlier detection.
  void PerformOutlierDetection()
      ABSL_LOCKS_EXCLUDED(state_mu_, outlier_detection_mu_);
  // Stops outlier detection.
  void StopOutlierDetection()
      ABSL_LOCKS_EXCLUDED(state_mu_, outlier_detection_mu_);

  // Protects the mutable state.
  absl::Mutex state_mu_;
  // The overall state of the protocol.
  ProtocolState protocol_state_ ABSL_GUARDED_BY(state_mu_);

  // Holds state of all clients. The length of the vector equals
  // to the number of clients accepted into the protocol.
  std::vector<ClientInfo> all_clients_ ABSL_GUARDED_BY(state_mu_);

  // Holds information about pending clients - when each client joined the
  // protocol. This helps to detect the outliers that remain in the pending
  // state for too long.
  absl::flat_hash_map<int64_t, absl::Time> pending_clients_
      ABSL_GUARDED_BY(state_mu_);

  // Calculates latency stats for clients that have successfully completed
  // the protocol. This provides data for calculating the threshold for
  // determining the outliers.
  LatencyAggregator latency_aggregator_ ABSL_GUARDED_BY(state_mu_);

  // Counters for various client states other than pending.
  // Note that the number of pending clients can be found by subtracting the
  // sum of the below counters from `client_states_.size()`.
  uint64_t num_clients_received_and_pending_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_aggregated_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_failed_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_aborted_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_discarded_ ABSL_GUARDED_BY(state_mu_) = 0;

  std::unique_ptr<CheckpointAggregator> checkpoint_aggregator_;
  const CheckpointParserFactory* const checkpoint_parser_factory_;
  const CheckpointBuilderFactory* const checkpoint_builder_factory_;
  ResourceResolver* const resource_resolver_;
  Clock* const clock_;
  // The result of the aggregation.
  absl::Cord result_ ABSL_GUARDED_BY(state_mu_);

  // Fields related to outlier detection.
  absl::Mutex outlier_detection_mu_;
  CancelationToken outlier_detection_cancelation_token_
      ABSL_GUARDED_BY(outlier_detection_mu_);
  std::optional<OutlierDetectionParameters> outlier_detection_parameters_
      ABSL_GUARDED_BY(outlier_detection_mu_);
};
}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_SIMPLE_AGGREGATION_PROTOCOL_H_
