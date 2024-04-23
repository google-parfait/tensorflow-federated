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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/aggregation_protocol_messages.pb.h"

namespace tensorflow_federated::aggregation {

// Describes a abstract aggregation protocol interface between a networking
// layer (e.g. a service that handles receiving and sending messages with the
// client devices) and an implementation of an aggregation algorithm.
//
// The design of the AggregationProtocol follows a Bridge Pattern
// (https://en.wikipedia.org/wiki/Bridge_pattern) in that it is meant to
// decouple an abstraction of the layers above and below the AggregationProtocol
// from the implementation.
//
// In this interface the receiving and sending contributing inputs or
// messages is abstracted from the actual mechanism for sending and receiving
// data over the network and from the actual aggregation mechanism.
//
// Client identification: the real client identities are hidden from the
// protocol implementations. Instead each client is identified by a client_id
// number in a range [0, num_clients) where num_clients is the number of clients
// the protocol started with or the extended number of clients, which is the
// sum of the starting num_clients and num_clients passed to each subsequent
// AddClients call.
//
// Thread safety: for any given client identified by a unique client_id, the
// protocol methods are expected to be called sequentially. But there are no
// assumptions about concurrent calls made for different clients. Specific
// implementations of AggregationProtocol are expected to handle concurrent
// calls. The caller side of the protocol isn't expected to queue messages.
class AggregationProtocol {
 public:
  AggregationProtocol() = default;
  virtual ~AggregationProtocol() = default;

  // Instructs the protocol to start with the specified number of clients.
  //
  // Depending on the protocol implementation, the starting number of clients
  // may be zero.  This method is guaranteed to be the first method called on
  // the protocol.
  //
  // The starting index of the batch of clients added is always 0.
  virtual absl::Status Start(int64_t num_clients) = 0;

  // Adds an additional batch of clients to the protocol.
  //
  // Depending on the protocol implementation, adding clients may not be allowed
  // and this method might return an error Status.
  //
  // Returns the starting index of the batch of clients added.
  virtual absl::StatusOr<int64_t> AddClients(int64_t num_clients) = 0;

  // Handles a message from a given client.
  //
  // Depending on the specific protocol implementation there may be multiple
  // messages exchanged with each clients.
  //
  // This method should return an error status only if there is an unrecoverable
  // error which must result in aborting the protocol.  Any client specific
  // error, like an invalid message, should result in closing the protocol with
  // that specific client only, but this method should still return OK status.
  virtual absl::Status ReceiveClientMessage(int64_t client_id,
                                            const ClientMessage& message) = 0;

  // Checks for outgoing messages to a given client.
  //
  // Returns a non-ok status if there is an error requiring the protocol to
  // abort, otherwise it either returns any message waiting for the client,
  // which could be null.
  virtual absl::StatusOr<std::optional<ServerMessage>> PollServerMessage(
      int64_t client_id) = 0;

  // Notifies the protocol about a communication with a given client being
  // closed, either normally or abnormally.
  //
  // The client_status indicates whether the client connection was closed
  // normally.
  //
  // No further calls (except `PollServerMessage`) specific to the given client
  // are expected after this method.
  virtual absl::Status CloseClient(int64_t client_id,
                                   absl::Status client_status) = 0;

  // Forces the protocol to complete.
  //
  // Once the protocol has completed successfully, the caller should invoke
  // `GetResult` to get the aggregation result.  If the protocol cannot be
  // completed in its current state, this method should return an error status.
  // It is also possible for the completion to fail eventually due to finishing
  // some asynchronous work.
  //
  // No further protocol method calls except Abort, GetStatus and
  // PollServerMessage are expected after this method.
  virtual absl::Status Complete() = 0;

  // Forces the protocol to Abort.
  //
  // No further protocol method calls except GetStatus and PollServerMessage are
  // expected after this method.
  virtual absl::Status Abort() = 0;

  // Called periodically to receive the protocol status.
  //
  // This method can still be called after the protocol has been completed or
  // aborted.
  virtual StatusMessage GetStatus() = 0;

  // Returns the result of the aggregation.
  //
  // Returns FAILED_PRECONDITION error if the protocol is not in COMPLETED state
  // when this method is invoked.
  virtual absl::StatusOr<absl::Cord> GetResult() = 0;

  // Returns true if the client has been closed in the protocol.
  virtual absl::StatusOr<bool> IsClientClosed(int64_t client_id) = 0;
};

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_H_
