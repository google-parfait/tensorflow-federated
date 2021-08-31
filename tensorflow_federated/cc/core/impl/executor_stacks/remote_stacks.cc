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

#include <chrono>  // NOLINT
#include <limits>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"
#include "grpc/include/grpcpp/support/time.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/remote_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"

namespace tensorflow_federated {

// This function queries the state of the incoming channels, filtering to those
// which are ready or idle. This function blocks up to
// `wait_connected_duration_millis` milliseconds, awaiting all channels in
// `channels` to be ready.
std::vector<const std::shared_ptr<grpc::ChannelInterface>>
FilterToLiveChannels_(
    absl::Span<const std::shared_ptr<grpc::ChannelInterface>> channels,
    int wait_connected_duration_millis = 1000) {
  std::vector<const std::shared_ptr<grpc::ChannelInterface>> live_channels;
  auto wait_connected =
      [&wait_connected_duration_millis](
          std::shared_ptr<grpc::ChannelInterface> channel) -> absl::Status {
    bool connected = channel->WaitForConnected(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(wait_connected_duration_millis));
    if (connected) {
      return absl::OkStatus();
    } else {
      return absl::UnavailableError("Channel not ready.");
    }
  };

  ParallelTasks wait_connected_tasks;
  for (const std::shared_ptr<grpc::ChannelInterface>& channel : channels) {
    wait_connected_tasks.add_task(std::bind(wait_connected, channel));
  }
  // Block up to WaitConnectedDuration for all tasks to complete.
  absl::Status all_channels_ready = wait_connected_tasks.WaitAll();

  for (const std::shared_ptr<grpc::ChannelInterface>& channel : channels) {
    // We can wait 0 duration here since we have already blocked for all
    // channels to attempt connection.
    bool connected =
        channel->WaitForConnected(std::chrono::system_clock::now());
    if (!connected) {
      // This channel is not yet ready to serve requests.
      continue;
    } else {
      live_channels.emplace_back(channel);
    }
  }
  return live_channels;
}

absl::StatusOr<std::shared_ptr<Executor>> CreateRemoteExecutorStack(
    absl::Span<const std::shared_ptr<grpc::ChannelInterface>> channels,
    const CardinalityMap& cardinalities) {
  auto rre_tf_leaf_executor = []() {
    return CreateReferenceResolvingExecutor(CreateTensorFlowExecutor());
  };
  ComposingChildFn composing_child_factory =
      [](std::shared_ptr<grpc::ChannelInterface> channel,
         const CardinalityMap& cardinalities)
      -> absl::StatusOr<ComposingChild> {
    return TFF_TRY(ComposingChild::Make(
        CreateRemoteExecutor(channel, cardinalities), cardinalities));
  };

  return CreateRemoteExecutorStack(
      channels, cardinalities, rre_tf_leaf_executor, composing_child_factory);
}

absl::StatusOr<std::shared_ptr<Executor>> CreateRemoteExecutorStack(
    absl::Span<const std::shared_ptr<grpc::ChannelInterface>> channels,
    const CardinalityMap& cardinalities, ExecutorFn leaf_executor_fn,
    ComposingChildFn composing_child_fn) {
  int num_clients = 0;
  auto cards_iterator = cardinalities.find(kClientsUri);
  if (cards_iterator != cardinalities.end()) {
    num_clients = cards_iterator->second;
  } else {
    return absl::InvalidArgumentError(
        "Num clients not specified in cardinalities.");
  }
  std::shared_ptr<Executor> server = TFF_TRY(leaf_executor_fn());
  int remaining_clients = num_clients;
  if (remaining_clients == 0) {
    auto federated_cardinalities = cardinalities;
    federated_cardinalities.insert_or_assign(kClientsUri, 0);
    return CreateReferenceResolvingExecutor(
        TFF_TRY(CreateFederatingExecutor(server, federated_cardinalities)));
  } else if (channels.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "A remote executor stack with nonzero number of clients must be "
        "configured with some remote worker. Found 0 remote channels but ",
        remaining_clients, " num_clients."));
  }

  auto live_channels = FilterToLiveChannels_(channels);
  int remaining_num_executors = live_channels.size();
  if (live_channels.empty()) {
    return absl::UnavailableError(
        "No TFF workers are ready; try again to reconnect");
  }
  std::vector<ComposingChild> remote_executors;
  for (const std::shared_ptr<grpc::ChannelInterface>& channel : live_channels) {
    int clients_for_executor = remaining_clients / remaining_num_executors;
    CardinalityMap cardinalities_for_executor = cardinalities;
    cardinalities_for_executor.insert_or_assign(kClientsUri,
                                                clients_for_executor);
    remote_executors.emplace_back(
        TFF_TRY(composing_child_fn(channel, cardinalities_for_executor)));
    remaining_clients -= clients_for_executor;
    remaining_num_executors -= 1;
  }
  VLOG(2) << "Addressing: " << remote_executors.size() << " Live TFF workers.";
  return CreateReferenceResolvingExecutor(
      CreateComposingExecutor(server, remote_executors));
}

}  // namespace tensorflow_federated
