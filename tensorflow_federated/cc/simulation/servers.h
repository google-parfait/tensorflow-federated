/* Copyright 2022, The TensorFlow Federated Authors.

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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_SIMULATION_SERVERS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_SIMULATION_SERVERS_H_

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "grpcpp/grpcpp.h"
#include "include/grpcpp/security/server_credentials.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_service.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

// Runs TFF ExecutorService backed by executors returned by the given
// executor_fn, listening on port. This function blocks, and will only
// return on error or shutdown.
void RunServer(std::function<absl::StatusOr<std::shared_ptr<Executor>>(
                   const CardinalityMap&)>
                   executor_fn,
               int port, std::shared_ptr<grpc::ServerCredentials> credentials,
               int grpc_max_message_length_megabytes);

// Runs a specialized version of RunServer above; the running executor service
// will execute federated computations on the local machine.
void RunWorker(
    int port, std::shared_ptr<grpc::ServerCredentials> credentials,
    int grpc_max_message_length_megabytes,
    absl::optional<uint32_t> max_concurrent_computation_calls = absl::nullopt);

}  // namespace tensorflow_federated
#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_SIMULATION_SERVERS_H_
