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

#include "tensorflow_federated/cc/simulation/servers.h"

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "include/grpcpp/security/server_credentials.h"
#include "include/grpcpp/server.h"
#include "include/grpcpp/server_builder.h"
#include "tensorflow_federated/cc/core/impl/executor_stacks/local_stacks.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_service.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

constexpr int MegabytesToBytes(int megabytes) {
  return megabytes * 1024 * 1024;
}

namespace tensorflow_federated {

namespace tff = ::tensorflow_federated;

void RunServer(std::function<absl::StatusOr<std::shared_ptr<Executor>>(
                   const CardinalityMap&)>
                   executor_fn,
               int port, std::shared_ptr<grpc::ServerCredentials> credentials,
               int grpc_max_message_length_megabytes) {
  std::string server_address = absl::StrCat("[::]:", port);

  grpc::ServerBuilder server_builder;
  server_builder.AddListeningPort(server_address, credentials);

  std::unique_ptr<tff::ExecutorService> executor_service;
  executor_service = std::make_unique<tff::ExecutorService>(executor_fn);
  server_builder.RegisterService(executor_service.get());

  // These server builder methods take their arguments in bytes.
  int grpc_message_length_bytes =
      MegabytesToBytes(grpc_max_message_length_megabytes);

  server_builder.SetMaxReceiveMessageSize(grpc_message_length_bytes);
  server_builder.SetMaxSendMessageSize(grpc_message_length_bytes);
  server_builder.SetMaxMessageSize(grpc_message_length_bytes);

  std::unique_ptr<grpc::Server> server(server_builder.BuildAndStart());
  if (server == nullptr) {
    LOG(ERROR) << "TFF ExecutorService failed to start. Check the logs above "
                  "for information.";
    return;
  }
  LOG(INFO) << "TFF ExecutorService started, listening on " << server_address;
  server->Wait();
}

void RunWorker(int port, std::shared_ptr<grpc::ServerCredentials> credentials,
               int grpc_max_message_length_megabytes,
               absl::optional<uint32_t> max_concurrent_computation_calls) {
  auto create_tf_executor_fn =
      [max_concurrent_computation_calls](
          absl::optional<int> unused) -> std::shared_ptr<Executor> {
    return CreateTensorFlowExecutor(max_concurrent_computation_calls);
  };
  auto create_local_executor_fn =
      [create_tf_executor_fn](const CardinalityMap& cardinality_map)
      -> absl::StatusOr<std::shared_ptr<Executor>> {
    return CreateLocalExecutor(cardinality_map, create_tf_executor_fn);
  };
  RunServer(create_local_executor_fn, port, credentials,
            grpc_max_message_length_megabytes);
}

}  // namespace tensorflow_federated
