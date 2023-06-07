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

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "include/grpcpp/security/server_credentials.h"
#include "include/grpcpp/server.h"
#include "include/grpcpp/server_builder.h"
#include "tensorflow_federated/cc/core/impl/executor_stacks/local_stacks.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_api.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_executor.h"
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

absl::StatusOr<
    std::function<absl::StatusOr<std::shared_ptr<Executor>>(int32_t)>>
CreateDTensorExecutorFn(TFE_Context* context, std::string serialized_mesh,
                        int32_t max_concurrent_computation_calls,
                        std::string_view dtensor_device_name) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  auto mesh = tensorflow::dtensor::Mesh::FromString(serialized_mesh);
  if (!mesh.ok()) {
    return tensorflow::ToAbslStatus(mesh.status());
  }

  TFE_DTENSOR_RegisterDTensorDevice(context, tensorflow::wrap(&mesh.value()),
                                    dtensor_device_name.data(), status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    return absl::InternalError(absl::StrCat(
        "Registering DTensor device failed ", TF_Message(status.get())));
  }
  auto create_dtensor_executor_fn =
      [context, max_concurrent_computation_calls, mesh,
       dtensor_device_name](int32_t unused) -> std::shared_ptr<Executor> {
    return CreateDTensorExecutor(
        context, std::string(dtensor_device_name), mesh.value(),
        /*dtensor_converter=*/nullptr, max_concurrent_computation_calls);
  };
  return create_dtensor_executor_fn;
}

void RunWorker(int port, std::shared_ptr<grpc::ServerCredentials> credentials,
               int grpc_max_message_length_megabytes,
               int32_t max_concurrent_computation_calls,
               std::string serialized_server_mesh,
               std::string serialized_client_mesh) {
  std::function<absl::StatusOr<std::shared_ptr<Executor>>(int32_t)>
      create_server_executor_fn = nullptr;
  std::function<absl::StatusOr<std::shared_ptr<Executor>>(int32_t)>
      create_client_executor_fn = nullptr;

  auto create_tf_executor_fn =
      [max_concurrent_computation_calls](
          std::optional<int> unused) -> std::shared_ptr<Executor> {
    return CreateTensorFlowExecutor(max_concurrent_computation_calls);
  };

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  // When enabling multi-client mesh configuration in future we need to add
  // the following
  // # Set server def
  // # enable collective ops

  // We can use TFE_HostAddressSpace to extract prefix for dtensor device in
  // future.
  // Since workers are mainly using local devices with common runtime, use
  // the default prefix "/job:localhost/replica:0/task:0/device:".

  // This assumes that no other custom devices with the name "CUSTOM:1" or
  // "CUSTOM:2" are created by this worker within the new TF context created.
  if (!serialized_server_mesh.empty()) {
    auto create_server_executor_fn_or = CreateDTensorExecutorFn(
        context.get(), serialized_server_mesh, max_concurrent_computation_calls,
        "/job:localhost/replica:0/task:0/device:CUSTOM:1");
    assert(create_server_executor_fn_or.status().ok());
    create_server_executor_fn = create_server_executor_fn_or.value();
  } else {
    create_server_executor_fn = create_tf_executor_fn;
  }

  if (!serialized_client_mesh.empty()) {
    auto create_client_executor_fn = CreateDTensorExecutorFn(
        context.get(), serialized_client_mesh, max_concurrent_computation_calls,
        "/job:localhost/replica:0/task:0/device:CUSTOM:2");
    assert(create_client_executor_fn.status().ok());
    create_client_executor_fn = create_client_executor_fn.value();
  } else {
    create_client_executor_fn = create_tf_executor_fn;
  }

  auto create_local_executor_fn =
      [create_server_executor_fn,
       create_client_executor_fn](const CardinalityMap& cardinality_map)
      -> absl::StatusOr<std::shared_ptr<Executor>> {
    return CreateLocalExecutor(cardinality_map, create_server_executor_fn,
                               create_client_executor_fn);
  };

  RunServer(create_local_executor_fn, port, credentials,
            grpc_max_message_length_megabytes);
}

}  // namespace tensorflow_federated
