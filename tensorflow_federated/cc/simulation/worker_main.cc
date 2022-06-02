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

#include <stdint.h>

#include <memory>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "include/grpcpp/security/server_credentials.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/simulation/servers.h"

ABSL_FLAG(uint32_t, port, 10000, "Port to run the executor service on");
ABSL_FLAG(uint32_t, grpc_max_message_length_megabytes, 10000,
          "Max gRPC message length in megabytes");

ABSL_FLAG(uint32_t, max_concurrent_computation_calls, -1,
          "The maximum number of parallel calls to a given computation. This"
          " will limit the number of parallel executions for intrinsics such"
          " as `federated_map(computation, value)`. This can be especially "
          "helpful for users running into OOMs when using GPUs. Non-positive"
          " values result in no limiting.");

// TODO(b/234160632): Add option for secure server connections here.

namespace tff = ::tensorflow_federated;

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::shared_ptr<grpc::ServerCredentials> credentials =
      grpc::InsecureServerCredentials();
  tff::RunWorker(absl::GetFlag(FLAGS_port), credentials,
                 absl::GetFlag(FLAGS_grpc_max_message_length_megabytes),
                 absl::GetFlag(FLAGS_max_concurrent_computation_calls));
}
