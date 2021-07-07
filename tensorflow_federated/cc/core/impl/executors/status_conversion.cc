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

#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"

#include "absl/status/status.h"

namespace tensorflow_federated {

absl::Status grpc_to_absl(grpc::Status status) {
  if (status.ok()) {
    return absl::OkStatus();
  }
  return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                      status.error_message());
}

grpc::Status absl_to_grpc(absl::Status status) {
  if (status.ok()) {
    return grpc::Status::OK;
  }
  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      std::string(status.message()));
}

}  // namespace tensorflow_federated
