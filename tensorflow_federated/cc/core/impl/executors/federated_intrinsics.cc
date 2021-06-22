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

#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"

#include "absl/status/statusor.h"

namespace tensorflow_federated {

absl::StatusOr<FederatedIntrinsic> FederatedIntrinsicFromUri(
    const absl::string_view uri) {
  if (uri == kFederatedMapAtClientsUri || uri == "federated_apply") {
    return FederatedIntrinsic::MAP;
  } else if (uri == kFederatedZipAtClientsUri ||
             uri == "federated_zip_at_server") {
    return FederatedIntrinsic::ZIP;
  } else if (uri == "federated_broadcast") {
    return FederatedIntrinsic::BROADCAST;
  } else if (uri == "federated_value_at_clients") {
    return FederatedIntrinsic::VALUE_AT_CLIENTS;
  } else if (uri == "federated_value_at_server") {
    return FederatedIntrinsic::VALUE_AT_SERVER;
  } else if (uri == kFederatedAggregateUri) {
    return FederatedIntrinsic::AGGREGATE;
  } else if (uri == kFederatedEvalAtClientsUri) {
    return FederatedIntrinsic::EVAL_AT_CLIENTS;
  } else if (uri == "federated_eval_at_server") {
    return FederatedIntrinsic::EVAL_AT_SERVER;
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported intrinsic URI: ", uri));
  }
}

}  // namespace tensorflow_federated
