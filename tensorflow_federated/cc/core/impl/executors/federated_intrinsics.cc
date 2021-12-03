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

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorflow_federated {

absl::StatusOr<FederatedIntrinsic> FederatedIntrinsicFromUri(
    const absl::string_view uri) {
  if (uri == kFederatedMapAtClientsUri || uri == "federated_apply" ||
      uri == "federated_map_all_equal") {
    return FederatedIntrinsic::MAP;
  } else if (uri == kFederatedZipAtClientsUri) {
    return FederatedIntrinsic::ZIP_AT_CLIENTS;
  } else if (uri == kFederatedZipAtServerUri) {
    return FederatedIntrinsic::ZIP_AT_SERVER;
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
  } else if (uri == "federated_select") {
    return absl::UnimplementedError(
        "`federated_select` simulation is not yet supported in the TFF C++ "
        "runtime. For `federated_select` support, consider opting into the "
        "Python runtime using "
        "`tff.backends.native.set_local_python_execution_context()` (or "
        "`tff.google.backends.native.set_borg_execution_context(...)` for "
        "multi-machine uses).");
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported intrinsic URI: ", uri));
  }
}

}  // namespace tensorflow_federated
