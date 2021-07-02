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

#include "tensorflow_federated/cc/core/impl/executors/value_validation.h"

#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/proto/v0/computation.proto.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

absl::StatusOr<FederatedKind> ValidateFederated(
    uint32_t num_clients, const v0::Value_Federated& federated) {
  const absl::string_view placement =
      federated.type().placement().value().uri();
  bool all_equal = federated.type().all_equal();
  if (all_equal) {
    if (federated.value_size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected all_equal federated value of type ",
          federated.type().DebugString(), " to have only one value, found ",
          federated.value().size(), " values."));
    }
  }
  if (placement == kServerUri) {
    if (!all_equal) {
      return absl::InvalidArgumentError(
          absl::StrCat("Server-placed values must be all-equal, found "
                       "non-all-equal server-placed type ",
                       federated.type().DebugString()));
    }
    return FederatedKind::SERVER;
  } else if (placement == kClientsUri) {
    if (all_equal) {
      return FederatedKind::CLIENTS_ALL_EQUAL;
    } else {
      if (federated.value_size() != num_clients) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected non-all_equal federated value of type ",
            federated.type().DebugString(),
            " to have a value for each client. The executor was configured "
            "for ",
            num_clients, " clients, but ", federated.value().size(),
            " values were provided."));
      }
      return FederatedKind::CLIENTS;
    }
  } else {
    return absl::UnknownError(absl::StrCat("Unknown placement ", placement));
  }
}

}  // namespace tensorflow_federated
