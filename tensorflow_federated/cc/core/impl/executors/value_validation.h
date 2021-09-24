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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_VALUE_VALIDATION_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_VALUE_VALIDATION_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

// The kind of an incoming validated federated value.
enum class FederatedKind {
  SERVER,
  CLIENTS,
  CLIENTS_ALL_EQUAL,
};

// Ensures that preconditions on federated values necessary for memory safety
// in value accesses are upheld.
//
// In particular, checks that `all_equal` corresponds to 1, and that
// server-placed values are `all_equal`.
absl::StatusOr<FederatedKind> ValidateFederated(
    uint32_t num_clients, const v0::Value_Federated& federated);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_VALUE_VALIDATION_H_
