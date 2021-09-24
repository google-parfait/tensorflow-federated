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

#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"

#include <utility>

#include "absl/status/statusor.h"

namespace tensorflow_federated {

absl::StatusOr<uint32_t> NumClientsFromCardinalities(
    const CardinalityMap& cardinalities) {
  auto entry_iter = cardinalities.find(kClientsUri);
  if (entry_iter == cardinalities.end()) {
    return absl::NotFoundError(
        "No cardinality provided for `clients` placement.");
  }
  return entry_iter->second;
}

}  // namespace tensorflow_federated
