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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_CARDINALITIES_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_CARDINALITIES_H_

#include <cstdint>
#include <string>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorflow_federated {

// `btree_map` is used rather than `flat_hash_map` to provide ordering.
//
// This ensures that simple string-joining of the entries in this map will
// produce a specific value for a given set of cardinalities independent of
// ordering. the `ExecutorService` uses this as an optimization to provide
// per-cardinality `ExecutorId`s.
using CardinalityMap = absl::btree_map<std::string, int>;
const absl::string_view kClientsUri = "clients";
const absl::string_view kServerUri = "server";

// Returns the number of clients specifed by the provided `cardinalities`.
absl::StatusOr<int> NumClientsFromCardinalities(
    const CardinalityMap& cardinalities);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_CARDINALITIES_H_
