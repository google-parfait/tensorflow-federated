/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_RESOURCE_RESOLVER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_RESOURCE_RESOLVER_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"

namespace tensorflow_federated::aggregation {

// Describes an abstract interface for resolving a resource from a given client
// and uri.
class ResourceResolver {
 public:
  virtual ~ResourceResolver() = default;

  // Retrieves a resource for the given `client_id` and `uri` combination.
  // The resource can be accessed exactly once and must be deleted (best-effort)
  // after it is returned.
  virtual absl::StatusOr<absl::Cord> RetrieveResource(
      int64_t client_id, const std::string& uri) = 0;
};
}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_RESOURCE_RESOLVER_H_
