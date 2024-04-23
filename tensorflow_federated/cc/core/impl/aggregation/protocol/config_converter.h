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
#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CONFIG_CONVERTER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CONFIG_CONVERTER_H_

#include <vector>

#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"

namespace tensorflow_federated {
namespace aggregation {

// Validates the Configuration.
// Returns INVALID_ARGUMENT if the configuration is invalid.
absl::Status ValidateConfiguration(const Configuration& configuration);

// Parses a Configuration proto into a vector of Intrinsic structs to
// represent the aggregation intrinsic independently from the proto.
StatusOr<std::vector<Intrinsic>> ParseFromConfig(const Configuration& config);

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CONFIG_CONVERTER_H_
