/*
 * Copyright 2026 Google LLC
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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_FEDSQL_UTIL_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_FEDSQL_UTIL_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"

namespace tensorflow_federated {
namespace aggregation {

// Updates the given intrinsic with the given DP parameters.
absl::Status PopulateDPParameters(
    Intrinsic& intrinsic,
    const absl::flat_hash_map<std::string, double>& parameters);

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_FEDSQL_UTIL_H_
