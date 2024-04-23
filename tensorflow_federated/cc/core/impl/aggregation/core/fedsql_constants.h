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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_FEDSQL_CONSTANTS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_FEDSQL_CONSTANTS_H_

namespace tensorflow_federated {
namespace aggregation {

// Constants related to intrinsic definitions that are used in multiple files.
// Ideally these would be marked inline to ensure a single copy of each variable
// but this requires c++17 which is not available when building for bare metal.

// URI of GroupByAggregator
constexpr char kGroupByUri[] = "fedsql_group_by";

// URI prefix of inner intrinsics
constexpr char kFedSqlPrefix[] = "GoogleSQL:";

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_FEDSQL_CONSTANTS_H_
