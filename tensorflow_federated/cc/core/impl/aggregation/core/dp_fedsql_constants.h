/*
 * Copyright 2024 Google LLC
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_FEDSQL_CONSTANTS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_FEDSQL_CONSTANTS_H_

namespace tensorflow_federated {
namespace aggregation {

// Constants related to intrinsic definitions that are used in DP-related files.
// Ideally these would be marked inline to ensure a single copy of each variable
// but this requires c++17 which is not available when building for bare metal.

// URI of DPGroupByAggregator
constexpr char kDPGroupByUri[] = "fedsql_dp_group_by";

// URI of DPGroupingFederatedSum
constexpr char kDPSumUri[] = "GoogleSQL:$differential_privacy_sum";

// URI of DPQuantileAggregator
constexpr char kDPQuantileUri[] =
    "GoogleSQL:$differential_privacy_percentile_cont";
// URI of a bundle of DPTensorAggregators
constexpr char kDPTensorAggregatorBundleUri[] =
    "differential_privacy_tensor_aggregator_bundle";
// Names of parameters in the DP intrinsics.
constexpr char kEpsilonName[] = "epsilon";
constexpr char kDeltaName[] = "delta";
constexpr char kMaxGroupsContributedName[] = "max_groups_contributed";
constexpr char kLinfName[] = "L_inf";
constexpr char kL1Name[] = "L_1";
constexpr char kL2Name[] = "L_2";

// In the DPGroupingFederatedSum intrinsic, the following are the indices of
// the norm bound parameters.
constexpr int kLinfinityIndex = 0;
constexpr int kL1Index = 1;
constexpr int kL2Index = 2;
constexpr int kNumDPSumParameters = 3;

// In the intrinsics that represent DP queries, the following are the indices of
// the epsilon and delta parameters.
constexpr int kEpsilonIndex = 0;
constexpr int kDeltaIndex = 1;

// The epsilon beyond which we will not use DP noise
constexpr double kEpsilonThreshold = 1e20;

// The maximum number of inputs that can be stored in the DPQuantileAggregator.
constexpr int kDPQuantileMaxInputs = 50000;

// The maximum magnitude of the output of the DPQuantileAggregator.
constexpr double kDPQuantileMaxOutputMagnitude = 1e20;

// Parameters that govern how DPQuantileAggregator buckets the inputs.
// First buckets are linear e.g. 0.0, 0.1, 0.2, ..., 100.0
// Then buckets are exponential e.g. 100 * 1.01^k.
constexpr double kDPQuantileLinearRate = 0.1;
constexpr double kDPQuantileExponentialRate = 1.01;
constexpr double kDPQuantileEndOfLinearGrowth = 100;

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_FEDSQL_CONSTANTS_H_
