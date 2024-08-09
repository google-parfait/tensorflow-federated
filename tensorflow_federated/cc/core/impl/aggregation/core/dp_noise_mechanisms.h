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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_NOISE_MECHANISMS_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_NOISE_MECHANISMS_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms.h"

// This file specifies DP noise mechanisms to be used in DPOpenDomainHistogram
// and DPClosedDomainHistogram. The functions build upon the
// differential_privacy::LaplaceMechanism and ::GaussianMechanism classes.
// They calculate replacement DP sensitivity from contribution bounds, which are
// expressed as l0, linfinity, l1, and l2 norm bounds.

namespace tensorflow_federated {
namespace aggregation {
namespace internal {

// Calculate L1 sensitivity from norm bounds, under replacement DP.
// A non-positive norm bound means it was not specified.
double CalculateL1Sensitivity(int64_t l0_bound, double linfinity_bound,
                              double l1_bound);

// Calculate L2 sensitivity from norm bounds, under replacement DP.
// A non-positive norm bound means it was not specified.
double CalculateL2Sensitivity(int64_t l0_bound, double linfinity_bound,
                              double l2_bound);

// Computes threshold needed when Laplace noise is used to ensure DP of open-
// domain histograms.
// Generalizes GaussianPartitionSelection::CalculateThresholdFromStddev, which
// assumes that linfinity_bound = 1 (we do not). The only role linfinity_bound
// plays is as an additive offset, so we simply shift the number it produces to
// compute the threshold.
absl::StatusOr<double> CalculateGaussianThreshold(
    double epsilon, double delta_for_noising, double delta_for_thresholding,
    int64_t l0_sensitivity, double linfinity_bound, double l2_sensitivity);

// Computes threshold needed when Laplace noise is used to ensure DP of open-
// domain histograms.
// Generalizes LaplacePartitionSelection from partition-selection.h, since it
// permits setting norm bounds beyond l0 (max_groups_contributed).
// l0_sensitivity and l1_sensitivity measure how much one user changes the l0
// and l1 norms, respectively, while linfinity_bound caps the magnitude of one
// user's contributions. This distinction is important for replacement DP.
absl::StatusOr<double> CalculateLaplaceThreshold(double epsilon, double delta,
                                                 int64_t l0_sensitivity,
                                                 double linfinity_bound,
                                                 double l1_sensitivity);
}  // namespace internal

// Because of substantial overlap in the logic for closed-domain and open-domain
// histogram algorithms, the following struct is used in both places.
struct DPHistogramBundle {
  // A pointer to a NumericalMechanism object which introduces noise for one
  // summation that satisfies replacement DP. The distribution will either be
  // Laplace or Gaussian, whichever has less variance.
  std::unique_ptr<differential_privacy::NumericalMechanism> mechanism = nullptr;

  // A threshold below which noisy sums will be erased. The thresholding step
  // consumes some or all of the delta that a customer provides. Only used in
  // the open-domain case.
  std::optional<double> threshold;

  // A boolean to indicate which noise is used.
  bool use_laplace = false;
};

// Given parameters for a DP aggregation, create a Gaussian mechanism for that
// aggregation (or return error status). If open_domain is true, then split
// delta and compute a post-aggregation threshold.
absl::StatusOr<DPHistogramBundle> CreateGaussianMechanism(
    double epsilon, double delta, int64_t l0_bound, double linfinity_bound,
    double l2_bound, bool open_domain);

// Given parameters for a DP aggregation, create a Laplace mechanism for that
// aggregation (or return error status).
absl::StatusOr<DPHistogramBundle> CreateLaplaceMechanism(
    double epsilon, double delta, int64_t l0_bound, double linfinity_bound,
    double l1_bound, bool open_domain);

// Given parameters for a DP aggregation, create a mechanism for it (or return
// an error status). The mechanism will be either Laplace or Gaussian, whichever
// has less variance for the same DP parameters.
// If it is not possible to make a mechanism, return an error status whose
// message includes the parameters of the aggregation and the provided index of
// the aggregation.
// If open_domain is true, then also compute a post-aggregation threshold.
//
// This function can be interpreted as an version of MinVarianceMechanismBuilder
// that takes L1 and L2 sensitivities.
absl::StatusOr<DPHistogramBundle> CreateDPHistogramBundle(
    double epsilon, double delta, int64_t l0_bound, double linfinity_bound,
    double l1_bound, double l2_bound, bool open_domain);

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_NOISE_MECHANISMS_H_
