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

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/partition-selection.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"

// This file specifies DP noise mechanisms to be used in DPOpenDomainHistogram
// and DPClosedDomainHistogram. The functions build upon the
// differential_privacy::LaplaceMechanism and ::GaussianMechanism classes.
// They calculate replacement DP sensitivity from contribution bounds, which are
// expressed as l0, linfinity, l1, and l2 norm bounds.

namespace tensorflow_federated {
namespace aggregation {
using differential_privacy::GaussianPartitionSelection;
using differential_privacy::NumericalMechanism;

// Because of substantial overlap in the logic for closed-domain and open-domain
// histogram algorithms, the following struct is used in both places.
template <typename OutputType>
struct DPHistogramBundle {
  // A pointer to a NumericalMechanism object which introduces noise for one
  // summation that satisfies replacement DP. The distribution will either be
  // Laplace or Gaussian, whichever has less variance.
  std::unique_ptr<NumericalMechanism> mechanism = nullptr;

  // A threshold below which noisy sums will be erased. The thresholding step
  // consumes some or all of the delta that a customer provides. Only used in
  // the open-domain case.
  OutputType threshold;

  // A boolean to indicate which noise is used.
  bool use_laplace;
};

// Calculate L1 sensitivity from norm bounds, under replacement DP.
// A non-positive norm bound means it was not specified.
double CalculateL1Sensitivity(int64_t l0_bound, double linfinity_bound,
                              double l1_bound);

// Calculate L2 sensitivity from norm bounds, under replacement DP.
// A non-positive norm bound means it was not specified.
double CalculateL2Sensitivity(int64_t l0_bound, double linfinity_bound,
                              double l2_bound);

// Given parameters for an DP aggregation, create a Gaussian mechanism for that
// aggregation (or return error status). If open_domain is true, then split
// delta and compute a post-aggregation threshold.
template <typename OutputType>
absl::StatusOr<DPHistogramBundle<OutputType>> CreateGaussianMechanism(
    double epsilon, double delta, int64_t l0_bound, OutputType linfinity_bound,
    double l2_bound, bool open_domain) {
  // The following parameter determines how much of delta is consumed for
  // thresholding (when open_domain is true). Currently set to 0.5, but this
  // could be optimized down the line.
  double fractionForThresholding = open_domain ? 0.5 : 0.0;
  double delta_for_thresholding = delta * fractionForThresholding;
  double delta_for_noising = delta - delta_for_thresholding;

  double l2_sensitivity =
      CalculateL2Sensitivity(l0_bound, linfinity_bound, l2_bound);

  differential_privacy::GaussianMechanism::Builder gaussian_builder;
  gaussian_builder.SetL2Sensitivity(l2_sensitivity)
      .SetEpsilon(epsilon)
      .SetDelta(delta_for_noising);

  DPHistogramBundle<OutputType> dp_histogram;
  TFF_ASSIGN_OR_RETURN(dp_histogram.mechanism, gaussian_builder.Build());
  dp_histogram.use_laplace = false;

  if (open_domain) {
    if (delta <= 0 || l0_bound <= 0 || linfinity_bound <= 0) {
      TFF_STATUS(INVALID_ARGUMENT) << "CreateGaussianMechanism: Open-domain DP "
                                      "histogram algorithm requires delta, "
                                      "l0_bound, and linfinity_bound.";
    }

    // Calculate the threshold which we will impose on noisy sums. We use
    // GaussianPartitionSelection::CalculateThresholdFromStddev. It assumes that
    // linfinity_bound = 1 but the only role linfinity_bound plays is as an
    // additive offset. So we can simply shift the number it produces to compute
    // the threshold.
    double stdev = differential_privacy::GaussianMechanism::CalculateStddev(
        epsilon, delta_for_noising, l2_sensitivity);
    TFF_ASSIGN_OR_RETURN(
        double library_threshold,
        GaussianPartitionSelection::CalculateThresholdFromStddev(
            stdev, delta_for_thresholding, 2 * l0_bound));
    // Use ceil to err on the side of caution:
    // if noisy_val is an integer less than (double) library_threshold,
    // a cast of library_threshold may make them appear equal
    if (std::is_integral<OutputType>::value) {
      library_threshold = ceil(library_threshold);
    }

    // GaussianPartitionSelection::CalculateThresholdFromStddev assumes that
    // linfinity_bound = 1 but the only role linfinity_bound plays is as an
    // additive offset. So we can simply shift the number it produces to compute
    // the threshold.
    dp_histogram.threshold =
        differential_privacy::SafeAdd<OutputType>(
            linfinity_bound - 1, static_cast<OutputType>(library_threshold))
            .value;
  }
  return std::move(dp_histogram);
}

// Computes threshold needed when Laplace noise is used to ensure DP.
// Generalizes LaplacePartitionSelection from partition-selection.h, since it
// permits setting norm bounds beyond l0 (max_groups_contributed).
// l0_sensitivity and l1_sensitivity measure how much one user changes the l0
// and l1 norms, respectively, while linfinity_bound caps the magnitude of one
// user's contributions. This distinction is important for replacement DP.
template <typename OutputType>
absl::StatusOr<OutputType> CalculateLaplaceThreshold(double epsilon,
                                                     double delta,
                                                     int64_t l0_sensitivity,
                                                     OutputType linfinity_bound,
                                                     double l1_sensitivity) {
  TFF_CHECK(epsilon > 0 && delta > 0 && l0_sensitivity > 0 &&
            linfinity_bound > 0 && l1_sensitivity > 0)
      << "CalculateThreshold: All inputs must be positive";
  TFF_CHECK(delta < 1) << "CalculateThreshold: delta must be less than 1";

  // If probability of failing to drop a small value is
  // 1- pow(1 - delta, 1 / l0_sensitivity)
  // then the overall privacy failure probability is delta
  // Below: numerically stable version of 1- pow(1 - delta, 1 / l0_sensitivity)
  // Adapted from PartitionSelectionStrategy::CalculateAdjustedDelta.
  double adjusted_delta = -std::expm1(log1p(-delta) / l0_sensitivity);

  OutputType laplace_tail_bound;
  if (adjusted_delta > 0.5) {
    laplace_tail_bound = static_cast<OutputType>(
        (l1_sensitivity / epsilon) * std::log(2 * (1 - adjusted_delta)));
  } else {
    laplace_tail_bound = static_cast<OutputType>(
        -(l1_sensitivity / epsilon) * (std::log(2 * adjusted_delta)));
  }

  return linfinity_bound + laplace_tail_bound;
}

// Given parameters for an DP aggregation, create a Laplace mechanism for that
// aggregation (or return error status).
template <typename OutputType>
absl::StatusOr<DPHistogramBundle<OutputType>> CreateLaplaceMechanism(
    double epsilon, double delta, int64_t l0_bound, OutputType linfinity_bound,
    double l1_bound, bool open_domain) {
  double l1_sensitivity =
      CalculateL1Sensitivity(l0_bound, linfinity_bound, l1_bound);

  differential_privacy::LaplaceMechanism::Builder laplace_builder;
  laplace_builder.SetL1Sensitivity(l1_sensitivity).SetEpsilon(epsilon);

  DPHistogramBundle<OutputType> dp_histogram;
  TFF_ASSIGN_OR_RETURN(dp_histogram.mechanism, laplace_builder.Build());
  dp_histogram.use_laplace = true;

  if (open_domain) {
    if (delta <= 0 || l0_bound <= 0 || linfinity_bound <= 0) {
      TFF_STATUS(INVALID_ARGUMENT) << "CreateLaplaceMechanism: Open-domain DP "
                                      "histogram algorithm requires delta, "
                                      "l0_bound, and linfinity_bound.";
    }

    // 3. Calculate the threshold which we will impose on noisy sums.
    // Note that l0_sensitivity = 2 * l0_bound because we target replacement DP.
    TFF_ASSIGN_OR_RETURN(
        double library_threshold,
        CalculateLaplaceThreshold<OutputType>(epsilon, delta, 2 * l0_bound,
                                              linfinity_bound, l1_sensitivity));
    // Use ceil to err on the side of caution:
    // if noisy_val is an integer less than (double) library_threshold,
    // a cast of library_threshold may make them appear equal
    if (std::is_integral<OutputType>::value) {
      library_threshold = ceil(library_threshold);
    }
    dp_histogram.threshold = static_cast<OutputType>(library_threshold);
  }

  return std::move(dp_histogram);
}

// Given parameters for an DP histogram aggregation, create a mechanism for that
// aggregation (or return error status). The mechanism will be either Laplace
// or Gaussian, whichever has less variance for the same DP parameters.
// If it is not possible to make a mechanism, return an error status whose
// message includes the parameters of the aggregation and the provided index of
// the aggregation.
// If open_domain is true, then also compute a post-aggregation threshold.
//
// This function can be interpreted as an version of MinVarianceMechanismBuilder
// that takes L1 and L2 sensitivities.
template <typename OutputType>
absl::StatusOr<DPHistogramBundle<OutputType>> CreateDPHistogramBundle(
    int64_t agg_index, double epsilon, double delta, int64_t l0_bound,
    OutputType linfinity_bound, double l1_bound, double l2_bound,
    bool open_domain) {
  // First we determine if we are able to make Gaussian or Laplace mechanisms
  // from the given parameters.
  double l1_sensitivity =
      CalculateL1Sensitivity(l0_bound, linfinity_bound, l1_bound);
  double l2_sensitivity =
      CalculateL2Sensitivity(l0_bound, linfinity_bound, l2_bound);
  bool laplace_is_possible =
      (epsilon > 0 && epsilon < kEpsilonThreshold &&
       l1_sensitivity != std::numeric_limits<double>::infinity());
  bool gaussian_is_possible =
      (epsilon > 0 && epsilon < kEpsilonThreshold && delta > 0 && delta < 1 &&
       l2_sensitivity != std::numeric_limits<double>::infinity());

  if (!laplace_is_possible && !gaussian_is_possible) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "CreateDPHistogramBundle: Unable to make either a Laplace or a"
              " Gaussian DP mechanism for aggregation "
           << agg_index << ". Relevant parameters:\n l0_bound: " << l0_bound
           << "\n linfinity_bound: " << linfinity_bound
           << "\n l1_bound: " << l1_bound << "\n l2_bound: " << l2_bound
           << "\n epsilon: " << epsilon << "\n delta: " << delta;
  }

  // When only one mechanism can be made, make it.
  if (!laplace_is_possible && gaussian_is_possible) {
    return CreateGaussianMechanism(epsilon, delta, l0_bound, linfinity_bound,
                                   l2_bound, open_domain);
  }
  if (laplace_is_possible && !gaussian_is_possible) {
    return CreateLaplaceMechanism(epsilon, delta, l0_bound, linfinity_bound,
                                  l1_bound, open_domain);
  }

  // When both mechanisms can be made, use the one with smaller variance.
  // This is a simple heuristic that will minimize average error across the
  // domain of composite keys. An alternative would be to minimize the
  // maximum error using tail bounds.

  TFF_ASSIGN_OR_RETURN(
      auto laplace_mechanism,
      CreateLaplaceMechanism(epsilon, delta, l0_bound, linfinity_bound,
                             l1_bound, open_domain));
  TFF_ASSIGN_OR_RETURN(
      auto gaussian_mechanism,
      CreateGaussianMechanism(epsilon, delta, l0_bound, linfinity_bound,
                              l2_bound, open_domain));

  if (gaussian_mechanism.mechanism->GetVariance() <
      laplace_mechanism.mechanism->GetVariance()) {
    return gaussian_mechanism;
  }
  return laplace_mechanism;
}

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_NOISE_MECHANISMS_H_
