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
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_noise_mechanisms.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/partition-selection.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"

namespace tensorflow_federated {
namespace aggregation {
namespace internal {
using differential_privacy::GaussianPartitionSelection;
using differential_privacy::SafeAdd;

constexpr double kMaxSensitivity = std::numeric_limits<double>::infinity();

// Calculate L1 sensitivity from norm bounds, under replacement DP.
// A non-positive norm bound means it was not specified.
double CalculateL1Sensitivity(int64_t l0_bound, double linfinity_bound,
                              double l1_bound) {
  double l1_sensitivity = kMaxSensitivity;
  if (l0_bound > 0 && linfinity_bound > 0) {
    l1_sensitivity = fmin(l1_sensitivity, 2.0 * l0_bound * linfinity_bound);
  }
  if (l1_bound > 0) {
    l1_sensitivity = fmin(l1_sensitivity, 2.0 * l1_bound);
  }
  return l1_sensitivity;
}

// Calculate L2 sensitivity from norm bounds, under replacement DP.
// A non-positive norm bound means it was not specified.
double CalculateL2Sensitivity(int64_t l0_bound, double linfinity_bound,
                              double l2_bound) {
  double l2_sensitivity = kMaxSensitivity;
  if (l0_bound > 0 && linfinity_bound > 0) {
    l2_sensitivity =
        fmin(l2_sensitivity, sqrt(2.0 * l0_bound) * linfinity_bound);
  }
  if (l2_bound > 0) {
    l2_sensitivity = fmin(l2_sensitivity, 2.0 * l2_bound);
  }
  return l2_sensitivity;
}

// Computes threshold needed when Gaussian noise is used to ensure DP of open-
// domain histograms.
absl::StatusOr<double> CalculateGaussianThreshold(
    double epsilon, double delta_for_noising, double delta_for_thresholding,
    int64_t l0_sensitivity, double linfinity_bound, double l2_sensitivity) {
  TFF_CHECK(epsilon > 0 && delta_for_noising > 0 &&
            delta_for_thresholding > 0 && l0_sensitivity > 0 &&
            linfinity_bound > 0 && l2_sensitivity > 0)
      << "CalculateGaussianThreshold: All inputs must be positive";
  TFF_CHECK(delta_for_noising < 1) << "CalculateGaussianThreshold: "
                                   << "delta_for_noising must be less than 1.";
  TFF_CHECK(delta_for_thresholding < 1)
      << "CalculateGaussianThreshold: "
      << "delta_for_thresholding must be less than 1.";

  double stdev = differential_privacy::GaussianMechanism::CalculateStddev(
      epsilon, delta_for_noising, l2_sensitivity);
  TFF_ASSIGN_OR_RETURN(double library_threshold,
                       GaussianPartitionSelection::CalculateThresholdFromStddev(
                           stdev, delta_for_thresholding, l0_sensitivity));

  return SafeAdd(linfinity_bound - 1, library_threshold).value;
}

// Computes threshold needed when Laplace noise is used to ensure DP of open-
// domain histograms.
absl::StatusOr<double> CalculateLaplaceThreshold(double epsilon, double delta,
                                                 int64_t l0_sensitivity,
                                                 double linfinity_bound,
                                                 double l1_sensitivity) {
  TFF_CHECK(epsilon > 0 && delta > 0 && l0_sensitivity > 0 &&
            linfinity_bound > 0 && l1_sensitivity > 0)
      << "CalculateLaplaceThreshold: All inputs must be positive";
  TFF_CHECK(delta < 1) << "CalculateLaplaceThreshold: delta must be less "
                       << "than 1";

  // If probability of failing to drop a small value is
  // 1- pow(1 - delta, 1 / l0_sensitivity)
  // then the overall privacy failure probability is delta
  // Below: numerically stable version of 1- pow(1 - delta, 1 / l0_sensitivity)
  // Adapted from PartitionSelectionStrategy::CalculateAdjustedDelta.
  double adjusted_delta = -std::expm1(log1p(-delta) / l0_sensitivity);

  double laplace_tail_bound;
  if (adjusted_delta > 0.5) {
    laplace_tail_bound =
        (l1_sensitivity / epsilon) * std::log(2 * (1 - adjusted_delta));
  } else {
    laplace_tail_bound =
        -(l1_sensitivity / epsilon) * (std::log(2 * adjusted_delta));
  }

  return linfinity_bound + laplace_tail_bound;
}
}  // namespace internal

// Given parameters for an DP aggregation, create a Gaussian mechanism for that
// aggregation (or return error status). If open_domain is true, then split
// delta and compute a post-aggregation threshold.
absl::StatusOr<DPHistogramBundle> CreateGaussianMechanism(
    double epsilon, double delta, int64_t l0_bound, double linfinity_bound,
    double l2_bound, bool open_domain) {
  if (epsilon <= 0 || epsilon >= kEpsilonThreshold) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "CreateGaussianMechanism: Epsilon must be positive "
              "and smaller than "
           << kEpsilonThreshold;
  }
  if (delta <= 0 || delta >= 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "CreateGaussianMechanism: Delta must lie within (0, 1).";
  }

  // The following parameter determines how much of delta is consumed for
  // thresholding (when open_domain is true). Currently set to 0.5, but this
  // could be optimized down the line.
  double fractionForThresholding = open_domain ? 0.5 : 0.0;
  double delta_for_thresholding = delta * fractionForThresholding;
  double delta_for_noising = delta - delta_for_thresholding;

  double l2_sensitivity =
      internal::CalculateL2Sensitivity(l0_bound, linfinity_bound, l2_bound);

  differential_privacy::GaussianMechanism::Builder gaussian_builder;
  gaussian_builder.SetL2Sensitivity(l2_sensitivity)
      .SetEpsilon(epsilon)
      .SetDelta(delta_for_noising);

  DPHistogramBundle dp_histogram;
  TFF_ASSIGN_OR_RETURN(dp_histogram.mechanism, gaussian_builder.Build());
  dp_histogram.use_laplace = false;

  if (open_domain) {
    if (l0_bound <= 0 || linfinity_bound <= 0) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "CreateGaussianMechanism: Open-domain DP "
                "histogram algorithm requires valid l0_bound "
                "and linfinity_bound.";
    }

    // Calculate the threshold which we will impose on noisy sums.
    // Note that l0_sensitivity = 2 * l0_bound because we target replacement DP.
    TFF_ASSIGN_OR_RETURN(
        dp_histogram.threshold,
        internal::CalculateGaussianThreshold(
            epsilon, delta_for_noising, delta_for_thresholding,
            /*l0_sensitivity=*/2 * l0_bound, linfinity_bound, l2_sensitivity));
  }
  return dp_histogram;
}

// Given parameters for an DP aggregation, create a Laplace mechanism for that
// aggregation (or return error status).
absl::StatusOr<DPHistogramBundle> CreateLaplaceMechanism(
    double epsilon, double delta, int64_t l0_bound, double linfinity_bound,
    double l1_bound, bool open_domain) {
  if (epsilon <= 0 || epsilon >= kEpsilonThreshold) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "CreateLaplaceMechanism: Epsilon must be positive "
              "and smaller than "
           << kEpsilonThreshold;
  }

  double l1_sensitivity =
      internal::CalculateL1Sensitivity(l0_bound, linfinity_bound, l1_bound);

  differential_privacy::LaplaceMechanism::Builder laplace_builder;
  laplace_builder.SetL1Sensitivity(l1_sensitivity).SetEpsilon(epsilon);

  DPHistogramBundle dp_histogram;
  TFF_ASSIGN_OR_RETURN(dp_histogram.mechanism, laplace_builder.Build());
  dp_histogram.use_laplace = true;

  if (open_domain) {
    if (delta <= 0 || delta >= 1 || l0_bound <= 0 || linfinity_bound <= 0) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "CreateLaplaceMechanism: Open-domain DP "
                "histogram algorithm requires valid delta, "
                "l0_bound, and linfinity_bound.";
    }

    // Calculate the threshold which we will impose on noisy sums.
    // Note that l0_sensitivity = 2 * l0_bound because we target replacement DP.
    TFF_ASSIGN_OR_RETURN(
        dp_histogram.threshold,
        internal::CalculateLaplaceThreshold(epsilon, delta, 2 * l0_bound,
                                            linfinity_bound, l1_sensitivity));
  }

  return dp_histogram;
}

// Given parameters for an DP histogram aggregation, create a mechanism for that
// aggregation (or return error status). The mechanism will be either Laplace
// or Gaussian, whichever has less variance for the same DP parameters.
absl::StatusOr<DPHistogramBundle> CreateDPHistogramBundle(
    double epsilon, double delta, int64_t l0_bound, double linfinity_bound,
    double l1_bound, double l2_bound, bool open_domain) {
  // First we determine if we are able to make Gaussian or Laplace mechanisms
  // from the given parameters.
  double l1_sensitivity =
      internal::CalculateL1Sensitivity(l0_bound, linfinity_bound, l1_bound);
  double l2_sensitivity =
      internal::CalculateL2Sensitivity(l0_bound, linfinity_bound, l2_bound);
  bool laplace_is_possible = (epsilon > 0 && epsilon < kEpsilonThreshold &&
                              l1_sensitivity != internal::kMaxSensitivity);
  bool gaussian_is_possible =
      (epsilon > 0 && epsilon < kEpsilonThreshold && delta > 0 && delta < 1 &&
       l2_sensitivity != internal::kMaxSensitivity);

  if (!laplace_is_possible && !gaussian_is_possible) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "CreateDPHistogramBundle: Unable to make either a Laplace or a"
              " Gaussian DP mechanism. Relevant parameters:"
           << "\n l0_bound: " << l0_bound
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
