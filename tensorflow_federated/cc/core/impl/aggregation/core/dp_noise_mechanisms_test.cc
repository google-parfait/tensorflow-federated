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

#include <cstdint>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::DoubleEq;
using ::testing::HasSubstr;
using ::testing::Ne;
constexpr double kSmallEpsilon = 0.01;

TEST(DPNoiseMechanismsTest, CreateLaplaceMechanismMissingEpsilon) {
  // The function needs epsilon.
  auto missing_epsilon = CreateLaplaceMechanism(-1, 1e-8, 10, 10, 10, false);
  EXPECT_THAT(missing_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_epsilon.status().message(),
              HasSubstr("Epsilon must be positive"));
}

TEST(DPNoiseMechanismsTest, CreateLaplaceMechanismMissingL1) {
  // The function returns an error if it is unable to bound the L1 sensitivity.
  auto missing_l1 = CreateLaplaceMechanism(1.0, 1e-8, -1, -1, -1, false);
  EXPECT_THAT(missing_l1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_l1.status().message(),
              HasSubstr("must be finite and positive"));
}

TEST(DPNoiseMechanismsTest, CreateLaplaceMechanismMissingParamsForOpenDomain) {
  // If the goal is to support open-domain histograms but there is no valid
  // delta, CreateLaplaceMechanism returns an error status. Same is true if we
  // are missing one of L0 and Linf.
  std::string kErrorMsg =
      "CreateLaplaceMechanism: Open-domain DP "
      "histogram algorithm requires valid delta, "
      "l0_bound, and linfinity_bound.";

  auto missing_delta = CreateLaplaceMechanism(1.0, -1, 10, 10, 10, true);
  EXPECT_THAT(missing_delta, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_delta.status().message(), HasSubstr(kErrorMsg));

  auto missing_L0 = CreateLaplaceMechanism(1.0, 1e-8, -1, 10, 10, true);
  EXPECT_THAT(missing_L0, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_L0.status().message(), HasSubstr(kErrorMsg));

  auto missing_Linf = CreateLaplaceMechanism(1.0, 1e-8, 10, -1, 10, true);
  EXPECT_THAT(missing_Linf, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_Linf.status().message(), HasSubstr(kErrorMsg));
}

TEST(DPNoiseMechanismsTest, CreateGaussianMechanismMissingEpsilon) {
  // The function needs epsilon and delta.
  auto missing_epsilon = CreateGaussianMechanism(-1, 1e-8, 10, 10, 10, false);
  EXPECT_THAT(missing_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_epsilon.status().message(),
              HasSubstr("Epsilon must be positive"));
}

TEST(DPNoiseMechanismsTest, CreateGaussianMechanismMissingDelta) {
  auto missing_delta = CreateGaussianMechanism(1.0, -1, 10, 10, 10, false);
  EXPECT_THAT(missing_delta, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_delta.status().message(),
              HasSubstr("Delta must lie within (0, 1)"));
}

TEST(DPNoiseMechanismsTest, CreateGaussianMechanismMissingL2) {
  // The function returns an error if it is unable to bound the L2 sensitivity.
  auto missing_l2 = CreateGaussianMechanism(1.0, 1e-8, -1, -1, -1, false);
  EXPECT_THAT(missing_l2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_l2.status().message(),
              HasSubstr("must be finite and positive"));
}

TEST(DPNoiseMechanismsTest, CreateGaussianMechanismMissingParamsForOpenDomain) {
  // If the goal is to support open-domain histograms but there is no valid
  // l0_bound or linfinity_bound, CreateGaussianMechanism returns an error
  // status.
  std::string kErrorMsg =
      "CreateGaussianMechanism: Open-domain DP "
      "histogram algorithm requires valid l0_bound "
      "and linfinity_bound.";

  auto missing_L0 = CreateGaussianMechanism(1.0, 1e-8, -1, 10, 10, true);
  EXPECT_THAT(missing_L0, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_L0.status().message(), HasSubstr(kErrorMsg));

  auto missing_Linf = CreateGaussianMechanism(1.0, 1e-8, 10, -1, 10, true);
  EXPECT_THAT(missing_Linf, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_Linf.status().message(), HasSubstr(kErrorMsg));
}

// If neither L1 nor L2 sensitivity can be computed, CreateDPHistogramBundle
// returns a bad status. Same is true if epsilon is invalid (< 0 or too big).
TEST(DPNoiseMechanismsTest, CreateDPHistogramBundleCatchesBadParameters) {
  std::string kErrorMsg =
      "CreateDPHistogramBundle: Unable to make either a "
      "Laplace or a Gaussian DP mechanism";
  // No norm bounds -> no sensitivity bound
  auto missing_bounds =
      CreateDPHistogramBundle(1.0, 1e-8, -1, -1, -1, -1, false);
  EXPECT_THAT(missing_bounds, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(missing_bounds.status().message(), HasSubstr(kErrorMsg));

  // Negative epsilon
  auto negative_epsilon =
      CreateDPHistogramBundle(-1, 1e-8, 10, 10, 10, 10, false);
  EXPECT_THAT(negative_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(negative_epsilon.status().message(), HasSubstr(kErrorMsg));

  // Too large epsilon
  auto too_large_epsilon =
      CreateDPHistogramBundle(kEpsilonThreshold, 1e-8, 10, 10, 10, 10, false);
  EXPECT_THAT(too_large_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(too_large_epsilon.status().message(), HasSubstr(kErrorMsg));
}

// If sensitivities and epsilon are valid, check that the function correctly
// switches between distributions.

// If only the L1 norm bound is given, Laplace should be used.
TEST(DPNoiseMechanismsTest, CreateDPHistogramBundleUsesLaplaceOnlyL1) {
  auto laplace_mechanism = CreateDPHistogramBundle(/*epsilon=*/1.0,
                                                   /*delta=*/1e-8,
                                                   /*l0_bound=*/-1,
                                                   /*linfinity_bound=*/-1,
                                                   /*l1_bound=*/10,
                                                   /*l2_bound=*/-1, false);
  EXPECT_THAT(laplace_mechanism, IsOk());
  EXPECT_TRUE(laplace_mechanism.value().use_laplace);
  EXPECT_THAT(laplace_mechanism.value().mechanism->GetVariance(),
              DoubleEq(800));
}

// If both Laplace and Gaussian can be used, Laplace should be used if its
// variance is smaller.
TEST(DPNoiseMechanismsTest, CreateDPHistogramBundleUsesLaplaceWhenAppropriate) {
  // L0 and Linf bounds are given, with Laplace variance smaller than Gaussian.
  auto agg1 = CreateDPHistogramBundle(1.0, 1e-10, 2, 10, -1, -1, false);
  EXPECT_THAT(agg1, IsOk());
  EXPECT_TRUE(agg1.value().use_laplace);

  // L1 and L2 bounds are given, with Laplace variance smaller than Gaussian.
  auto agg2 = CreateDPHistogramBundle(1.0, 1e-8, -1, -1, 10, 10, false);
  EXPECT_THAT(agg2, IsOk());
  EXPECT_TRUE(agg2.value().use_laplace);
}

// If only the L2 norm bound is given, Gaussian should be used.
TEST(DPNoiseMechanismsTest, CreateDPHistogramBundleUsesGaussianOnlyL2) {
  auto gaussian_mechanism = CreateDPHistogramBundle(/*epsilon=*/1.0,
                                                    /*delta=*/1e-8,
                                                    /*l0_bound=*/-1,
                                                    /*linfinity_bound=*/-1,
                                                    /*l1_bound=*/-1,
                                                    /*l2_bound=*/10, false);
  EXPECT_THAT(gaussian_mechanism, IsOk());
  EXPECT_FALSE(gaussian_mechanism.value().use_laplace);
}

TEST(DPNoiseMechanismsTest,
     CreateDPHistogramBundleUsesGaussianWhenAppropriate) {
  // L0 and Linf plus L2 bound are given; Laplace variance larger than Gaussian.
  auto agg1 = CreateDPHistogramBundle(1.0, 1e-10, /*l0_bound=*/2,
                                      /*linfinity_bound=*/10, -1, 2, false);
  EXPECT_THAT(agg1, IsOk());
  EXPECT_FALSE(agg1.value().use_laplace);

  // L1 and L2 bounds are given, with Gaussian variance smaller than Laplace.
  auto agg2 = CreateDPHistogramBundle(1.0, 1e-8, -1, -1, 46, 10, false);
  EXPECT_THAT(agg2, IsOk());
  EXPECT_FALSE(agg2.value().use_laplace);
}

TEST(DPNoiseMechanismsTest, CreateDPHistogramBundleUsesGaussianForLargeL0) {
  // If a user can contribute to L0 = x groups and there is only an L_inf bound,
  // Laplace noise is linear in x while Gaussian noise scales with sqrt(x).
  // Hence, we should use Gaussian when we loosen x (from 2 to 20)
  auto agg3 = CreateDPHistogramBundle(1.0, /*delta=*/1e-10, /*l0_bound=*/20,
                                      /*linfinity_bound=*/10, -1, -1, false);
  EXPECT_THAT(agg3, IsOk());
  EXPECT_FALSE(agg3.value().use_laplace);
}

TEST(DPNoiseMechanismsTest, CreateDPHistogramBundleUsesGaussianForLargeDelta) {
  // Gaussian noise should also be used if delta was loosened enough
  auto agg4 = CreateDPHistogramBundle(1.0, /*delta=*/1e-3, /*l0_bound=*/2,
                                      /*linfinity_bound=*/10, -1, -1, false);
  EXPECT_THAT(agg4, IsOk());
  EXPECT_FALSE(agg4.value().use_laplace);
}

// Check that noise is added at all: the noised sum should not be the same as
// the unnoised sum. The chance of a false negative shrinks with epsilon.
TEST(DPNoiseMechanismsTest, LaplaceNoiseAddedForSmallEpsilons) {
  // Laplace
  auto bundle = CreateDPHistogramBundle(kSmallEpsilon, 1e-8, -1, -1,
                                        /*l1_bound=*/1, -1, false);
  EXPECT_THAT(bundle, IsOk());
  int val = 1000;
  auto noisy_val = bundle.value().mechanism->AddNoise(val);
  EXPECT_THAT(noisy_val, Ne(val));
}

TEST(DPNoiseMechanismsTest, GaussianNoiseAddedForSmallEpsilons) {
  // Gaussian
  auto bundle = CreateDPHistogramBundle(kSmallEpsilon, 1e-8, -1, -1, -1,
                                        /*l2_bound=*/1, false);
  EXPECT_THAT(bundle, IsOk());
  int val = 1000;
  auto noisy_val = bundle.value().mechanism->AddNoise(val);
  EXPECT_THAT(noisy_val, Ne(val));
}

// Check that CalculateLaplaceThreshold computes the right threshold
// Case 1: adjusted delta less than 1/2
TEST(DPNoiseMechanismsTest, CalculateLaplaceThresholdSucceedsSmallDelta) {
  double delta = 0.468559;  // = 1-(9/10)^6
  double linfinity_bound = 1;
  int64_t l0_bound = 1;

  // under replacement DP:
  int64_t l0_sensitivity = 2 * l0_bound;
  double l1_sensitivity = 2;  // = min(2 * l0_bound * linf_bound, 2 * l1_bound)

  // We'll work with eps = 1 for simplicity
  auto threshold_wrapper = internal::CalculateLaplaceThreshold(
      /*epsilon=*/1.0, delta, l0_sensitivity, linfinity_bound, l1_sensitivity);
  TFF_ASSERT_OK(threshold_wrapper.status());

  double laplace_tail_bound = 1.22497855;
  // = -(l1_sensitivity / 1.0) * std::log(2.0 * adjusted_delta),
  // where adjusted_delta = 1 - sqrt(1-delta) = 1 - (9/10)^3 = 1 - 0.729 = 0.271

  EXPECT_NEAR(threshold_wrapper.value(), linfinity_bound + laplace_tail_bound,
              1e-5);
}

// Case 2: adjusted delta greater than 1/2
TEST(DPNoiseMechanismsTest, CalculateLaplaceThresholdSucceedsLargeDelta) {
  double delta = 0.77123207545039;  // 1-(9/10)^14

  double linfinity_bound = 1;
  int64_t l0_bound = 1;

  // under replacement DP:
  int64_t l0_sensitivity = 2 * l0_bound;
  double l1_sensitivity = 2;  // = min(2 * l0_bound * linf_bound, 2 * l1_bound)

  auto threshold_wrapper = internal::CalculateLaplaceThreshold(
      /*epsilon=*/1.0, delta, l0_sensitivity, linfinity_bound, l1_sensitivity);
  TFF_ASSERT_OK(threshold_wrapper.status());

  double laplace_tail_bound = -0.0887529;
  // = (l1_sensitivity / 1.0) * std::log(2.0 - 2.0 * adjusted_delta),
  // where adjusted_delta = 1 - sqrt(1-delta) = 1 - (9/10)^7 = 0.5217031
  EXPECT_NEAR(threshold_wrapper.value(), linfinity_bound + laplace_tail_bound,
              1e-5);
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
