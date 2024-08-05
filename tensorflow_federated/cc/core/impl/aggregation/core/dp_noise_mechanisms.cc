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
#include <limits>

namespace tensorflow_federated {
namespace aggregation {
// Calculate L1 sensitivity from norm bounds, under replacement DP.
// A non-positive norm bound means it was not specified.
double CalculateL1Sensitivity(int64_t l0_bound, double linfinity_bound,
                              double l1_bound) {
  double l1_sensitivity = std::numeric_limits<double>::infinity();
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
  double l2_sensitivity = std::numeric_limits<double>::infinity();
  if (l0_bound > 0 && linfinity_bound > 0) {
    l2_sensitivity =
        fmin(l2_sensitivity, sqrt(2.0 * l0_bound) * linfinity_bound);
  }
  if (l2_bound > 0) {
    l2_sensitivity = fmin(l2_sensitivity, 2.0 * l2_bound);
  }
  return l2_sensitivity;
}
}  // namespace aggregation
}  // namespace tensorflow_federated
