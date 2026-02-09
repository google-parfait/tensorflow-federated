# Copyright 2022, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differential privacy methods for federated analytics."""

import dp_accounting


def analytic_gauss_stddev(
    epsilon: float, delta: float, norm_bound: float, tol: float = 1.0e-12
):
  """Compute the stddev for the Gaussian mechanism with the given DP params.

  Arguments:
    epsilon: The target epsilon.
    delta: The target delta.
    norm_bound: Upper bound on L2 global sensitivity.
    tol: Error tolerance for search.

  Returns:
    sigma: Standard deviation of Gaussian noise needed to achieve
      (epsilon,delta)-DP under the given norm_bound.
  """
  if epsilon < 0:
    raise ValueError(f'epsilon must be non-negative, got epsilon={epsilon}.')
  if not 0 <= delta <= 1:
    raise ValueError(f'delta must be in [0, 1], got delta={delta}.')
  if norm_bound < 0:
    raise ValueError(f'norm_bound must be non-negative, got {norm_bound}.')

  if norm_bound == 0:
    return 0

  scaled_tol = tol / norm_bound
  return norm_bound * dp_accounting.get_sigma_gaussian(
      epsilon, delta, scaled_tol
  )
