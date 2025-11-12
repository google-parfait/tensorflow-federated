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

import numpy as np
from scipy import optimize
from scipy import stats


def _log_sub(x, y):
  """Stable computation of log(exp(x) - exp(y))."""
  return x + np.log1p(-np.exp(y - x)) if y <= x else -np.inf


def _get_log_delta(noise, eps):
  # See https://arxiv.org/pdf/1805.06530, Eq. (6).
  t_star = eps * noise + 1 / (2 * noise)
  return _log_sub(
      stats.norm.logcdf(1 / noise - t_star),
      eps + stats.norm.logcdf(-t_star),
  )


def get_epsilon_gaussian(
    noise: float, delta: float, tol: float = 1.0e-12
) -> float:
  """Compute the epsilon for the Gaussian mechanism with the given DP params.

  Args:
    noise: The standard deviation of the Gaussian noise.
    delta: The target delta (0 < delta < 1).
    tol: Error tolerance for search.

  Returns:
    The epsilon for the Gaussian mechanism with the given DP params.
  """
  if noise < 0:
    raise ValueError(f'noise must be non-negative, got noise={noise}.')
  if not 0 <= delta <= 1:
    raise ValueError(f'delta must be in [0, 1], got delta={delta}.')

  if delta == 1:
    return 0

  if noise == 0:
    return np.inf
  elif noise == np.inf:
    return 0

  if delta == 0:
    return np.inf

  log_delta = np.log(delta)
  if _get_log_delta(noise, 0) < log_delta:
    # We have (0, delta)-DP.
    return 0

  eps_lo, eps_hi = 0, 1
  while _get_log_delta(noise, eps_hi) > log_delta:
    eps_lo, eps_hi = eps_hi, eps_hi * 2

  return optimize.brentq(
      lambda eps: _get_log_delta(noise, eps) - log_delta,
      eps_lo,
      eps_hi,
      xtol=tol,
  )


def get_noise_gaussian(
    epsilon: float, delta: float, tol: float = 1.0e-12
) -> float:
  """Compute the noise for the Gaussian mechanism with the given DP params.

  Args:
    epsilon: The target epsilon.
    delta: The target delta.
    tol: Error tolerance for search.

  Returns:
    The noise for the Gaussian mechanism with the given DP params.
  """
  if epsilon < 0:
    raise ValueError(f'epsilon must be non-negative, got epsilon={epsilon}.')
  if not 0 <= delta <= 1:
    raise ValueError(f'delta must be in [0, 1], got delta={delta}.')

  if delta == 1:
    return 0

  if epsilon == 0:
    return np.inf
  elif epsilon == np.inf:
    return 0

  if delta == 0:
    return np.inf

  log_delta = np.log(delta)
  noise_lo, noise_hi = 1e-1, 1
  while _get_log_delta(noise_lo, epsilon) < log_delta:
    noise_hi, noise_lo = noise_lo, noise_lo / 10
  while _get_log_delta(noise_hi, epsilon) > log_delta:
    noise_lo, noise_hi = noise_hi, noise_hi * 10

  return optimize.brentq(
      lambda noise: _get_log_delta(noise, epsilon) - log_delta,
      noise_lo,
      noise_hi,
      xtol=tol,
  )


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
  if norm_bound < 0:
    raise ValueError(f'norm_bound must be non-negative, got {norm_bound}.')

  if norm_bound == 0:
    return 0

  scaled_tol = tol / norm_bound
  return get_noise_gaussian(epsilon, delta, scaled_tol) * norm_bound
