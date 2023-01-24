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

import math


def analytic_gauss_stddev(epsilon, delta, norm_bound, tol=1.0e-12):
  """Compute the stddev for the Gaussian mechanism with the given DP params.

  Calibrate a Gaussian perturbation for differential privacy using the
  analytic Gaussian mechanism of [Balle and Wang, ICML'18].

  Reference: http://proceedings.mlr.press/v80/balle18a/balle18a.pdf.

  Arguments:
    epsilon: Target epsilon (0 < epsilon <= 500). The epsilon value is limited
      to at most 500 in this implementation because large epsilons causes
      overflow in python.
    delta: Target delta (0 < delta < 1).
    norm_bound: Upper bound on L2 global sensitivity (norm_bound >= 0).
    tol: Error tolerance for binary search (tol > 0).

  Returns:
    sigma: Standard deviation of Gaussian noise needed to achieve
      (epsilon,delta)-DP under the given norm_bound.
  """

  if epsilon <= 0 or epsilon >= 500:
    raise ValueError(f'epsilon must be in (0, 500], got epsilon={epsilon}.')
  if delta <= 0 or delta >= 1:
    raise ValueError(f'delta must be in (0, 1), got delta={delta}.')
  if norm_bound <= 0:
    raise ValueError(
        f'norm_bound must be positive, got norm_bound={norm_bound}.'
    )
  if tol <= 0:
    raise ValueError(f'tol must be positive, got tol={tol}.')

  exp = math.exp
  sqrt = math.sqrt

  def phi(t):
    return 0.5 * (1.0 + math.erf(float(t) / sqrt(2.0)))

  # $B_{\epsilon}^+(v)$ in Alg 1 of [Balle and Wang, ICML'18].
  def case_one(eps, s):
    return phi(sqrt(eps * s)) - exp(eps) * phi(-sqrt(eps * (s + 2.0)))

  # $B_{\epsilon}^-(u)$ in Alg 1 of [Balle and Wang, ICML'18].
  def case_two(eps, s):
    return phi(-sqrt(eps * s)) - exp(eps) * phi(-sqrt(eps * (s + 2.0)))

  def search_s_bounds(predicate_stop, s_inf, s_sup):
    while not predicate_stop(s_sup):
      s_inf = s_sup
      s_sup = 2.0 * s_inf
    return s_inf, s_sup

  def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
    s_mid = s_inf + (s_sup - s_inf) / 2.0
    while not predicate_stop(s_mid):
      if predicate_left(s_mid):
        s_sup = s_mid
      else:
        s_inf = s_mid
      s_mid = s_inf + (s_sup - s_inf) / 2.0
    return s_mid

  delta_thr = case_one(epsilon, 0.0)

  if delta == delta_thr:
    alpha = 1.0

  else:
    if delta > delta_thr:
      predicate_stop_dt = lambda s: case_one(epsilon, s) >= delta
      function_s_to_delta = lambda s: case_one(epsilon, s)
      predicate_left_bs = lambda s: function_s_to_delta(s) > delta
      function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

    else:
      predicate_stop_dt = lambda s: case_two(epsilon, s) <= delta
      function_s_to_delta = lambda s: case_two(epsilon, s)
      predicate_left_bs = lambda s: function_s_to_delta(s) < delta
      function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

    predicate_stop_bs = lambda s: abs(function_s_to_delta(s) - delta) <= tol

    s_inf, s_sup = search_s_bounds(predicate_stop_dt, 0.0, 1.0)
    s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
    alpha = function_s_to_alpha(s_final)

  sigma = alpha * norm_bound / sqrt(2.0 * epsilon)
  return sigma
