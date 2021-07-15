# Copyright 2020, The TensorFlow Federated Authors.
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
"""Functions to create differentially private tree aggregation factory.

The factory is created by wrapping proper tfp.DPQuery in
`DifferentiallyPrivateFactory`.
"""

import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.aggregators import sum_factory


def create_central_hierarchical_histogram_factory(
    stddev: float = 0.0,
    arity: int = 2,
    max_records_per_user: int = 10,
    secure_sum: bool = False):
  """Creates aggregator for hierarchical histograms with differential privacy.

  Args:
    stddev: The standard deviation of noise added to each node of the central
      tree.
    arity: The branching factor of the tree.
    max_records_per_user: The maximum of records each user can upload in their
      local histogram.
    secure_sum: A boolean deciding whether to use secure aggregation. Defaults
      to `False`.

  Returns:
    `tff.aggregators.UnWeightedAggregationFactory`.

  Raises:
    `ValueError`: If 'stddev < 0', `arity < 2`, `max_records_per_user < 1` or
    `inner_agg_factory` is illegal.
  """
  if stddev < 0:
    raise ValueError(f"Standard deviation should be greater than zero."
                     f"stddev={stddev} is given.")

  if arity < 2:
    raise ValueError(f"Arity should be at least 2." f"arity={arity} is given.")

  if max_records_per_user < 1:
    raise ValueError(f"Maximum records per user should be at least 1."
                     f"max_records_per_user={max_records_per_user} is given.")

  central_tree_agg_query = tfp.privacy.dp_query.tree_aggregation_query.CentralTreeSumQuery(
      stddev=stddev, arity=arity, l1_bound=max_records_per_user)

  if secure_sum:
    inner_agg_factory = secure.SecureSumFactory(
        upper_bound_threshold=float(max_records_per_user),
        lower_bound_threshold=0.)
  else:
    inner_agg_factory = sum_factory.SumFactory()

  return differential_privacy.DifferentiallyPrivateFactory(
      central_tree_agg_query, inner_agg_factory)
