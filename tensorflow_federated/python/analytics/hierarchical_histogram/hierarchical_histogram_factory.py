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


def create_central_hierarchical_histogram_factory(stddev: float = 0.0,
                                                  arity: int = 2,
                                                  max_records_per_user: int = 10
                                                 ):
  central_tree_agg_query = tfp.privacy.dp_query.tree_aggregation_query.CentralTreeSumQuery(
      stddev=stddev, arity=arity, l1_bound=max_records_per_user)
  return differential_privacy.DifferentiallyPrivateFactory(
      central_tree_agg_query)


def create_distributed_hierarchical_histogram_factory(
    stddev: float = 0.0, arity: int = 2, max_records_per_user: int = 10):
  distributed_tree_agg_query = tfp.privacy.dp_query.tree_aggregation_query.DistributedTreeSumQuery(
      stddev=stddev, arity=arity, l1_bound=max_records_per_user)
  return differential_privacy.DifferentiallyPrivateFactory(
      distributed_tree_agg_query)
