# Copyright 2019, The TensorFlow Federated Authors.
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
"""Utility class for saving and loading simulation metrics."""

import abc
from typing import Any, Dict


class MetricsManager(metaclass=abc.ABCMeta):
  """An abstract base class for metrics managers.

  A `MetricManager` is a utility to log metric data across a number of
  rounds of some simulation.
  """

  @abc.abstractmethod
  def update_metrics(self, round_num: int, metrics_to_append: Dict[str, Any]):
    """Updates the metrics manager with metrics for a given round.

    This method updates the MetricsManager with a given nested structure of
    tensors `metrics_to_append`, at a given round number `round_num`. This
    method should only support strictly increasing, nonnegative round numbers,
    but not necessarily contiguous round numbers.

    For example, calling this method with `round_num = 3` then `round_num = 7`
    is acceptable, but calling the method with `round_num = 6` then
    `round_num = 6` (or anything less than 6) is not supported. he `round_num`
    must also be nonnegative, so `round_num = 0` is supported, but
    `round_num < 0` is not.

    The `metrics_to_append` can be any nested structure of tensors. The actual
    metrics that are recorded are the leaves of this nested structure, with
    names given by the path to the leaf.

    Args:
      round_num: A nonnegative integer representing the round number associated
        with `metrics_to_append`.
      metrics_to_append: A nested structure of tensors.
    """
    raise NotImplementedError
