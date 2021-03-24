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
from typing import Any, Mapping


class MetricsManager(metaclass=abc.ABCMeta):
  """An abstract base class for metrics managers.

  A `tff.simulation.MetricsManager` is a utility to save metric data across a
  number of rounds of some simulation.
  """

  @abc.abstractmethod
  def save_metrics(self, metrics: Mapping[str, Any], round_num: int) -> None:
    """Saves metrics data for a given round.

    Note that different implementations may save metrics in markedly different
    ways, including skipping metrics depending on their type or `round_num`.

    Args:
      metrics: A mapping with string valued keys.
      round_num: A nonnegative integer representing the round number associated
        with `metrics`.
    """
    raise NotImplementedError

  def clear_metrics(self, round_num: int) -> None:
    """Clear out metrics at or after a given starting `round_num`.

    Note that since `save_metrics` is only compatible with nonnegative integer
    round numbers, `clear_metrics(round_num=0)` corresponds to clearing all
    metrics previously saved via `save_metrics`.

    Args:
      round_num: A nonnegative integer representing the starting round number
        for clearing metrics.
    """
    pass
