# Copyright 2021, The TensorFlow Federated Authors.
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
"""Utility class for saving metrics in memory using dictionaries."""

import bisect
import collections
from typing import Any, Dict, Mapping, Optional

from tensorflow_federated.python.simulation import metrics_manager


class DictionaryMetricsManager(metrics_manager.MetricsManager):
  """A manager for keeping metrics in memory using an ordered dictionary.

  Note that this class stores all metrics in memory, and may be prohibitively
  expensive in large-scale simulations, especiall those storing large tensor
  metrics.
  """

  def __init__(self):
    """Returns an initialized `DictionaryMetricsManager`.

    This class will maintain metrics in a dictionary held in memory, where the
    keys will be integer round numbers, and the values are the metrics for the
    given round.
    """
    self._latest_round_num = None
    self._metrics = collections.OrderedDict()

  def save_metrics(self, metrics: Mapping[str, Any], round_num: int) -> None:
    """Updates the stored metrics data with metrics for a specific round.

    Args:
      metrics: A nested structure of metrics collected during `round_num`.
      round_num: Integer round at which `metrics` was collected.

    Raises:
      ValueError: If `round_num` is negative.
      ValueError: If `round_num` is less than or equal to the latest round
        number used to save metrics.
    """
    if not isinstance(round_num, int) or round_num < 0:
      raise ValueError(
          f'round_num must be a nonnegative integer, received {round_num}.')
    if self._latest_round_num and round_num <= self._latest_round_num:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'but metrics already exist through round '
                       f'{self._latest_round_num}.')

    self._metrics[round_num] = metrics
    self._latest_round_num = round_num

  def clear_metrics(self, round_num: int) -> None:
    """Clear out metrics at and after a given starting `round_num`.

    Note that if `clear_metrics(round_num=0)` is called, all metrics are cleared
    in a more performant manner. Rather than removing all keys associated to
    round numbers after `round_num`, we simply clear the entire dictionary.

    Args:
      round_num: A nonnegative integer indicating the starting round number for
        clearing metrics from the manager's associated dictionary.

    Raises:
      ValueError: If `round_num` is negative.
    """
    if round_num < 0:
      raise ValueError('Attempting to clear metrics after round '
                       f'{round_num}, which is negative.')
    round_numbers = list(self._metrics.keys())
    removal_index = bisect.bisect_left(round_numbers, round_num)
    if removal_index == 0:
      self._metrics.clear()
      self._latest_round_num = None
    else:
      for x in round_numbers[removal_index:]:
        del self._metrics[x]
      self._latest_round_num = round_numbers[removal_index - 1]

  @property
  def metrics(self) -> Dict[int, Any]:
    """Retrieve the stored experiment metrics data for all rounds."""
    return self._metrics.copy()

  @property
  def latest_round_num(self) -> Optional[int]:
    """The last round number passed to `save_metrics`.

    If no metrics have been written, this will be `None`, otherwise it will
    be a nonnegative integer.
    """
    return self._latest_round_num
