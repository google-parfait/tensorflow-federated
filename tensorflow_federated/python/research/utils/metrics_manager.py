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
"""Utility class for saving and loading scalar experiment metrics."""

import collections
import os.path
from typing import Any, Dict

import pandas as pd
import tensorflow as tf
import tree

from tensorflow_federated.python.research.utils import utils_impl


class ScalarMetricsManager():
  """Utility class for saving/loading scalar experiment metrics.

  The metrics are backed by CSVs stored on the file system.
  """

  def __init__(self,
               root_metrics_dir: str = '/tmp',
               prefix: str = 'experiment',
               use_bz2: bool = True):
    """Returns an initialized `ScalarMetricsManager`.

    This class will maintain metrics in a CSV file in the filesystem. The path
    of the file is {`root_metrics_dir`}/{`prefix`}.metrics.csv (if use_bz2 is
    set to False) or {`root_metrics_dir`}/{`prefix`}.metrics.csv.bz2 (if
    use_bz2 is set to True). To use this class upon restart of an experiment at
    an earlier round number, you can initialize and then call the
    clear_rounds_after() method to remove all rows for round numbers later than
    the restart round number. This ensures that no duplicate rows of data exist
    in the CSV.

    Args:
      root_metrics_dir: A path on the filesystem to store CSVs.
      prefix: A string to use as the prefix of filename. Usually the name of a
        specific run in a larger grid of experiments sharing a common
        `root_metrics_dir`.
      use_bz2: A boolean indicating whether to zip the result metrics csv using
        bz2.

    Raises:
      ValueError: If `root_metrics_dir` is empty string.
      ValueError: If `prefix` is empty string.
      ValueError: If the specified metrics csv file already exists but does not
        contain a `round_num` column.
    """
    super().__init__()
    if not root_metrics_dir:
      raise ValueError('Empty string passed for root_metrics_dir argument.')
    if not prefix:
      raise ValueError('Empty string passed for prefix argument.')

    if use_bz2:
      # Using .bz2 rather than .zip due to
      # https://github.com/pandas-dev/pandas/issues/26023
      self._metrics_filename = os.path.join(root_metrics_dir,
                                            f'{prefix}.metrics.csv.bz2')
    else:
      self._metrics_filename = os.path.join(root_metrics_dir,
                                            f'{prefix}.metrics.csv')
    if not tf.io.gfile.exists(self._metrics_filename):
      utils_impl.atomic_write_to_csv(pd.DataFrame(), self._metrics_filename)

    self._metrics = utils_impl.atomic_read_from_csv(self._metrics_filename)
    if not self._metrics.empty and 'round_num' not in self._metrics.columns:
      raise ValueError(
          f'The specified csv file ({self._metrics_filename}) already exists '
          'but was not created by ScalarMetricsManager (it does not contain a '
          '`round_num` column.')

    self._latest_round_num = (None if self._metrics.empty else
                              self._metrics.round_num.max(axis=0))

  def update_metrics(self, round_num,
                     metrics_to_append: Dict[str, Any]) -> Dict[str, float]:
    """Updates the stored metrics data with metrics for a specific round.

    The specified `round_num` must be later than the latest round number for
    which metrics exist in the stored metrics data. This method will atomically
    update the stored CSV file. Also, if stored metrics already exist and
    `metrics_to_append` contains a new, previously unseen metric name, a new
    column in the dataframe will be added for that metric, and all previous rows
    will fill in with NaN values for the metric.

    Args:
      round_num: Communication round at which `metrics_to_append` was collected.
      metrics_to_append: A dictionary of metrics collected during `round_num`.
        These metrics can be in a nested structure, but the nesting will be
        flattened for storage in the CSV (with the new keys equal to the paths
        in the nested structure).

    Returns:
      A `collections.OrderedDict` of the data just added in a new row to the
        pandas.DataFrame. Compared with the input `metrics_to_append`, this data
        is flattened, with the key names equal to the path in the nested
        structure. Also, `round_num` has been added as an additional key.

    Raises:
      ValueError: If the provided round number is negative.
      ValueError: If the provided round number is less than or equal to the
        latest round number in the stored metrics data.
    """
    if round_num < 0:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'which is negative.')
    if self._latest_round_num and round_num <= self._latest_round_num:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'but metrics already exist through round '
                       f'{self._latest_round_num}.')

    # Add the round number to the metrics before storing to csv file. This will
    # be used if a restart occurs, to identify which metrics to trim in the
    # _clear_invalid_rounds() method.
    metrics_to_append['round_num'] = round_num

    flat_metrics = tree.flatten_with_path(metrics_to_append)
    flat_metrics = [
        ('/'.join(map(str, path)), item) for path, item in flat_metrics
    ]
    flat_metrics = collections.OrderedDict(flat_metrics)
    self._metrics = self._metrics.append(flat_metrics, ignore_index=True)
    utils_impl.atomic_write_to_csv(self._metrics, self._metrics_filename)
    self._latest_round_num = round_num

    return flat_metrics

  def get_metrics(self) -> pd.DataFrame:
    """Retrieve the stored experiment metrics data for all rounds.

    Returns:
      A `pandas.DataFrame` containing experiment metrics data for all rounds.
        This DataFrame is in `wide` format: a row for each round and a column
        for each metric. The data has been flattened, with the column names
        equal to the path in the original nested metric structure. There is a
        column (`round_num`) to indicate the round number.
    """
    return self._metrics

  def clear_all_rounds(self) -> None:
    """Existing metrics for all rounds are cleared out.

    This method will atomically update the stored CSV file.
    """
    self._metrics = pd.DataFrame()
    utils_impl.atomic_write_to_csv(self._metrics, self._metrics_filename)
    self._latest_round_num = None

  def clear_rounds_after(self, last_valid_round_num: int) -> None:
    """Metrics for rounds greater than `last_valid_round_num` are cleared out.

    By using this method, this class can be used upon restart of an experiment
    at `last_valid_round_num` to ensure that no duplicate rows of data exist in
    the CSV file. This method will atomically update the stored CSV file.

    Args:
      last_valid_round_num: All metrics for rounds later than this are expunged.

    Raises:
      RuntimeError: If metrics do not exist (none loaded during construction '
        nor recorded via `update_metrics()` and `last_valid_round_num` is not
        zero.
      ValueError: If `last_valid_round_num` is negative.
    """
    if last_valid_round_num < 0:
      raise ValueError('Attempting to clear metrics after round '
                       f'{last_valid_round_num}, which is negative.')
    if self._latest_round_num is None:
      if last_valid_round_num == 0:
        return
      raise RuntimeError('Metrics do not exist yet.')
    self._metrics = self._metrics.drop(
        self._metrics[self._metrics.round_num > last_valid_round_num].index)
    utils_impl.atomic_write_to_csv(self._metrics, self._metrics_filename)
    self._latest_round_num = last_valid_round_num

  @property
  def metrics_filename(self) -> str:
    return self._metrics_filename
