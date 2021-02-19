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
"""Utility class for saving and loading simulation metrics via CSV."""

import collections
import csv
import os.path
import shutil
import tempfile
from typing import Any, Dict, List, Mapping, Tuple, Sequence, Set, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.simulation import metrics_manager

_QUOTING = csv.QUOTE_NONNUMERIC


def _create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _read_from_csv(
    file_name: str) -> Tuple[Sequence[str], List[Dict[str, Any]]]:
  """Returns a list of fieldnames and a list of metrics from a given CSV."""
  with tf.io.gfile.GFile(file_name, 'r') as csv_file:
    reader = csv.DictReader(csv_file, quoting=_QUOTING)
    fieldnames = reader.fieldnames
    csv_metrics = list(reader)
  return fieldnames, csv_metrics


def _write_to_csv(metrics: List[Dict[str, Any]], file_name: str,
                  fieldnames: Union[Sequence[str], Set[str]]):
  """Writes a list of metrics to CSV in an atomic fashion."""
  tmp_dir = tempfile.mkdtemp(prefix='atomic_write_to_csv_tmp')
  tmp_name = os.path.join(tmp_dir, os.path.basename(file_name))
  assert not tf.io.gfile.exists(tmp_name), 'File [{!s}] already exists'.format(
      tmp_name)

  # Write to a temporary GFile.
  with tf.io.gfile.GFile(tmp_name, 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=_QUOTING)
    writer.writeheader()
    for metric_row in metrics:
      writer.writerow(metric_row)

  # Copy to a temporary GFile next to the target, allowing for an atomic move.
  tmp_gfile_name = os.path.join(
      os.path.dirname(file_name), '{}.tmp{}'.format(
          os.path.basename(file_name),
          np.random.randint(0, 2**63, dtype=np.int64)))
  tf.io.gfile.copy(src=tmp_name, dst=tmp_gfile_name, overwrite=True)

  # Finally, do an atomic rename and clean up.
  tf.io.gfile.rename(tmp_gfile_name, file_name, overwrite=True)
  shutil.rmtree(tmp_dir)


def _append_to_csv(metrics_to_append: Dict[str, Any], file_name: str):
  """Appends `metrics` to a CSV.

  Here, the CSV is assumed to have first row representing the CSV's fieldnames.
  If `metrics` contains keys not in the CSV fieldnames, the CSV is re-written
  in order using the union of the fieldnames and `metrics.keys`.

  Args:
    metrics_to_append: A dictionary of metrics.
    file_name: The CSV file_name.

  Returns:
    A list of fieldnames for the updated CSV.
  """
  new_fieldnames = metrics_to_append.keys()
  with tf.io.gfile.GFile(file_name, 'a+') as csv_file:
    reader = csv.DictReader(csv_file, quoting=_QUOTING)
    current_fieldnames = reader.fieldnames

    if current_fieldnames is None:
      writer = csv.DictWriter(
          csv_file, fieldnames=list(new_fieldnames), quoting=_QUOTING)
      writer.writeheader()
      writer.writerow(metrics_to_append)
      return new_fieldnames
    elif new_fieldnames <= set(current_fieldnames):
      writer = csv.DictWriter(
          csv_file, fieldnames=current_fieldnames, quoting=_QUOTING)
      writer.writerow(metrics_to_append)
      return current_fieldnames
    else:
      metrics = list(reader)

  expanded_fieldnames = set(current_fieldnames).union(new_fieldnames)
  metrics.append(metrics_to_append)
  _write_to_csv(metrics, file_name, expanded_fieldnames)
  return expanded_fieldnames


def _flatten_nested_dict(struct: Mapping[str, Any]) -> Dict[str, Any]:
  """Flattens a given nested structure of tensors, sorting by flattened keys.

  For example, if we have the nested dictionary {'d':3, 'a': {'b': 1, 'c':2}, },
  this will produce the (ordered) dictionary {'a/b': 1, 'a/c': 2, 'd': 3}. This
  will unpack lists, so that {'a': [3, 4, 5]} will be flattened to the ordered
  dictionary {'a/0': 3, 'a/1': 4, 'a/2': 5}. The resulting values of the
  flattened dictionary will be the leaf nodetensors in the original struct.

  Args:
    struct: A (possibly nested) mapping.

  Returns:
    A `collections.OrderedDict` representing a flattened version of `struct`.
  """
  flat_struct = tree.flatten_with_path(struct)
  flat_struct = [('/'.join(map(str, path)), item) for path, item in flat_struct]
  return collections.OrderedDict(sorted(flat_struct))


class CSVMetricsManager(metrics_manager.MetricsManager):
  """Utility class for saving/loading experiment metrics via a CSV file."""

  def __init__(self, csv_filepath: str):
    """Returns an initialized `CSVMetricsManager`.

    This class will maintain metrics in a CSV file in the filesystem. The path
    of the file is {`root_metrics_dir`}/{`prefix`}.metrics.csv. To use this
    class upon restart of an experiment at an earlier round number, you can
    initialize and then call the clear_metrics() method to remove all rows
    for round numbers later than the restart round number. This ensures that no
    duplicate rows of data exist in the CSV.

    In order to reduce metric writing time, metrics passed to `save_metrics`
    are appended to the underlying CSV file (as long as they do not introduce
    any new key values). This may be incompatible with zipped files, such as
    `.bz2` formats, or encoded directories.

    Args:
      csv_filepath: A string specifying the file to write and read metrics from.

    Raises:
      ValueError: If `csv_filepath` is an empty string.
      ValueError: If the file at `csv_filepath` already exists but does not
        contain a `round_num` column.
    """
    super().__init__()
    if not csv_filepath:
      raise ValueError('Empty string passed for csv_filepath argument.')

    self._metrics_file = csv_filepath
    if not tf.io.gfile.exists(self._metrics_file):
      tf.io.gfile.makedirs(os.path.dirname(self._metrics_file))
      with tf.io.gfile.GFile(self._metrics_file, 'w') as file_object:
        writer = csv.DictWriter(
            file_object, fieldnames=['round_num'], quoting=_QUOTING)
        writer.writeheader()

    current_fieldnames, current_metrics = _read_from_csv(self._metrics_file)

    if current_metrics and 'round_num' not in current_fieldnames:
      raise ValueError(
          f'The specified csv file ({self._metrics_file}) already exists '
          'but was not created by CSVMetricsManager (it does not contain a '
          '`round_num` column.')

    if not current_metrics:
      self._latest_round_num = None
    else:
      self._latest_round_num = current_metrics[-1]['round_num']

  def save_metrics(self, round_num, metrics: Mapping[str, Any]) -> None:
    """Updates the stored metrics data with metrics for a specific round.

    The specified `round_num` must be nonnegative, and larger than the latest
    round number for which metrics exist in the stored metrics data. For
    example, calling this method with `round_num = 3` then `round_num = 7`
    is acceptable, but calling the method with `round_num = 6` then
    `round_num = 6` (or anything less than 6) is not supported.

    This method will atomically update the stored CSV file. Also, if stored
    metrics already exist and `metrics` contains a new, previously unseen metric
    name, a new column in the dataframe will be added for that metric, and all
    previous rows will fill in with NaN values for the metric.

    The metrics written are the leaf node tensors of the metrics_to_append
    structure. Purely scalar tensors will be written as scalars in the CSV,
    while tensors with non-zero rank will be written as a list of lists. For
    example, the tensor `tf.ones([2, 2])` will be written to the CSV as
    `'[[1.0, 1.0], [1.0, 1.0]'`.

    Args:
      round_num: Communication round at which `metrics_to_append` was collected.
      metrics: A nested structure of metrics collected during `round_num`. The
        nesting will be flattened for storage in the CSV (with the new keys
        equal to the paths in the nested structure).

    Raises:
      ValueError: If the provided round number is negative.
      ValueError: If the provided round number is less than or equal to the
        latest round number in the stored metrics data.
    """
    if not isinstance(round_num, int) or round_num < 0:
      raise ValueError(
          f'round_num must be a nonnegative integer, received {round_num}.')
    if self._latest_round_num and round_num <= self._latest_round_num:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'but metrics already exist through round '
                       f'{self._latest_round_num}.')

    flat_metrics = _flatten_nested_dict(metrics)
    flat_metrics_as_list = collections.OrderedDict()
    for key, value in flat_metrics.items():
      flat_metrics_as_list[key] = np.array(value).tolist()

    # Add the round number to the metrics before storing to csv file. This will
    # be used if a restart occurs, to identify which metrics to trim in the
    # clear_metrics() method.
    flat_metrics_as_list['round_num'] = round_num
    _append_to_csv(flat_metrics_as_list, self._metrics_file)
    self._latest_round_num = round_num

  def get_metrics(self) -> Tuple[Sequence[str], List[Dict[str, Any]]]:
    """Retrieve the stored experiment metrics data for all rounds.

    Returns:
      A sequence representing all possible keys for the metrics, and a list
      containing experiment metrics data for all rounds. Each entry in the list
      is a dictionary corresponding to a given round. The data has been
      flattened, with the column names equal to the path in the original nested
      metric structure. There is a fieldname `round_num` to indicate the round
      number.
    """
    return _read_from_csv(self._metrics_file)

  def _clear_all_rounds(self) -> None:
    """Existing metrics for all rounds are cleared out.

    This method will atomically update the stored CSV file.
    """
    with tf.io.gfile.GFile(self._metrics_file, 'w') as csv_file:
      writer = csv.DictWriter(
          csv_file, fieldnames=['round_num'], quoting=_QUOTING)
      writer.writeheader()
    self._latest_round_num = None

  def _clear_rounds_after(self, round_num: int) -> None:
    """Metrics for rounds greater than `round_num` are cleared out.

    Args:
      round_num: All metrics for rounds later than this are expunged.

    Raises:
      RuntimeError: If `self._last_valid_round_num` is `None`, indicating that
        no metrics have yet been written.
    """
    if self._latest_round_num is None:
      raise RuntimeError('Metrics do not exist yet.')

    reduced_fieldnames = set(['round_num'])
    _, metrics = _read_from_csv(self._metrics_file)
    reduced_metrics = []

    latest_round_num = None
    for metric_row in metrics:
      metric_row_round_num = metric_row['round_num']
      if metric_row_round_num <= round_num:
        reduced_fieldnames = reduced_fieldnames.union(metric_row.keys())
        reduced_metrics.append(metric_row)
        if (latest_round_num is None) or (latest_round_num <
                                          metric_row_round_num):
          latest_round_num = metric_row_round_num

    _write_to_csv(reduced_metrics, self._metrics_file, reduced_fieldnames)
    self._latest_round_num = latest_round_num

  def clear_metrics(self, round_num: int) -> None:
    """Clear out metrics at and after a given starting `round_num`.

    By using this method, this class can be used upon restart of an experiment
    at `round_num` to ensure that no duplicate rows of data exist in
    the CSV file. This method will atomically update the stored CSV file.

    Note that if `clear_metrics(round_num=0)` is called, all metrics are cleared
    in a more performant manner. Rather than removing all rows of the CSV with
    round numbers greater than or equal to `round_num`, we simply remove all
    rows of the CSV when `round_num=0`.

    Args:
      round_num: A nonnegative integer indicating the starting round number for
        clearing metrics from the manager's associated CSV.

    Raises:
      ValueError: If `round_num` is negative.
    """
    if round_num < 0:
      raise ValueError('Attempting to clear metrics after round '
                       f'{round_num}, which is negative.')
    if self._latest_round_num is not None:
      if round_num == 0:
        self._clear_all_rounds()
      else:
        self._clear_rounds_after(round_num - 1)

  @property
  def metrics_filename(self) -> str:
    return self._metrics_file
