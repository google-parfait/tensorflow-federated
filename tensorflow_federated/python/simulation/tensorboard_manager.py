# Copyright 2020, Google LLC.
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
"""Utility class for logging metrics and hyperparameters to TensorBoard."""

import collections
from typing import Any, Dict, Mapping

from absl import logging
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.simulation import metrics_manager


def _create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _flatten_nested_dict(struct: Mapping[str, Any]) -> Dict[str, Any]:
  """Flattens a given nested structure of tensors, sorting by flattened keys.

  For example, if we have the nested dictionary {'d':3, 'a': {'b': 1, 'c':2}, },
  this will produce the (ordered) dictionary {'a/b': 1, 'a/c': 2, 'd': 3}. This
  will unpack lists, so that {'a': [3, 4, 5]} will be flattened to the ordered
  dictionary {'a/0': 3, 'a/1': 4, 'a/2': 5}. The values of the resulting
  flattened dictionary will be the tensors at the corresponding leaf nodes
  of the original struct.

  Args:
    struct: A nested dictionary.

  Returns:
    A `collections.OrderedDict` representing a flattened version of `struct`.
  """
  flat_struct = tree.flatten_with_path(struct)
  flat_struct = [('/'.join(map(str, path)), item) for path, item in flat_struct]
  return collections.OrderedDict(sorted(flat_struct))


class TensorBoardManager(metrics_manager.MetricsManager):
  """Utility class for saving metrics using `tf.summary`.

  This class is intended to log metrics so that they can be used with
  TensorBoard. Note that this supports both scalar and series data, which are
  logged via `tf.summary.scalar` and `tf.summary.histogram`, respectively.
  """

  def __init__(self, summary_dir: str = '/tmp/logdir'):
    """Returns an initialized `SummaryWriterManager`.

    This class will write metrics to `summary_dir` using a
    `tf.summary.SummaryWriter`, created via `tf.summary.create_file_writer`.

    Args:
      summary_dir: A path on the filesystem containing all outputs of the
        associated summary writer.

    Raises:
      ValueError: If `root_metrics_dir` is an empty string.
      ValueError: If `summary_dir` is an empty string.
    """
    super().__init__()
    if not summary_dir:
      raise ValueError('Empty string passed for summary_dir argument.')

    self._logdir = summary_dir
    _create_if_not_exists(self._logdir)
    self._summary_writer = tf.summary.create_file_writer(self._logdir)
    self._latest_round_num = None

  def save_metrics(self, round_num: int, metrics: Mapping[str, Any]):
    """Updates the stored metrics data with metrics for a specific round.

    The specified `round_num` must be later than the latest round number
    previously used with `save_metrics`. Note that we do not check whether
    the underlying summary writer has previously written any metrics with the
    given `round_num`. Thus, if the `TensorboardManager` is created from a
    directory containing previously written metrics, it may overwrite them. This
    is intended usage, allowing one to restart and resume experiments from
    previous rounds.

    The metrics written by the underlying `tf.summary.SummaryWriter` will be the
    leaf node tensors of the metrics_to_append structure. Purely scalar tensors
    will be written using `tf.summary.scalar`, while tensors with non-zero rank
    will be written using `tf.summary.histogram`.

    Args:
      round_num: Communication round at which `metrics` was collected.
      metrics: A nested structure of metrics collected during `round_num`. The
        nesting will be flattened for purposes of writing to TensorBoard.

    Returns:
      A `collections.OrderedDict` of the metrics used to update the manager.
      Compared with the input `metrics`, this data is flattened, with the key
      names equal to the path in the nested structure, and `round_num` has been
      added as an additional key (overwriting the value if already present in
      the input `metrics`). The `OrderedDict` is sorted by the flattened keys.

    Raises:
      ValueError: If `round_num` is negative, or `round_num` is less than or
        equal to the last round number used with `save_metrics`.
    """
    if not isinstance(round_num, int) or round_num < 0:
      raise ValueError(
          f'round_num must be a nonnegative integer, received {round_num}.')
    if self._latest_round_num and round_num <= self._latest_round_num:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'but metrics already exist through round '
                       f'{self._latest_round_num}.')

    flat_metrics = _flatten_nested_dict(metrics)
    flat_metrics['round_num'] = round_num
    with self._summary_writer.as_default():
      for name, val in flat_metrics.items():
        val_array = np.array(val)
        if val_array.shape:
          tf.summary.histogram(name, val, step=round_num)
        else:
          tf.summary.scalar(name, val, step=round_num)

    self._latest_round_num = round_num
    return flat_metrics
