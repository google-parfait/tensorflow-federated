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

import collections
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.utils import metrics_manager
from tensorflow_federated.python.research.utils import utils_impl


def _create_dummy_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': 1.0,
          'c': 2.0,
      }),
  ])


def _create_dummy_metrics_with_extra_column():
  metrics = _create_dummy_metrics()
  metrics['a']['d'] = 3.0
  return metrics


class ScalarMetricsManagerTest(tf.test.TestCase):

  def test_metrics_are_appended(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics = metrics_mngr.get_metrics()
    self.assertTrue(metrics.empty)

    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics = metrics_mngr.get_metrics()
    self.assertEqual(1, len(metrics.index))

    metrics_mngr.update_metrics(1, _create_dummy_metrics())
    metrics = metrics_mngr.get_metrics()
    self.assertEqual(2, len(metrics.index))

  def test_update_metrics_returns_flat_dict(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    input_data_dict = _create_dummy_metrics()
    appended_data_dict = metrics_mngr.update_metrics(0, input_data_dict)
    self.assertEqual({
        'a/b': 1.0,
        'a/c': 2.0,
        'round_num': 0.0
    }, appended_data_dict)

  def test_column_names(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics = metrics_mngr.get_metrics()
    self.assertEqual(['a/b', 'a/c', 'round_num'], metrics.columns.tolist())

  def test_update_metrics_adds_column_if_previously_unseen_metric_added(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics_mngr.update_metrics(1, _create_dummy_metrics_with_extra_column())
    metrics = metrics_mngr.get_metrics()
    self.assertTrue(np.isnan(metrics.at[0, 'a/d']))

  def test_update_metrics_adds_nan_if_previously_seen_metric_not_provided(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics_with_extra_column())
    metrics_mngr.update_metrics(1, _create_dummy_metrics())
    metrics = metrics_mngr.get_metrics()
    self.assertTrue(np.isnan(metrics.at[1, 'a/d']))

  def test_csvfile_is_saved(self):
    temp_dir = self.get_temp_dir()
    metrics_manager.ScalarMetricsManager(temp_dir, prefix='foo')
    self.assertEqual(set(os.listdir(temp_dir)), set(['foo.metrics.csv.bz2']))

  def test_reload_of_csvfile(self):
    temp_dir = self.get_temp_dir()
    metrics_mngr = metrics_manager.ScalarMetricsManager(temp_dir, prefix='bar')
    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics_mngr.update_metrics(5, _create_dummy_metrics())

    new_metrics_mngr = metrics_manager.ScalarMetricsManager(
        temp_dir, prefix='bar')
    metrics = new_metrics_mngr.get_metrics()
    self.assertEqual(2, len(metrics.index),
                     'There should be 2 rows of metrics (for rounds 0 and 5).')
    self.assertEqual(5, metrics['round_num'].iloc[-1],
                     'Last metrics are for round 5.')

    self.assertEqual(set(os.listdir(temp_dir)), set(['bar.metrics.csv.bz2']))

  def test_update_metrics_raises_value_error_if_round_num_is_negative(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    with self.assertRaises(ValueError):
      metrics_mngr.update_metrics(-1, _create_dummy_metrics())

  def test_update_metrics_raises_value_error_if_round_num_is_out_of_order(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    metrics_mngr.update_metrics(1, _create_dummy_metrics())

    with self.assertRaises(ValueError):
      metrics_mngr.update_metrics(0, _create_dummy_metrics())

  def test_clear_rounds_after_raises_runtime_error_if_no_metrics(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    # Clear is allowed with no metrics if no rounds have yet completed.
    metrics_mngr.clear_rounds_after(last_valid_round_num=0)

    with self.assertRaises(RuntimeError):
      # Raise exception with no metrics if no rounds have yet completed.
      metrics_mngr.clear_rounds_after(last_valid_round_num=1)

  def test_clear_rounds_after_raises_value_error_if_round_num_is_negative(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics())

    with self.assertRaises(ValueError):
      metrics_mngr.clear_rounds_after(last_valid_round_num=-1)

  def test_rows_are_cleared_and_last_round_num_is_reset(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics_mngr.update_metrics(5, _create_dummy_metrics())
    metrics_mngr.update_metrics(10, _create_dummy_metrics())
    metrics = metrics_mngr.get_metrics()
    self.assertEqual(
        3, len(metrics.index),
        'There should be 3 rows of metrics (for rounds 0, 5, and 10).')

    metrics_mngr.clear_rounds_after(last_valid_round_num=7)

    metrics = metrics_mngr.get_metrics()
    self.assertEqual(
        2, len(metrics.index),
        'After clearing all rounds after last_valid_round_num=7, should be 2 '
        'rows of metrics (for rounds 0 and 5).')
    self.assertEqual(5, metrics['round_num'].iloc[-1],
                     'Last metrics retained are for round 5.')

    # The internal state of the manager knows the last round number is 7, so it
    # raises an exception if a user attempts to add new metrics at round 7, ...
    with self.assertRaises(ValueError):
      metrics_mngr.update_metrics(7, _create_dummy_metrics())

    # ... but allows a user to add new metrics at a round number greater than 7.
    metrics_mngr.update_metrics(8, _create_dummy_metrics())  # (No exception.)

  def test_rows_are_cleared_is_reflected_in_saved_file(self):
    temp_dir = self.get_temp_dir()
    metrics_mngr = metrics_manager.ScalarMetricsManager(temp_dir, prefix='foo')

    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics_mngr.update_metrics(5, _create_dummy_metrics())
    metrics_mngr.update_metrics(10, _create_dummy_metrics())

    file_contents_before = utils_impl.atomic_read_from_csv(
        os.path.join(temp_dir, 'foo.metrics.csv.bz2'))
    self.assertEqual(3, len(file_contents_before.index))

    metrics_mngr.clear_rounds_after(last_valid_round_num=7)

    file_contents_after = utils_impl.atomic_read_from_csv(
        os.path.join(temp_dir, 'foo.metrics.csv.bz2'))
    self.assertEqual(2, len(file_contents_after.index))

  def test_constructor_raises_value_error_if_csvfile_is_invalid(self):
    dataframe_missing_round_num = pd.DataFrame.from_dict(
        _create_dummy_metrics())

    temp_dir = self.get_temp_dir()
    # This csvfile is 'invalid' in that it was not originally created by an
    # instance of ScalarMetricsManager, and is missing a column for
    # round_num.
    invalid_csvfile = os.path.join(temp_dir, 'foo.metrics.csv.bz2')
    utils_impl.atomic_write_to_csv(dataframe_missing_round_num, invalid_csvfile)

    with self.assertRaises(ValueError):
      metrics_manager.ScalarMetricsManager(temp_dir, prefix='foo')


if __name__ == '__main__':
  tf.test.main()
