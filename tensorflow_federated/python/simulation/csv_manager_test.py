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
import csv
import os

import tensorflow as tf

from tensorflow_federated.python.simulation import csv_manager


def _create_scalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': 1.0,
          'c': 2.0,
      }),
  ])


def _create_nonscalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': tf.ones([1]),
          'c': tf.zeros([2, 2]),
      }),
  ])


def _create_scalar_metrics_with_extra_column():
  metrics = _create_scalar_metrics()
  metrics['a']['d'] = 3.0
  return metrics


class CSVMetricsManager(tf.test.TestCase):

  def test_scalar_metrics_are_appended(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    _, metrics = csv_mngr.get_metrics()
    self.assertEmpty(metrics)

    csv_mngr.update_metrics(0, _create_scalar_metrics())
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 1)

    csv_mngr.update_metrics(1, _create_scalar_metrics())
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 2)

  def test_nonscalar_metrics_are_appended(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    _, metrics = csv_mngr.get_metrics()
    self.assertEmpty(metrics)

    csv_mngr.update_metrics(0, _create_nonscalar_metrics())
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 1)

    csv_mngr.update_metrics(1, _create_nonscalar_metrics())
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 2)

  def test_update_metrics_returns_flat_dict_with_scalars(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    input_data_dict = _create_scalar_metrics()
    appended_data_dict = csv_mngr.update_metrics(0, input_data_dict)
    self.assertEqual(
        collections.OrderedDict({
            'a/b': 1.0,
            'a/c': 2.0,
            'round_num': 0.0
        }), appended_data_dict)

  def test_update_metrics_returns_flat_dict_with_nonscalars(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    input_data_dict = _create_nonscalar_metrics()
    appended_data_dict = csv_mngr.update_metrics(0, input_data_dict)
    expected_dict = collections.OrderedDict({
        'a/b': tf.ones([1]),
        'a/c': tf.zeros([2, 2]),
        'round_num': 0.0
    })
    self.assertListEqual(
        list(expected_dict.keys()), list(appended_data_dict.keys()))
    self.assertEqual(expected_dict['round_num'],
                     appended_data_dict['round_num'])
    self.assertAllEqual(expected_dict['a/b'], appended_data_dict['a/b'])
    self.assertAllEqual(expected_dict['a/c'], appended_data_dict['a/c'])

  def test_column_names(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, _create_scalar_metrics())
    fieldnames, _ = csv_mngr.get_metrics()
    self.assertCountEqual(['a/b', 'a/c', 'round_num'], fieldnames)

  def test_column_names_with_list(self):
    metrics_to_append = {'a': [3, 4, 5], 'b': 6}
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, metrics_to_append)
    fieldnames, _ = csv_mngr.get_metrics()
    self.assertCountEqual(['a/0', 'a/1', 'a/2', 'b', 'round_num'], fieldnames)

  def test_update_metrics_adds_column_if_previously_unseen_metric_added(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, _create_scalar_metrics())
    fieldnames, metrics = csv_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c'])
    self.assertNotIn('a/d', metrics[0].keys())

    csv_mngr.update_metrics(1, _create_scalar_metrics_with_extra_column())
    fieldnames, metrics = csv_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c', 'a/d'])
    self.assertEqual(metrics[0]['a/d'], '')

  def test_update_metrics_adds_empty_str_if_previous_column_not_provided(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, _create_scalar_metrics_with_extra_column())
    csv_mngr.update_metrics(1, _create_scalar_metrics())
    _, metrics = csv_mngr.get_metrics()
    self.assertEqual(metrics[1]['a/d'], '')

  def test_csvfile_is_saved(self):
    temp_dir = os.path.join(self.get_temp_dir(), 'test_dir')
    csv_file = os.path.join(temp_dir, 'metrics.csv')
    csv_manager.CSVMetricsManager(csv_file)
    self.assertEqual(set(os.listdir(temp_dir)), set(['metrics.csv']))

  def test_reload_of_csvfile(self):
    temp_dir = os.path.join(self.get_temp_dir(), 'test_dir')
    csv_file = os.path.join(temp_dir, 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, _create_scalar_metrics())
    csv_mngr.update_metrics(5, _create_scalar_metrics())

    new_csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    fieldnames, metrics = new_csv_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c'])
    self.assertLen(metrics, 2, 'There should be 2 rows (for rounds 0 and 5).')
    self.assertEqual(5, metrics[-1]['round_num'],
                     'Last metrics are for round 5.')

    self.assertEqual(set(os.listdir(temp_dir)), set(['metrics.csv']))

  def test_update_metrics_raises_value_error_if_round_num_is_negative(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    with self.assertRaises(ValueError):
      csv_mngr.update_metrics(-1, _create_scalar_metrics())

  def test_update_metrics_raises_value_error_if_round_num_is_out_of_order(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(1, _create_scalar_metrics())
    with self.assertRaises(ValueError):
      csv_mngr.update_metrics(0, _create_scalar_metrics())

  def test_clear_rounds_after_raises_runtime_error_if_no_metrics(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    # Clear is allowed with no metrics if no rounds have yet completed.
    csv_mngr.clear_rounds_after(last_valid_round_num=0)
    with self.assertRaises(RuntimeError):
      # Raise exception with no metrics if no rounds have yet completed.
      csv_mngr.clear_rounds_after(last_valid_round_num=1)

  def test_clear_rounds_after_raises_value_error_if_round_num_is_negative(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, _create_scalar_metrics())
    with self.assertRaises(ValueError):
      csv_mngr.clear_rounds_after(last_valid_round_num=-1)

  def test_rows_are_cleared_and_last_round_num_is_reset(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, _create_scalar_metrics())
    csv_mngr.update_metrics(5, _create_scalar_metrics())
    csv_mngr.update_metrics(10, _create_scalar_metrics())
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 3,
                   'There should be 3 rows (for rounds 0, 5, and 10).')
    csv_mngr.clear_rounds_after(last_valid_round_num=7)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(
        metrics, 2,
        'After clearing all rounds after last_valid_round_num=7, should be 2 '
        'rows of metrics (for rounds 0 and 5).')
    self.assertEqual(5, metrics[-1]['round_num'],
                     'Last metrics retained are for round 5.')
    # The internal state of the manager knows the last round number is 7, so it
    # raises an exception if a user attempts to add new metrics at round 7, ...
    with self.assertRaises(ValueError):
      csv_mngr.update_metrics(7, _create_scalar_metrics())
    # ... but allows a user to add new metrics at a round number greater than 7.
    csv_mngr.update_metrics(8, _create_scalar_metrics())  # (No exception.)

  def test_rows_are_cleared_is_reflected_in_saved_file(self):
    temp_dir = os.path.join(self.get_temp_dir(), 'test_dir')
    csv_file = os.path.join(temp_dir, 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, _create_scalar_metrics())
    csv_mngr.update_metrics(5, _create_scalar_metrics())
    csv_mngr.update_metrics(10, _create_scalar_metrics())
    filename = os.path.join(temp_dir, 'metrics.csv')
    with tf.io.gfile.GFile(filename, 'r') as csvfile:
      num_lines_before = len(csvfile.readlines())
    # The CSV file should have 4 lines, one for the fieldnames, and 3 for each
    # call to `update_metrics`.
    self.assertEqual(num_lines_before, 4)
    csv_mngr.clear_rounds_after(last_valid_round_num=7)
    with tf.io.gfile.GFile(filename, 'r') as csvfile:
      num_lines_after = len(csvfile.readlines())
    # The CSV file should have 3 lines, one for the fieldnames, and 2 for the
    # calls to `update_metrics` with round_nums less <= 7.
    self.assertEqual(num_lines_after, 3)

  def test_constructor_raises_value_error_if_csvfile_is_invalid(self):
    metrics_missing_round_num = _create_scalar_metrics()
    temp_dir = self.get_temp_dir()
    # This csvfile is 'invalid' in that it was not originally created by an
    # instance of CSVMetricsManager, and is missing a column for
    # round_num.
    invalid_csvfile = os.path.join(temp_dir, 'invalid_metrics.csv')
    with tf.io.gfile.GFile(invalid_csvfile, 'w') as csvfile:
      writer = csv.DictWriter(
          csvfile, fieldnames=metrics_missing_round_num.keys())
      writer.writeheader()
      writer.writerow(metrics_missing_round_num)
    with self.assertRaises(ValueError):
      csv_manager.CSVMetricsManager(invalid_csvfile)

  def test_get_metrics_with_nonscalars_returns_list_of_lists(self):
    metrics_to_append = {
        'a': tf.ones([1], dtype=tf.int32),
        'b': tf.zeros([2, 2], dtype=tf.int32)
    }
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.update_metrics(0, metrics_to_append)
    _, metrics = csv_mngr.get_metrics()
    self.assertEqual(metrics[0]['a'], '[1]')
    self.assertEqual(metrics[0]['b'], '[[0, 0], [0, 0]]')


if __name__ == '__main__':
  tf.test.main()
