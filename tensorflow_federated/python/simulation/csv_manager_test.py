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

    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 1)

    csv_mngr.save_metrics(_create_scalar_metrics(), 1)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 2)

  def test_nonscalar_metrics_are_appended(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    _, metrics = csv_mngr.get_metrics()
    self.assertEmpty(metrics)

    csv_mngr.save_metrics(_create_nonscalar_metrics(), 0)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 1)

    csv_mngr.save_metrics(_create_nonscalar_metrics(), 1)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 2)

  def test_flatten_nested_dict_with_scalars(self):
    input_data_dict = _create_scalar_metrics()
    flattened_data = csv_manager._flatten_nested_dict(input_data_dict)
    self.assertEqual(
        collections.OrderedDict({
            'a/b': 1.0,
            'a/c': 2.0,
        }), flattened_data)

  def test_flatten_nested_dict_with_nonscalars(self):
    input_data_dict = _create_nonscalar_metrics()
    flattened_data = csv_manager._flatten_nested_dict(input_data_dict)
    expected_dict = collections.OrderedDict({
        'a/b': tf.ones([1]),
        'a/c': tf.zeros([2, 2]),
    })
    self.assertListEqual(
        list(expected_dict.keys()), list(flattened_data.keys()))
    self.assertAllEqual(expected_dict['a/b'], flattened_data['a/b'])
    self.assertAllEqual(expected_dict['a/c'], flattened_data['a/c'])

  def test_column_names(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    fieldnames, _ = csv_mngr.get_metrics()
    self.assertCountEqual(['a/b', 'a/c', 'round_num'], fieldnames)

  def test_column_names_with_list(self):
    metrics_to_append = {'a': [3, 4, 5], 'b': 6}
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(metrics_to_append, 0)
    fieldnames, _ = csv_mngr.get_metrics()
    self.assertCountEqual(['a/0', 'a/1', 'a/2', 'b', 'round_num'], fieldnames)

  def test_save_metrics_adds_column_if_previously_unseen_metric_added(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    fieldnames, metrics = csv_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c'])
    self.assertNotIn('a/d', metrics[0].keys())

    csv_mngr.save_metrics(_create_scalar_metrics_with_extra_column(), 1)
    fieldnames, metrics = csv_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c', 'a/d'])
    self.assertEqual(metrics[0]['a/d'], '')

  def test_save_metrics_adds_empty_str_if_previous_column_not_provided(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics_with_extra_column(), 0)
    csv_mngr.save_metrics(_create_scalar_metrics(), 1)
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
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    csv_mngr.save_metrics(_create_scalar_metrics(), 5)

    new_csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    fieldnames, metrics = new_csv_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c'])
    self.assertLen(metrics, 2, 'There should be 2 rows (for rounds 0 and 5).')
    self.assertEqual(5, metrics[-1]['round_num'],
                     'Last metrics are for round 5.')

    self.assertEqual(set(os.listdir(temp_dir)), set(['metrics.csv']))

  def test_save_metrics_raises_value_error_if_round_num_is_negative(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    with self.assertRaises(ValueError):
      csv_mngr.save_metrics(_create_scalar_metrics(), -1)

  def test_save_metrics_raises_value_error_if_round_num_is_out_of_order(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 1)
    with self.assertRaises(ValueError):
      csv_mngr.save_metrics(_create_scalar_metrics(), 0)

  def test_clear_metrics_raises_value_error_if_round_num_is_negative(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    with self.assertRaises(ValueError):
      csv_mngr.clear_metrics(round_num=-1)

  def test_clear_metrics_removes_rounds_after_input_arg(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    csv_mngr.save_metrics(_create_scalar_metrics(), 5)
    csv_mngr.save_metrics(_create_scalar_metrics(), 10)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 3,
                   'There should be 3 rows (for rounds 0, 5, and 10).')
    csv_mngr.clear_metrics(round_num=7)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(
        metrics, 2, 'After clearing all rounds after round_num=7, should be 2 '
        'rows of metrics (for rounds 0 and 5).')
    self.assertEqual(5, metrics[-1]['round_num'],
                     'Last metrics retained are for round 5.')
    self.assertEqual(5, csv_mngr._latest_round_num)

  def test_clear_metrics_removes_rounds_equal_to_input_arg(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    csv_mngr.save_metrics(_create_scalar_metrics(), 5)
    csv_mngr.save_metrics(_create_scalar_metrics(), 10)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 3,
                   'There should be 3 rows (for rounds 0, 5, and 10).')
    csv_mngr.clear_metrics(round_num=5)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(
        metrics, 1, 'After clearing all rounds starting at round_num=5, there '
        'should be 1 row of metrics (for round 0).')
    self.assertEqual(0, metrics[-1]['round_num'],
                     'Last metrics retained are for round 0.')
    self.assertEqual(0, csv_mngr._latest_round_num)

  def test_clear_metrics_with_round_zero_removes_all_metrics(self):
    csv_file = os.path.join(self.get_temp_dir(), 'test_dir', 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    csv_mngr.save_metrics(_create_scalar_metrics(), 5)
    csv_mngr.save_metrics(_create_scalar_metrics(), 10)
    _, metrics = csv_mngr.get_metrics()
    self.assertLen(metrics, 3,
                   'There should be 3 rows (for rounds 0, 5, and 10).')
    csv_mngr.clear_metrics(round_num=0)
    _, metrics = csv_mngr.get_metrics()
    self.assertEmpty(metrics)
    self.assertIsNone(csv_mngr._latest_round_num)

  def test_rows_are_cleared_is_reflected_in_saved_file(self):
    temp_dir = os.path.join(self.get_temp_dir(), 'test_dir')
    csv_file = os.path.join(temp_dir, 'metrics.csv')
    csv_mngr = csv_manager.CSVMetricsManager(csv_file)
    csv_mngr.save_metrics(_create_scalar_metrics(), 0)
    csv_mngr.save_metrics(_create_scalar_metrics(), 5)
    csv_mngr.save_metrics(_create_scalar_metrics(), 10)
    filename = os.path.join(temp_dir, 'metrics.csv')
    with tf.io.gfile.GFile(filename, 'r') as csvfile:
      num_lines_before = len(csvfile.readlines())
    # The CSV file should have 4 lines, one for the fieldnames, and 3 for each
    # call to `save_metrics`.
    self.assertEqual(num_lines_before, 4)
    csv_mngr.clear_metrics(round_num=7)
    with tf.io.gfile.GFile(filename, 'r') as csvfile:
      num_lines_after = len(csvfile.readlines())
    # The CSV file should have 3 lines, one for the fieldnames, and 2 for the
    # calls to `save_metrics` with round_nums less <= 7.
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
    csv_mngr.save_metrics(metrics_to_append, 0)
    _, metrics = csv_mngr.get_metrics()
    self.assertEqual(metrics[0]['a'], '[1]')
    self.assertEqual(metrics[0]['b'], '[[0, 0], [0, 0]]')


if __name__ == '__main__':
  tf.test.main()
