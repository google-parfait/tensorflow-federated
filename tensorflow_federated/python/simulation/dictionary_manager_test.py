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

import collections

import tensorflow as tf

from tensorflow_federated.python.simulation import dictionary_manager


class DictionaryManagerTest(tf.test.TestCase):

  def test_metrics_are_saved(self):
    dict_manager = dictionary_manager.DictionaryMetricsManager()
    actual_metrics = dict_manager.metrics
    self.assertEqual(actual_metrics, collections.OrderedDict())
    self.assertIsNone(dict_manager.latest_round_num)

    round_0_metrics = {'a': 2, 'b': 5}
    expected_metrics = collections.OrderedDict([(0, round_0_metrics)])
    dict_manager.save_metrics(round_0_metrics, 0)
    actual_metrics = dict_manager.metrics
    self.assertEqual(actual_metrics, expected_metrics)
    self.assertEqual(dict_manager.latest_round_num, 0)

    round_5_metrics = {'c': 5}
    expected_metrics = collections.OrderedDict([(0, round_0_metrics),
                                                (5, round_5_metrics)])
    dict_manager.save_metrics(round_5_metrics, 5)
    actual_metrics = dict_manager.metrics
    self.assertEqual(actual_metrics, expected_metrics)
    self.assertEqual(dict_manager.latest_round_num, 5)

  def test_get_metrics_returns_copy(self):
    dict_manager = dictionary_manager.DictionaryMetricsManager()
    round_0_metrics = {'a': 2, 'b': 5}
    dict_manager.save_metrics(round_0_metrics, 0)
    dict_manager.metrics[1] = 'foo'
    expected_metrics = collections.OrderedDict([(0, round_0_metrics)])
    self.assertEqual(dict_manager.metrics, expected_metrics)

  def test_save_metrics_raises_if_round_num_is_negative(self):
    dict_manager = dictionary_manager.DictionaryMetricsManager()
    with self.assertRaises(ValueError):
      dict_manager.save_metrics({'a': 1}, -1)

  def test_clear_metrics_raises_if_round_num_is_negative(self):
    dict_manager = dictionary_manager.DictionaryMetricsManager()
    with self.assertRaises(ValueError):
      dict_manager.clear_metrics(round_num=-1)

  def test_clear_metrics_removes_rounds_after_input_arg(self):
    dict_manager = dictionary_manager.DictionaryMetricsManager()
    dict_manager.save_metrics({'a': 1}, 0)
    dict_manager.save_metrics({'b': 2}, 5)
    dict_manager.save_metrics({'c': 3}, 10)
    dict_manager.clear_metrics(round_num=7)
    expected_metrics = collections.OrderedDict([(0, {'a': 1}), (5, {'b': 2})])
    self.assertEqual(dict_manager.metrics, expected_metrics)
    self.assertEqual(dict_manager.latest_round_num, 5)

  def test_clear_metrics_removes_rounds_equal_to_input_arg(self):
    dict_manager = dictionary_manager.DictionaryMetricsManager()
    dict_manager.save_metrics({'a': 1}, 0)
    dict_manager.save_metrics({'b': 2}, 5)
    dict_manager.save_metrics({'c': 3}, 10)
    dict_manager.clear_metrics(round_num=10)
    expected_metrics = collections.OrderedDict([(0, {'a': 1}), (5, {'b': 2})])
    self.assertEqual(dict_manager.metrics, expected_metrics)
    self.assertEqual(dict_manager.latest_round_num, 5)

  def test_clear_all_metrics(self):
    dict_manager = dictionary_manager.DictionaryMetricsManager()
    dict_manager.save_metrics({'a': 1}, 0)
    dict_manager.save_metrics({'b': 2}, 5)
    dict_manager.save_metrics({'c': 3}, 10)
    dict_manager.clear_metrics(round_num=0)
    self.assertEqual(dict_manager.metrics, collections.OrderedDict())
    self.assertIsNone(dict_manager.latest_round_num)


if __name__ == '__main__':
  tf.test.main()
