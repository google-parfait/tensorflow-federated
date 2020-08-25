# Copyright 2020, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.research.analytics.heavy_hitters import heavy_hitters_utils as hh_utils


class HeavyHittersUtilsTest(tf.test.TestCase):

  def test_top_k(self):
    signal = ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'e']
    signal = dict(collections.Counter(signal))
    self.assertEqual(hh_utils.top_k(signal, 1), {'a': 3})
    self.assertEqual(hh_utils.top_k(signal, 2), {'a': 3, 'b': 2})

  def test_precision(self):
    signal = {'a': 3, 'b': 2, 'c': 1, 'd': 0}

    ground_truth = {'a': 3, 'b': 2, 'c': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.precision(ground_truth, signal, 2), 1.0)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.precision(ground_truth, signal, 2), 0.5)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.precision(ground_truth, signal, 3), 1.0)

    ground_truth = {'a': 3, 'd': 2, 'b': 2, 'c': 2}
    self.assertAlmostEqual(hh_utils.precision(ground_truth, signal, 3), 1.0)

  def test_recall(self):
    signal = {'a': 3, 'b': 2, 'c': 1, 'd': 0}

    ground_truth = {'a': 3, 'b': 2, 'c': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.recall(ground_truth, signal, 2), 1.0)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.recall(ground_truth, signal, 2), 0.5)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.recall(ground_truth, signal, 3), 1.0)

    ground_truth = {'a': 3, 'd': 2, 'b': 2, 'c': 2}
    self.assertAlmostEqual(hh_utils.recall(ground_truth, signal, 3), 1.0)

  def test_f1_score(self):
    signal = {'a': 3, 'b': 2, 'c': 1, 'd': 0}

    ground_truth = {'a': 3, 'b': 2, 'c': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.f1_score(ground_truth, signal, 2), 1.0)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.f1_score(ground_truth, signal, 2), 0.5)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.f1_score(ground_truth, signal, 3), 1.0)

    ground_truth = {'a': 3, 'd': 2, 'b': 2, 'c': 2}
    self.assertAlmostEqual(hh_utils.f1_score(ground_truth, signal, 3), 1.0)

  def test_compute_threshold_leakage(self):
    # The counts of discovered heavy hitters do not affect the results.
    signal = {'a': 5, 'b': 5, 'c': 5, 'd': 5, 'e': 5}
    ground_truth = {'a': 7, 'b': 6, 'c': 4, 'd': 4, 'e': 3, 'f': 2, 'g': 1}

    threshold = 0
    expected_false_positive_rate = {}
    expected_false_discovery_rate = {}
    expected_harmonic_mean_fpr_fdr = {}
    false_positive_rate, false_discovery_rate, harmonic_mean_fpr_fdr = hh_utils.compute_threshold_leakage(
        ground_truth, signal, threshold)
    self.assertEqual(false_positive_rate, expected_false_positive_rate)
    self.assertEqual(false_discovery_rate, expected_false_discovery_rate)
    self.assertEqual(harmonic_mean_fpr_fdr, expected_harmonic_mean_fpr_fdr)

    threshold = 10
    expected_false_positive_rate = {
        10: 5.0 / 7,
        9: 5.0 / 7,
        8: 5.0 / 7,
        7: 4.0 / 6,
        6: 3.0 / 5,
        5: 3.0 / 5,
        4: 1.0 / 3,
        3: 0.0,
        2: 0.0,
        1: 0.0
    }
    expected_false_discovery_rate = {
        10: 5.0 / 5,
        9: 5.0 / 5,
        8: 5.0 / 5,
        7: 4.0 / 5,
        6: 3.0 / 5,
        5: 3.0 / 5,
        4: 1.0 / 5,
        3: 0.0,
        2: 0.0,
        1: 0.0
    }
    expected_harmonic_mean_fpr_fdr = {
        10: 2 * (5.0 / 5) * (5.0 / 7) / (5.0 / 5 + 5.0 / 7),
        9: 2 * (5.0 / 5) * (5.0 / 7) / (5.0 / 5 + 5.0 / 7),
        8: 2 * (5.0 / 5) * (5.0 / 7) / (5.0 / 5 + 5.0 / 7),
        7: 2 * (4.0 / 5) * (4.0 / 6) / (4.0 / 5 + 4.0 / 6),
        6: 2 * (3.0 / 5) * (3.0 / 5) / (3.0 / 5 + 3.0 / 5),
        5: 2 * (3.0 / 5) * (3.0 / 5) / (3.0 / 5 + 3.0 / 5),
        4: 2 * (1.0 / 5) * (1.0 / 3) / (1.0 / 5 + 1.0 / 3),
        3: 0.0,
        2: 0.0,
        1: 0.0
    }

    false_positive_rate, false_discovery_rate, harmonic_mean_fpr_fdr = hh_utils.compute_threshold_leakage(
        ground_truth, signal, threshold)
    self.assertEqual(false_positive_rate, expected_false_positive_rate)
    self.assertEqual(false_discovery_rate, expected_false_discovery_rate)
    for threshold in range(1, 11):
      self.assertAlmostEqual(harmonic_mean_fpr_fdr[threshold],
                             expected_harmonic_mean_fpr_fdr[threshold])

    threshold = 5
    signal = {}
    ground_truth = {'a': 7, 'b': 6, 'c': 4, 'd': 4, 'e': 3, 'f': 2, 'g': 1}
    expected_false_positive_rate = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    expected_false_discovery_rate = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    expected_harmonic_mean_fpr_fdr = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    false_positive_rate, false_discovery_rate, harmonic_mean_fpr_fdr = hh_utils.compute_threshold_leakage(
        ground_truth, signal, threshold)
    self.assertEqual(false_positive_rate, expected_false_positive_rate)
    self.assertEqual(false_discovery_rate, expected_false_discovery_rate)
    for threshold in range(1, 5):
      self.assertAlmostEqual(harmonic_mean_fpr_fdr[threshold],
                             expected_harmonic_mean_fpr_fdr[threshold])


class GetTopElementsTest(tf.test.TestCase):

  def test_empty_dataset(self):
    ds = tf.data.Dataset.from_tensor_slices([])
    top_elements = hh_utils.get_top_elements(ds, max_user_contribution=10)
    self.assertEmpty(top_elements)

  def test_under_max_contribution(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    top_elements = hh_utils.get_top_elements(ds, max_user_contribution=10)
    self.assertCountEqual(top_elements.numpy(), [b'a', b'b', b'c'])

  def test_over_max_contribution(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'c', 'b', 'c', 'c'])
    top_elements = hh_utils.get_top_elements(ds, max_user_contribution=2)
    self.assertCountEqual(top_elements.numpy(), [b'a', b'c'])


if __name__ == '__main__':
  tf.test.main()
