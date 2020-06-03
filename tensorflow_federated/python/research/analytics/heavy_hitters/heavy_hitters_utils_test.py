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

import tensorflow as tf

from tensorflow_federated.python.research.analytics.heavy_hitters import heavy_hitters_utils as hh_utils


class HeavyHittersUtilsTest(tf.test.TestCase):

  def test_top_k(self):
    signal = ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'e']
    self.assertEqual(hh_utils.top_k(signal, 1), {'a': 3})
    self.assertEqual(hh_utils.top_k(signal, 2), {'a': 3, 'b': 2, 'c': 2})

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
    self.assertAlmostEqual(hh_utils.recall(ground_truth, signal, 3), 0.75)

  def test_f1_score(self):
    signal = {'a': 3, 'b': 2, 'c': 1, 'd': 0}

    ground_truth = {'a': 3, 'b': 2, 'c': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.f1_score(ground_truth, signal, 2), 1.0)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.f1_score(ground_truth, signal, 2), 0.5)

    ground_truth = {'a': 3, 'c': 2, 'b': 1, 'd': 0}
    self.assertAlmostEqual(hh_utils.f1_score(ground_truth, signal, 3), 1.0)

    ground_truth = {'a': 3, 'd': 2, 'b': 2, 'c': 2}
    self.assertAlmostEqual(
        hh_utils.f1_score(ground_truth, signal, 3), 0.85714285)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
