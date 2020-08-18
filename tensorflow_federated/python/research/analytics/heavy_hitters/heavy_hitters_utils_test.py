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
