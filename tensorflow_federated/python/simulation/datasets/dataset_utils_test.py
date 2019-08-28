# Lint as: python3
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
"""Tests for tensorflow_federated.python.simulation.datasets.dataset_utils."""

import collections

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import dataset_utils


class DatasetUtilsTest(tf.test.TestCase):

  def test_deterministic_dataset_mixture(self):
    a = tf.data.Dataset.range(5)
    b = tf.data.Dataset.range(5).map(lambda x: x + 5)
    mixture = dataset_utils.build_dataset_mixture(a, b, a_probability=0.5)
    expected_examples = [0, 1, 2, 3, 9]
    actual_examples = [self.evaluate(x) for x in mixture]
    self.assertAllEqual(expected_examples, actual_examples)

  def test_deterministic_dataset_mixture_distribution(self):
    # Create a dataset of infinite fives.
    a = tf.data.Dataset.from_tensor_slices([8]).repeat(None)
    # Create a normal sampling of integers around mean=5
    b = tf.data.Dataset.from_tensor_slices(
        tf.cast(tf.random.normal(shape=[1000], mean=5, stddev=2.0), tf.int32))
    # Create a mixture of 1000 integers (bounded by the size of `b` since `a` is
    # infinite).
    mixture = dataset_utils.build_dataset_mixture(a, b, a_probability=0.8)

    # Count each label. Expect approximately 800 values of '8', then the
    # remaining 200 normally distributed around 5.
    counts = collections.Counter(self.evaluate(x) for x in mixture)
    self.assertEqual(
        {
            8: 804,
            5: 47,
            3: 37,
            4: 36,
            6: 27,
            7: 16,
            2: 14,
            1: 10,
            0: 5,
            9: 3,
            10: 1,
        },
        counts,
        msg=str(counts))

  def test_filter_single_label_dataset(self):
    # Create a uniform sampling of integers in [0, 10).
    d = tf.data.Dataset.from_tensor_slices({
        'label':
            tf.random.uniform(shape=[1000], minval=0, maxval=9, dtype=tf.int32),
    })

    filtered_d = dataset_utils.build_single_label_dataset(
        d, label_key='label', desired_label=6)
    filtered_examples = [self.evaluate(x) for x in filtered_d]
    # Expect close to 1000 / 10  = 100 examples.
    self.assertLen(filtered_examples, 103)
    self.assertTrue(all(x['label'] == 6 for x in filtered_d))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
