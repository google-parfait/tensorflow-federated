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
"""Tests for utils."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import build_tree_from_leaf


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('binary_no_padding', [1., 1., 1., 1.], 2, [[4.], [2., 2.],
                                                  [1., 1., 1., 1.]]),
      ('binary_padding', [1., 1., 1.], 2, [[3.], [2., 1.], [1., 1., 1., 0.]]),
      ('ternary_no_padding', [1., 1., 1.], 3, [[3.], [1., 1., 1.]]),
      ('ternary_padding', [1., 1.], 3, [[2.], [1., 1., 0.]]),
  )
  def test_create_hierarchical_hist(self, histogram, arity,
                                    expected_hierarchical_histogram):
    hierarchical_histogram = build_tree_from_leaf.create_hierarchical_histogram(
        histogram, arity)
    self.assertAllClose(hierarchical_histogram, expected_hierarchical_histogram)


if __name__ == '__main__':
  tf.test.main()
