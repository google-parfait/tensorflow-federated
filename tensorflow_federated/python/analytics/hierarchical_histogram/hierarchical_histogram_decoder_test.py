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
"""Tests for hierarchical_histogram_decoder."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_decoder as hi_hist_decoder


def _create_hierarchical_histogram(arity, depth):
  """Utility function to create the hierarchical histogram."""
  hist = np.arange(arity**(depth - 1))
  hi_hist = hi_hist_decoder._create_hierarchical_histogram(hist, arity, depth)
  return tf.ragged.constant(hi_hist)


class HierarchicalHistogramDecoderTest(tf.test.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters(
      ('binary_4_layer_ne', 2, 4, False, 1, 1),
      ('binary_8_layer_ne', 2, 8, False, 2, 2),
      ('binary_4_layer_e', 2, 4, True, 1, 1),
      ('binary_8_layer_e', 2, 8, True, 2, 2),
      ('ternary_4_layer_ne', 3, 4, False, 1, 2),
      ('ternary_8_layer_ne', 3, 8, False, 2, 4),
      ('ternary_4_layer_e', 3, 4, True, 1, 2),
      ('ternary_8_layer_e', 3, 8, True, 2, 4),
  )
  def test_node_query(self, arity, depth, use_efficient, layer, index):
    hi_hist = _create_hierarchical_histogram(arity, depth)
    decoder = hi_hist_decoder.HierarchicalHist(
        hi_hist, arity=arity, use_efficient=use_efficient)
    node_value = decoder.node_query(layer, index)

    self.assertAllClose(node_value, hi_hist[layer][index])

  @parameterized.named_parameters(
      ('test_1', -1, 0),
      ('test_2', 2, 0),
      ('test_3', 0, -1),
      ('test_4', 0, 1),
  )
  def test_node_query_raises(self, layer, index):
    hi_hist = _create_hierarchical_histogram(arity=2, depth=1)
    decoder = hi_hist_decoder.HierarchicalHist(hi_hist, arity=2)
    with self.assertRaises(ValueError):
      decoder.node_query(layer, index)

  @parameterized.named_parameters(
      ('binary_4_layer_ne_1_3', 2, 4, False, 1, 3),
      ('binary_8_layer_ne_0_7', 2, 8, False, 0, 7),
      ('binary_4_layer_e_1_3', 2, 4, True, 1, 3),
      ('binary_8_layer_e_0_7', 2, 8, True, 0, 7),
      ('ternary_4_layer_ne_1_3', 3, 4, False, 1, 3),
      ('ternary_8_layer_ne_0_7', 3, 8, False, 0, 7),
      ('ternary_4_layer_e_1_3', 3, 4, True, 1, 3),
      ('ternary_4_layer_e_0_7', 3, 8, True, 0, 7),
  )
  def test_range_query(self, arity, depth, use_efficient, left, right):
    hi_hist = _create_hierarchical_histogram(arity, depth)
    decoder = hi_hist_decoder.HierarchicalHist(
        hi_hist, arity=arity, use_efficient=use_efficient)
    range_sum = decoder.range_query(left, right)
    reference_range_sum = (left + right) * (right - left + 1) // 2
    self.assertAllClose(range_sum, reference_range_sum)

  @parameterized.named_parameters(
      ('test_1', -1, 0),
      ('test_2', 0, 2),
      ('test_3', 1, 0),
  )
  def test_range_query_raises(self, left, right):
    hi_hist = _create_hierarchical_histogram(arity=2, depth=2)
    decoder = hi_hist_decoder.HierarchicalHist(hi_hist, arity=2)
    with self.assertRaises(ValueError):
      decoder.range_query(left, right)


if __name__ == '__main__':
  tf.test.main()
