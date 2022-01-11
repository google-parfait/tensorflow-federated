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

from tensorflow_federated.python.analytics.hierarchical_histogram import build_tree_from_leaf
from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_decoder


def _create_hierarchical_histogram(arity, depth):
  """Utility function to create the hierarchical histogram."""
  histogram = np.arange(arity**(depth - 1))
  hierarchical_histogram = build_tree_from_leaf.create_hierarchical_histogram(
      histogram, arity)
  return tf.ragged.constant(hierarchical_histogram)


def _create_noisy_hierarchical_histogram(arity, depth, scale=1.0):
  """Utility function to create the hierarchical histogram with noise."""
  hist = np.arange(arity**(depth - 1))
  hierarchical_histogram = build_tree_from_leaf.create_hierarchical_histogram(
      hist, arity)
  noisy_hierarchical_histogram = [
      v + np.random.normal(size=(len(v),), scale=scale)
      for v in hierarchical_histogram
  ]
  return tf.ragged.constant(noisy_hierarchical_histogram)


class HierarchicalHistogramDecoderTest(tf.test.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters([
      ('0', True, 2, 2),
      ('1', True, 2, 4),
      ('2', True, 2, 8),
      ('3', True, 3, 2),
      ('4', True, 3, 4),
      ('5', True, 3, 8),
      ('6', False, 2, 2),
      ('7', False, 2, 4),
      ('8', False, 2, 8),
      ('9', False, 3, 2),
      ('10', False, 3, 4),
      ('11', False, 3, 8),
  ])
  def test_check_consistency(self, is_consistent, arity, depth):
    if is_consistent:
      hierarchical_histogram = _create_hierarchical_histogram(arity, depth)
    else:
      hierarchical_histogram = _create_noisy_hierarchical_histogram(
          arity, depth)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram, 0, arity**(depth - 1))
    self.assertEqual(decoder._check_consistency(), is_consistent)

  @parameterized.named_parameters([
      ('0', 2, 2, 0.1),
      ('1', 2, 2, 1.0),
      ('2', 2, 2, 4.0),
      ('3', 2, 3, 0.1),
      ('4', 2, 3, 1.0),
      ('5', 2, 3, 4.0),
      ('6', 3, 2, 0.1),
      ('7', 3, 2, 1.0),
      ('8', 3, 2, 4.0),
      ('9', 3, 3, 0.1),
      ('10', 3, 3, 1.0),
      ('11', 3, 3, 4.0),
  ])
  def test_enforce_consistency(self, arity, depth, scale):
    noisy_hierarchical_histogram = _create_noisy_hierarchical_histogram(
        arity, depth, scale)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        noisy_hierarchical_histogram, 0, arity**(depth - 1))
    decoder.enforce_consistency()

    self.assertTrue(decoder._check_consistency())

  @parameterized.named_parameters(
      ('test_1', tf.ragged.constant([[0., 0.]])),
      ('test_2', tf.ragged.constant([[0., 0.], [0., 0., 0., 0.]])),
      ('test_3', tf.ragged.constant([
          [0.],
          [0., 0.],
          [0., 0., 0., 0., 0.],
      ])),
      ('test_4',
       tf.ragged.constant([
           [0.],
           [0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
       ])),
  )
  def test_init_raises(self, hierarchical_histogram):
    with self.assertRaises(ValueError):
      hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
          hierarchical_histogram, 0, 0)

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
    hierarchical_histogram = _create_hierarchical_histogram(arity, depth)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram,
        0,
        arity**(depth - 1),
        use_efficient=use_efficient)
    node_value = decoder.node_query(layer, index)
    # The test histogram is from 0 to the length of the histogram, so the
    # expected node value is as follows.
    expected_node_value = 0.
    reverse_depth = depth - layer - 1
    expansion_factor = arity**reverse_depth
    for i in range(index * expansion_factor, (index + 1) * expansion_factor):
      expected_node_value += i

    self.assertAllClose(node_value, expected_node_value)

  @parameterized.named_parameters(
      ('test_1', -1, 0),
      ('test_2', 2, 0),
      ('test_3', 0, -1),
      ('test_4', 0, 1),
  )
  def test_node_query_raises(self, layer, index):
    hierarchical_histogram = _create_hierarchical_histogram(arity=2, depth=1)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram, 0, 1)
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
    hierarchical_histogram = _create_hierarchical_histogram(arity, depth)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram,
        0,
        arity**(depth - 1),
        use_efficient=use_efficient)
    range_sum = decoder.range_query(left, right)
    # The test histogram is from 0 to the length of the histogram, so the
    # expected node value is as follows.
    expected_range_sum = (left + right) * (right - left + 1) // 2
    self.assertAllClose(range_sum, expected_range_sum)

  @parameterized.named_parameters(
      ('test_1', -1, 0),
      ('test_2', 0, 2),
      ('test_3', 1, 0),
  )
  def test_range_query_raises(self, left, right):
    hierarchical_histogram = _create_hierarchical_histogram(arity=2, depth=2)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram, 0, 2)
    with self.assertRaises(ValueError):
      decoder.range_query(left, right)

  @parameterized.named_parameters(
      ('binary_0', 2, 0.0, (0, 1)),
      ('binary_0_25', 2, 0.25, (4, 5)),
      ('binary_0_5', 2, 0.5, (5, 6)),
      ('binary_0_75', 2, 0.75, (6, 7)),
      ('binary_1', 2, 1.0, (7, 8)),
      ('ternary_0', 3, 0.0, (0, 1)),
      ('ternary_0_25', 3, 0.25, (13, 14)),
      ('ternary_0_5', 3, 0.5, (19, 20)),
      ('ternary_0_75', 3, 0.75, (23, 24)),
      ('ternary_1', 3, 1.0, (26, 27)),
  )
  def test_quantile_query(self, arity, q, expected_quantile):
    hierarchical_histogram = _create_hierarchical_histogram(arity, depth=4)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram, 0, arity**3)
    quantile = decoder.quantile_query(q)
    self.assertEqual(quantile, expected_quantile)

  @parameterized.named_parameters([
      ('0', -0.1),
      ('1', 1.1),
  ])
  def test_quantile_query_raises_invalid_quantile(self, q):
    hierarchical_histogram = _create_hierarchical_histogram(arity=2, depth=2)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram, 0, 2)
    with self.assertRaises(ValueError):
      decoder.quantile_query(q)

  def test_quantile_query_raises_inconsistent(self):
    hierarchical_histogram = _create_noisy_hierarchical_histogram(
        arity=2, depth=2)
    decoder = hierarchical_histogram_decoder.HierarchicalHistogramDecoder(
        hierarchical_histogram, 0, 2)
    with self.assertRaises(ValueError):
      decoder.quantile_query(0.5)


if __name__ == '__main__':
  tf.test.main()
