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
"""Decoder class for hierarchical histograms."""

import math

import numpy as np
import tensorflow as tf


def _create_hierarchical_histogram(hist, arity, depth=None):
  """Utility function to create the hierarchical histogram."""
  if depth is None:
    depth = math.ceil(math.log(len(hist), arity)) + 1

  def _shrink_hist(hist):
    return np.sum((np.reshape(hist, (-1, arity))), axis=1)

  hi_hist = [hist]
  for _ in range(depth - 1, 0, -1):
    hi_hist = [_shrink_hist(hi_hist[0])] + hi_hist

  return hi_hist


class HierarchicalHist():
  """Hierarchical histogram decoder executing range queries and other statistics on a hierarchical histogram."""

  def __init__(self,
               hi_hist: tf.RaggedTensor,
               arity: int = 2,
               use_efficient: bool = False):
    """Initializer for `HierarchicalHist`.

    Args:
      hi_hist: a `tf.RaggedTensor` for the hierarchical histogram.
      arity: the arity of the hierarchical histogram, i.e. how many bins in the
        (i+1)th layer sum up to one bin in the ith layer.
      use_efficient: boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "'Efficient Use of
        Differentially Private Binary Trees'. James Honaker".
    """

    self._size = len(hi_hist[-1])
    self._arity = arity
    self._num_layers = math.ceil(math.log(self._size, arity)) + 1
    self._hi_hist = hi_hist.to_list()
    self._use_efficient = use_efficient

  def _left_right_most_leaf(self, layer, index):
    """Utility function returning the leftmost and rightmost leaves from a node.

    Return a tuple of the leftmost and rightmost leaves within the subtree of
    the node indexed by (layer, index).

    Args:
      layer: A `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: A `int` for the inner_layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      `Tuple[int, int]`. The first element being the leftmost leaf. The second
      element being the rightmost leaf.
    """
    reverse_depth = self._num_layers - 1 - layer
    left_most_child = index * (self._arity**reverse_depth)
    right_most_child = (index + 1) * (self._arity**reverse_depth) - 1
    return left_most_child, right_most_child

  def _check_index(self, layer: int, index: int):
    """Utility function checking whether a node index is legal.

    Args:
      layer: A `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: A `int` for the inner_layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      `None`

    Raises:
      `ValueError` if (layer, index) does not point to an existing node.
    """
    if layer < 0 or layer >= self._num_layers:
      raise ValueError(f'Cross-layer index {layer} is out of legal range. '
                       f'Expected to be within [0, {self._num_layers}).')
    if index < 0 or index >= self._arity**layer:
      raise ValueError(f'Inner-layer index {index} is out of legal range. '
                       f'Expected to be within [0, {self._arity**layer}).')

  def node_query(self, layer: int, index: int):
    """Query the value of a bin in the hierarchical histogram.

    Args:
      layer: `int`, the cross-layer index of the bin.
      index: `int`, the inside-layer index of the bin.

    Returns:
      `float`, the value of the bin indexed by (layer, index).

    Raises:
      `ValueError` if (layer, index) does not point to an existing node.
    """

    self._check_index(layer, index)

    if self._use_efficient:

      def from_below(layer, index):
        if layer == self._num_layers - 1:
          return self._hi_hist[layer][index]
        else:
          node_value = self._hi_hist[layer][index]
          below_value = 0
          for child_index in range(index * self._arity,
                                   (index + 1) * self._arity):
            below_value += from_below(layer + 1, child_index)
          weight = self._arity / (self._arity + 1)
          return weight * node_value + (1 - weight) * below_value

      def from_above(layer, index):
        if layer == 0:
          return self._hi_hist[layer][index]
        else:
          node_value = self._hi_hist[layer][index]
          parent_index = index // self._arity
          above_value = from_above(layer - 1, parent_index)
          for sibling_index in range(parent_index * self._arity,
                                     (parent_index + 1) * self._arity):
            if sibling_index != index:
              above_value -= from_below(layer, sibling_index)

          weight = 1 / (3 - math.pow(self._arity, layer - self._num_layers + 1))
          return weight * node_value + (1 - weight) * above_value

      below_value = from_below(layer, index)
      if layer == 0:
        return below_value
      parent_index = index // self._arity
      above_value = from_above(layer - 1, parent_index)
      for sibling_index in range(parent_index * self._arity,
                                 (parent_index + 1) * self._arity):
        if sibling_index != index:
          above_value -= from_below(layer, sibling_index)
      below_variance = 1 / (2 - (self._arity**(layer - self._num_layers + 1)))
      above_variance = 1 / (3 - (self._arity**(layer - self._num_layers + 1)))
      weight = (1 / below_variance) / ((1 / below_variance) +
                                       (1 / (below_variance + above_variance)))
      return weight * below_value + (1 - weight) * above_value
    else:
      return self._hi_hist[layer][index]

  def _range_query(self, left: int, right: int, layer: int, index: int):
    """Recursively query the cumulative value within a range [left, right].

    Args:
      left: `int`, the inclusive left end of the range query.
      right: `int`, the inclusive right end of the range query.
      layer: `int`, the cross-layer index for the starting point of the search.
      index: `int`, the inside-layer index for the starting point of the search.

    Returns:
      `float`, the sum of the bins within the range of [left, right] starting
      from bin (layer, index).
    """

    left_most_leaf, right_most_leaf = self._left_right_most_leaf(layer, index)

    if left > right_most_leaf or right < left_most_leaf or left > right:
      return 0

    elif left <= left_most_leaf and right >= right_most_leaf:
      return self.node_query(layer, index)

    else:
      range_sum = 0
      for child_index in range(index * self._arity, (index + 1) * self._arity):
        interval_left, interval_right = self._left_right_most_leaf(
            layer + 1, child_index)
        interval_left = max(left, interval_left)
        interval_right = min(right, interval_right)
        range_sum += self._range_query(interval_left, interval_right, layer + 1,
                                       child_index)
      return range_sum

  def range_query(self, left: int, right: int):
    """Range query in a hierarchical historgram.

    Implemented as a wrapper around _range_query starting from the root.

    Args:
      left: `int`, the inclusive left end of the range query.
      right: `int`, the inclusive right end of the range query.

    Returns:
      `float`, the sum of the bins within the range of [left, right].
    """

    if left < 0 or right >= self._size:
      raise ValueError(f'[{left}, {right}] is outside the legal range '
                       f'[0, {self._size}).')

    if left > right:
      raise ValueError(f'left {left} is expected to be less than or equal to '
                       f'right {right}.')

    return self._range_query(left, right, 0, 0)
