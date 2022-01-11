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
"""Decoder class for hierarchical histograms."""

import math
from typing import Tuple

import numpy as np

import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import build_tree_from_leaf

# TODO(b/194028367): Re-organize the below class into several functions.

# TODO(b/193976902): Add lazy tag to the hierarchical histogram when it is not
# consistent and `use_efficient` is `True` to improve amortized performance on
# multiple queries.


def _check_hierarchical_histogram_shape(hierarchical_histogram):
  """Checks whether the input hierarchical histogram is valid."""
  if len(hierarchical_histogram) < 1:
    raise ValueError('Input hierarchical histogram is invalid.')
  if len(hierarchical_histogram[0]) != 1:
    raise ValueError('Input hierarchical histogram is invalid.')
  depth = len(hierarchical_histogram)
  if depth == 1:
    return
  arity = len(hierarchical_histogram[1]) / len(hierarchical_histogram[0])
  for layer in range(1, depth - 1):
    if len(hierarchical_histogram[layer + 1]) % len(
        hierarchical_histogram[layer]) != 0:
      raise ValueError('Input hierarchical histogram is invalid.')
    elif len(hierarchical_histogram[layer + 1]) / len(
        hierarchical_histogram[layer]) != arity:
      raise ValueError('Input hierarchical histogram is invalid.')


class HierarchicalHistogramDecoder():
  """Hierarchical histogram decoder.

  Decodes the output hierarchical histogram of the function returned by
  `build_central_hierarchical_histogram_computation` to answer node queries /
  range queries / other statistics. A hierarchical histogram is a
  `tf.RaggedTensor`, `tree`, in which the jth node in the ith layer can be
  accessed via directly indexing `tree[i][j]`. When the hierarchical histogram
  is differentially private, provides optional optimization tricks to improve
  query accuracy.
  """

  def __init__(self,
               hierarchical_histogram: tf.RaggedTensor,
               lower_bound: float,
               upper_bound: float,
               use_efficient: bool = False):
    """Initializer for `HierarchicalHistogramDecoder`.

    `use_efficient` decides whether to use the accuracy optimization trick from
    the paper ["Efficient Use of Differentially Private Binary Trees. James
    Honaker".](https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf)
    for differentially private hierarchical histogram. The optimization trick
    leverages redudant information in the hierarchical histogram to optimize the
    accuracy of node queries.

    Args:
      hierarchical_histogram: A `tf.RaggedTensor` for the hierarchical
        histogram.
      lower_bound: A `float` representing the lower bound of the hierarchical
        histogram data.
      upper_bound: A `float` representing the upper bound of the hierarchical
        histogram data.
      use_efficient: A boolean indicating the usage of the efficient tree
        aggregation algorithm.
    """

    self._hierarchical_histogram = hierarchical_histogram.to_list()
    self._lower_bound = lower_bound
    self._upper_bound = upper_bound
    _check_hierarchical_histogram_shape(self._hierarchical_histogram)
    if len(self._hierarchical_histogram) == 1:
      self._arity = 2
    else:
      self._arity = int(
          len(self._hierarchical_histogram[1]) /
          len(self._hierarchical_histogram[0]))
    self._size = len(hierarchical_histogram[-1])
    self._num_layers = math.ceil(math.log(self._size, self._arity)) + 1
    self._use_efficient = use_efficient

  def _right_most_leaf(self, layer: int, index: int) -> int:
    """Returns the rightmost leaf from a node.

    Args:
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inner_layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      An `int` representing the the rightmost leaf within the subtree of the
      node indexed by (layer, index).
    """
    reverse_depth = self._num_layers - 1 - layer
    right_most_child = int((index + 1) * (self._arity**reverse_depth) - 1)
    return right_most_child

  def _left_right_most_leaf(self, layer: int, index: int) -> Tuple[int, int]:
    """Returns the leftmost and rightmost leaves from a node.

    Returns a tuple of the leftmost and rightmost leaves within the subtree of
    the node indexed by (layer, index).

    Args:
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inner-layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      A tuple of two integers, representing the leftmost and rightmost leaves,
      respectively.
    """
    reverse_depth = self._num_layers - 1 - layer
    left_most_child = int(index * (self._arity**reverse_depth))
    right_most_child = int((index + 1) * (self._arity**reverse_depth) - 1)
    return left_most_child, right_most_child

  def _flatten_hierarhical_hist(self):
    """Flattens the hierarchical histogram."""
    flattened_hh = []
    for layer in range(self._num_layers):
      flattened_hh.extend(self._hierarchical_histogram[layer])
    return np.array(flattened_hh)

  def _construct_matrix(self):
    """Constructs the matrix for the least square optimization."""
    row_size = (self._arity**self._num_layers - 1) // (self._arity - 1)
    col_size = self._arity**(self._num_layers - 1)
    matrix = np.zeros([row_size, col_size])

    row_index = 0
    for layer in range(self._num_layers):
      node_num = self._arity**layer
      for index in range(0, node_num):
        left, right = self._left_right_most_leaf(layer, index)
        for col_index in range(left, right + 1):
          matrix[row_index][col_index] = 1
        row_index += 1
    return matrix

  def _check_consistency(self) -> bool:
    """Checks whether the hierarchical histogram is consistent."""
    for layer in range(self._num_layers - 1):
      for index in range(self._arity**layer):
        parent = self._hierarchical_histogram[layer][index]
        children_sum = 0.
        for child in range(index * self._arity, (index + 1) * self._arity):
          children_sum += self._hierarchical_histogram[layer + 1][child]
        if not np.isclose(parent, children_sum):
          return False
    return True

  def enforce_consistency(self):
    """Make the tree consistent by solving a least square optimization problem.

    See 'Answering Range Queries Under Local Differential Privacy. Graham
    Cormode, Tejas Kulkarni, Divesh Srivastava' for details. Improve the
    accuracy of range queries. When this function is invoked, `use_efficient`
    will be automatically set to `False` because using both optimizations does
    not further improve the accuracy.
    """

    # As consistency enforcement and Honaker trick together does not further
    # improve the query accuracy. Honaker trick will be automatically disabled
    # if this function is called.
    self._use_efficient = False

    if self._check_consistency():
      return

    ls_matrix = self._construct_matrix()
    ls_rhs = self._flatten_hierarhical_hist()
    consistent_hist = np.linalg.lstsq(ls_matrix, ls_rhs)[0]
    self._hierarchical_histogram = build_tree_from_leaf.create_hierarchical_histogram(
        consistent_hist, self._arity)

  def _check_index(self, layer: int, index: int):
    """Checks whether a node index is legal.

    Args:
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inner-layer index of a node (i.e. index^th node in
        the layer).

    Raises:
      `ValueError` if (layer, index) does not point to an existing node.
    """
    if layer < 0 or layer >= self._num_layers:
      raise ValueError(f'Cross-layer index {layer} is out of valid range. '
                       f'Expected to be within [0, {self._num_layers}).')
    if index < 0 or index >= self._arity**layer:
      raise ValueError(f'Inner-layer index {index} is out of valid range. '
                       f'Expected to be within [0, {self._arity**layer}).')

  def _from_below(self, layer: int, index: int) -> float:
    """Returns an estimate of a node value from its subtree.

    Returns a more accurate estimation of a node using information from its
    subtree. Used if `use_efficient=True`. See "'Efficient Use of
    Differentially Private Binary Trees'. James Honaker". for more details.

    Args:
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inner-layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      A `float` representing the estimation of the value of the bin indexed by
      (layer, index) using information from its subtree.
    """
    if layer == self._num_layers - 1:
      return self._hierarchical_histogram[layer][index]
    else:
      node_value = self._hierarchical_histogram[layer][index]
      below_value = 0
      for child_index in range(index * self._arity, (index + 1) * self._arity):
        below_value += self._from_below(layer + 1, child_index)
      weight = self._arity / (self._arity + 1)
      return weight * node_value + (1 - weight) * below_value

  def _from_above(self, layer: int, index: int) -> float:
    """Returns an estimate of a node value from nodes not in its subtree.

    Returns a more accurate estimation of a node using information from the
    nodes not in its subtree. Used if `use_efficient=True`.  See "'Efficient Use
    of Differentially Private Binary Trees'. James Honaker". for more details.

    Args:
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inner-layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      A `float` representing the estimation of the value of the bin indexed by
      (layer, index) using information from the nodes not in its subtree.
    """
    if layer == 0:
      return self._hierarchical_histogram[layer][index]
    else:
      node_value = self._hierarchical_histogram[layer][index]
      parent_index = index // self._arity
      above_value = self._from_above(layer - 1, parent_index)
      for sibling_index in range(parent_index * self._arity,
                                 (parent_index + 1) * self._arity):
        if sibling_index != index:
          above_value -= self._from_below(layer, sibling_index)

      weight = 1 / (3 - math.pow(self._arity, layer - self._num_layers + 1))
      return weight * node_value + (1 - weight) * above_value

  def node_query(self, layer: int, index: int) -> float:
    """Queries the value of a bin in the hierarchical histogram.

    Args:
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inne-layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      A `float` representing the estimation of the value of the bin indexed by
      (layer, index).

    Raises:
      `ValueError` if (layer, index) does not point to an existing node.
    """

    self._check_index(layer, index)

    if not self._use_efficient:
      return self._hierarchical_histogram[layer][index]
    else:
      below_value = self._from_below(layer, index)
      if layer == 0:
        return below_value
      parent_index = index // self._arity
      above_value = self._from_above(layer - 1, parent_index)
      for sibling_index in range(parent_index * self._arity,
                                 (parent_index + 1) * self._arity):
        if sibling_index != index:
          above_value -= self._from_below(layer, sibling_index)
      below_variance = 1 / (2 - (self._arity**(layer - self._num_layers + 1)))
      above_variance = 1 / (3 - (self._arity**(layer - self._num_layers + 1)))
      weight = (1 / below_variance) / ((1 / below_variance) +
                                       (1 / (below_variance + above_variance)))
      return weight * below_value + (1 - weight) * above_value

  def _range_query(self, left: int, right: int, layer: int,
                   index: int) -> float:
    """Recursively query the cumulative value within a range [left, right].

    Args:
      left: An `int` representing the inclusive left end of the range query.
      right: An `int` representing the inclusive right end of the range query.
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inner-layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      A `float` representing the sum of the bins within the range of [left,
      right] in the subtree of bin (layer, index).
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

  def range_query(self, left: int, right: int) -> float:
    """Returns a range query of a hierarchical historgram.

    Args:
      left: An `int` representing the inclusive left end of the range query.
      right: An `int` representing the inclusive right end of the range query.

    Returns:
      An `float` representing the sum of the bins within the range of [left,
      right].
    """

    if left < 0 or right >= self._size:
      raise ValueError(f'[{left}, {right}] is outside the valid range '
                       f'[0, {self._size}).')

    if left > right:
      raise ValueError(f'left {left} is expected to be less than or equal to '
                       f'right {right}.')

    return self._range_query(left, right, 0, 0)

  def _quantile_query(self, expected_weight: float, layer: int,
                      index: int) -> int:
    """Recursively queries the q-quantile in an ordinal histogram.

    Args:
      expected_weight: A `float` specifying the expected weight before the
        q-quantile.
      layer: An `int` for the cross-layer index of a node (i.e. which layer the
        node is in).
      index: An `int` for the inner-layer index of a node (i.e. index^th node in
        the layer).

    Returns:
      An `int` representing the index of the leftmost leaf such that the sum of
      the bins before it is at least `expected_weight`.
    """

    if layer == self._num_layers - 1:
      return index

    # TODO(b/193976080): make this a binary search to improve performance when
    # arity is large.
    exclusive_left_children_sum = 0.
    for child_index in range(index * self._arity, (index + 1) * self._arity):
      left_children_sum = exclusive_left_children_sum + self.node_query(
          layer + 1, child_index)
      if left_children_sum == expected_weight:
        quantile = self._right_most_leaf(layer + 1, child_index)
        break
      elif left_children_sum > expected_weight:
        quantile = self._quantile_query(
            expected_weight - exclusive_left_children_sum, layer + 1,
            child_index)
        break
      exclusive_left_children_sum = left_children_sum

    return quantile

  def quantile_query(self, q) -> Tuple[float, float]:
    """Queries the q-quantile in a hierarchical historgram.

    Args:
      q: A `float` specifying the wanted q-quantile.

    Returns:
      A `tuple` representing the range of values within the bin such that the
      sum of bins before it accounts for at least `q` proportion of the total
      sum.


    Raises:
      `ValueError` if the hierarchical histogram is not consistent (e.g. with DP
      noise) or `q` is not within [0, 1].
    """

    if not self._check_consistency():
      raise ValueError(
          'Quantile query can only be run on consistent hierarchical histogram.'
          ' Call `enforce_consistency` before running `quantile_query`.')

    if not 0 <= q <= 1:
      raise ValueError(f'`q={q}` is expected to be within [0, 1].')

    total_weight = self.range_query(0, self._size - 1)
    bin_index = self._quantile_query(q * total_weight, 0, 0)
    bin_size = (self._upper_bound - self._lower_bound) / self._size
    bin_lower_bound = self._lower_bound + bin_index * bin_size
    return bin_lower_bound, bin_lower_bound + bin_size
