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
"""Utilities for hierarchical histogram."""

import math
import numpy as np


def create_hierarchical_histogram(histogram, arity: int):
  """Converts a histogram to its hierarchical representation.

  Args:
    histogram: An array_like representing the input histogram.
    arity: The branching factor of the hierarchical histogram.

  Returns:
    A list of 1-D lists. Each inner list represents one layer of the
    hierarchical histogram.
  """
  if arity < 2:
    raise ValueError(f'Arity should be at least 2.arity={arity} is given.')

  depth = math.ceil(math.log(len(histogram), arity)) + 1
  size_ = arity ** (depth - 1)
  histogram = np.pad(
      histogram, (0, size_ - len(histogram)), 'constant', constant_values=(0, 0)
  ).tolist()

  def _shrink_histogram(histogram):
    return np.sum((np.reshape(histogram, (-1, arity))), axis=1).tolist()

  hierarchical_histogram = [histogram]
  for _ in range(depth - 1):
    hierarchical_histogram = [
        _shrink_histogram(hierarchical_histogram[0])
    ] + hierarchical_histogram

  return hierarchical_histogram
