# Copyright 2018, The TensorFlow Federated Authors.
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
"""Utilities for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(b/118783928) Fix BUILD target visibility.
from tensorflow.python.util import nest


def assert_nested_struct_eq(x, y):
  """Asserts that nested structures 'x' and 'y' are the same.

  Args:
    x: One nested structure.
    y: Another nested structure.

  Raises:
    ValueError: if the structures are not the same.
  """
  nest.assert_same_structure(x, y)
  xl = nest.flatten(x)
  yl = nest.flatten(y)
  if len(xl) != len(yl):
    raise ValueError('The sizes of structures {} and {} mismatch.'.format(
        str(len(xl)), str(len(yl))))
  for xe, ye in zip(xl, yl):
    if xe != ye:
      raise ValueError('Mismatching elements {} and {}.'.format(
          str(xe), str(ye)))
