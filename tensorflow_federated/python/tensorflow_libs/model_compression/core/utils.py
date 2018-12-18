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
"""Utilities for implementation of the `model_compression` module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six
import tensorflow as tf

from tensorflow.python.framework import tensor_util


def static_or_dynamic_shape(tensor):
  """Returns shape of the input `Tensor`.

  If the shape is statically known, it returns a Python object. Otherwise,
  returns result of `tf.shape(tensor)`.

  Args:
    tensor: A `Tensor`.

  Returns:
    Static or dynamic shape of `tensor`.

  Raises:
    TypeError:
      If the input is not a `Tensor`.
  """
  if not tensor_util.is_tensor(tensor):
    raise TypeError('The provided input is not a Tensor.')
  return tensor.shape if tensor.shape.is_fully_defined() else tf.shape(tensor)


def split_dict_py_tf(dictionary):
  """Splits dictionary based on Python and TensorFlow values.

  Args:
    dictionary: An arbitrary `dict`. Any `dict` objects in values will be
      processed recursively.

  Returns:
    A tuple `(d_py, d_tf)`, where
    d_py: A `dict` of the same structure as `dictionary`, with TensorFlow values
      replaced by `None`, recursively.
    d_tf: A `dict` of the same structure as `dictionary`, with non-TensorFlow
      values replaced by `None`, recursively.

  Raises:
    TypeError:
      If the input is not a `dict` object.
  """
  if not isinstance(dictionary, dict):
    raise TypeError
  d_py, d_tf = {}, {}
  for k, v in six.iteritems(dictionary):
    if isinstance(v, dict):
      d_py[k], d_tf[k] = split_dict_py_tf(v)
    else:
      if tensor_util.is_tensor(v):
        d_py[k], d_tf[k] = None, v
      else:
        d_py[k], d_tf[k] = v, None
  return d_py, d_tf


def merge_same_structure_dicts(dict1, dict2):
  """Merges dictionaries of the same structure.

  This method merges two dictionaries of the same structure, of which each
  element is `None` in exactly one the dictionaries.

  This method is mainly to be used together with the `split_dict_py_tf` method.

  Args:
    dict1: An arbitrary `dict`. Any `dict` objects in values will be processed
      recursively.
    dict2: A `dict`, with the same structure as `dict1`.

  Returns:
    A `dict` of the same structure as the inputs, with merged values.

  Raises:
    TypeError:
      If either of the input arguments is not a dictionary.
    ValueError:
      If the input dictionaries do not have the same structure, or if a certain
      value is set (not equal to `None`) in both of the dictionaries.
  """
  merged_dict = {}
  if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
    raise TypeError
  if len(dict1) != len(dict2):
    raise ValueError('Dictionaries must have the same structure.')

  if {(type(k), k) for k in dict1} != {(type(k), k) for k in dict2}:
    raise ValueError('Dictionaries must have the same structure.')
  for k in dict1:
    v1, v2 = dict1[k], dict2[k]
    if isinstance(v1, dict) != isinstance(v2, dict):
      raise ValueError('Dictionaries must have the same structure.')

    if isinstance(v1, dict):
      merged_dict[k] = merge_same_structure_dicts(v1, v2)
    elif v1 is None:
      merged_dict[k] = v2  # The value v2 could also be None.
    elif v2 is None:
      merged_dict[k] = v1
    else:
      raise ValueError('At least one value must be None.')

  return merged_dict
