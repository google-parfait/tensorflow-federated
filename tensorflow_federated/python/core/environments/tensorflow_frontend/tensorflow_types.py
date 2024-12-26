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
"""Defines functions and classes for building and manipulating TFF types."""

import federated_language
import numpy as np
import tensorflow as tf
import tree


def _tensorflow_dtype_to_numpy_dtype(
    dtype: tf.dtypes.DType,
) -> type[np.generic]:
  """Returns a numpy dtype for the `dtype`."""
  # TensorFlow converts a dtype of `tf.string` into a dtype of `np.object_`.
  # However, this is not a valid dtype for TFF, instead use `np.str_`.
  if dtype.base_dtype == tf.string:
    dtype = np.str_
  else:
    dtype = dtype.base_dtype.as_numpy_dtype
  return dtype


def _tensor_shape_to_array_shape(
    tensor_shape: tf.TensorShape,
) -> federated_language.ArrayShape:
  """Returns a `federated_language.ArrayShape` for the `tensor_shape`."""
  if tensor_shape.rank is not None:
    shape = tensor_shape.as_list()
  else:
    shape = None
  return shape


def _tensor_spec_to_type(tensor_spec: tf.TensorSpec) -> federated_language.Type:
  """Returns a `federated_language.Type` for the `tensor_spec`."""
  dtype = _tensorflow_dtype_to_numpy_dtype(tensor_spec.dtype)
  shape = _tensor_shape_to_array_shape(tensor_spec.shape)
  return federated_language.TensorType(dtype, shape)


def _dataset_spec_to_type(
    dataset_spec: tf.data.DatasetSpec,
) -> federated_language.Type:
  """Returns a `federated_language.Type` for the `dataset_spec`."""
  element_type = to_type(dataset_spec.element_spec)
  return federated_language.SequenceType(element_type)


def to_type(obj: object) -> federated_language.Type:
  """Returns a `federated_language.Type` for an `obj` containing TensorFlow type specs.

  This function extends `federated_language.to_type` to handle TensorFlow type
  specs and
  Python structures containing TensorFlow type specs:

  *   `tf.dtypes.DType`
  *   tensor-like objects (e.g. `(tf.int32, tf.TensorShape([2, 3]))`)
  *   `tf.TensorSpec`
  *   `tf.data.DatasetSpec`

  For example:

  >>> to_type(tf.int32)
  federated_language.TensorType(np.int32)

  >>> to_type((tf.int32, tf.TensorShape([2, 3])))
  federated_language.TensorType(np.int32, (2, 3))

  >>> spec = tf.TensorSpec(shape=[2, 3], dtype=tf.int32)
  >>> to_type(spec)
  federated_language.TensorType(np.int32, (2, 3))

  >>> spec = tf.data.DatasetSpec(tf.TensorSpec([2, 3], dtype=tf.int32))
  >>> to_type(spec)
  federated_language.SequenceType(federated_language.TensorType(np.int32, (2,
  3)))

  Args:
    obj: A `federated_language.Type` or an argument convertible to a
      `federated_language.Type`.
  """

  def _to_type(obj):
    if isinstance(obj, tf.dtypes.DType):
      return _tensorflow_dtype_to_numpy_dtype(obj)
    elif isinstance(obj, tf.TensorShape):
      shape = _tensor_shape_to_array_shape(obj)
      if shape is None:
        return tree.MAP_TO_NONE
      else:
        return shape
    elif isinstance(obj, tf.TensorSpec):
      return _tensor_spec_to_type(obj)
    elif isinstance(obj, tf.data.DatasetSpec):
      return _dataset_spec_to_type(obj)
    else:
      return None

  partial_type = tree.traverse(_to_type, obj)
  return federated_language.to_type(partial_type)
