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
# limitations under the License.
"""Utilities for type conversion, type checking, type inference, etc."""

import collections
from collections.abc import Callable
from typing import Optional

import attrs
import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import typed_object


def _tensor_to_type(tensor: tf.Tensor) -> computation_types.Type:
  """Returns a `tff.Type` for the `tensor`."""
  return tensorflow_types.to_type((tensor.dtype, tensor.shape))


def _variable_to_type(variable: tf.Variable) -> computation_types.Type:
  """Returns a `tff.Type` for the `variable`."""
  return tensorflow_types.to_type((variable.dtype, variable.shape))


def _dataset_to_type(dataset: tf.data.Dataset) -> computation_types.Type:
  """Returns a `tff.Type` for the `dataset`."""
  dataset_spec = tf.data.DatasetSpec.from_value(dataset)
  return tensorflow_types.to_type(dataset_spec)


def tensorflow_infer_type(obj: object) -> Optional[computation_types.Type]:
  """Returns a `tff.Type` for an `obj` containing TensorFlow values.

  This function extends `type_conversions.infer_type` to handle TensorFlow
  values and Python structures containing TensorFlow values:

  *   `tf.Tensor`
  *   `tf.Variable`
  *   `tf.data.Dataset`

  For example:

  >>> tensor = tf.ones(shape=[2, 3], dtype=tf.int32)
  >>> tensorflow_infer_type(tensor)
  tff.TensorType(np.int32, (2, 3))

  >>> tensor = tf.ones(shape=[2, 3], dtype=tf.int32)
  >>> variable = tf.Variable(tensor)
  >>> tensorflow_infer_type(variable)
  tff.TensorType(np.int32, (2, 3))

  >>> tensor = tf.ones(shape=[2, 3], dtype=tf.int32)
  >>> dataset = tf.data.Dataset.from_tensors(tensor)
  >>> tensorflow_infer_type(dataset)
  tff.SequenceType(tff.TensorType(np.int32, (2, 3)))

  Args:
    obj: An object to infer a `tff.Type`.
  """

  class _Placeholder(typed_object.TypedObject):

    def __init__(self, type_signature: computation_types.Type):
      self._type_signature = type_signature

    @property
    def type_signature(self) -> computation_types.Type:
      return self._type_signature

  def _infer_type(obj):
    if isinstance(obj, tf.Tensor):
      type_spec = _tensor_to_type(obj)
    elif isinstance(obj, tf.Variable):
      type_spec = _variable_to_type(obj)
    elif isinstance(obj, tf.data.Dataset):
      type_spec = _dataset_to_type(obj)
    else:
      type_spec = None

    # Return a `TypedObject` instead of the `tff.Type` because `infer_type` does
    # not know how to infer the type of a `tff.Type`.
    if type_spec is not None:
      return _Placeholder(type_spec)
    else:
      return None

  partial = tree.traverse(_infer_type, obj)
  return type_conversions.infer_type(partial)


def _type_to_tf_dtypes_and_shapes(type_spec: computation_types.Type):
  """Returns nested structures of tensor dtypes and shapes for a given TFF type.

  The returned dtypes and shapes match those used by `tf.data.Dataset`s to
  indicate the type and shape of their elements. They can be used, e.g., as
  arguments in constructing an iterator over a string handle.

  Args:
    type_spec: A `computation_types.Type`, the type specification must be
      composed of only named tuples and tensors. In all named tuples that appear
      in the type spec, all the elements must be named.

  Returns:
    A pair of parallel nested structures with the dtypes and shapes of tensors
    defined in `type_spec`. The layout of the two structures returned is the
    same as the layout of the nested type defined by `type_spec`. Named tuples
    are represented as dictionaries.

  Raises:
    ValueError: if the `type_spec` is composed of something other than named
      tuples and tensors, or if any of the elements in named tuples are unnamed.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    shape = tf.TensorShape(type_spec.shape)
    return (type_spec.dtype, shape)
  elif isinstance(type_spec, computation_types.StructType):
    elements = structure.to_elements(type_spec)
    if not elements:
      output_dtypes = []
      output_shapes = []
    elif elements[0][0] is not None:
      output_dtypes = collections.OrderedDict()
      output_shapes = collections.OrderedDict()
      for e in elements:
        element_name = e[0]
        element_spec = e[1]
        if element_name is None:
          raise ValueError(
              'When a sequence appears as a part of a parameter to a section '
              'of TensorFlow code, in the type signature of elements of that '
              'sequence all named tuples must have their elements explicitly '
              'named, and this does not appear to be the case in {}.'.format(
                  type_spec
              )
          )
        output_dtype, output_shape = _type_to_tf_dtypes_and_shapes(element_spec)
        output_dtypes[element_name] = output_dtype
        output_shapes[element_name] = output_shape
    else:
      output_dtypes = []
      output_shapes = []
      for e in elements:
        element_name = e[0]
        element_spec = e[1]
        if element_name is not None:
          raise ValueError(
              'When a sequence appears as a part of a parameter to a section '
              'of TensorFlow code, in the type signature of elements of that '
              'sequence all named tuples must have their elements explicitly '
              'named, and this does not appear to be the case in {}.'.format(
                  type_spec
              )
          )
        output_dtype, output_shape = _type_to_tf_dtypes_and_shapes(element_spec)
        output_dtypes.append(output_dtype)
        output_shapes.append(output_shape)
    if type_spec.python_container is not None:
      container_type = type_spec.python_container

      def build_py_container(elements):
        if isinstance(
            container_type, py_typecheck.SupportsNamedTuple
        ) or attrs.has(container_type):
          return container_type(**dict(elements))
        else:
          return container_type(elements)  # pylint: disable=too-many-function-args

      output_dtypes = build_py_container(output_dtypes)
      output_shapes = build_py_container(output_shapes)
    else:
      output_dtypes = tuple(output_dtypes)
      output_shapes = tuple(output_shapes)
    return (output_dtypes, output_shapes)
  else:
    raise ValueError(
        'Unsupported type {}.'.format(py_typecheck.type_string(type(type_spec)))
    )


def type_to_tf_tensor_specs(type_spec: computation_types.Type):
  """Returns nested structure of `tf.TensorSpec`s for a given TFF type.

  The dtypes and shapes of the returned `tf.TensorSpec`s match those used by
  `tf.data.Dataset`s to indicate the type and shape of their elements. They can
  be used, e.g., as arguments in constructing an iterator over a string handle.

  Args:
    type_spec: A `computation_types.Type`, the type specification must be
      composed of only named tuples and tensors. In all named tuples that appear
      in the type spec, all the elements must be named.

  Returns:
    A nested structure of `tf.TensorSpec`s with the dtypes and shapes of tensors
    defined in `type_spec`. The layout of the structure returned is the same as
    the layout of the nested type defined by `type_spec`. Named tuples are
    represented as dictionaries.
  """
  dtypes, shapes = _type_to_tf_dtypes_and_shapes(type_spec)
  return tree.map_structure(
      lambda dtype, shape: tf.TensorSpec(shape, dtype), dtypes, shapes
  )


def type_to_tf_structure(type_spec: computation_types.Type):
  """Returns nested `tf.data.experimental.Structure` for a given TFF type.

  Args:
    type_spec: A `computation_types.Type`, the type specification must be
      composed of only named tuples and tensors. In all named tuples that appear
      in the type spec, all the elements must be named.

  Returns:
    An instance of `tf.data.experimental.Structure`, possibly nested, that
    corresponds to `type_spec`.

  Raises:
    ValueError: if the `type_spec` is composed of something other than named
      tuples and tensors, or if any of the elements in named tuples are unnamed.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return tf.TensorSpec(type_spec.shape, type_spec.dtype)
  elif isinstance(type_spec, computation_types.StructType):
    elements = structure.to_elements(type_spec)
    if not elements:
      return ()
    element_outputs = [(k, type_to_tf_structure(v)) for k, v in elements]
    named = element_outputs[0][0] is not None
    if not all((e[0] is not None) == named for e in element_outputs):
      raise ValueError('Tuple elements inconsistently named.')
    if type_spec.python_container is None:
      if named:
        return collections.OrderedDict(element_outputs)
      else:
        return tuple(v for _, v in element_outputs)
    else:
      container_type = type_spec.python_container
      if isinstance(
          container_type, py_typecheck.SupportsNamedTuple
      ) or attrs.has(container_type):
        return container_type(**dict(element_outputs))
      elif named:
        return container_type(element_outputs)  # pylint: disable=too-many-function-args
      else:
        return container_type(
            e if e[0] is not None else e[1] for e in element_outputs  # pylint: disable=too-many-function-args
        )
  else:
    raise ValueError(
        'Unsupported type {}.'.format(py_typecheck.type_string(type(type_spec)))
    )


def _structure_from_tensor_type_tree_inner(
    fn, type_spec: computation_types.Type
):
  """Helper for `structure_from_tensor_type_tree`."""
  if isinstance(type_spec, computation_types.StructType):
    def _map_element(element):
      name, nested_type = element
      return (name, _structure_from_tensor_type_tree_inner(fn, nested_type))

    return structure.Struct(
        map(_map_element, structure.iter_elements(type_spec))
    )
  elif isinstance(type_spec, computation_types.TensorType):
    return fn(type_spec)
  else:
    raise ValueError(
        'Expected tensor or structure type, found type:\n'
        + type_spec.formatted_representation()
    )


def structure_from_tensor_type_tree(
    fn: Callable[[computation_types.TensorType], object], type_spec
) -> object:
  """Constructs a structure from a `type_spec` tree of `tff.TensorType`s.

  Args:
    fn: A callable used to generate the elements with which to fill the
      resulting structure. `fn` will be called exactly once per leaf
      `tff.TensorType` in the order they appear in the `type_spec` structure.
    type_spec: A TFF type or value convertible to TFF type. Once converted,
      `type_spec` must be a `tff.TensorType` or `tff.StructType` containing only
      other `tff.TensorType`s and `tff.StructType`s.

  Returns:
    A structure with the same shape and Python containers as `type_spec` but
    with the `tff.TensorType` elements replaced with the results of `fn`.

  Raises:
    ValueError: if the provided `type_spec` is not a structural or tensor type.
  """
  type_spec = computation_types.to_type(type_spec)
  non_python_typed = _structure_from_tensor_type_tree_inner(fn, type_spec)
  return type_conversions.type_to_py_container(non_python_typed, type_spec)
