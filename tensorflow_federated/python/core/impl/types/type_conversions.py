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
from collections.abc import Callable, Hashable, Mapping
import typing
from typing import Optional, Union

import attrs
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import dtype_utils
from tensorflow_federated.python.core.impl.types import typed_object


def infer_type(arg: object) -> Optional[computation_types.Type]:
  """Infers the TFF type of the argument (a `computation_types.Type` instance).

  Warning: This function is only partially implemented.

  The kinds of arguments that are currently correctly recognized:
  * tensors, variables, and data sets
  * things that are convertible to tensors (including `numpy` arrays, builtin
    types, as well as `list`s and `tuple`s of any of the above, etc.)
  * nested lists, `tuple`s, `namedtuple`s, anonymous `tuple`s, `dict`,
    `OrderedDict`s, `attrs` classes, and `tff.TypedObject`s

  Args:
    arg: The argument, the TFF type of which to infer.

  Returns:
    Either an instance of `computation_types.Type`, or `None` if the argument is
    `None`.
  """
  # TODO: b/224484886 - Downcasting to all handled types.
  arg = typing.cast(
      Union[
          None,
          typed_object.TypedObject,
          structure.Struct,
          py_typecheck.SupportsNamedTuple,
          Mapping[Hashable, object],
          tuple[object, ...],
          list[object],
      ],
      arg,
  )
  if arg is None:
    return None
  elif isinstance(arg, typed_object.TypedObject):
    return arg.type_signature
  elif isinstance(arg, structure.Struct):
    return computation_types.StructType([
        (k, infer_type(v)) if k else infer_type(v)
        for k, v in structure.iter_elements(arg)
    ])
  elif attrs.has(type(arg)):
    items = attrs.asdict(arg, recurse=False).items()
    return computation_types.StructWithPythonType(
        [(k, infer_type(v)) for k, v in items], type(arg)
    )
  elif isinstance(arg, py_typecheck.SupportsNamedTuple):
    items = arg._asdict().items()
    return computation_types.StructWithPythonType(
        [(k, infer_type(v)) for k, v in items], type(arg)
    )
  elif isinstance(arg, Mapping):
    items = arg.items()
    return computation_types.StructWithPythonType(
        [(k, infer_type(v)) for k, v in items], type(arg)
    )
  elif isinstance(arg, (tuple, list)):
    elements = []
    all_elements_named = True
    for element in arg:
      all_elements_named &= py_typecheck.is_name_value_pair(
          element, name_type=str
      )
      elements.append(infer_type(element))
    # If this is a tuple of (name, value) pairs, the caller most likely intended
    # this to be a StructType, so we avoid storing the Python container.
    if elements and all_elements_named:
      return computation_types.StructType(elements)
    else:
      return computation_types.StructWithPythonType(elements, type(arg))
  elif isinstance(arg, (np.ndarray, np.generic)):
    return computation_types.TensorType(arg.dtype, arg.shape)
  elif isinstance(arg, (bool, int, float, complex, str, bytes)):
    dtype = dtype_utils.infer_dtype(arg)
    return computation_types.TensorType(dtype)
  else:
    raise NotImplementedError(f'Unexpected type found: {type(arg)}.')


def _tensor_to_type(tensor: tf.Tensor) -> computation_types.Type:
  """Returns a `tff.Type` for the `tensor`."""
  return computation_types.tensorflow_to_type((tensor.dtype, tensor.shape))


def _variable_to_type(variable: tf.Variable) -> computation_types.Type:
  """Returns a `tff.Type` for the `variable`."""
  return computation_types.tensorflow_to_type((variable.dtype, variable.shape))


def _dataset_to_type(dataset: tf.data.Dataset) -> computation_types.Type:
  """Returns a `tff.Type` for the `dataset`."""
  dataset_spec = tf.data.DatasetSpec.from_value(dataset)
  return computation_types.tensorflow_to_type(dataset_spec)


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
  return infer_type(partial)


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
          return container_type(elements)

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
        return container_type(element_outputs)
      else:
        return container_type(
            e if e[0] is not None else e[1] for e in element_outputs
        )
  else:
    raise ValueError(
        'Unsupported type {}.'.format(py_typecheck.type_string(type(type_spec)))
    )


def _is_container_type_without_names(container_type: type[object]) -> bool:
  """Returns whether `container_type`'s elements are unnamed."""
  return issubclass(container_type, (list, tuple)) and not isinstance(
      container_type, py_typecheck.SupportsNamedTuple
  )


def _is_container_type_with_names(container_type: type[object]) -> bool:
  """Returns whether `container_type`'s elements are named."""
  return (
      isinstance(container_type, py_typecheck.SupportsNamedTuple)
      or attrs.has(container_type)
      or issubclass(container_type, dict)
  )


def type_to_py_container(value, type_spec: computation_types.Type):
  """Recursively convert `structure.Struct`s to Python containers.

  This is in some sense the inverse operation to
  `structure.from_container`.

  This function assumes some unique behavior with regards to `tff.SequenceType`.
  If the `value` is a list, it may yield other `tff.StructTypes` as well as
  Python types. Otherwise, it may only yield Python types.

  Args:
    value: A structure of anonymous tuples of values corresponding to
      `type_spec`.
    type_spec: The `tff.Type` to which value should conform, possibly including
      `computation_types.StructWithPythonType`.

  Returns:
    The input value, with containers converted to appropriate Python
    containers as specified by the `type_spec`.

  Raises:
    ValueError: If the conversion is not possible due to a mix of named
      and unnamed values, or if `value` contains names that are mismatched or
      not present in the corresponding index of `type_spec`.
  """
  if isinstance(type_spec, computation_types.FederatedType):
    if type_spec.all_equal:
      structure_type_spec = type_spec.member
    else:
      if not isinstance(value, list):
        raise TypeError(
            'Unexpected Python type for non-all-equal TFF type '
            f'{type_spec}: expected `list`, found `{type(value)}`.'
        )
      return [
          type_to_py_container(element, type_spec.member) for element in value
      ]
  else:
    structure_type_spec = type_spec

  if isinstance(structure_type_spec, computation_types.SequenceType):
    element_type = structure_type_spec.element
    if isinstance(value, list):
      return [type_to_py_container(element, element_type) for element in value]
    else:
      # Assume that the type of value does not understand `Struct` and that the
      # value must yield Python containers.
      return value

  if not isinstance(structure_type_spec, computation_types.StructType):
    return value

  if not isinstance(value, structure.Struct):
    # NOTE: When encountering non-`structure.Struct`s, we assume that
    # this means that we're attempting to re-convert a value that
    # already has the proper containers, and we short-circuit to
    # avoid re-converting. This is a possibly dangerous assumption.
    return value

  container_type = structure_type_spec.python_container

  # Ensure that names are only added, not mismatched or removed
  names_from_value = structure.name_list_with_nones(value)
  names_from_type_spec = structure.name_list_with_nones(structure_type_spec)
  for value_name, type_name in zip(names_from_value, names_from_type_spec):
    if value_name is not None:
      if value_name != type_name:
        raise ValueError(
            f'Cannot convert value with field name `{value_name}` into a '
            f'type with field name `{type_name}`.'
        )

  num_named_elements = len(dir(structure_type_spec))
  num_unnamed_elements = len(structure_type_spec) - num_named_elements
  if num_named_elements > 0 and num_unnamed_elements > 0:
    raise ValueError(
        f'Cannot represent value {value} with a Python container because it '
        'contains a mix of named and unnamed elements.\n\nNote: this was '
        'previously allowed when using the `tff.structure.Struct` container. '
        'This support has been removed: please change to use structures with '
        'either all-named or all-unnamed fields.'
    )
  if container_type is None:
    if num_named_elements:
      container_type = collections.OrderedDict
    else:
      container_type = tuple

  # Avoid projecting the `structure.StructType`d TFF value into a Python
  # container that is not supported.
  if num_named_elements > 0 and _is_container_type_without_names(
      container_type
  ):
    raise ValueError(
        'Cannot represent value {} with named elements '
        "using container type {} which does not support names. In TFF's "
        'typesystem, this corresponds to an implicit downcast'.format(
            value, container_type
        )
    )
  if _is_container_type_with_names(container_type) and len(
      dir(structure_type_spec)
  ) != len(value):
    # If the type specifies the names, we have all the information we need.
    # Otherwise we must raise here.
    raise ValueError(
        'When packaging as a Python value which requires names, '
        'the TFF type spec must have all names specified. Found '
        '{} names in type spec {} of length {}, with requested'
        'python type {}.'.format(
            len(dir(structure_type_spec)),
            structure_type_spec,
            len(value),
            container_type,
        )
    )

  elements = []
  for index, (elem_name, elem_type) in enumerate(
      structure.iter_elements(structure_type_spec)
  ):
    element = type_to_py_container(value[index], elem_type)

    if elem_name is None:
      elements.append(element)
    else:
      elements.append((elem_name, element))

  if (
      isinstance(container_type, py_typecheck.SupportsNamedTuple)
      or attrs.has(container_type)
  ):
    # The namedtuple and attr.s class constructors cannot interpret a list of
    # (name, value) tuples; instead call constructor using kwargs. Note that
    # these classes already define an order of names internally, so order does
    # not matter.
    return container_type(**dict(elements))
  else:
    # E.g., tuple and list when elements only has values, but also `dict`,
    # `collections.OrderedDict`, or `structure.Struct` when
    # elements has (name, value) tuples.
    return container_type(elements)  # pytype: disable=wrong-arg-count


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
  return type_to_py_container(non_python_typed, type_spec)


def type_to_non_all_equal(type_spec):
  """Constructs a non-`all_equal` version of the federated type `type_spec`.

  Args:
    type_spec: An instance of `tff.FederatedType`.

  Returns:
    A federated type with the same member and placement, but `all_equal=False`.
  """
  py_typecheck.check_type(type_spec, computation_types.FederatedType)
  return computation_types.FederatedType(
      type_spec.member, type_spec.placement, all_equal=False
  )
