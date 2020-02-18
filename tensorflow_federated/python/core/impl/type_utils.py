# Lint as: python3
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
from typing import Any, Callable, Dict, Type, TypeVar

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl.compiler import placement_literals

TF_DATASET_REPRESENTATION_TYPES = (tf.data.Dataset, tf.compat.v1.data.Dataset,
                                   tf.compat.v2.data.Dataset)


def infer_type(arg):
  """Infers the TFF type of the argument (a `computation_types.Type` instance).

  WARNING: This function is only partially implemented.

  The kinds of arguments that are currently correctly recognized:
  - tensors, variables, and data sets,
  - things that are convertible to tensors (including numpy arrays, builtin
    types, as well as lists and tuples of any of the above, etc.),
  - nested lists, tuples, namedtuples, anonymous tuples, dict, and OrderedDicts.

  Args:
    arg: The argument, the TFF type of which to infer.

  Returns:
    Either an instance of `computation_types.Type`, or `None` if the argument is
    `None`.
  """
  # TODO(b/113112885): Implement the remaining cases here on the need basis.
  if arg is None:
    return None
  elif isinstance(arg, typed_object.TypedObject):
    return arg.type_signature
  elif tf.is_tensor(arg):
    return computation_types.TensorType(arg.dtype.base_dtype, arg.shape)
  elif isinstance(arg, TF_DATASET_REPRESENTATION_TYPES):
    return computation_types.SequenceType(
        computation_types.to_type(arg.element_spec))
  elif isinstance(arg, anonymous_tuple.AnonymousTuple):
    return computation_types.NamedTupleType([
        (k, infer_type(v)) if k else infer_type(v)
        for k, v in anonymous_tuple.iter_elements(arg)
    ])
  elif py_typecheck.is_attrs(arg):
    items = attr.asdict(
        arg, dict_factory=collections.OrderedDict, recurse=False)
    return computation_types.NamedTupleTypeWithPyContainerType(
        [(k, infer_type(v)) for k, v in items.items()], type(arg))
  elif py_typecheck.is_named_tuple(arg):
    items = arg._asdict()
    return computation_types.NamedTupleTypeWithPyContainerType(
        [(k, infer_type(v)) for k, v in items.items()], type(arg))
  elif isinstance(arg, dict):
    if isinstance(arg, collections.OrderedDict):
      items = arg.items()
    else:
      items = sorted(arg.items())
    return computation_types.NamedTupleTypeWithPyContainerType(
        [(k, infer_type(v)) for k, v in items], type(arg))
  elif isinstance(arg, (tuple, list)):
    elements = []
    all_elements_named = True
    for element in arg:
      all_elements_named &= py_typecheck.is_name_value_pair(element)
      elements.append(infer_type(element))
    # If this is a tuple of (name, value) pairs, the caller most likely intended
    # this to be a NamedTupleType, so we avoid storing the Python container.
    if all_elements_named:
      return computation_types.NamedTupleType(elements)
    else:
      return computation_types.NamedTupleTypeWithPyContainerType(
          elements, type(arg))
  elif isinstance(arg, str):
    return computation_types.TensorType(tf.string)
  elif isinstance(arg, (np.generic, np.ndarray)):
    return computation_types.TensorType(
        tf.dtypes.as_dtype(arg.dtype), arg.shape)
  else:
    dtype = {bool: tf.bool, int: tf.int32, float: tf.float32}.get(type(arg))
    if dtype:
      return computation_types.TensorType(dtype)
    else:
      # Now fall back onto the heavier-weight processing, as all else failed.
      # Use make_tensor_proto() to make sure to handle it consistently with
      # how TensorFlow is handling values (e.g., recognizing int as int32, as
      # opposed to int64 as in NumPy).
      try:
        # TODO(b/113112885): Find something more lightweight we could use here.
        tensor_proto = tf.make_tensor_proto(arg)
        return computation_types.TensorType(
            tf.dtypes.as_dtype(tensor_proto.dtype),
            tf.TensorShape(tensor_proto.tensor_shape))
      except TypeError as err:
        raise TypeError('Could not infer the TFF type of {}: {}'.format(
            py_typecheck.type_string(type(arg)), err))


def to_canonical_value(value):
  """Converts a Python object to a canonical TFF value for a given type.

  Args:
    value: The object to convert.

  Returns:
    The canonical TFF representation of `value` for a given type.
  """
  if value is None:
    return None
  elif isinstance(value, dict):
    if isinstance(value, collections.OrderedDict):
      items = value.items()
    else:
      items = sorted(value.items())
    return anonymous_tuple.AnonymousTuple(
        (k, to_canonical_value(v)) for k, v in items)
  elif isinstance(value, (tuple, list)):
    return [to_canonical_value(e) for e in value]
  return value


def check_type(val, type_spec):
  """Checks whether `val` is of TFF type `type_spec`.

  Args:
    val: The object to check.
    type_spec: An instance of `tff.Type` or something convertible to it that the
      `val` is checked against.

  Raises:
    TypeError: If the inefferred type of `val` is not `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  val_type = infer_type(val)
  if not is_assignable_from(type_spec, val_type):
    raise TypeError(
        'Expected TFF type {}, which is not assignable from {}.'.format(
            type_spec, val_type))


def tf_dtypes_and_shapes_to_type(dtypes, shapes):
  """Returns computation_types.Type for the given TF (dtypes, shapes) tuple.

  The returned dtypes and shapes match those used by `tf.data.Dataset`s to
  indicate the type and shape of their elements. They can be used, e.g., as
  arguments in constructing an iterator over a string handle. Note that the
  nested structure of dtypes and shapes must be identical.

  Args:
    dtypes: A nested structure of dtypes, such as what is returned by Dataset's
      output_dtypes property.
    shapes: A nested structure of shapes, such as what is returned by Dataset's
      output_shapes property.

  Returns:
    The corresponding instance of computation_types.Type.

  Raises:
    TypeError: if the arguments are of types that weren't recognized.
  """
  tf.nest.assert_same_structure(dtypes, shapes)

  def _parallel_dict_to_element_list(dtype_dict, shape_dict):
    return [(name, tf_dtypes_and_shapes_to_type(dtype_elem, shape_dict[name]))
            for name, dtype_elem in dtype_dict.items()]

  if isinstance(dtypes, tf.DType):
    return computation_types.TensorType(dtypes, shapes)
  elif py_typecheck.is_named_tuple(dtypes):
    # Special handling needed for collections.namedtuple due to the lack of
    # a base class. Note this must precede the test for being a list.
    dtype_dict = dtypes._asdict()
    shape_dict = shapes._asdict()
    return computation_types.NamedTupleTypeWithPyContainerType(
        _parallel_dict_to_element_list(dtype_dict, shape_dict), type(dtypes))
  elif py_typecheck.is_attrs(dtypes):
    dtype_dict = attr.asdict(
        dtypes, dict_factory=collections.OrderedDict, recurse=False)
    shapes_dict = attr.asdict(
        shapes, dict_factory=collections.OrderedDict, recurse=False)
    return computation_types.NamedTupleTypeWithPyContainerType(
        _parallel_dict_to_element_list(dtype_dict, shapes_dict), type(dtypes))
  elif isinstance(dtypes, dict):
    if isinstance(dtypes, collections.OrderedDict):
      items = dtypes.items()
    else:
      items = sorted(dtypes.items())
    elements = [(name, tf_dtypes_and_shapes_to_type(dtypes_elem, shapes[name]))
                for name, dtypes_elem in items]
    return computation_types.NamedTupleTypeWithPyContainerType(
        elements, type(dtypes))
  elif isinstance(dtypes, (list, tuple)):
    return computation_types.NamedTupleTypeWithPyContainerType([
        tf_dtypes_and_shapes_to_type(dtypes_elem, shapes[idx])
        for idx, dtypes_elem in enumerate(dtypes)
    ], type(dtypes))
  else:
    raise TypeError('Unrecognized: dtypes {}, shapes {}.'.format(
        dtypes, shapes))


def type_to_tf_dtypes_and_shapes(type_spec):
  """Returns nested structures of tensor dtypes and shapes for a given TFF type.

  The returned dtypes and shapes match those used by `tf.data.Dataset`s to
  indicate the type and shape of their elements. They can be used, e.g., as
  arguments in constructing an iterator over a string handle.

  Args:
    type_spec: Type specification, either an instance of
      `computation_types.Type`, or something convertible to it. The type
      specification must be composed of only named tuples and tensors. In all
      named tuples that appear in the type spec, all the elements must be named.

  Returns:
    A pair of parallel nested structures with the dtypes and shapes of tensors
    defined in `type_spec`. The layout of the two structures returned is the
    same as the layout of the nested type defined by `type_spec`. Named tuples
    are represented as dictionaries.

  Raises:
    ValueError: if the `type_spec` is composed of something other than named
      tuples and tensors, or if any of the elements in named tuples are unnamed.
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(type_spec, computation_types.TensorType):
    return (type_spec.dtype, type_spec.shape)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    elements = anonymous_tuple.to_elements(type_spec)
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
                  type_spec))
        element_output = type_to_tf_dtypes_and_shapes(element_spec)
        output_dtypes[element_name] = element_output[0]
        output_shapes[element_name] = element_output[1]
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
                  type_spec))
        element_output = type_to_tf_dtypes_and_shapes(element_spec)
        output_dtypes.append(element_output[0])
        output_shapes.append(element_output[1])
    if isinstance(type_spec,
                  computation_types.NamedTupleTypeWithPyContainerType):
      container_type = computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
          type_spec)

      def build_py_container(elements):
        if (py_typecheck.is_named_tuple(container_type) or
            py_typecheck.is_attrs(container_type)):
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
    raise ValueError('Unsupported type {}.'.format(
        py_typecheck.type_string(type(type_spec))))


def type_to_tf_tensor_specs(type_spec):
  """Returns nested structure of `tf.TensorSpec`s for a given TFF type.

  The dtypes and shapes of the returned `tf.TensorSpec`s match those used by
  `tf.data.Dataset`s to indicate the type and shape of their elements. They can
  be used, e.g., as arguments in constructing an iterator over a string handle.

  Args:
    type_spec: Type specification, either an instance of
      `computation_types.Type`, or something convertible to it. Ther type
      specification must be composed of only named tuples and tensors. In all
      named tuples that appear in the type spec, all the elements must be named.

  Returns:
    A nested structure of `tf.TensorSpec`s with the dtypes and shapes of tensors
    defined in `type_spec`. The layout of the structure returned is the same as
    the layout of the nested type defined by `type_spec`. Named tuples are
    represented as dictionaries.
  """
  dtypes, shapes = type_to_tf_dtypes_and_shapes(type_spec)
  return tf.nest.map_structure(lambda dtype, shape: tf.TensorSpec(shape, dtype),
                               dtypes, shapes)


def type_to_tf_structure(type_spec):
  """Returns nested `tf.data.experimental.Structure` for a given TFF type.

  Args:
    type_spec: Type specification, either an instance of
      `computation_types.Type`, or something convertible to it. Ther type
      specification must be composed of only named tuples and tensors. In all
      named tuples that appear in the type spec, all the elements must be named.

  Returns:
    An instance of `tf.data.experimental.Structure`, possibly nested, that
    corresponds to `type_spec`.

  Raises:
    ValueError: if the `type_spec` is composed of something other than named
      tuples and tensors, or if any of the elements in named tuples are unnamed.
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(type_spec, computation_types.TensorType):
    return tf.TensorSpec(type_spec.shape, type_spec.dtype)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    elements = anonymous_tuple.to_elements(type_spec)
    if not elements:
      raise ValueError('Empty tuples are unsupported.')
    element_outputs = [(k, type_to_tf_structure(v)) for k, v in elements]
    named = element_outputs[0][0] is not None
    if not all((e[0] is not None) == named for e in element_outputs):
      raise ValueError('Tuple elements inconsistently named.')
    if not isinstance(type_spec,
                      computation_types.NamedTupleTypeWithPyContainerType):
      if named:
        output = collections.OrderedDict(element_outputs)
      else:
        output = tuple(v for _, v in element_outputs)
    else:
      container_type = computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
          type_spec)
      if (py_typecheck.is_named_tuple(container_type) or
          py_typecheck.is_attrs(container_type)):
        output = container_type(**dict(element_outputs))
      elif named:
        output = container_type(element_outputs)
      else:
        output = container_type(
            e if e[0] is not None else e[1] for e in element_outputs)
    return output
  else:
    raise ValueError('Unsupported type {}.'.format(
        py_typecheck.type_string(type(type_spec))))


def type_from_tensors(tensors):
  """Builds a `tff.Type` from supplied tensors.

  Args:
    tensors: A nested structure of tensors.

  Returns:
    The nested TensorType structure.
  """

  def _mapping_fn(x):
    if not tf.is_tensor(x):
      x = tf.convert_to_tensor(x)
    return computation_types.TensorType(x.dtype.base_dtype, x.shape)

  if isinstance(tensors, anonymous_tuple.AnonymousTuple):
    return computation_types.to_type(
        anonymous_tuple.map_structure(_mapping_fn, tensors))
  else:
    return computation_types.to_type(
        tf.nest.map_structure(_mapping_fn, tensors))


def get_named_tuple_element_type(type_spec, name):
  """Returns the type of a named tuple member.

  Args:
    type_spec: Type specification, either an instance of computation_types.Type
      or something convertible to it by computation_types.to_type().
    name: The string name of the named tuple member.

  Returns:
    The TFF type of the element.

  Raises:
    TypeError: if arguments are of the wrong computation_types.
    ValueError: if the tuple does not have an element with the given name.
  """
  py_typecheck.check_type(name, str)
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
  elements = anonymous_tuple.to_elements(type_spec)
  for elem_name, elem_type in elements:
    if name == elem_name:
      return elem_type
  raise ValueError('The name \'{}\' of the element does not correspond to any '
                   'of the names {} in the named tuple type.'.format(
                       name, [e[0] for e in elements if e[0]]))

T = TypeVar('T')


def preorder_call(given_type: Any, fn: Callable[[Any, T], T], arg: T):
  """Recursively calls `fn` on the possibly nested structure `given_type`.

  Walks the tree in a preorder manner. Updates `arg` on the way down with
  the appropriate information, as defined in `fn`.

  Args:
    given_type: Possibly nested `computation_types.Type` or object convertible
      to it by `computation_types.to_type`.
    fn: Function to apply to each of the constituent elements of `given_type`
      with the argument `arg`. Must return an updated version of `arg` which
      incorporated the information we'd like to track as we move down the nested
      type tree.
    arg: Initial state of information to be passed down the tree.
  """
  type_signature = computation_types.to_type(given_type)
  arg = fn(type_signature, arg)
  if isinstance(type_signature, computation_types.FederatedType):
    preorder_call(type_signature.member, fn, arg)
  elif isinstance(type_signature, computation_types.SequenceType):
    preorder_call(type_signature.element, fn, arg)
  elif isinstance(type_signature, computation_types.FunctionType):
    preorder_call(type_signature.parameter, fn, arg)
    preorder_call(type_signature.result, fn, arg)
  elif isinstance(type_signature, computation_types.NamedTupleType):
    for element in anonymous_tuple.iter_elements(type_signature):
      preorder_call(element[1], fn, arg)


def check_well_formed(type_spec):
  """Checks that `type_spec` represents a well-formed type.

  Performs the following checks of well-formedness for `type_spec`:
    1. If `type_spec` contains a  `computation_types.FederatedType`, checks
    that its `member` contains nowhere in its structure intances
    of `computation_types.FunctionType` or `computation_types.FederatedType`.
    2. If `type_spec` contains a `computation_types.SequenceType`, checks that
    its `element` contains nowhere in its structure instances of
    `computation_types.SequenceType`,  `computation_types.FederatedType`
    or `computation_types.FunctionType`.

  Args:
    type_spec: The type specification to check, either an instance of
      `computation_types.Type` or something convertible to it by
      `computation_types.to_type()`.

  Raises:
    TypeError: if `type_spec` is not a well-formed TFF type.
  """

  # TODO(b/113112885): Reinstate a call to `check_all_abstract_types_are_bound`
  # after revising the definition of well-formedness.
  type_signature = computation_types.to_type(type_spec)

  def _check_for_disallowed_type(
      type_to_check: Any,
      disallowed_types: Dict[Type[Any], str],
  ) -> Dict[Type[Any], str]:
    """Checks subtree of `type_to_check` for `disallowed_types`."""
    for disallowed_type, disallowed_context in disallowed_types.items():
      if isinstance(type_to_check, disallowed_type):
        raise TypeError('{} has been encountered in the type signature {}. '
                        '{} is disallowed inside of {}.'.format(
                            type_to_check,
                            type_signature,
                            disallowed_type,
                            disallowed_context,
                        ))
    if isinstance(type_to_check, computation_types.FederatedType):
      context = 'federated types (types placed @CLIENT or @SERVER)'
      disallowed_types = {
          **disallowed_types, computation_types.FederatedType: context,
          computation_types.FunctionType: context
      }
    if isinstance(type_to_check, computation_types.SequenceType):
      context = 'sequence types'
      disallowed_types = {
          **disallowed_types, computation_types.FederatedType: context,
          computation_types.SequenceType: context
      }
    return disallowed_types

  preorder_call(type_signature, _check_for_disallowed_type, dict())


def type_tree_contains_only(type_spec, whitelisted_types):
  """Checks whether `type_spec` contains only instances of `whitelisted_types`.

  Args:
    type_spec: The type specification to check, either an instance of
      `computation_types.Type` or something convertible to it by
      `computation_types.to_type()`.
    whitelisted_types: The singleton or tuple of types for which we wish to
      check `type_spec`. Contains subclasses of `computation_types.Type`. Uses
      similar syntax to `isinstance`; allows for single argument or `tuple` of
      multiple arguments.

  Returns:
    True if `type_spec` contains only types in `whitelisted_types`, and
    `False` otherwise.
  """
  type_signature = computation_types.to_type(type_spec)

  class WhitelistTracker(object):
    """Simple callable to track Boolean through nested structure."""

    def __init__(self):
      self.whitelisted = True

    def __call__(self, type_to_check, whitelist):
      """Checks subtree of `type_to_check` for `whitelist`."""
      if not isinstance(type_to_check, whitelist):
        self.whitelisted = False
      return whitelist

  tracker = WhitelistTracker()
  preorder_call(type_signature, tracker, whitelisted_types)
  return tracker.whitelisted


def is_tensorflow_compatible_type(type_spec):
  """Checks `type_spec` against an explicit whitelist for `tf_computation`."""
  if type_spec is None:
    return True
  tf_comp_whitelist = (computation_types.TensorType,
                       computation_types.SequenceType,
                       computation_types.NamedTupleType)
  return type_tree_contains_only(type_spec, tf_comp_whitelist)


def check_tensorflow_compatible_type(type_spec):
  if not is_tensorflow_compatible_type(type_spec):
    raise TypeError(
        'Expected type to be compatible with TensorFlow (i.e. tensor, '
        'sequence, or tuple types), found {}.'.format(type_spec))


def is_generic_op_compatible_type(type_spec):
  """Checks `type_spec` against an explicit whitelist for generic operators."""
  if type_spec is None:
    return True
  tf_comp_whitelist = (computation_types.TensorType,
                       computation_types.NamedTupleType)
  return type_tree_contains_only(type_spec, tf_comp_whitelist)


def is_binary_op_with_upcast_compatible_pair(possibly_nested_type,
                                             type_to_upcast):
  """Checks unambiguity in applying `type_to_upcast` to `possibly_nested_type`.

  That is, checks that either these types are equivalent and contain only
  tuples and tensors, or that
  `possibly_nested_type` is perhaps a nested structure containing only tensors
  with `dtype` of `type_to_upcast` at the leaves, where `type_to_upcast` must
  be a scalar tensor type. Notice that this relationship is not symmetric,
  since binary operators need not respect this symmetry in general.
  For example, it makes perfect sence to divide a nested structure of tensors
  by a scalar, but not the other way around.

  Args:
    possibly_nested_type: Convertible to `computation_types.Type`.
    type_to_upcast: Convertible to `computation_types.Type`.

  Returns:
    Boolean indicating whether `type_to_upcast` can be upcast to
    `possibly_nested_type` in the manner described above.
  """
  possibly_nested_type = computation_types.to_type(possibly_nested_type)
  type_to_upcast = computation_types.to_type(type_to_upcast)
  if not (is_generic_op_compatible_type(possibly_nested_type) and
          is_generic_op_compatible_type(type_to_upcast)):
    return False
  if are_equivalent_types(possibly_nested_type, type_to_upcast):
    return True
  if not (isinstance(type_to_upcast, computation_types.TensorType) and
          type_to_upcast.shape == tf.TensorShape(())):
    return False

  types_are_ok = [True]

  only_allowed_dtype = type_to_upcast.dtype

  def _check_tensor_types(type_spec):
    if isinstance(
        type_spec,
        computation_types.TensorType) and type_spec.dtype != only_allowed_dtype:
      types_are_ok[0] = False
    return type_spec, False

  transform_type_postorder(possibly_nested_type, _check_tensor_types)

  return types_are_ok[0]


def type_tree_contains_types(type_spec, blacklisted_types):
  """Checks whether `type_spec` contains any instances of `blacklisted_types`.

  Args:
    type_spec: The type specification to check, either an instance of
      `computation_types.Type` or something convertible to it by
      `computation_types.to_type()`.
    blacklisted_types: The singleton or tuple of types for which we wish to
      check in `type_spec`. Contains subclasses of `computation_types.Type`.
      Uses similar syntax to `isinstance`; allows for single argument or `tuple`
      of multiple arguments.

  Returns:
    True if `type_spec` contains any types in `blacklisted_types`, and
    `False` otherwise.
  """
  type_signature = computation_types.to_type(type_spec)

  class BlacklistTracker(object):
    """Simple callable to track Boolean through nested structure."""

    def __init__(self):
      self.blacklisted = False

    def __call__(self, type_to_check, blacklist):
      """Checks subtree of `type_to_check` for `blacklist`."""
      if isinstance(type_to_check, blacklist):
        self.blacklisted = True
      return blacklist

  tracker = BlacklistTracker()
  preorder_call(type_signature, tracker, blacklisted_types)
  return tracker.blacklisted


def check_all_abstract_types_are_bound(type_spec):
  """Checks that all abstract types labels appearing in 'type_spec' are bound.

  For abstract types to be bound, it means that type labels appearing on the
  result side of functional type signatures must also appear on the parameter
  side. This check is intended to verify that abstract types are only used to
  model template-like type signatures, and can always be reduce to a concrete
  type by specializing templates to work with specific sets of arguments.

  Examples of valid types that pass this check successfully:

    int32
    (int32 -> int32)
    ( -> int32)
    (T -> T)
    ((T -> T) -> bool)
    (( -> T) -> T)
    (<T*, ((T, T) -> T)> -> T)
    (T* -> int32)
    ( -> (T -> T))
    <T, (U -> U), U> -> <T, U>

  Examples of invalid types that fail this check because 'T' is unbound:

    T
    (int32 -> T)
    ( -> T)
    (T -> U)

  Args:
    type_spec: An instance of computation_types.Type, or something convertible
      to it.

  Raises:
    TypeError: if arguments are of the wrong types, or if unbound type labels
      occur in 'type_spec'.
  """

  def _check_or_get_unbound_abstract_type_labels(type_spec, bound_labels,
                                                 check):
    """Checks or collects abstract type labels from 'type_spec'.

    This is a helper function used by 'check_abstract_types_are_bound', not to
    be exported out of this module.

    Args:
      type_spec: An instance of computation_types.Type.
      bound_labels: A set of string labels that refer to 'bound' abstract types,
        i.e., ones that appear on the parameter side of a functional type.
      check: A bool value. If True, no new unbound type labels are permitted,
        and if False, any new labels encountered are returned as a set.

    Returns:
      If check is False, a set of new abstract type labels introduced in
      'type_spec' that don't yet appear in the set 'bound_labels'. If check is
      True, always returns an empty set.

    Raises:
      TypeError: if unbound labels are found and check is True.
    """
    py_typecheck.check_type(type_spec, computation_types.Type)
    if isinstance(type_spec, computation_types.TensorType):
      return set()
    elif isinstance(type_spec, computation_types.SequenceType):
      return _check_or_get_unbound_abstract_type_labels(type_spec.element,
                                                        bound_labels, check)
    elif isinstance(type_spec, computation_types.FederatedType):
      return _check_or_get_unbound_abstract_type_labels(type_spec.member,
                                                        bound_labels, check)
    elif isinstance(type_spec, computation_types.NamedTupleType):
      return set().union(*[
          _check_or_get_unbound_abstract_type_labels(v, bound_labels, check)
          for _, v in anonymous_tuple.iter_elements(type_spec)
      ])
    elif isinstance(type_spec, computation_types.AbstractType):
      if type_spec.label in bound_labels:
        return set()
      elif not check:
        return set([type_spec.label])
      else:
        raise TypeError('Unbound type label \'{}\'.'.format(type_spec.label))
    elif isinstance(type_spec, computation_types.FunctionType):
      if type_spec.parameter is None:
        parameter_labels = set()
      else:
        parameter_labels = _check_or_get_unbound_abstract_type_labels(
            type_spec.parameter, bound_labels, False)
      result_labels = _check_or_get_unbound_abstract_type_labels(
          type_spec.result, bound_labels.union(parameter_labels), check)
      return parameter_labels.union(result_labels)

  _check_or_get_unbound_abstract_type_labels(
      computation_types.to_type(type_spec), set(), True)


def is_numeric_dtype(dtype):
  """Returns True iff `dtype` is numeric.

  Args:
    dtype: An instance of tf.DType.

  Returns:
    True iff `dtype` is numeric, i.e., integer, float, or complex.
  """
  py_typecheck.check_type(dtype, tf.DType)
  return dtype.is_integer or dtype.is_floating or dtype.is_complex


def is_sum_compatible(type_spec):
  """Determines if `type_spec` is a type that can be added to itself.

  Types that are sum-compatible are composed of scalars of numeric types,
  possibly packaged into nested named tuples, and possibly federated. Types
  that are sum-incompatible include sequences, functions, abstract types,
  and placements.

  Args:
    type_spec: Either an instance of computation_types.Type, or something
      convertible to it.

  Returns:
    `True` iff `type_spec` is sum-compatible, `False` otherwise.
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(type_spec, computation_types.TensorType):
    return is_numeric_dtype(type_spec.dtype)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    return all(
        is_sum_compatible(v)
        for _, v in anonymous_tuple.iter_elements(type_spec))
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_sum_compatible(type_spec.member)
  else:
    return False


def check_is_sum_compatible(type_spec):
  if not is_sum_compatible(type_spec):
    raise TypeError(
        'Expected a type which is compatible with the sum operator, found {}.'
        .format(type_spec))


def is_structure_of_integers(type_spec):
  """Determines if `type_spec` is a structure of integers.

  Args:
    type_spec: Either an instance of computation_types.Type, or something
      convertible to it.

  Returns:
    `True` iff `type_spec` is a structure of integers, otherwise `False`.
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(type_spec, computation_types.TensorType):
    py_typecheck.check_type(type_spec.dtype, tf.DType)
    return type_spec.dtype.is_integer
  elif isinstance(type_spec, computation_types.NamedTupleType):
    return all(
        is_structure_of_integers(v)
        for _, v in anonymous_tuple.iter_elements(type_spec))
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_structure_of_integers(type_spec.member)
  else:
    return False


def check_is_structure_of_integers(type_spec):
  if not is_sum_compatible(type_spec):
    raise TypeError(
        'Expected a type which is structure of integers, found {}.'.format(
            type_spec))


def check_federated_type(type_spec,
                         member=None,
                         placement=None,
                         all_equal=None):
  """Checks that `type_spec` is a federated type with the given parameters.

  Args:
    type_spec: The `tff.Type` to check (or something convertible to it).
    member: The expected member type, or `None` if unspecified.
    placement: The desired placement, or `None` if unspecified.
    all_equal: The desired result of accessing the property
      `tff.FederatedType.all_equal` of `type_spec`, or `None` if left
      unspecified.

  Raises:
    TypeError: if `type_spec` is not a federated type of the given kind.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.FederatedType)
  if member is not None:
    member = computation_types.to_type(member)
    py_typecheck.check_type(member, computation_types.Type)
    check_assignable_from(member, type_spec.member)
  if placement is not None:
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    if type_spec.placement is not placement:
      raise TypeError(
          'Expected federated type placed at {}, got one placed at {}.'.format(
              placement, type_spec.placement))
  if all_equal is not None:
    py_typecheck.check_type(all_equal, bool)
    if type_spec.all_equal != all_equal:
      raise TypeError(
          'Expected federated type with all_equal {}, got one with {}.'.format(
              all_equal, type_spec.all_equal))


def is_average_compatible(type_spec):
  """Determines if `type_spec` can be averaged.

  Types that are average-compatible are composed of numeric tensor types,
  either floating-point or complex, possibly packaged into nested named tuples,
  and possibly federated.

  Args:
    type_spec: An instance of `types.Type`, or something convertible to it.

  Returns:
    `True` iff `type_spec` is average-compatible, `False` otherwise.
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(type_spec, computation_types.TensorType):
    return type_spec.dtype.is_floating or type_spec.dtype.is_complex
  elif isinstance(type_spec, computation_types.NamedTupleType):
    return all(
        is_average_compatible(v)
        for _, v in anonymous_tuple.iter_elements(type_spec))
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_average_compatible(type_spec.member)
  else:
    return False


def is_assignable_from(target_type, source_type):
  """Determines whether `target_type` is assignable from `source_type`.

  Args:
    target_type: The expected type (that of the target of the assignment).
    source_type: The actual type (that of the source of the assignment), tested
      for being a specialization of the `target_type`.

  Returns:
    `True` iff `target_type` is assignable from `source_type`, or else `False`.

  Raises:
    TypeError: If the arguments are not TFF types.
  """
  target_type = computation_types.to_type(target_type)
  source_type = computation_types.to_type(source_type)
  py_typecheck.check_type(target_type, computation_types.Type)
  py_typecheck.check_type(source_type, computation_types.Type)
  if isinstance(target_type, computation_types.TensorType):

    def _shape_is_assignable_from(x, y):

      def _dimension_is_assignable_from(x, y):
        return (x.value is None) or (x.value == y.value)

      # TODO(b/123764922): See if we can pass to TensorShape's
      # `is_compatible_with`.
      return ((x.ndims == y.ndims) and ((x.dims is None) or all(
          _dimension_is_assignable_from(x.dims[k], y.dims[k])
          for k in range(x.ndims)))) or x.ndims is None

    return (isinstance(source_type, computation_types.TensorType) and
            (target_type.dtype == source_type.dtype) and
            _shape_is_assignable_from(target_type.shape, source_type.shape))
  elif isinstance(target_type, computation_types.NamedTupleType):
    if not isinstance(source_type, computation_types.NamedTupleType):
      return False
    target_elements = anonymous_tuple.to_elements(target_type)
    source_elements = anonymous_tuple.to_elements(source_type)
    return ((len(target_elements) == len(source_elements)) and all(
        ((source_elements[k][0] in [target_elements[k][0], None]) and
         is_assignable_from(target_elements[k][1], source_elements[k][1]))
        for k in range(len(target_elements))))
  elif isinstance(target_type, computation_types.SequenceType):
    return (isinstance(source_type, computation_types.SequenceType) and
            is_assignable_from(target_type.element, source_type.element))
  elif isinstance(target_type, computation_types.FunctionType):
    return (isinstance(source_type, computation_types.FunctionType) and
            (((source_type.parameter is None) and
              (target_type.parameter is None)) or
             ((source_type.parameter is not None) and
              (target_type.parameter is not None) and is_assignable_from(
                  source_type.parameter, target_type.parameter)) and
             is_assignable_from(target_type.result, source_type.result)))
  elif isinstance(target_type, computation_types.AbstractType):
    # TODO(b/113112108): Revise this to extend the relation of assignability to
    # abstract types.
    raise TypeError('Abstract types are not comparable.')
  elif isinstance(target_type, computation_types.PlacementType):
    return isinstance(source_type, computation_types.PlacementType)
  elif isinstance(target_type, computation_types.FederatedType):
    if (not isinstance(source_type, computation_types.FederatedType) or
        not is_assignable_from(target_type.member, source_type.member) or
        target_type.all_equal and not source_type.all_equal):
      return False
    for val in [target_type, source_type]:
      py_typecheck.check_type(val.placement,
                              placement_literals.PlacementLiteral)
    return target_type.placement is source_type.placement
  else:
    raise TypeError('Unexpected target type {}.'.format(target_type))


def check_assignable_from(target, source):
  target = computation_types.to_type(target)
  source = computation_types.to_type(source)
  if not is_assignable_from(target, source):
    raise TypeError(
        'The target type {} is not assignable from source type {}.'.format(
            target, source))


def are_equivalent_types(type1, type2):
  """Determines whether `type1` and `type2` are equivalent.

  We define equivaence in this context as both types being assignable from
  one-another.

  Args:
    type1: One type.
    type2: Another type.

  Returns:
    `True` iff `type1` anf `type2` are equivalent, or else `False`.
  """
  if type1 is None:
    return type2 is None
  else:
    return type2 is not None and (is_assignable_from(type1, type2) and
                                  is_assignable_from(type2, type1))


def check_equivalent_types(type1, type2):
  """Checks that `type1` and `type2` are equivalent.

  Args:
    type1: One type.
    type2: Another type.

  Raises:
    TypeError: If `not are_equivalent_types(type1, type2)`.
  """
  if not are_equivalent_types(type1, type2):
    raise TypeError('Types {} and {} are not equivalent.'.format(type1, type2))


def is_anon_tuple_with_py_container(value, type_spec):
  return (isinstance(value, anonymous_tuple.AnonymousTuple) and isinstance(
      type_spec, computation_types.NamedTupleTypeWithPyContainerType))


def convert_to_py_container(value, type_spec):
  """Recursively convert `AnonymousTuple`s to Python containers.

  This is in some sense the inverse operation to
  `anonymous_tuple.from_container`.

  Args:
    value: An `AnonymousTuple`, in which case this method recurses, replacing
      all `AnonymousTuple`s with the appropriate Python containers if possible
      (and keeping AnonymousTuple otherwise); or some other value, in which case
      that value is returned unmodified immediately (terminating the recursion).
    type_spec: The `tff.Type` to which value should conform, possibly including
      `NamedTupleTypeWithPyContainerType`.

  Returns:
    The input value, with containers converted to appropriate Python
    containers as specified by the `type_spec`.

  Raises:
    ValueError: If the conversion is not possible due to a mix of named
    and unnamed values.
  """
  if not isinstance(value, anonymous_tuple.AnonymousTuple):
    return value

  anon_tuple = value
  py_typecheck.check_type(type_spec, computation_types.NamedTupleType)

  def is_container_type_without_names(container_type):
    return (issubclass(container_type, (list, tuple)) and
            not py_typecheck.is_named_tuple(container_type))

  def is_container_type_with_names(container_type):
    return (py_typecheck.is_named_tuple(container_type) or
            py_typecheck.is_attrs(container_type) or
            issubclass(container_type, dict))

  if isinstance(type_spec, computation_types.NamedTupleTypeWithPyContainerType):
    container_type = (
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            type_spec))
    container_is_anon_tuple = False
  else:
    # TODO(b/133228705): Consider requiring NamedTupleTypeWithPyContainerType.
    container_is_anon_tuple = True
    container_type = anonymous_tuple.AnonymousTuple

  # Avoid projecting the AnonymousTuple into a Python container that is not
  # supported.
  if not container_is_anon_tuple:
    num_named_elements = len(dir(anon_tuple))
    num_unnamed_elements = len(anon_tuple) - num_named_elements
    if num_named_elements > 0 and num_unnamed_elements > 0:
      raise ValueError('Cannot represent value {} with container type {}, '
                       'because value contains a mix of named and unnamed '
                       'elements.'.format(anon_tuple, container_type))
    if (num_named_elements > 0 and
        is_container_type_without_names(container_type)):
      # Note: This could be relaxed in some cases if needed.
      raise ValueError(
          'Cannot represent value {} with named elements '
          'using container type {} which does not support names.'.format(
              anon_tuple, container_type))
    if (num_unnamed_elements > 0 and
        is_container_type_with_names(container_type)):
      # Note: This could be relaxed in some cases if needed.
      raise ValueError('Cannot represent value {} with unnamed elements '
                       'with container type {} which requires names.'.format(
                           anon_tuple, container_type))

  elements = []
  for index, (elem_name,
              elem_type) in enumerate(anonymous_tuple.to_elements(type_spec)):
    value = convert_to_py_container(anon_tuple[index], elem_type)

    if elem_name is None and not container_is_anon_tuple:
      elements.append(value)
    else:
      elements.append((elem_name, value))

  if (py_typecheck.is_named_tuple(container_type) or
      py_typecheck.is_attrs(container_type)):
    # The namedtuple and attr.s class constructors cannot interpret a list of
    # (name, value) tuples; instead call constructor using kwargs. Note
    # that these classes already define an order of names internally,
    # so order does not matter.
    return container_type(**dict(elements))
  else:
    # E.g., tuple and list when elements only has values,
    # but also dict, OrderedDict, or AnonymousTuple when
    # elements has (name, value) tuples.
    return container_type(elements)


def is_concrete_instance_of(type_with_concrete_elements,
                            type_with_abstract_elements):
  """Checks whether abstract types can be concretized via a parallel structure.

  This function builds up a new concrete structure via the bindings encountered
  in `type_with_abstract_elements` in a postorder fashion. That is, it walks the
  type trees in parallel, caching bindings for abstract types on the way. When
  it encounters a previously bound abstract type, it simply inlines this cached
  value. Finally, `abstract_types_can_be_concretized` delegates checking type
  equivalence to `are_equivalent_types`, passing in the created concrete
  structure for comparison with `type_with_concrete_elements`.

  Args:
    type_with_concrete_elements: Instance of `computation_types.Type` of
      parallel structure to `type_with_concrete_elements`, containing only
      concrete types, to test for equivalence with a concretization of
      `type_with_abstract_elements`.
    type_with_abstract_elements: Instance of `computation_types.Type` which may
      contain abstract types, to check for possibility of concretizing according
      to `type_with_concrete_elements`.

  Returns:
    `True` if `type_with_abstract_elements` can be concretized to
    `type_with_concrete_elements`. Returns `False` if they are of the same
    structure but some conflicting assignment exists in
    `type_with_concrete_elements`.

  Raises:
    TypeError: If `type_with_abstract_elements` and
    `type_with_concrete_elements` are not structurally equivalent; that is,
    their type trees are of different structure; or if
    `type_with_concrete_elements` contains abstract elements.
  """
  py_typecheck.check_type(type_with_abstract_elements, computation_types.Type)
  py_typecheck.check_type(type_with_concrete_elements, computation_types.Type)

  if type_tree_contains_types(type_with_concrete_elements,
                              computation_types.AbstractType):
    raise TypeError(
        '`type_with_concrete_elements` must contain no abstract types. You '
        'have passed {}'.format(type_with_concrete_elements))

  bound_abstract_types = {}
  type_error_string = ('Structural mismatch encountered while concretizing '
                       'abstract types. The structure of {} does not match the '
                       'structure of {}').format(type_with_abstract_elements,
                                                 type_with_concrete_elements)

  def _concretize_abstract_types(abstract_type_spec, concrete_type_spec):
    """Recursive helper function to construct concrete type spec."""
    if isinstance(abstract_type_spec, computation_types.AbstractType):
      bound_type = bound_abstract_types.get(str(abstract_type_spec.label))
      if bound_type:
        return bound_type
      else:
        bound_abstract_types[str(abstract_type_spec.label)] = concrete_type_spec
        return concrete_type_spec
    elif isinstance(abstract_type_spec, computation_types.TensorType):
      return abstract_type_spec
    elif isinstance(abstract_type_spec, computation_types.NamedTupleType):
      if not isinstance(concrete_type_spec, computation_types.NamedTupleType):
        raise TypeError(type_error_string)
      abstract_elements = anonymous_tuple.to_elements(abstract_type_spec)
      concrete_elements = anonymous_tuple.to_elements(concrete_type_spec)
      if len(abstract_elements) != len(concrete_elements):
        raise TypeError(type_error_string)
      concretized_tuple_elements = []
      for k in range(len(abstract_elements)):
        if abstract_elements[k][0] != concrete_elements[k][0]:
          raise TypeError(type_error_string)
        concretized_tuple_elements.append(
            (abstract_elements[k][0],
             _concretize_abstract_types(abstract_elements[k][1],
                                        concrete_elements[k][1])))
      return computation_types.NamedTupleType(concretized_tuple_elements)
    elif isinstance(abstract_type_spec, computation_types.SequenceType):
      if not isinstance(concrete_type_spec, computation_types.SequenceType):
        raise TypeError(type_error_string)
      return computation_types.SequenceType(
          _concretize_abstract_types(abstract_type_spec.element,
                                     concrete_type_spec.element))
    elif isinstance(abstract_type_spec, computation_types.FunctionType):
      if not isinstance(concrete_type_spec, computation_types.FunctionType):
        raise TypeError(type_error_string)
      concretized_param = _concretize_abstract_types(
          abstract_type_spec.parameter, concrete_type_spec.parameter)
      concretized_result = _concretize_abstract_types(abstract_type_spec.result,
                                                      concrete_type_spec.result)
      return computation_types.FunctionType(concretized_param,
                                            concretized_result)
    elif isinstance(abstract_type_spec, computation_types.PlacementType):
      if not isinstance(concrete_type_spec, computation_types.PlacementType):
        raise TypeError(type_error_string)
      return abstract_type_spec
    elif isinstance(abstract_type_spec, computation_types.FederatedType):
      if not isinstance(concrete_type_spec, computation_types.FederatedType):
        raise TypeError(type_error_string)
      new_member = _concretize_abstract_types(abstract_type_spec.member,
                                              concrete_type_spec.member)
      return computation_types.FederatedType(new_member,
                                             abstract_type_spec.placement,
                                             abstract_type_spec.all_equal)
    elif abstract_type_spec is None:
      if concrete_type_spec is not None:
        raise TypeError(type_error_string)
      return None
    else:
      raise TypeError(
          'Unexpected abstract typespec {}.'.format(abstract_type_spec))

  concretized_abstract_type = _concretize_abstract_types(
      type_with_abstract_elements, type_with_concrete_elements)

  return are_equivalent_types(concretized_abstract_type,
                              type_with_concrete_elements)


def transform_type_postorder(type_signature, transform_fn):
  """Walks type tree of `type_signature` postorder, calling `transform_fn`.

  Args:
    type_signature: Instance of `computation_types.Type` to transform
      recursively.
    transform_fn: Transformation function to apply to each node in the type tree
      of `type_signature`. Must be instance of Python function type.

  Returns:
    A possibly transformed version of `type_signature`, with each node in its
    tree the result of applying `transform_fn` to the corresponding node in
    `type_signature`.

  Raises:
    TypeError: If the types don't match the specification above.
  """
  # TODO(b/134525440): Investigate unifying the recursive methods in type_utils,
  # rather than proliferating them.
  # TODO(b/134595038): Revisit the change here to add a mutated flag.
  py_typecheck.check_type(type_signature, computation_types.Type)
  py_typecheck.check_callable(transform_fn)
  if isinstance(type_signature, computation_types.FederatedType):
    transformed_member, member_mutated = transform_type_postorder(
        type_signature.member, transform_fn)
    if member_mutated:
      type_signature = computation_types.FederatedType(transformed_member,
                                                       type_signature.placement,
                                                       type_signature.all_equal)
    fed_type_signature, type_signature_mutated = transform_fn(type_signature)
    return fed_type_signature, type_signature_mutated or member_mutated
  elif isinstance(type_signature, computation_types.SequenceType):
    transformed_element, element_mutated = transform_type_postorder(
        type_signature.element, transform_fn)
    if element_mutated:
      type_signature = computation_types.SequenceType(transformed_element)
    seq_type_signature, type_signature_mutated = transform_fn(type_signature)
    return seq_type_signature, type_signature_mutated or element_mutated
  elif isinstance(type_signature, computation_types.FunctionType):
    transformed_param, param_mutated = transform_type_postorder(
        type_signature.parameter, transform_fn)
    transformed_result, result_mutated = transform_type_postorder(
        type_signature.result, transform_fn)
    if param_mutated or result_mutated:
      type_signature = computation_types.FunctionType(transformed_param,
                                                      transformed_result)
    fn_type_signature, fn_mutated = transform_fn(type_signature)
    return fn_type_signature, fn_mutated or param_mutated or result_mutated
  elif isinstance(type_signature, computation_types.NamedTupleType):
    elems = []
    elems_mutated = False
    for element in anonymous_tuple.iter_elements(type_signature):
      transformed_element, element_mutated = transform_type_postorder(
          element[1], transform_fn)
      elems_mutated = elems_mutated or element_mutated
      elems.append((element[0], transformed_element))
    if elems_mutated:
      if isinstance(type_signature,
                    computation_types.NamedTupleTypeWithPyContainerType):
        type_signature = computation_types.NamedTupleTypeWithPyContainerType(
            elems,
            computation_types.NamedTupleTypeWithPyContainerType
            .get_container_type(type_signature))
      else:
        type_signature = computation_types.NamedTupleType(elems)
    tuple_type_signature, tuple_mutated = transform_fn(type_signature)
    return tuple_type_signature, elems_mutated or tuple_mutated
  elif isinstance(type_signature,
                  (computation_types.AbstractType, computation_types.TensorType,
                   computation_types.PlacementType)):
    return transform_fn(type_signature)


def reconcile_value_with_type_spec(value, type_spec):
  """Reconciles the type of `value` with the given `type_spec`.

  The currently implemented logic only performs reconciliation of `value` and
  `type` for values that implement `tff.TypedObject`. Future extensions may
  perform reconciliation for a greater range of values; the caller should not
  depend on the limited implementation. This method may fail in case of any
  incompatibility between `value` and `type_spec`. In any case, the method is
  going to fail if the type cannot be determined.

  Args:
    value: An object that represents a value.
    type_spec: An instance of `tff.Type` or something convertible to it.

  Returns:
    An instance of `tff.Type`. If `value` is not a `tff.TypedObject`, this is
    the same as `type_spec`, which in this case must not be `None`. If `value`
    is a `tff.TypedObject`, and `type_spec` is `None`, this is simply the type
    signature of `value.` If the `value` is a `tff.TypedObject` and `type_spec`
    is not `None`, this is `type_spec` to the extent that it is eqiuvalent to
    the type signature of `value`, otherwise an exception is raised.

  Raises:
    TypeError: If the `value` type and `type_spec` are incompatible, or if the
      type cannot be determined..
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(value, typed_object.TypedObject):
    return reconcile_value_type_with_type_spec(value.type_signature, type_spec)
  elif type_spec is not None:
    return type_spec
  else:
    raise TypeError(
        'Cannot derive an eager representation for a value of an unknown type.')


def reconcile_value_type_with_type_spec(value_type, type_spec):
  """Reconciles a pair of types.

  Args:
    value_type: An instance of `tff.Type` or something convertible to it. Must
      not be `None`.
    type_spec: An instance of `tff.Type`, something convertible to it, or
      `None`.

  Returns:
    Either `value_type` if `type_spec` is `None`, or `type_spec` if `type_spec`
    is not `None` and rquivalent with `value_type`.

  Raises:
    TypeError: If arguments are of incompatible types.
  """
  value_type = computation_types.to_type(value_type)
  py_typecheck.check_type(value_type, computation_types.Type)
  if type_spec is None:
    return value_type
  else:
    type_spec = computation_types.to_type(type_spec)
    if are_equivalent_types(value_type, type_spec):
      return type_spec
    else:
      raise TypeError('Expected a value of type {}, found {}.'.format(
          type_spec, value_type))


def to_non_all_equal(type_spec):
  """Constructs a non-`all_equal` version of the federated type `type_spec`.

  Args:
    type_spec: An instance of `tff.FederatedType`.

  Returns:
    A federated type with the same member and placement, but `all_equal=False`.
  """
  py_typecheck.check_type(type_spec, computation_types.FederatedType)
  return computation_types.FederatedType(
      type_spec.member, type_spec.placement, all_equal=False)


def check_valid_federated_weighted_mean_argument_tuple_type(type_spec):
  """Checks that `type_spec` is a valid type of a federated weighted mean arg.

  Args:
    type_spec: An instance of `tff.Type` or something convertible to it.

  Raises:
    TypeError: If the check fails.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_not_none(type_spec)
  py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
  if len(type_spec) != 2:
    raise TypeError('Expected a 2-tuple, found {}.'.format(type_spec))
  for _, v in anonymous_tuple.iter_elements(type_spec):
    check_federated_type(v, None, placement_literals.CLIENTS, False)
    if not is_average_compatible(v.member):
      raise TypeError(
          'Expected average-compatible args, got {} from argument of type {}.'
          .format(v.member, type_spec))
  w_type = type_spec[1].member
  py_typecheck.check_type(w_type, computation_types.TensorType)
  if w_type.shape.ndims != 0:
    raise TypeError('Expected scalar weight, got {}.'.format(w_type))
