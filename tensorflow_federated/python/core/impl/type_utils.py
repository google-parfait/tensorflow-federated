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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import numpy as np
from six import string_types
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import types
from tensorflow_federated.python.core.api import value_base

from tensorflow_federated.python.core.impl import anonymous_tuple


def infer_type(arg):
  """Infers the TFF type of the argument (an instance of types.Type).

  WARNING: This function is only partially implemented.

  The kinds of arguments that are currently correctly recognized:
  - tensors, variables, and data sets,
  - things that are convertible to tensors (including numpy arrays, builtin
    types, as well as lists and tuples of any of the above, etc.),
  - nested lists, tuples, namedtuples, anonymous tuples, dict, and OrderedDicts.

  Args:
    arg: The argument, the TFF type of which to infer.

  Returns:
    Either an instance of types.Type, or None if the argument is None.
  """
  # TODO(b/113112885): Implement the remaining cases here on the need basis.
  if arg is None:
    return None
  elif isinstance(arg, value_base.Value):
    return arg.type_signature
  elif tf.contrib.framework.is_tensor(arg):
    return types.TensorType(arg.dtype.base_dtype, arg.shape)
  elif isinstance(arg, (np.generic, np.ndarray)):
    return types.TensorType(tf.as_dtype(arg.dtype), arg.shape)
  elif isinstance(arg, tf.data.Dataset):
    return types.SequenceType(
        tf_dtypes_and_shapes_to_type(arg.output_types, arg.output_shapes))
  elif isinstance(arg, anonymous_tuple.AnonymousTuple):
    return types.NamedTupleType([
        (k, infer_type(v)) if k else infer_type(v) for k, v in (
            anonymous_tuple.to_elements(arg))])
  elif '_asdict' in type(arg).__dict__:
    # Special handling needed for collections.namedtuple.
    return infer_type(arg._asdict())
  elif isinstance(arg, dict):
    # This also handles 'OrderedDict', as it inherits from 'dict'.
    return types.NamedTupleType([
        (k, infer_type(v)) for k, v in arg.iteritems()])
  # Quickly try special-casing a few very common built-in scalar types before
  # applying any kind of heavier-weight processing.
  elif isinstance(arg, string_types):
    return types.TensorType(tf.string)
  else:
    dtype = {bool: tf.bool, int: tf.int32, float: tf.float32}.get(type(arg))
    if dtype:
      return types.TensorType(dtype)
    else:
      # Now fall back onto the heavier-weight processing, as all else failed.
      # Use make_tensor_proto() to make sure to handle it consistently with
      # how TensorFlow is handling values (e.g., recognizing int as int32, as
      # opposed to int64 as in NumPy).
      try:
        # TODO(b/113112885): Find something more lightweight we could use here.
        tensor_proto = tf.make_tensor_proto(arg)
        return types.TensorType(
            tf.DType(tensor_proto.dtype),
            tf.TensorShape(tensor_proto.tensor_shape))
      except TypeError as err:
        # We could not convert to a tensor type. First, check if we are dealing
        # with a list or tuple, as those can be reinterpreted as named tuples.
        if isinstance(arg, (tuple, list)):
          return types.NamedTupleType([infer_type(e) for e in arg])
        else:
          # If neiter a tuple nor a list, we are out of options.
          raise TypeError('Could not infer the TFF type of {}: {}.'.format(
              py_typecheck.type_string(type(arg)), str(err)))


def tf_dtypes_and_shapes_to_type(dtypes, shapes):
  """Returns types.Type for the givem TensorFlows's (dtypes, shapes) combo.

  The returned dtypes and shapes match those used by tf.Datasets to indicate
  the type and shape of their elements. They can be used, e.g., as arguments in
  constructing an iterator over a string handle. Note that the nested structure
  of dtypes and shapes must be identical.

  Args:
    dtypes: A nested structure of dtypes, such as what is returned by Dataset's
      output_dtypes property.
    shapes: A nested structure of shapes, such as what is returned by Dataset's
      output_shapes property.

  Returns:
    The corresponding instance of types.Type.

  Raises:
    TypeError: if the arguments are of types that weren't recognized.
  """
  tf.contrib.framework.nest.assert_same_structure(dtypes, shapes)
  if isinstance(dtypes, tf.DType):
    return types.TensorType(dtypes, shapes)
  elif '_asdict' in type(dtypes).__dict__:
    # Special handling needed for collections.namedtuple due to the lack of
    # a base class. Note this must precede the test for being a list.
    return tf_dtypes_and_shapes_to_type(dtypes._asdict(), shapes._asdict())
  elif isinstance(dtypes, dict):
    # This also handles 'OrderedDict', as it inherits from 'dict'.
    return types.NamedTupleType([
        (name, tf_dtypes_and_shapes_to_type(dtypes_elem, shapes[name]))
        for name, dtypes_elem in dtypes.iteritems()])
  elif isinstance(dtypes, (list, tuple)):
    return types.NamedTupleType([
        tf_dtypes_and_shapes_to_type(dtypes_elem, shapes[idx])
        for idx, dtypes_elem in enumerate(dtypes)])
  else:
    raise TypeError(
        'Unrecognized: dtypes {}, shapes {}.'.format(str(dtypes), str(shapes)))


def type_to_tf_dtypes_and_shapes(type_spec):
  """Returns nested structures of tensor dtypes and shapes for a given TFF type.

  The returned dtypes and shapes match those used by tf.Datasets to indicate
  the type and shape of their elements. They can be used, e.g., as arguments in
  constructing an iterator over a string handle.

  Args:
    type_spec: Type specification, either an instance of types.Type, or
      something convertible to it. Ther type specification must be composed of
      only named tuples and tensors. In all named tuples that appear in the
      type spec, all the elements must be named.

  Returns:
    A pair of parallel nested structures with the dtypes and shapes of tensors
    defined in type_spec. The layout of the two structures returned is the same
    as the layout of the nesdted type defined by type_spec. Named tuples are
    represented as dictionaries.

  Raises:
    ValueError: if the type_spec is composed of something other than named
      tuples and tensors, or if any of the elements in named tuples are unnamed.
  """
  type_spec = types.to_type(type_spec)
  if isinstance(type_spec, types.TensorType):
    return (type_spec.dtype, type_spec.shape)
  elif isinstance(type_spec, types.NamedTupleType):
    output_dtypes = collections.OrderedDict()
    output_shapes = collections.OrderedDict()
    for e in type_spec.elements:
      element_name = e[0]
      element_spec = e[1]
      if element_name is None:
        # TODO(b/113112108): Possibly remove this limitation.
        raise ValueError(
            'When a sequence appears as a part of a parameter to a section of '
            'TensorFlow code, in the type signature of elements of that '
            'sequence all named tuples must have their elements explicitly '
            'named, and this does not appear to be the case in {}.'.format(
                str(type_spec)))
      element_output = type_to_tf_dtypes_and_shapes(element_spec)
      output_dtypes[element_name] = element_output[0]
      output_shapes[element_name] = element_output[1]
    return (output_dtypes, output_shapes)
  else:
    raise ValueError('Unsupported type {}.'.format(
        py_typecheck.type_string(type(type_spec))))


def get_named_tuple_element_type(type_spec, name):
  """Returns the type of a named tuple member.

  Args:
    type_spec: Type specification, either an instance of types.Type or something
      convertible to it by types.to_type().
    name: The string name of the named tuple member.

  Returns:
    The TFF type of the element.

  Raises:
    TypeError: if arguments are of the wrong types.
    ValueError: if the tuple does not have an element with the given name.
  """
  py_typecheck.check_type(name, string_types)
  type_spec = types.to_type(type_spec)
  py_typecheck.check_type(type_spec, types.NamedTupleType)
  elements = type_spec.elements
  for elem_name, elem_type in elements:
    if name == elem_name:
      return elem_type
  raise ValueError(
      'The name \'{}\' of the element does not correspond to '
      'any of the names {} in the named tuple type.'.format(
          name, str([e[0] for e in elements if e[0]])))


def check_well_formed(type_spec):
  """Checks that 'type_spec' represents a well-formed type.

  WARNING: This function is only partially implemented. See the documentation
  in 'computation.proto' for what factors determine well-formedness.

  Args:
    type_spec: The type specification to check, either an instance of
      types.Type or something convertible to it by types.to_type().

  Raises:
    TypeError: if arguments are of the wrong types, or if 'type_spec' is not a
      well-formed TFF type.
  """
  # TODO(b/113112885): Implement the remainder of this function, and document
  # the conditions for well-formedness here as well.
  check_all_abstract_types_are_bound(type_spec)


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
    type_spec: An instance of types.Type, or something convertible to it.

  Raises:
    TypeError: if arguments are of the wrong types, or if unbound type labels
      occur in 'type_spec'.
  """
  def _check_or_get_unbound_abstract_type_labels(
      type_spec, bound_labels, check):
    """Checks or collects abstract type labels from 'type_spec'.

    This is a helper function used by 'check_abstract_types_are_bound', not to
    be exported out of this module.

    Args:
      type_spec: An instance of types.Type.
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
    py_typecheck.check_type(type_spec, types.Type)
    if isinstance(type_spec, types.TensorType):
      return set()
    elif isinstance(type_spec, types.SequenceType):
      return _check_or_get_unbound_abstract_type_labels(
          type_spec.element, bound_labels, check)
    elif isinstance(type_spec, types.NamedTupleType):
      return set().union(*[
          _check_or_get_unbound_abstract_type_labels(v, bound_labels, check)
          for _, v in type_spec.elements])
    elif isinstance(type_spec, types.AbstractType):
      if type_spec.label in bound_labels:
        return set()
      elif not check:
        return set([type_spec.label])
      else:
        raise TypeError('Unbound type label \'{}\'.'.format(type_spec.label))
    elif isinstance(type_spec, types.FunctionType):
      parameter_labels = (
          set() if not type_spec.parameter
          else _check_or_get_unbound_abstract_type_labels(
              type_spec.parameter, bound_labels, False))
      result_labels = _check_or_get_unbound_abstract_type_labels(
          type_spec.result, bound_labels.union(parameter_labels), check)
      return parameter_labels.union(result_labels)

  _check_or_get_unbound_abstract_type_labels(
      types.to_type(type_spec), set(), True)
