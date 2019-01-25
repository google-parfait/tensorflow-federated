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

import numpy as np
import six
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import placement_literals


def infer_type(arg):
  """Infers the TFF type of the argument (a computation_types.Type instance).

  WARNING: This function is only partially implemented.

  The kinds of arguments that are currently correctly recognized:
  - tensors, variables, and data sets,
  - things that are convertible to tensors (including numpy arrays, builtin
    types, as well as lists and tuples of any of the above, etc.),
  - nested lists, tuples, namedtuples, anonymous tuples, dict, and OrderedDicts.

  Args:
    arg: The argument, the TFF type of which to infer.

  Returns:
    Either an instance of computation_types.Type, or None if the argument is
    None.
  """
  # TODO(b/113112885): Implement the remaining cases here on the need basis.
  if arg is None:
    return None
  elif isinstance(arg, (value_base.Value, computation_base.Computation)):
    return arg.type_signature
  elif tf.contrib.framework.is_tensor(arg):
    return computation_types.TensorType(arg.dtype.base_dtype, arg.shape)
  elif isinstance(arg, (np.generic, np.ndarray)):
    return computation_types.TensorType(tf.as_dtype(arg.dtype), arg.shape)
  elif isinstance(arg, tf.data.Dataset):
    return computation_types.SequenceType(
        tf_dtypes_and_shapes_to_type(arg.output_types, arg.output_shapes))
  elif isinstance(arg, anonymous_tuple.AnonymousTuple):
    return computation_types.NamedTupleType(
        [(k, infer_type(v)) if k else infer_type(v)
         for k, v in anonymous_tuple.to_elements(arg)])
  elif py_typecheck.is_named_tuple(arg):
    # Special handling needed for collections.namedtuple.
    return infer_type(arg._asdict())
  elif isinstance(arg, dict):
    if isinstance(arg, collections.OrderedDict):
      items = six.iteritems(arg)
    else:
      items = sorted(six.iteritems(arg))
    return computation_types.NamedTupleType(
        [(k, infer_type(v)) for k, v in items])
  # Quickly try special-casing a few very common built-in scalar types before
  # applying any kind of heavier-weight processing.
  elif isinstance(arg, six.string_types):
    return computation_types.TensorType(tf.string)
  elif isinstance(arg, (tuple, list)):
    return computation_types.NamedTupleType([infer_type(e) for e in arg])
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
            tf.DType(tensor_proto.dtype),
            tf.TensorShape(tensor_proto.tensor_shape))
      except TypeError as err:
        raise TypeError('Could not infer the TFF type of {}: {}.'.format(
            py_typecheck.type_string(type(arg)), str(err)))


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
      items = six.iteritems(value)
    else:
      items = sorted(six.iteritems(value))
    return anonymous_tuple.AnonymousTuple(
        [(k, to_canonical_value(v)) for k, v in items])
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
            str(type_spec), str(val_type)))


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
  tf.contrib.framework.nest.assert_same_structure(dtypes, shapes)
  if isinstance(dtypes, tf.DType):
    return computation_types.TensorType(dtypes, shapes)
  elif py_typecheck.is_named_tuple(dtypes):
    # Special handling needed for collections.namedtuple due to the lack of
    # a base class. Note this must precede the test for being a list.
    return tf_dtypes_and_shapes_to_type(dtypes._asdict(), shapes._asdict())
  elif isinstance(dtypes, dict):
    if isinstance(dtypes, collections.OrderedDict):
      items = six.iteritems(dtypes)
    else:
      items = sorted(six.iteritems(dtypes))
    elements = [(name, tf_dtypes_and_shapes_to_type(dtypes_elem, shapes[name]))
                for name, dtypes_elem in items]
    return computation_types.NamedTupleType(elements)
  elif isinstance(dtypes, (list, tuple)):
    return computation_types.NamedTupleType([
        tf_dtypes_and_shapes_to_type(dtypes_elem, shapes[idx])
        for idx, dtypes_elem in enumerate(dtypes)
    ])
  else:
    raise TypeError('Unrecognized: dtypes {}, shapes {}.'.format(
        str(dtypes), str(shapes)))


def type_to_tf_dtypes_and_shapes(type_spec):
  """Returns nested structures of tensor dtypes and shapes for a given TFF type.

  The returned dtypes and shapes match those used by `tf.data.Dataset`s to
  indicate the type and shape of their elements. They can be used, e.g., as
  arguments in constructing an iterator over a string handle.

  Args:
    type_spec: Type specification, either an instance of
      `computation_types.Type`, or something convertible to it. Ther type
      specification must be composed of only named tuples and tensors. In all
      named tuples that appear in the type spec, all the elements must be named.

  Returns:
    A pair of parallel nested structures with the dtypes and shapes of tensors
    defined in `type_spec`. The layout of the two structures returned is the
    same as the layout of the nesdted type defined by `type_spec`. Named tuples
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
    if elements[0][0] is not None:
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
                  str(type_spec)))
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
                  str(type_spec)))
        element_output = type_to_tf_dtypes_and_shapes(element_spec)
        output_dtypes.append(element_output[0])
        output_shapes.append(element_output[1])
    return (output_dtypes, output_shapes)
  else:
    raise ValueError('Unsupported type {}.'.format(
        py_typecheck.type_string(type(type_spec))))


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
  py_typecheck.check_type(name, six.string_types)
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
  elements = anonymous_tuple.to_elements(type_spec)
  for elem_name, elem_type in elements:
    if name == elem_name:
      return elem_type
  raise ValueError('The name \'{}\' of the element does not correspond to '
                   'any of the names {} in the named tuple type.'.format(
                       name, str([e[0] for e in elements if e[0]])))


def preorder_call(given_type, func, arg):
  """Recursively calls `func` on the possibly nested structure `given_type`.

  Walks the tree in a preorder manner. Updates `arg` on the way down with
  the appropriate information, as defined in `func`.

  Args:
    given_type: Possibly nested `computation_types.Type` or object convertible
      to it by `computation_types.to_type`.
    func: Function to apply to each of the constituent elements of `given_type`
      with the argument `arg`. Must return an updated version of `arg` which
      incorporated the information we'd like to track as we move down the nested
      type tree.
    arg: Initial state of information to be passed down the tree.
  """
  type_signature = computation_types.to_type(given_type)
  arg = func(type_signature, arg)
  if isinstance(type_signature, computation_types.FederatedType):
    preorder_call(type_signature.member, func, arg)
  elif isinstance(type_signature, computation_types.SequenceType):
    preorder_call(type_signature.element, func, arg)
  elif isinstance(type_signature, computation_types.FunctionType):
    preorder_call(type_signature.parameter, func, arg)
    preorder_call(type_signature.result, func, arg)
  elif isinstance(type_signature, computation_types.NamedTupleType):
    for element in anonymous_tuple.to_elements(type_signature):
      preorder_call(element[1], func, arg)


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

  Returns:
    True iff `type_spec` represents a well-formed TFF type.

  Raises:
    TypeError: if `type_spec` is not a well-formed TFF type.
  """

  # TODO(b/113112885): Reinstate a call to `check_all_abstract_types_are_bound`
  # after revising the definition of well-formedness.
  type_signature = computation_types.to_type(type_spec)

  def _check_for_disallowed_type(type_to_check, disallowed_types):
    """Checks subtree of `type_to_check` for nonlocal `disallowed_types`."""
    if isinstance(type_to_check, tuple(disallowed_types)):
      raise TypeError('A {} has been encountered in the given type signature, '
                      ' but {} is disallowed.'.format(type_to_check,
                                                      disallowed_types))
    if isinstance(type_to_check, computation_types.FederatedType):
      disallowed_types = set(
          [computation_types.FederatedType,
           computation_types.FunctionType]).union(disallowed_types)
    if isinstance(type_to_check, computation_types.SequenceType):
      disallowed_types = set([
          computation_types.FederatedType, computation_types.FunctionType,
          computation_types.SequenceType
      ]).union(disallowed_types)
    return disallowed_types

  preorder_call(type_signature, _check_for_disallowed_type, set())
  return True


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
          for _, v in anonymous_tuple.to_elements(type_spec)
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
        is_sum_compatible(v) for _, v in anonymous_tuple.to_elements(type_spec))
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_sum_compatible(type_spec.member)
  else:
    return False


def check_federated_type(type_spec, member=None, placement=None,
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
    if type_spec.placement != placement:
      raise TypeError(
          'Expected federated type placed at {}, got one placed at {}.'.format(
              str(placement), str(type_spec.placement)))
  if all_equal is not None:
    py_typecheck.check_type(all_equal, bool)
    if type_spec.all_equal != all_equal:
      raise TypeError(
          'Expected federated type with all_equial {}, got one with {}.'.format(
              str(all_equal), str(type_spec.all_equal)))


def check_federated_value_placement(value, placement, label=None):
  """Checks that `value` is a federated value placed at `placement`.

  Args:
    value: The value to check, an instance of value_base.Value.
    placement: The expected placement.
    label: An optional string label that describes `value`.

  Raises:
    TypeError: if `value` is not a value_base.Value of a federated type with
      the expected placement `placement`.
  """
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, computation_types.FederatedType)
  if label is not None:
    py_typecheck.check_type(label, six.string_types)
  if value.type_signature.placement is not placement:
    raise TypeError('The {} should be placed at {}, but it '
                    'is placed at {}.'.format(
                        label if label else 'value', str(placement),
                        str(value.type_signature.placement)))


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
        for _, v in anonymous_tuple.to_elements(type_spec))
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

      return ((x.ndims == y.ndims) and ((x.dims is None) or all(
          _dimension_is_assignable_from(x.dims[k], y.dims[k])
          for k in range(x.ndims))))

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
    raise TypeError('Unexpected target type {}.'.format(str(target_type)))


def check_assignable_from(target, source):
  target = computation_types.to_type(target)
  source = computation_types.to_type(source)
  if not is_assignable_from(target, source):
    raise TypeError(
        'The target type {} is not assignable from source type {}.'.format(
            str(target), str(source)))


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
