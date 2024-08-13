# Copyright 2019, The TensorFlow Federated Authors.
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
"""A library of static analysis functions for computation types."""

import collections
from collections.abc import Callable
from typing import Optional

import ml_dtypes
import numpy as np

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import array_shape
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_transformations

_TypePredicate = Callable[[computation_types.Type], bool]


def preorder_types(type_signature: computation_types.Type):
  """Yields each type in `type_signature` in a preorder fashion."""
  yield type_signature
  for child in type_signature.children():
    yield from preorder_types(child)


def count(
    type_signature: computation_types.Type, predicate: _TypePredicate
) -> int:
  """Returns the number of types in `type_signature` matching `predicate`.

  Args:
    type_signature: A tree of `computation_type.Type`s to count.
    predicate: A Python function that takes a type as a parameter and returns a
      boolean value.
  """
  one_or_zero = lambda t: 1 if predicate(t) else 0
  return sum(map(one_or_zero, preorder_types(type_signature)))


def contains(
    type_signature: computation_types.Type, predicate: _TypePredicate
) -> bool:
  """Checks if `type_signature` contains any types that pass `predicate`."""
  for t in preorder_types(type_signature):
    if predicate(t):
      return True
  return False


def contains_federated_types(type_signature):
  """Returns whether or not `type_signature` contains a federated type."""
  return contains(
      type_signature, lambda t: isinstance(t, computation_types.FederatedType)
  )


def contains_tensor_types(type_signature):
  """Returns whether or not `type_signature` contains a tensor type."""
  return contains(
      type_signature, lambda t: isinstance(t, computation_types.TensorType)
  )


def contains_only(
    type_signature: computation_types.Type,
    predicate: _TypePredicate,
) -> bool:
  """Checks if `type_signature` contains only types that pass `predicate`."""
  return not contains(type_signature, lambda t: not predicate(t))


def check_type(value: object, type_spec: computation_types.Type):
  """Checks whether `val` is of TFF type `type_spec`.

  Args:
    value: The object to check.
    type_spec: A `computation_types.Type`, the type that `value` is checked
      against.

  Raises:
    TypeError: If the inferred type of `value` is not assignable to `type_spec`.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  value_type = type_conversions.infer_type(value)
  if not type_spec.is_assignable_from(value_type):
    raise TypeError(
        computation_types.type_mismatch_error_message(
            value_type,
            type_spec,
            computation_types.TypeRelation.ASSIGNABLE,
            second_is_expected=True,
        )
    )


def is_tensorflow_compatible_type(type_spec):
  """Checks `type_spec` against an explicit list of `tf_computation`."""
  if type_spec is None:
    return True

  def _predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.SequenceType,
            computation_types.StructType,
            computation_types.TensorType,
        ),
    )

  return contains_only(type_spec, _predicate)


def is_structure_of_tensors(type_spec: computation_types.Type) -> bool:
  def _predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.StructType,
            computation_types.TensorType,
        ),
    )

  return contains_only(type_spec, _predicate)


def check_tensorflow_compatible_type(type_spec):
  if not is_tensorflow_compatible_type(type_spec):
    raise TypeError(
        'Expected type to be compatible with TensorFlow (i.e. tensor, '
        'sequence, or tuple types), found {}.'.format(type_spec)
    )


def is_generic_op_compatible_type(type_spec):
  """Checks `type_spec` against an explicit list of generic operators."""
  if type_spec is None:
    return False

  def _predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.TensorType,
            computation_types.StructType,
        ),
    )

  return contains_only(type_spec, _predicate)


def is_binary_op_with_upcast_compatible_pair(
    possibly_nested_type: Optional[computation_types.Type],
    type_to_upcast: computation_types.Type,
) -> bool:
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
    possibly_nested_type: A `computation_types.Type`, or `None`.
    type_to_upcast: A `computation_types.Type`, or `None`.

  Returns:
    Boolean indicating whether `type_to_upcast` can be upcast to
    `possibly_nested_type` in the manner described above.
  """
  if possibly_nested_type is not None:
    py_typecheck.check_type(possibly_nested_type, computation_types.Type)
  if type_to_upcast is not None:
    py_typecheck.check_type(type_to_upcast, computation_types.Type)
  if not (
      is_generic_op_compatible_type(possibly_nested_type)
      and is_generic_op_compatible_type(type_to_upcast)
  ):
    return False
  if possibly_nested_type is None:
    return type_to_upcast is None
  if possibly_nested_type.is_equivalent_to(type_to_upcast):
    return True
  if not isinstance(
      type_to_upcast, computation_types.TensorType
  ) or not array_shape.is_shape_scalar(type_to_upcast.shape):
    return False

  types_are_ok = [True]

  only_allowed_dtype = type_to_upcast.dtype  # pytype: disable=attribute-error

  def _check_tensor_types(type_spec):
    if (
        isinstance(type_spec, computation_types.TensorType)
        and type_spec.dtype != only_allowed_dtype
    ):  # pytype: disable=attribute-error
      types_are_ok[0] = False
    return type_spec, False

  type_transformations.transform_type_postorder(
      possibly_nested_type, _check_tensor_types
  )

  return types_are_ok[0]


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

  def _check_or_get_unbound_abstract_type_labels(
      type_spec, bound_labels, check
  ):
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
      return _check_or_get_unbound_abstract_type_labels(
          type_spec.element, bound_labels, check
      )
    elif isinstance(type_spec, computation_types.FederatedType):
      return _check_or_get_unbound_abstract_type_labels(
          type_spec.member, bound_labels, check
      )
    elif isinstance(type_spec, computation_types.StructType):
      return set().union(
          *[
              _check_or_get_unbound_abstract_type_labels(v, bound_labels, check)
              for _, v in structure.iter_elements(type_spec)
          ]
      )
    elif isinstance(type_spec, computation_types.AbstractType):
      if type_spec.label in bound_labels:
        return set()
      elif not check:
        return set([type_spec.label])
      else:
        raise TypeError("Unbound type label '{}'.".format(type_spec.label))
    elif isinstance(type_spec, computation_types.FunctionType):
      if type_spec.parameter is None:
        parameter_labels = set()
      else:
        parameter_labels = _check_or_get_unbound_abstract_type_labels(
            type_spec.parameter, bound_labels, False
        )
      result_labels = _check_or_get_unbound_abstract_type_labels(
          type_spec.result, bound_labels.union(parameter_labels), check
      )
      return parameter_labels.union(result_labels)

  _check_or_get_unbound_abstract_type_labels(type_spec, set(), True)


class SumIncompatibleError(TypeError):

  def __init__(self, type_spec, type_spec_context, reason):
    message = (
        'Expected a type which is compatible with the sum operator, found\n'
        f'{type_spec_context}\nwhich contains\n{type_spec}\nwhich is not '
        f'sum-compatible because {reason}.'
    )
    super().__init__(message)


def check_is_sum_compatible(type_spec, type_spec_context=None):
  """Determines if `type_spec` is a type that can be added to itself.

  Types that are sum-compatible are composed of scalars of numeric types,
  possibly packaged into nested named tuples, and possibly federated. Types
  that are sum-incompatible include sequences, functions, abstract types,
  and placements.

  Args:
    type_spec: A `computation_types.Type`.
    type_spec_context: An optional parent type to include in the error message.

  Raises:
     SumIncompatibleError: if `type_spec` is not sum-compatible.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if type_spec_context is None:
    type_spec_context = type_spec
  py_typecheck.check_type(type_spec_context, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    if not (
        np.issubdtype(type_spec.dtype, np.number)
        or type_spec.dtype == ml_dtypes.bfloat16
    ):
      raise SumIncompatibleError(
          type_spec, type_spec_context, f'{type_spec.dtype} is not numeric'
      )
    if not array_shape.is_shape_fully_defined(type_spec.shape):
      raise SumIncompatibleError(
          type_spec,
          type_spec_context,
          f'{type_spec.shape} is not fully defined',
      )
  elif isinstance(type_spec, computation_types.StructType):
    for _, element_type in structure.iter_elements(type_spec):
      check_is_sum_compatible(element_type, type_spec_context)
  elif isinstance(type_spec, computation_types.FederatedType):
    check_is_sum_compatible(type_spec.member, type_spec_context)
  else:
    raise SumIncompatibleError(
        type_spec,
        type_spec_context,
        'only structures of tensors (possibly federated) may be summed',
    )


def is_structure_of_floats(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is a structure of floats.

  Note that an empty `computation_types.StructType` will return `True`, as it
  does not contain any non-floating types.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is a structure of floats, otherwise `False`.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec.dtype, np.floating)
  elif isinstance(type_spec, computation_types.StructType):
    return all(
        is_structure_of_floats(v) for _, v in structure.iter_elements(type_spec)
    )
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_structure_of_floats(type_spec.member)
  else:
    return False


def check_is_structure_of_floats(type_spec):
  if not is_structure_of_floats(type_spec):
    raise TypeError(
        'Expected a type which is structure of floats, found {}.'.format(
            type_spec
        )
    )


def is_structure_of_integers(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is a structure of integers.

  Note that an empty `computation_types.StructType` will return `True`, as it
  does not contain any non-integer types.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is a structure of integers, otherwise `False`.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec.dtype, np.integer)
  elif isinstance(type_spec, computation_types.StructType):
    return all(
        is_structure_of_integers(v)
        for _, v in structure.iter_elements(type_spec)
    )
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_structure_of_integers(type_spec.member)
  else:
    return False


def check_is_structure_of_integers(type_spec):
  if not is_structure_of_integers(type_spec):
    raise TypeError(
        'Expected a type which is structure of integers, found {}.'.format(
            type_spec
        )
    )


def is_single_integer_or_matches_structure(
    type_sig: computation_types.Type, shape_type: computation_types.Type
) -> bool:
  """If `type_sig` is an integer or integer structure matching `shape_type`."""

  py_typecheck.check_type(type_sig, computation_types.Type)
  py_typecheck.check_type(shape_type, computation_types.Type)

  if isinstance(type_sig, computation_types.TensorType):
    # This condition applies to both `shape_type` being a tensor or structure,
    # as the same integer bitwidth can be used for all values in the structure.
    return (
        np.issubdtype(type_sig.dtype, np.integer)
        and array_shape.num_elements_in_shape(type_sig.shape) == 1
    )
  elif isinstance(shape_type, computation_types.StructType) and isinstance(
      type_sig, computation_types.StructType
  ):
    bitwidth_name_and_types = list(structure.iter_elements(type_sig))
    shape_name_and_types = list(structure.iter_elements(shape_type))
    if len(type_sig) != len(shape_name_and_types):
      return False
    for (inner_name, type_sig), (inner_shape_name, inner_shape_type) in zip(
        bitwidth_name_and_types, shape_name_and_types
    ):
      if inner_name != inner_shape_name:
        return False
      if not is_single_integer_or_matches_structure(type_sig, inner_shape_type):
        return False
    return True
  else:
    return False


def check_federated_type(
    type_spec: computation_types.FederatedType,
    member: Optional[computation_types.Type] = None,
    placement: Optional[placements.PlacementLiteral] = None,
    all_equal: Optional[bool] = None,
):
  """Checks that `type_spec` is a federated type with the given parameters.

  Args:
    type_spec: The `tff.FederatedType` to check.
    member: The expected member type, or `None` if unspecified.
    placement: The desired placement, or `None` if unspecified.
    all_equal: The desired result of accessing the property
      `tff.FederatedType.all_equal` of `type_spec`, or `None` if left
      unspecified.

  Raises:
    TypeError: if `type_spec` is not a federated type of the given kind.
  """
  py_typecheck.check_type(type_spec, computation_types.FederatedType)
  if member is not None:
    py_typecheck.check_type(member, computation_types.Type)
    member.check_assignable_from(type_spec.member)
  if placement is not None:
    py_typecheck.check_type(placement, placements.PlacementLiteral)
    if type_spec.placement is not placement:
      raise TypeError(
          'Expected federated type placed at {}, got one placed at {}.'.format(
              placement, type_spec.placement
          )
      )
  if all_equal is not None:
    py_typecheck.check_type(all_equal, bool)
    if type_spec.all_equal != all_equal:
      raise TypeError(
          'Expected federated type with all_equal {}, got one with {}.'.format(
              all_equal, type_spec.all_equal
          )
      )


def is_average_compatible(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` can be averaged.

  Types that are average-compatible are composed of numeric tensor types,
  either floating-point or complex, possibly packaged into nested named tuples,
  and possibly federated.

  Args:
    type_spec: a `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is average-compatible, `False` otherwise.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec, np.inexact)
  elif isinstance(type_spec, computation_types.StructType):
    return all(
        is_average_compatible(v) for _, v in structure.iter_elements(type_spec)
    )
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_average_compatible(type_spec.member)
  else:
    return False


def is_min_max_compatible(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is min/max compatible.

  Types that are min/max-compatible are composed of integer or floating tensor
  types, possibly packaged into nested tuples and possibly federated.

  Args:
    type_spec: a `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is min/max compatible, `False` otherwise.
  """
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec.dtype, np.integer) or np.issubdtype(
        type_spec.dtype, np.floating
    )
  elif isinstance(type_spec, computation_types.StructType):
    return all(
        is_min_max_compatible(v) for _, v in structure.iter_elements(type_spec)
    )
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_min_max_compatible(type_spec.member)
  else:
    return False


def is_struct_with_py_container(value, type_spec):
  return isinstance(value, structure.Struct) and isinstance(
      type_spec, computation_types.StructWithPythonType
  )


class NotConcreteTypeError(TypeError):

  def __init__(self, full_type, found_abstract):
    message = (
        'Expected concrete type containing no abstract types, but '
        f'found abstract type {found_abstract} in {full_type}.'
    )
    super().__init__(message)


class MismatchedConcreteTypesError(TypeError):
  """Raised when there is a mismatch between two types."""

  def __init__(
      self,
      full_concrete,
      full_generic,
      abstract_label,
      first_concrete,
      second_concrete,
  ):
    message = (
        f'Expected concrete type {full_concrete} to be a valid substitution '
        f'for generic type {full_generic}, but abstract type {abstract_label} '
        f'had substitutions {first_concrete} and {second_concrete}, which are '
        'not equivalent.'
    )
    super().__init__(message)


class UnassignableConcreteTypesError(TypeError):
  """Raised when one type can not be assigned to another type."""

  def __init__(
      self,
      full_concrete,
      full_generic,
      abstract_label,
      definition,
      not_assignable_from,
  ):
    message = (
        f'Expected concrete type {full_concrete} to be a valid substitution '
        f'for generic type {full_generic}, but abstract type {abstract_label} '
        f'was defined as {definition}, and later used as {not_assignable_from} '
        'which cannot be assigned from the former.'
    )
    super().__init__(message)


class MismatchedStructureError(TypeError):
  """Raised when there is a mismatch between the structures of two types."""

  def __init__(
      self,
      full_concrete,
      full_generic,
      concrete_member,
      generic_member,
      mismatch,
  ):
    message = (
        f'Expected concrete type {full_concrete} to be a valid substitution '
        f'for generic type {full_generic}, but their structures do not match: '
        f'{concrete_member} differs in {mismatch} from {generic_member}.'
    )
    super().__init__(message)


class MissingDefiningUsageError(TypeError):

  def __init__(self, generic_type, label_name):
    message = (
        f'Missing defining use of abstract type {label_name} in type '
        f'{generic_type}. See `check_concrete_instance_of` documentation for '
        'details on what counts as a defining use.'
    )
    super().__init__(message)


def check_concrete_instance_of(
    concrete_type: computation_types.Type, generic_type: computation_types.Type
):
  """Checks whether `concrete_type` is a valid substitution of `generic_type`.

  This function determines whether `generic_type`'s type parameters can be
  substituted such that it is equivalent to `concrete type`.

  Note that passing through argument-position of function type swaps the
  variance of abstract types. Argument-position types can be assigned *from*
  other instances of the same type, but are not equivalent to it.

  Due to this variance issue, only abstract types must include at least one
  "defining" usage. "Defining" uses are those which are encased in function
  parameter position an odd number of times. These usages must all be
  equivalent. Non-defining usages need not compare equal but must be assignable
  *from* defining usages.

  Args:
    concrete_type: A type containing no `computation_types.AbstractType`s to
      check against `generic_type`'s shape.
    generic_type: A type which may contain `computation_types.AbstractType`s.

  Raises:
    TypeError: If `concrete_type` is not a valid substitution of `generic_type`.
  """
  py_typecheck.check_type(concrete_type, computation_types.Type)
  py_typecheck.check_type(generic_type, computation_types.Type)

  for t in preorder_types(concrete_type):
    if isinstance(t, computation_types.AbstractType):
      raise NotConcreteTypeError(concrete_type, t)

  type_bindings = {}
  non_defining_usages = collections.defaultdict(list)

  def _check_helper(
      generic_type_member: computation_types.Type,
      concrete_type_member: computation_types.Type,
      defining: bool,
  ):
    """Recursive helper function."""

    def _raise_structural(mismatch):
      raise MismatchedStructureError(
          concrete_type,
          generic_type,
          concrete_type_member,
          generic_type_member,
          mismatch,
      )

    def _both_are(predicate):
      if predicate(generic_type_member):
        if predicate(concrete_type_member):
          return True
        else:
          _raise_structural('kind')
      else:
        return False

    if isinstance(generic_type_member, computation_types.AbstractType):
      label = str(generic_type_member.label)
      if not defining:
        non_defining_usages[label].append(concrete_type_member)
      else:
        bound_type = type_bindings.get(label)
        if bound_type is not None:
          if not concrete_type_member.is_equivalent_to(bound_type):
            raise MismatchedConcreteTypesError(
                concrete_type,
                generic_type,
                label,
                bound_type,
                concrete_type_member,
            )
        else:
          type_bindings[label] = concrete_type_member
    elif _both_are(lambda t: isinstance(t, computation_types.TensorType)):
      if generic_type_member != concrete_type_member:
        _raise_structural('tensor types')
    elif _both_are(lambda t: isinstance(t, computation_types.PlacementType)):
      if generic_type_member != concrete_type_member:
        _raise_structural('placements')
    elif _both_are(lambda t: isinstance(t, computation_types.StructType)):
      generic_elements = structure.to_elements(generic_type_member)  # pytype: disable=wrong-arg-types
      concrete_elements = structure.to_elements(concrete_type_member)  # pytype: disable=wrong-arg-types
      if len(generic_elements) != len(concrete_elements):
        _raise_structural('length')
      for generic_element, concrete_element in zip(
          generic_elements, concrete_elements
      ):
        if generic_element[0] != concrete_element[0]:
          _raise_structural('element names')
        _check_helper(generic_element[1], concrete_element[1], defining)
    elif _both_are(lambda t: isinstance(t, computation_types.SequenceType)):
      _check_helper(
          generic_type_member.element,  # pytype: disable=attribute-error
          concrete_type_member.element,  # pytype: disable=attribute-error
          defining,
      )
    elif _both_are(lambda t: isinstance(t, computation_types.FunctionType)):
      if generic_type_member.parameter is None:  # pytype: disable=attribute-error
        if concrete_type_member.parameter is not None:  # pytype: disable=attribute-error
          _raise_structural('parameter')
      else:
        _check_helper(
            generic_type_member.parameter,  # pytype: disable=attribute-error
            concrete_type_member.parameter,  # pytype: disable=attribute-error
            not defining,
        )
      _check_helper(
          generic_type_member.result,  # pytype: disable=attribute-error
          concrete_type_member.result,  # pytype: disable=attribute-error
          defining,
      )
    elif _both_are(lambda t: isinstance(t, computation_types.FederatedType)):
      if generic_type_member.placement != concrete_type_member.placement:  # pytype: disable=attribute-error
        _raise_structural('placement')
      if generic_type_member.all_equal != concrete_type_member.all_equal:  # pytype: disable=attribute-error
        _raise_structural('all equal')
      _check_helper(
          generic_type_member.member,  # pytype: disable=attribute-error
          concrete_type_member.member,  # pytype: disable=attribute-error
          defining,
      )
    else:
      raise TypeError(f'Unexpected type kind {generic_type}.')

  _check_helper(generic_type, concrete_type, False)

  for label, usages in non_defining_usages.items():
    bound_type = type_bindings.get(label)
    if bound_type is None:
      if len(usages) == 1:
        # Single-use abstract types can't be wrong.
        # Note: we could also add an exception here for cases where every usage
        # is equivalent to the first usage. However, that's not currently
        # needed since the only intrinsic that doesn't have a defining use is
        # GENERIC_ZERO, which has only a single-use type parameter.
        pass
      else:
        raise MissingDefiningUsageError(generic_type, label)
    else:
      for usage in usages:
        if not usage.is_assignable_from(bound_type):
          raise UnassignableConcreteTypesError(
              concrete_type, generic_type, label, bound_type, usage
          )


def check_valid_federated_weighted_mean_argument_tuple_type(
    type_spec: computation_types.StructType,
):
  """Checks that `type_spec` is a valid type of a federated weighted mean arg.

  Args:
    type_spec: A `computation_types.StructType`.

  Raises:
    TypeError: If the check fails.
  """
  py_typecheck.check_type(type_spec, computation_types.StructType)
  if len(type_spec) != 2:
    raise TypeError('Expected a 2-tuple, found {}.'.format(type_spec))
  for _, v in structure.iter_elements(type_spec):
    check_federated_type(v, None, placements.CLIENTS, False)
    if not is_average_compatible(v.member):
      raise TypeError(
          'Expected average-compatible args, got {} from argument of type {}.'
          .format(v.member, type_spec)
      )
  w_type = type_spec[1].member
  if (
      not isinstance(w_type, computation_types.TensorType)
      or w_type.shape is None
      or w_type.shape
  ):
    raise TypeError('Expected scalar weight, got {}.'.format(w_type))


def count_tensors_in_type(
    type_spec: computation_types.Type,
    tensor_filter: Optional[
        Callable[[computation_types.TensorType], bool]
    ] = None,
) -> collections.OrderedDict[str, int]:
  """Counts tensors and fully-specified elements under `type_spec`.

  Args:
    type_spec: Instance of `computation_types.Type` to count tensors under.
    tensor_filter: Optional filtering function. Callable which takes an argument
      of type `computation_types.TensorType` and returns a boolean. If
      specified, only tensor type which pass this filter (IE, on which this
      function returns `True`) will be counted.

  Returns:
    A `collections.OrderedDict` with three parameters. The first, `tensors`, is
    the count of all `computation_types.TensorType` (passing `tensor_filter`
    if this argument is specified). The second, `parameters`, is the count
    of all fully-specified parameters of these tensors. Note that this implies
    any tensor with a `None` dimension (IE, of unspecified size) will not be
    counted. The third counts how many tensors fall into this category (that
    is, now many have unspecified size).
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if tensor_filter is None:
    tensor_filter = lambda _: True

  tensors_and_params = collections.OrderedDict(
      num_tensors=0, parameters=0, num_unspecified_tensors=0
  )

  def _capture_tensors(type_signature):
    if isinstance(
        type_signature, computation_types.TensorType
    ) and tensor_filter(type_signature):
      tensors_and_params['num_tensors'] += 1
      num_parameters = array_shape.num_elements_in_shape(type_signature.shape)
      if num_parameters is not None:
        tensors_and_params['parameters'] += num_parameters
      else:
        tensors_and_params['num_unspecified_tensors'] += 1
    return type_signature, False

  type_transformations.transform_type_postorder(type_spec, _capture_tensors)
  return tensors_and_params
