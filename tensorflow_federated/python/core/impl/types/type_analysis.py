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

from typing import Any, Callable, Optional

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_transformations

_TypePredicate = Callable[[computation_types.Type], bool]


def count(type_signature: computation_types.Type,
          predicate: _TypePredicate) -> int:
  """Returns the number of types in `type_signature` matching `predicate`.

  Args:
    type_signature: A tree of `computation_type.Type`s to count.
    predicate: A Python function that takes a type as a parameter and returns a
      boolean value.
  """
  counter = 1 if predicate(type_signature) else 0
  counter += sum(count(child, predicate) for child in type_signature.children())
  return counter


def contains(type_signature: computation_types.Type,
             predicate: _TypePredicate) -> bool:
  """Checks if `type_signature` contains any types that pass `predicate`."""
  if predicate(type_signature):
    return True
  for child in type_signature.children():
    if contains(child, predicate):
      return True
  return False


def contains_federated_types(type_signature):
  """Returns whether or not `type_signature` contains a federated type."""
  return contains(type_signature, lambda t: t.is_federated())


def contains_tensor_types(type_signature):
  """Returns whether or not `type_signature` contains a tensor type."""
  return contains(type_signature, lambda t: t.is_tensor())


def contains_only(
    type_signature: computation_types.Type,
    predicate: _TypePredicate,
) -> bool:
  """Checks if `type_signature` contains only types that pass `predicate`."""
  return not contains(type_signature, lambda t: not predicate(t))


def check_type(value: Any, type_spec: computation_types.Type):
  """Checks whether `val` is of TFF type `type_spec`.

  Args:
    value: The object to check.
    type_spec: A `computation_types.Type`, the type that `value` is checked
      against.

  Raises:
    TypeError: If the infferred type of `value` is not `type_spec`.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  value_type = type_conversions.infer_type(value)
  if not type_spec.is_assignable_from(value_type):
    raise TypeError(
        'Expected TFF type {}, which is not assignable from {}.'.format(
            type_spec, value_type))


def is_tensorflow_compatible_type(type_spec):
  """Checks `type_spec` against an explicit list of `tf_computation`."""
  if type_spec is None:
    return True
  return contains_only(
      type_spec, lambda t: t.is_struct() or t.is_sequence() or t.is_tensor())


def check_tensorflow_compatible_type(type_spec):
  if not is_tensorflow_compatible_type(type_spec):
    raise TypeError(
        'Expected type to be compatible with TensorFlow (i.e. tensor, '
        'sequence, or tuple types), found {}.'.format(type_spec))


def is_generic_op_compatible_type(type_spec):
  """Checks `type_spec` against an explicit list of generic operators."""
  if type_spec is None:
    return True
  return contains_only(type_spec, lambda t: t.is_struct() or t.is_tensor())


def is_binary_op_with_upcast_compatible_pair(
    possibly_nested_type: Optional[computation_types.Type],
    type_to_upcast: computation_types.Type) -> bool:
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
  if not (is_generic_op_compatible_type(possibly_nested_type) and
          is_generic_op_compatible_type(type_to_upcast)):
    return False
  if possibly_nested_type is None:
    return type_to_upcast is None
  if possibly_nested_type.is_equivalent_to(type_to_upcast):
    return True
  if not (type_to_upcast.is_tensor() and type_to_upcast.shape == tf.TensorShape(
      ())):
    return False

  types_are_ok = [True]

  only_allowed_dtype = type_to_upcast.dtype

  def _check_tensor_types(type_spec):
    if type_spec.is_tensor() and type_spec.dtype != only_allowed_dtype:
      types_are_ok[0] = False
    return type_spec, False

  type_transformations.transform_type_postorder(possibly_nested_type,
                                                _check_tensor_types)

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
    if type_spec.is_tensor():
      return set()
    elif type_spec.is_sequence():
      return _check_or_get_unbound_abstract_type_labels(type_spec.element,
                                                        bound_labels, check)
    elif type_spec.is_federated():
      return _check_or_get_unbound_abstract_type_labels(type_spec.member,
                                                        bound_labels, check)
    elif type_spec.is_struct():
      return set().union(*[
          _check_or_get_unbound_abstract_type_labels(v, bound_labels, check)
          for _, v in structure.iter_elements(type_spec)
      ])
    elif type_spec.is_abstract():
      if type_spec.label in bound_labels:
        return set()
      elif not check:
        return set([type_spec.label])
      else:
        raise TypeError('Unbound type label \'{}\'.'.format(type_spec.label))
    elif type_spec.is_function():
      if type_spec.parameter is None:
        parameter_labels = set()
      else:
        parameter_labels = _check_or_get_unbound_abstract_type_labels(
            type_spec.parameter, bound_labels, False)
      result_labels = _check_or_get_unbound_abstract_type_labels(
          type_spec.result, bound_labels.union(parameter_labels), check)
      return parameter_labels.union(result_labels)

  _check_or_get_unbound_abstract_type_labels(type_spec, set(), True)


def is_numeric_dtype(dtype):
  """Returns True iff `dtype` is numeric.

  Args:
    dtype: An instance of tf.DType.

  Returns:
    True iff `dtype` is numeric, i.e., integer, float, or complex.
  """
  py_typecheck.check_type(dtype, tf.DType)
  return dtype.is_integer or dtype.is_floating or dtype.is_complex


def is_sum_compatible(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is a type that can be added to itself.

  Types that are sum-compatible are composed of scalars of numeric types,
  possibly packaged into nested named tuples, and possibly federated. Types
  that are sum-incompatible include sequences, functions, abstract types,
  and placements.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is sum-compatible, `False` otherwise.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if type_spec.is_tensor():
    return is_numeric_dtype(type_spec.dtype)
  elif type_spec.is_struct():
    return all(
        is_sum_compatible(v) for _, v in structure.iter_elements(type_spec))
  elif type_spec.is_federated():
    return is_sum_compatible(type_spec.member)
  else:
    return False


def check_is_sum_compatible(type_spec):
  if not is_sum_compatible(type_spec):
    raise TypeError(
        'Expected a type which is compatible with the sum operator, found {}.'
        .format(type_spec))


def is_structure_of_integers(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is a structure of integers.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is a structure of integers, otherwise `False`.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if type_spec.is_tensor():
    py_typecheck.check_type(type_spec.dtype, tf.DType)
    return type_spec.dtype.is_integer
  elif type_spec.is_struct():
    return all(
        is_structure_of_integers(v)
        for _, v in structure.iter_elements(type_spec))
  elif type_spec.is_federated():
    return is_structure_of_integers(type_spec.member)
  else:
    return False


def check_is_structure_of_integers(type_spec):
  if not is_structure_of_integers(type_spec):
    raise TypeError(
        'Expected a type which is structure of integers, found {}.'.format(
            type_spec))


def is_valid_bitwidth_type_for_value_type(
    bitwidth_type: computation_types.Type,
    value_type: computation_types.Type) -> bool:
  """Whether or not `bitwidth_type` is a valid bitwidth type for `value_type`."""

  # NOTE: this function is primarily a helper for `intrinsic_factory.py`'s
  # `federated_secure_sum` function.
  py_typecheck.check_type(bitwidth_type, computation_types.Type)
  py_typecheck.check_type(value_type, computation_types.Type)

  if value_type.is_tensor() and bitwidth_type.is_tensor():
    # Here, `value_type` refers to a tensor. Rather than check that
    # `bitwidth_type` is exactly the same, we check that it is a single integer,
    # since we want a single bitwidth integer per tensor.
    return bitwidth_type.dtype.is_integer and (
        bitwidth_type.shape.num_elements() == 1)
  elif value_type.is_struct() and bitwidth_type.is_struct():
    bitwidth_name_and_types = list(structure.iter_elements(bitwidth_type))
    value_name_and_types = list(structure.iter_elements(value_type))
    if len(bitwidth_name_and_types) != len(value_name_and_types):
      return False
    for (inner_bitwidth_name,
         inner_bitwidth_type), (inner_value_name, inner_value_type) in zip(
             bitwidth_name_and_types, value_name_and_types):
      if inner_bitwidth_name != inner_value_name:
        return False
      if not is_valid_bitwidth_type_for_value_type(inner_bitwidth_type,
                                                   inner_value_type):
        return False
    return True
  else:
    return False


def check_federated_type(
    type_spec: computation_types.Type,
    member: Optional[computation_types.Type] = None,
    placement: Optional[placement_literals.PlacementLiteral] = None,
    all_equal: Optional[bool] = None):
  """Checks that `type_spec` is a federated type with the given parameters.

  Args:
    type_spec: The `tff.Type` to check.
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
  if type_spec.is_tensor():
    return type_spec.dtype.is_floating or type_spec.dtype.is_complex
  elif type_spec.is_struct():
    return all(
        is_average_compatible(v) for _, v in structure.iter_elements(type_spec))
  elif type_spec.is_federated():
    return is_average_compatible(type_spec.member)
  else:
    return False


def is_struct_with_py_container(value, type_spec):
  return (type_spec.is_struct_with_python() and
          isinstance(value, structure.Struct))


def is_concrete_instance_of(type_with_concrete_elements,
                            type_with_abstract_elements):
  """Checks whether abstract types can be concretized via a parallel structure.

  This function builds up a new concrete structure via the bindings encountered
  in `type_with_abstract_elements` in a postorder fashion. That is, it walks the
  type trees in parallel, caching bindings for abstract types on the way. When
  it encounters a previously bound abstract type, it simply inlines this cached
  value. Finally, `abstract_types_can_be_concretized` delegates checking type
  equivalence to `computation_types.Type.is_equivalent_to`, passing in the
  created concrete structure for comparison with `type_with_concrete_elements`.

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

  if contains(type_with_concrete_elements, lambda t: t.is_abstract()):
    raise TypeError(
        '`type_with_concrete_elements` must contain no abstract types. You '
        'have passed {}'.format(type_with_concrete_elements))

  bound_abstract_types = {}
  type_error_string = ('Structural mismatch encountered while concretizing '
                       'abstract types. The structure of {} does not match the '
                       'structure of {}').format(type_with_abstract_elements,
                                                 type_with_concrete_elements)

  def _concretize_abstract_types(
      abstract_type_spec: computation_types.Type,
      concrete_type_spec: computation_types.Type) -> computation_types.Type:
    """Recursive helper function to construct concrete type spec."""
    if abstract_type_spec.is_abstract():
      bound_type = bound_abstract_types.get(str(abstract_type_spec.label))
      if bound_type:
        return bound_type
      else:
        bound_abstract_types[str(abstract_type_spec.label)] = concrete_type_spec
        return concrete_type_spec
    elif abstract_type_spec.is_tensor():
      return abstract_type_spec
    elif abstract_type_spec.is_struct():
      if not concrete_type_spec.is_struct():
        raise TypeError(type_error_string)
      abstract_elements = structure.to_elements(abstract_type_spec)
      concrete_elements = structure.to_elements(concrete_type_spec)
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
      return computation_types.StructType(concretized_tuple_elements)
    elif abstract_type_spec.is_sequence():
      if not concrete_type_spec.is_sequence():
        raise TypeError(type_error_string)
      return computation_types.SequenceType(
          _concretize_abstract_types(abstract_type_spec.element,
                                     concrete_type_spec.element))
    elif abstract_type_spec.is_function():
      if not concrete_type_spec.is_function():
        raise TypeError(type_error_string)
      if abstract_type_spec.parameter is None:
        if concrete_type_spec.parameter is not None:
          return TypeError(type_error_string)
        concretized_param = None
      else:
        concretized_param = _concretize_abstract_types(
            abstract_type_spec.parameter, concrete_type_spec.parameter)
      concretized_result = _concretize_abstract_types(abstract_type_spec.result,
                                                      concrete_type_spec.result)
      return computation_types.FunctionType(concretized_param,
                                            concretized_result)
    elif abstract_type_spec.is_placement():
      if not concrete_type_spec.is_placement():
        raise TypeError(type_error_string)
      return abstract_type_spec
    elif abstract_type_spec.is_federated():
      if not concrete_type_spec.is_federated():
        raise TypeError(type_error_string)
      new_member = _concretize_abstract_types(abstract_type_spec.member,
                                              concrete_type_spec.member)
      return computation_types.FederatedType(new_member,
                                             abstract_type_spec.placement,
                                             abstract_type_spec.all_equal)
    else:
      raise TypeError(
          'Unexpected abstract typespec {}.'.format(abstract_type_spec))

  concretized_abstract_type = _concretize_abstract_types(
      type_with_abstract_elements, type_with_concrete_elements)

  return concretized_abstract_type.is_equivalent_to(type_with_concrete_elements)


def check_valid_federated_weighted_mean_argument_tuple_type(
    type_spec: computation_types.StructType):
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
    check_federated_type(v, None, placement_literals.CLIENTS, False)
    if not is_average_compatible(v.member):
      raise TypeError(
          'Expected average-compatible args, got {} from argument of type {}.'
          .format(v.member, type_spec))
  w_type = type_spec[1].member
  py_typecheck.check_type(w_type, computation_types.TensorType)
  if w_type.shape.ndims != 0:
    raise TypeError('Expected scalar weight, got {}.'.format(w_type))
