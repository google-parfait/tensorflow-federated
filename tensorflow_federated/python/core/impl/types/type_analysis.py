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

from typing import Callable, Optional, Tuple, Type, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_transformations

_TypeOrTupleOfTypes = Union[Type[computation_types.Type],
                            Tuple[Type[computation_types.Type], ...]]


def _visit_postorder(type_signature: computation_types.Type,
                     function: Callable[[computation_types.Type], None]):
  py_typecheck.check_type(type_signature, computation_types.Type)

  def _visit(inner_type):
    function(inner_type)
    return inner_type, False

  type_transformations.transform_type_postorder(type_signature, _visit)


def count(
    type_signature: computation_types.Type,
    predicate: Optional[Callable[[computation_types.Type],
                                 bool]] = None) -> int:
  """Returns the number of types in `type_signature` matching `predicate`.

  Args:
    type_signature: A tree of `computation_type.Type`s to count.
    predicate: An optional Python function that takes a type as a parameter and
      returns a boolean value. If `None`, all types are counted.
  """
  counter = 0

  def _function(inner_type):
    nonlocal counter
    counter += 1 if predicate(inner_type) else 0

  _visit_postorder(type_signature, _function)
  return counter


def count_types(type_signature: computation_types.Type,
                types: _TypeOrTupleOfTypes) -> int:
  """Returns the number of instances of `types` in `type_signature`.

  Args:
    type_signature: A tree of `computation_type.Type`s to count.
    types: A `computation_types.Type` type or a tuple of
      `computation_types.Type` types; the same as what is accepted by
      `isinstance`.
  """
  return count(type_signature, lambda x: isinstance(x, types))


def contains_types(type_signature: computation_types.Type,
                   types: _TypeOrTupleOfTypes) -> bool:
  """Checks if `type_signature` contains any instance of `types`.

  Args:
    type_signature: A tree of `computation_type.Type` to test.
    types: A `computation_types.Type` type or a tuple of
      `computation_types.Type` types; the same as what is accepted by
      `isinstance`.

  Returns:
    `True` if `type_signature` contains any instance of `types`, otherwise
    `False`.
  """
  return count_types(type_signature, types) > 0


def contains_federated_types(type_signature):
  """Returns whether or not `type_signature` contains a federated type."""
  return contains_types(type_signature, computation_types.FederatedType)


def contains_tensor_types(type_signature):
  """Returns whether or not `type_signature` contains a tensor type."""
  return contains_types(type_signature, computation_types.TensorType)


def contains_only_types(type_signature: computation_types.Type,
                        types: _TypeOrTupleOfTypes) -> bool:
  """Checks if `type_signature` contains only instances of `types`.

  Args:
    type_signature: A tree of `computation_type.Type` to test.
    types: A `computation_types.Type` type or a tuple of
      `computation_types.Type` types; the same as what is accepted by
      `isinstance`.

  Returns:
    `True` if `type_signature` contains only instance of `types`, otherwise
    `False`.
  """
  return count(type_signature, lambda x: not isinstance(x, types)) == 0


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

  def _check_for_disallowed_type(type_to_check, disallowed_types):
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
          **disallowed_types,
          computation_types.FederatedType: context,
          computation_types.FunctionType: context,
      }
    if isinstance(type_to_check, computation_types.SequenceType):
      context = 'sequence types'
      disallowed_types = {
          **disallowed_types,
          computation_types.FederatedType: context,
          computation_types.SequenceType: context,
      }
    return disallowed_types

  type_transformations.visit_preorder(type_signature,
                                      _check_for_disallowed_type, dict())


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
  val_type = type_conversions.infer_type(val)
  if not is_assignable_from(type_spec, val_type):
    raise TypeError(
        'Expected TFF type {}, which is not assignable from {}.'.format(
            type_spec, val_type))


def is_tensorflow_compatible_type(type_spec):
  """Checks `type_spec` against an explicit whitelist for `tf_computation`."""
  if type_spec is None:
    return True
  return contains_only_types(type_spec, (
      computation_types.NamedTupleType,
      computation_types.SequenceType,
      computation_types.TensorType,
  ))


def check_tensorflow_compatible_type(type_spec):
  if not is_tensorflow_compatible_type(type_spec):
    raise TypeError(
        'Expected type to be compatible with TensorFlow (i.e. tensor, '
        'sequence, or tuple types), found {}.'.format(type_spec))


def is_generic_op_compatible_type(type_spec):
  """Checks `type_spec` against an explicit whitelist for generic operators."""
  if type_spec is None:
    return True
  return contains_only_types(type_spec, (
      computation_types.NamedTupleType,
      computation_types.TensorType,
  ))


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
  if not is_structure_of_integers(type_spec):
    raise TypeError(
        'Expected a type which is structure of integers, found {}.'.format(
            type_spec))


def is_valid_bitwidth_type_for_value_type(bitwidth_type, value_type):
  """Whether or not `bitwidth_type` is a valid bitwidth type for `value_type`."""

  # NOTE: this function is primarily a helper for `intrinsic_factory.py`'s
  # `federated_secure_sum` function.

  def _both_are_type(first, second, ty):
    """Whether or not `first` and `second` are both instances of `ty`."""
    return isinstance(first, ty) and isinstance(second, ty)

  bitwidth_type = computation_types.to_type(bitwidth_type)
  value_type = computation_types.to_type(value_type)

  if _both_are_type(value_type, bitwidth_type, computation_types.TensorType):
    # Here, `value_type` refers to a tensor. Rather than check that
    # `bitwidth_type` is exactly the same, we check that it is a single integer,
    # since we want a single bitwidth integer per tensor.
    return bitwidth_type.dtype.is_integer and (
        bitwidth_type.shape.num_elements() == 1)
  elif _both_are_type(value_type, bitwidth_type,
                      computation_types.NamedTupleType):
    bitwidth_name_and_types = list(anonymous_tuple.iter_elements(bitwidth_type))
    value_name_and_types = list(anonymous_tuple.iter_elements(value_type))
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

  if contains_types(type_with_concrete_elements,
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
