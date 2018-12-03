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
"""Defines intrinsics for use in composing federated computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import types
from tensorflow_federated.python.core.api import value_base

from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


def federated_average(value, weight=None):
  """Computes a `SERVER` average of `value` placed on `CLIENTS`.

  Args:
    value: The value to be averaged. Must be of a TFF federated type placed at
      `CLIENTS`. The value may be structured, e.g., its member constituents can
      be named tuples. The tensor types that the value is composed of must be
      floating-point or complex.

    weight: An optional weight, a TFF federated integer or floating-point tensor
      value, also placed at `CLIENTS`.

  Returns:
    A representation at the `SERVER` of an average of the member constituents
    of `value`, optionally weighted with `weight` if specified (otherwise, the
    member constituents contributed by all clients are equally weighted).

  Raises:
    TypeError: if `value` is not a federated TFF value placed at `CLIENTS`, or
      if `weight` is not a federated integer or a floating-point tensor with
      the matching placement.
  """
  # TODO(b/113112108): Possibly relax the constraints on numeric types, and
  # inject implicit casts where appropriate. For instance, we might want to
  # allow `tf.int32` values as the input, and automatically cast them to
  # `tf.float321 before invoking the average, thus producing a floating-point
  # result.

  # TODO(b/120439632): Possibly allow the weight to be either structured or
  # non-scalar, e.g., for the case of averaging a convolutional layer, when
  # we would want to use a different weight for every filter, and where it
  # might be cumbersome for users to have to manually slice and assemble a
  # variable.

  value = value_impl.to_value(value)
  type_utils.check_federated_value_placement(
      value, placements.CLIENTS, 'value to be averaged')
  if not type_utils.is_average_compatible(value.type_signature):
    raise TypeError(
        'The value type {} is not compatible with the average operator.'.format(
            str(value.type_signature)))

  if weight is not None:
    weight = value_impl.to_value(weight)
    type_utils.check_federated_value_placement(
        weight, placements.CLIENTS, 'weight to use in averaging')
    py_typecheck.check_type(weight.type_signature.member, types.TensorType)
    if weight.type_signature.member.shape.ndims != 0:
      raise TypeError('The weight type {} is not a federated scalar.'.format(
          str(weight.type_signature)))
    if not (weight.type_signature.member.dtype.is_integer or
            weight.type_signature.member.dtype.is_floating):
      raise TypeError(
          'The weight type {} is not a federated integer or '
          'floating-point tensor.'.format(str(weight.type_signature)))

  result_type = types.FederatedType(
      value.type_signature.member, placements.SERVER, True)

  if weight is not None:
    intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_WEIGHTED_AVERAGE.uri,
        types.FunctionType(
            [value.type_signature, weight.type_signature], result_type)))
    return intrinsic(value, weight)
  else:
    intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_AVERAGE.uri,
        types.FunctionType(value.type_signature, result_type)))
    return intrinsic(value)


def federated_broadcast(value):
  """Broadcasts a federated value from the `SERVER` to the `CLIENTS`.

  Args:
    value: A value of a TFF federated type placed at the `SERVER`, all members
      of which are equal (the `all_equal` property of the federated type of
     `value` is True).

  Returns:
    A representation of the result of broadcasting: a value of a TFF federated
    type placed at the `CLIENTS`, all members of which are equal.

  Raises:
    TypeError: if the argument is not a federated TFF value placed at the
      `SERVER`.
  """
  value = value_impl.to_value(value)
  type_utils.check_federated_value_placement(
      value, placements.SERVER, 'value to be broadcasted')

  if not value.type_signature.all_equal:
    raise TypeError('The broadcasted value should be equal at all locations.')

  # TODO(b/113112108): Replace this hand-crafted logic here and below with
  # a call to a helper function that handles it in a uniform manner after
  # implementing support for correctly typechecking federated template types
  # and instantiating template types on concrete arguments.
  result_type = types.FederatedType(
      value.type_signature.member, placements.CLIENTS, True)
  intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_BROADCAST.uri,
      types.FunctionType(value.type_signature, result_type)))
  return intrinsic(value)


def federated_collect(value):
  """Materializes a federated value from `CLIENTS` as a `SERVER` sequence.

  Args:
    value: A value of a TFF federated type placed at the `CLIENTS`.

  Returns:
    A stream of the same type as the member constituents of `value` placed at
    the `SERVER`.

  Raises:
    TypeError: if the argument is not a federated TFF value placed at `CLIENTS`.
  """
  value = value_impl.to_value(value)
  type_utils.check_federated_value_placement(
      value, placements.CLIENTS, 'value to be collected')

  result_type = types.FederatedType(
      types.SequenceType(value.type_signature.member), placements.SERVER, True)
  intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_COLLECT.uri,
      types.FunctionType(value.type_signature, result_type)))
  return intrinsic(value)


def federated_map(value, mapping_fn):
  """Maps a federated value on CLIENTS pointwise using a given mapping function.

  Args:
    value: A value of a TFF federated type placed at the `CLIENTS`.
    mapping_fn: A mapping function to apply pointwise to member constituents of
      `value` on each of the participants in `CLIENTS`. The parameter of this
      function must be of the same type as the member constituents of `value`.

  Returns:
    A federated value on `CLIENTS` that represents the result of mapping.

  Raises:
    TypeError: if the arguments are not of the appropriates types.
  """
  # TODO(b/113112108): Extend this to auto-zip the `value` argument if needed.

  # TODO(b/113112108): Possibly lift the restriction that the mapped value must
  # be placed at the clients after adding support for placement labels in the
  # federated types, and expanding the type specification of the intrinsic this
  # is based on to work with federated values of arbitrary placement.

  value = value_impl.to_value(value)
  type_utils.check_federated_value_placement(
      value, placements.CLIENTS, 'value to be mapped')

  # TODO(b/113112108): Add support for polymorphic templates auto-instantiated
  # here based on the actual type of the argument.
  mapping_fn = value_impl.to_value(mapping_fn)

  py_typecheck.check_type(mapping_fn, value_base.Value)
  py_typecheck.check_type(mapping_fn.type_signature, types.FunctionType)
  if not mapping_fn.type_signature.parameter.is_assignable_from(
      value.type_signature.member):
    raise TypeError(
        'The mapping function expects a parameter of type {}, but member '
        'constituents of the mapped value are of incompatible type {}.'.format(
            str(mapping_fn.type_signature.parameter_type),
            str(value.type_signature.member)))

  # TODO(b/113112108): Replace this as noted above.
  result_type = types.FederatedType(
      mapping_fn.type_signature.result,
      placements.CLIENTS,
      value.type_signature.all_equal)
  intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP.uri,
      types.FunctionType(value.type_signature, result_type)))
  return intrinsic(value)


def federated_reduce(value, zero, op):
  """Reduces `value` from `CLIENTS` to `SERVER` using a reduction operator `op`.

  This method reduces a set of member constituents of a `value` of federated
  type `T@CLIENTS` for some `T`, using a given `zero` in the algebra (i.e., the
  result of reducing an empty set) of some type `U`, and a reduction operator
  `op` with type signature `(<U,T> -> U)` that incorporates a single `T`-typed
  member constituent of `value` into the `U`-typed result of partial reduction.
  In the special case of `T` equal to `U`, this corresponds to the classical
  notion of reduction of a set using a commutative associative binary operator.
  The generalized reduction (with `T` not equal to `U`) requires that repeated
  application of `op` to reduce a set of `T` always yields the same `U`-typed
  result, regardless of the order in which elements of `T` are processed in the
  course of the reduction.

  Args:
    value: A value of a TFF federated type placed at the `CLIENTS`.
    zero: The result of reducing a value with no constituents.
    op: An operator with type signature `(<U,T> -> U)`, where `T` is the type
      of the constituents of `value` and `U` is the type of `zero` to be used
      in performing the reduction.

  Returns:
    A representation on the `SERVER` of the result of reducing the set of all
    member constituents of `value` using the operator `op` into a single item.

  Raises:
    TypeError: if the arguments are not of the types specified above.
  """
  # TODO(b/113112108): Since in most cases, it can be assumed that CLIENTS is
  # a non-empty collective (or else, the computation fails), specifying zero
  # at this level of the API should probably be optional. TBD.

  value = value_impl.to_value(value)
  type_utils.check_federated_value_placement(
      value, placements.CLIENTS, 'value to be reduced')

  zero = value_impl.to_value(zero)
  py_typecheck.check_type(zero, value_base.Value)

  # TODO(b/113112108): We need a check here that zero does not have federated
  # constituents.

  op = value_impl.to_value(op)
  py_typecheck.check_type(op, value_base.Value)
  py_typecheck.check_type(op.type_signature, types.FunctionType)
  op_type_expected = type_utils.reduction_op(
      zero.type_signature, value.type_signature.member)
  if not op_type_expected.is_assignable_from(op.type_signature):
    raise TypeError('Expected an operator of type {}, got {}.'.format(
        str(op_type_expected), str(op.type_signature)))

  # TODO(b/113112108): Replace this as noted above.
  result_type = types.FederatedType(
      zero.type_signature, placements.SERVER, True)
  intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_REDUCE.uri,
      types.FunctionType(
          [value.type_signature, zero.type_signature, op_type_expected],
          result_type)))
  return intrinsic(value, zero, op)


def federated_sum(value):
  """Computes a sum at `SERVER` of a federated value placed on the `CLIENTS`.

  Args:
    value: A value of a TFF federated type placed at the `CLIENTS`.

  Returns:
    A representation of the sum of the member constituents of `value` placed
    on the `SERVER`.

  Raises:
    TypeError: if the argument is not a federated TFF value placed at `CLIENTS`.
  """
  value = value_impl.to_value(value)
  type_utils.check_federated_value_placement(
      value, placements.CLIENTS, 'value to be summed')

  if not type_utils.is_sum_compatible(value.type_signature):
    raise TypeError(
        'The value type {} is not compatible with the sum operator.'.format(
            str(value.type_signature)))

  # TODO(b/113112108): Replace this as noted above.
  result_type = types.FederatedType(
      value.type_signature.member, placements.SERVER, True)
  intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SUM.uri,
      types.FunctionType(value.type_signature, result_type)))
  return intrinsic(value)


def federated_zip(value):
  """Converts a 2-tuple of federated values into a federated 2-tuple value.

  Args:
    value: A value of a TFF named tuple type with two elements, both of which
      are federated values placed at the `CLIENTS`.

  Returns:
    A federated value placed at the `CLIENTS` in which every member component
    at the given client is a two-element named tuple that consists of the pair
    of the corresponding member components of the elements of `value` residing
    at that client.

  Raises:
    TypeError: if the argument is not a named tuple of federated values placed
    at 'CLIENTS`.
  """
  # TODO(b/113112108): Extend this to accept named tuples of arbitrary length.

  # TODO(b/113112108): Extend this to accept *args.

  value = value_impl.to_value(value)
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, types.NamedTupleType)
  num_elements = len(value.type_signature.elements)
  if num_elements != 2:
    raise TypeError(
        'The federated zip operator currently only supports zipping '
        'two-element tuples, but the tuple given as argument has {} '
        'elements.'.format(num_elements))
  for _, elem in value.type_signature.elements:
    py_typecheck.check_type(elem, types.FederatedType)
    if elem.placement is not placements.CLIENTS:
      raise TypeError(
          'The elements of the named tuple to zip must be placed at CLIENTS.')

  # TODO(b/113112108): Replace this as noted above.
  result_type = types.FederatedType(
      [e.member for _, e in value.type_signature.elements],
      placements.CLIENTS,
      all(e.all_equal for _, e in value.type_signature.elements))
  intrinsic = value_impl.ValueImpl(computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_ZIP.uri,
      types.FunctionType(value.type_signature, result_type)))
  return intrinsic(value)
