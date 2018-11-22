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
from tensorflow_federated.python.core.impl import value_impl


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
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, types.FederatedType)
  if value.type_signature.placement is not placements.SERVER:
    raise TypeError(
        'The value to be broadcasted should be placed at the SERVER, but it '
        'is placed at {}.'.format(str(value.type_signature.placement)))
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
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, types.FederatedType)
  if value.type_signature.placement is not placements.CLIENTS:
    raise TypeError(
        'The value to be mapped should be placed at the CLIENTS, but it '
        'is placed at {}.'.format(str(value.type_signature.placement)))

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
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, types.FederatedType)
  if value.type_signature.placement is not placements.CLIENTS:
    raise TypeError(
        'The value to be summed should be placed at the CLIENTS, but it '
        'is placed at {}.'.format(str(value.type_signature.placement)))

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
