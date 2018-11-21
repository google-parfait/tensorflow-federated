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

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import types
from tensorflow_federated.python.core.api import value_base

from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import value_impl


# TODO(b/113112108): Add the formal TFF type signatures after overhauling the
# typechecking logic to transform these from polymorphic callables into actual
# federated computations.


@computations.federated_computation
def federated_broadcast(value):
  """Broadcasts a value from the `SERVER` to the `CLIENTS`.

  Args:
    value: A value of a TFF type placed at the `SERVER`.

  Returns:
    A representation of the result of broadcasting of 'value' to `CLIENTS`.

  Raises:
    TypeError: if the argument is not a `SERVER`-side federated TFF value.
  """
  # TODO(b/113112108): Replace this manual typechecking with generic one after
  # adding support for typechecking federated template types.
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, types.FederatedType)
  if value.type_signature.placement is not placements.SERVER:
    raise TypeError('The broadcasted value must reside at the SERVER.')
  if not value.type_signature.all_equal:
    raise TypeError('The broadcasted value must be equal at all locations.')

  # TODO(b/113112108): Replace this manual construction with generic one after
  # adding support for typechecking federated template types.
  result_type = types.FederatedType(
      value.type_signature.member, placements.CLIENTS, True)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_BROADCAST.uri,
      types.FunctionType(value.type_signature, result_type))
  assert isinstance(value, value_impl.ValueImpl)
  return value_impl.ValueImpl(
      computation_building_blocks.Call(
          intrinsic, value_impl.ValueImpl.get_comp(value)))


@computations.federated_computation
def federated_map(value, mapping_fn):
  """Maps constituents of a federated value using a given mapping function.

  Args:
    value: A value of a TFF type placed at the `CLIENTS`.
    mapping_fn: A mapping function to apply pointwise to member constituents of
      'value' on each of the participants in `CLIENTS`. The parameter and
      result of this function must be of the same type as the mmeber constitents
      of `value`.

  Returns:
    A federated value on `CLIENTS` that represents the result of mapping.

  Raises:
    TypeError: if the arguments are not of the appropriates types.
  """
  # TODO(b/113112108): Replace this manual typechecking with generic one after
  # adding support for typechecking federated template types.
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, types.FederatedType)
  if value.type_signature.placement is not placements.CLIENTS:
    raise TypeError('The value to be mapped must reside at the CLIENTS.')
  if not isinstance(mapping_fn, value_base.Value):
    mapping_fn = value_impl.to_value(mapping_fn)
  assert isinstance(mapping_fn, value_base.Value)
  py_typecheck.check_type(mapping_fn.type_signature, types.FunctionType)
  if not mapping_fn.type_signature.parameter.is_assignable_from(
      value.type_signature.member):
    raise TypeError(
        'The mapping function expects a parameter of type {}, but member '
        'constituents of the mapped value are of incompatible type {}.'.format(
            str(mapping_fn.type_signature.parameter_type),
            str(value.type_signature.member)))

  # TODO(b/113112108): Replace this manual construction with generic one after
  # adding support for typechecking federated template types.
  result_type = types.FederatedType(
      mapping_fn.type_signature.result, placements.CLIENTS, False)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP.uri,
      types.FunctionType(value.type_signature, result_type))
  assert isinstance(value, value_impl.ValueImpl)
  return value_impl.ValueImpl(
      computation_building_blocks.Call(
          intrinsic, value_impl.ValueImpl.get_comp(value)))


@computations.federated_computation
def federated_sum(value):
  """Computes a `SERVER`-side sum of a federated value placed on the `CLIENTS`.

  Args:
    value: A value of a TFF type placed at the CLIENTS.

  Returns:
    A representation of the sum of member constituents of 'value' on the
    `SERVER`.

  Raises:
    TypeError: if the argument is not a `CLIENT`-side federated TFF value.
  """
  # TODO(b/113112108): Replace this manual typechecking with generic one after
  # adding support for typechecking federated template types.
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, types.FederatedType)
  if value.type_signature.placement is not placements.CLIENTS:
    raise TypeError('The broadcasted value must reside at the CLIENTS.')

  # TODO(b/113112108): Replace this manual construction with generic one after
  # adding support for typechecking federated template types.
  result_type = types.FederatedType(
      value.type_signature.member, placements.SERVER, True)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SUM.uri,
      types.FunctionType(value.type_signature, result_type))
  assert isinstance(value, value_impl.ValueImpl)
  return value_impl.ValueImpl(
      computation_building_blocks.Call(
          intrinsic, value_impl.ValueImpl.get_comp(value)))
