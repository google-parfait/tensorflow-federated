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
"""Library implementing reusable `computation_building_blocks` constructs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals


def construct_federated_getitem_call(arg, idx):
  """Calls intrinsic `ValueImpl`, passing getitem to a federated value.

  The main piece of orchestration plugging __getitem__ call together with a
  federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `idx`.
    idx: Index, instance of `int` or `slice` used to address the
      `computation_types.NamedTupleType` underlying `arg`.

  Returns:
    Returns an instance of `ValueImpl` of type `computation_types.FederatedType`
    of same placement as `arg`, the result of applying or mapping the
    appropriate `__getitem__` function, as defined by `idx`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(idx, (int, slice))
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getitem_comp = construct_federated_getitem_comp(arg, idx)
  intrinsic = construct_map_or_apply(getitem_comp, arg)
  call = computation_building_blocks.Call(
      intrinsic, computation_building_blocks.Tuple([getitem_comp, arg]))
  return call


def construct_federated_getattr_call(arg, name):
  """Calls intrinsic `ValueImpl`, passing getattr to a federated value.

  The main piece of orchestration plugging __getattr__ call together with a
  federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `name`.
    name: String name to address the `computation_types.NamedTupleType`
      underlying `arg`.

  Returns:
    Returns an instance of `ValueImpl` of type `computation_types.FederatedType`
    of same placement as `arg`, the result of applying or mapping the
    appropriate `__getattr__` function, as defined by `name`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getattr_comp = construct_federated_getattr_comp(arg, name)
  intrinsic = construct_map_or_apply(getattr_comp, arg)
  call = computation_building_blocks.Call(
      intrinsic, computation_building_blocks.Tuple([getattr_comp, arg]))
  return call


def construct_map_or_apply(fn, arg):
  """Injects intrinsic to allow application of `fn` to federated `arg`.

  Args:
    fn: `value_base.Value` instance of non-federated type to be wrapped with
      intrinsic in order to call on `arg`.
    arg: `computation_building_blocks.ComputationBuildingBlock` instance of
      federated type for which to construct intrinsic in order to call `fn` on
      `value`.

  Returns:
    Returns `value_base.Value` instance wrapping
      `computation_building_blocks.Intrinsic` which can call `fn` on `arg`.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                arg.type_signature.placement,
                                                arg.type_signature.all_equal)
  if arg.type_signature.placement == placement_literals.SERVER:
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_APPLY.uri,
        computation_types.FunctionType([fn.type_signature, arg.type_signature],
                                       result_type))
  elif arg.type_signature.placement == placement_literals.CLIENTS:
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MAP.uri,
        computation_types.FunctionType([fn.type_signature, arg.type_signature],
                                       result_type))
  return intrinsic


def construct_federated_getattr_comp(comp, name):
  """Function to construct computation for `federated_apply` of `__getattr__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `name` from its argument, of type `comp.type_signature.member`,
  an instance of `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `ValueImpl` or
      `computation_building_blocks.ComputationBuildingBlock` with type signature
      `computation_types.FederatedType` whose `member` attribute is of type
      `computation_types.NamedTupleType`.
    name: String name of attribute to grab.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs attribute
      according to `name` of its argument.
  """
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  element_names = [
      x for x, _ in anonymous_tuple.to_elements(comp.type_signature.member)
  ]
  if name not in element_names:
    raise ValueError('The federated value {} has no element of name {}'.format(
        comp, name))
  apply_input = computation_building_blocks.Reference(
      'x', comp.type_signature.member)
  selected = computation_building_blocks.Selection(apply_input, name=name)
  apply_lambda = computation_building_blocks.Lambda(
      'x', apply_input.type_signature, selected)
  return apply_lambda


def construct_federated_getitem_comp(comp, key):
  """Function to construct computation for `federated_apply` of `__getitem__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `key` from its argument, of type `comp.type_signature.member`,
  of type `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `ValueImpl` or
      `computation_building_blocks.ComputationBuildingBlock` with type signature
      `computation_types.FederatedType` whose `member` attribute is of type
      `computation_types.NamedTupleType`.
    key: Instance of `int` or `slice`, key used to grab elements from the member
      of `comp`. implementation of slicing for `ValueImpl` objects with
      `type_signature` `computation_types.NamedTupleType`.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs slice
      according to `key` of its argument.
  """
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(key, (int, slice))
  apply_input = computation_building_blocks.Reference(
      'x', comp.type_signature.member)
  if isinstance(key, int):
    selected = computation_building_blocks.Selection(apply_input, index=key)
  else:
    elems = anonymous_tuple.to_elements(comp.type_signature.member)
    index_range = six.moves.range(*key.indices(len(elems)))
    elem_list = []
    for k in index_range:
      elem_list.append((elems[k][0],
                        computation_building_blocks.Selection(
                            apply_input, index=k)))
    selected = computation_building_blocks.Tuple(elem_list)
  apply_lambda = computation_building_blocks.Lambda(
      'x', apply_input.type_signature, selected)
  return apply_lambda
