# Lint as: python3
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
from six.moves import range

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_utils


def construct_federated_getitem_call(arg, idx):
  """Constructs computation building block passing getitem to federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `idx`.
    idx: Index, instance of `int` or `slice` used to address the
      `computation_types.NamedTupleType` underlying `arg`.

  Returns:
    Returns a `computation_building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`, the result
    of applying or mapping the appropriate `__getitem__` function, as defined
    by `idx`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(idx, (int, slice))
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getitem_comp = construct_federated_getitem_comp(arg, idx)
  return create_federated_map_or_apply(getitem_comp, arg)


def construct_federated_getattr_call(arg, name):
  """Constructs computation building block passing getattr to federated value.

  Args:
    arg: Instance of `computation_building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `name`.
    name: String name to address the `computation_types.NamedTupleType`
      underlying `arg`.

  Returns:
    Returns a `computation_building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`,
    the result of applying or mapping the appropriate `__getattr__` function,
    as defined by `name`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getattr_comp = construct_federated_getattr_comp(arg, name)
  return create_federated_map_or_apply(getattr_comp, arg)


def construct_federated_setattr_call(federated_comp, name, value_comp):
  """Returns building block for `setattr(name, value_comp)` on `federated_comp`.

  Constructs an appropriate communication intrinsic (either `federated_map` or
  `federated_apply`) as well as a `computation_building_blocks.Lambda`
  representing setting the `name` attribute of `federated_comp`'s `member` to
  `value_comp`, and stitches these together in a call.

  Notice that `federated_comp`'s `member` must actually define a `name`
  attribute; this is enforced to avoid the need to worry about theplacement of a
  previously undefined name.

  Args:
    federated_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock` of type
      `computation_types.FederatedType`, with member of type
      `computation_types.NamedTupleType` whose attribute `name` we wish to set
      to `value_comp`.
    name: String name of the attribute we wish to overwrite in `federated_comp`.
    value_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock`, the value to
      assign to `federated_comp`'s `member`'s `name` attribute.

  Returns:
    Instance of `computation_building_blocks.ComputationBuildingBlock`
    representing `federated_comp` with its `member`'s `name` attribute set to
    `value`.
  """
  py_typecheck.check_type(federated_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(federated_comp.type_signature,
                          computation_types.FederatedType)
  py_typecheck.check_type(federated_comp.type_signature.member,
                          computation_types.NamedTupleType)
  named_tuple_type_signature = federated_comp.type_signature.member
  setattr_lambda = construct_named_tuple_setattr_lambda(
      named_tuple_type_signature, name, value_comp)
  return create_federated_map_or_apply(setattr_lambda, federated_comp)


def construct_named_tuple_setattr_lambda(named_tuple_signature, name,
                                         value_comp):
  """Constructs a building block for replacing one attribute in a named tuple.

  Returns an instance of `computation_building_blocks.Lambda` which takes an
  argument of type `computation_types.NamedTupleType` and returns a
  `computation_building_blocks.Tuple` which contains all the same elements as
  the argument, except the attribute `name` now has value `value_comp`. The
  Lambda constructed is the analogue of Python's `setattr` for the concrete
  type `named_tuple_signature`.

  Args:
    named_tuple_signature: Instance of `computation_types.NamedTupleType`, the
      type of the argument to the constructed
      `computation_building_blocks.Lambda`.
    name: String name of the attribute in the `named_tuple_signature` to replace
      with `value_comp`. Must be present as a name in `named_tuple_signature;
      otherwise we will raise an `AttributeError`.
    value_comp: Instance of
      `computation_building_blocks.ComputationBuildingBlock`, the value to place
      as attribute `name` in the argument of the returned function.

  Returns:
    An instance of `computation_building_blocks.Block` of functional type
    representing setting attribute `name` to value `value_comp` in its argument
    of type `named_tuple_signature`.

  Raises:
    TypeError: If the types of the arguments don't match the assumptions above.
    AttributeError: If `name` is not present as a named element in
      `named_tuple_signature`
  """
  py_typecheck.check_type(named_tuple_signature,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp,
                          computation_building_blocks.ComputationBuildingBlock)
  value_comp_placeholder = computation_building_blocks.Reference(
      'value_comp_placeholder', value_comp.type_signature)
  lambda_arg = computation_building_blocks.Reference('lambda_arg',
                                                     named_tuple_signature)
  if name not in dir(named_tuple_signature):
    raise AttributeError(
        'There is no such attribute as \'{}\' in this federated tuple. '
        'TFF does not allow for assigning to a nonexistent attribute. '
        'If you want to assign to \'{}\', you must create a new named tuple '
        'containing this attribute.'.format(name, name))
  elements = []
  for idx, (key, element_type) in enumerate(
      anonymous_tuple.to_elements(named_tuple_signature)):
    if key == name:
      if not type_utils.is_assignable_from(element_type,
                                           value_comp.type_signature):
        raise TypeError(
            '`setattr` has attempted to set element {} of type {} with incompatible type {}'
            .format(key, element_type, value_comp.type_signature))
      elements.append((key, value_comp_placeholder))
    else:
      elements.append(
          (key, computation_building_blocks.Selection(lambda_arg, index=idx)))
  return_tuple = computation_building_blocks.Tuple(elements)
  lambda_to_return = computation_building_blocks.Lambda(lambda_arg.name,
                                                        named_tuple_signature,
                                                        return_tuple)
  symbols = ((value_comp_placeholder.name, value_comp),)
  return computation_building_blocks.Block(symbols, lambda_to_return)


def construct_federated_getattr_comp(comp, name):
  """Function to construct computation for `federated_apply` of `__getattr__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `name` from its argument, of type `comp.type_signature.member`,
  an instance of `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      with type signature `computation_types.FederatedType` whose `member`
      attribute is of type `computation_types.NamedTupleType`.
    name: String name of attribute to grab.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs attribute
      according to `name` of its argument.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  element_names = [
      x for x, _ in anonymous_tuple.to_elements(comp.type_signature.member)
  ]
  if name not in element_names:
    raise ValueError('The federated value {} has no element of name {}'.format(
        comp, name))
  apply_input = computation_building_blocks.Reference(
      'x', comp.type_signature.member)
  selected = computation_building_blocks.Selection(apply_input, name=name)
  apply_lambda = computation_building_blocks.Lambda('x',
                                                    apply_input.type_signature,
                                                    selected)
  return apply_lambda


def construct_federated_getitem_comp(comp, key):
  """Function to construct computation for `federated_apply` of `__getitem__`.

  Constructs a `computation_building_blocks.ComputationBuildingBlock`
  which selects `key` from its argument, of type `comp.type_signature.member`,
  of type `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      with type signature `computation_types.FederatedType` whose `member`
      attribute is of type `computation_types.NamedTupleType`.
    key: Instance of `int` or `slice`, key used to grab elements from the member
      of `comp`. implementation of slicing for `ValueImpl` objects with
      `type_signature` `computation_types.NamedTupleType`.

  Returns:
    Instance of `computation_building_blocks.Lambda` which grabs slice
      according to `key` of its argument.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
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
    index_range = range(*key.indices(len(elems)))
    elem_list = []
    for k in index_range:
      elem_list.append(
          (elems[k][0],
           computation_building_blocks.Selection(apply_input, index=k)))
    selected = computation_building_blocks.Tuple(elem_list)
  apply_lambda = computation_building_blocks.Lambda('x',
                                                    apply_input.type_signature,
                                                    selected)
  return apply_lambda


def create_computation_appending(comp1, comp2):
  r"""Returns a block appending `comp2` to `comp1`.

                Block
               /     \
  [comps=Tuple]       Tuple
         |            |
    [Comp, Comp]      [Sel(0), ...,  Sel(0),   Sel(1)]
                             \             \         \
                              Sel(0)        Sel(n)    Ref(comps)
                                    \             \
                                     Ref(comps)    Ref(comps)

  Args:
    comp1: A `computation_building_blocks.ComputationBuildingBlock` with a
      `type_signature` of type `computation_type.NamedTupleType`.
    comp2: A `computation_building_blocks.ComputationBuildingBlock` or a named
      computation (a tuple pair of name, computation) representing a single
      element of an `anonymous_tuple.AnonymousTuple`.

  Returns:
    A `computation_building_blocks.Block`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(comp1,
                          computation_building_blocks.ComputationBuildingBlock)
  if isinstance(comp2, computation_building_blocks.ComputationBuildingBlock):
    name2 = None
  elif py_typecheck.is_name_value_pair(
      comp2,
      name_required=False,
      value_type=computation_building_blocks.ComputationBuildingBlock):
    name2, comp2 = comp2
  else:
    raise TypeError('Unexpected tuple element: {}.'.format(comp2))
  comps = computation_building_blocks.Tuple((comp1, comp2))
  ref = computation_building_blocks.Reference('comps', comps.type_signature)
  sel_0 = computation_building_blocks.Selection(ref, index=0)
  elements = []
  named_type_signatures = anonymous_tuple.to_elements(comp1.type_signature)
  for index, (name, _) in enumerate(named_type_signatures):
    sel = computation_building_blocks.Selection(sel_0, index=index)
    elements.append((name, sel))
  sel_1 = computation_building_blocks.Selection(ref, index=1)
  elements.append((name2, sel_1))
  result = computation_building_blocks.Tuple(elements)
  symbols = ((ref.name, comps),)
  return computation_building_blocks.Block(symbols, result)


def create_federated_aggregate(value, zero, accumulate, merge, report):
  r"""Creates a called federated aggregate.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp, Comp, Comp]

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.
    zero: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      initial value.
    accumulate: A `computation_building_blocks.ComputationBuildingBlock` to use
      as the accumulate function.
    merge: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the merge function.
    report: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the report function.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(accumulate,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(merge,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(report,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(report.type_signature.result,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType((
      value.type_signature,
      zero.type_signature,
      accumulate.type_signature,
      merge.type_signature,
      report.type_signature,
  ), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_AGGREGATE.uri, intrinsic_type)
  values = computation_building_blocks.Tuple(
      (value, zero, accumulate, merge, report))
  return computation_building_blocks.Call(intrinsic, values)


def create_federated_apply(fn, arg):
  r"""Creates a called federated apply.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_APPLY.uri, intrinsic_type)
  values = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, values)


def create_federated_broadcast(value):
  r"""Creates a called federated broadcast.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placement_literals.CLIENTS,
                                                all_equal=True)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_BROADCAST.uri, intrinsic_type)
  return computation_building_blocks.Call(intrinsic, value)


def create_federated_collect(value):
  r"""Creates a called federated collect.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  type_signature = computation_types.SequenceType(value.type_signature.member)
  result_type = computation_types.FederatedType(type_signature,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_COLLECT.uri, intrinsic_type)
  return computation_building_blocks.Call(intrinsic, value)


def create_federated_map(fn, arg):
  r"""Creates a called federated map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  parameter_type = computation_types.FederatedType(arg.type_signature.member,
                                                   placement_literals.CLIENTS)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placement_literals.CLIENTS)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP.uri, intrinsic_type)
  values = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, values)


def create_federated_map_or_apply(fn, arg):
  r"""Creates a called federated map or apply depending on `arg`s placement.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  if arg.type_signature.placement is placement_literals.CLIENTS:
    return create_federated_map(fn, arg)
  elif arg.type_signature.placement is placement_literals.SERVER:
    return create_federated_apply(fn, arg)
  else:
    raise TypeError('Unsupported placement {}.'.format(
        arg.type_signature.placement))


def create_federated_mean(value, weight):
  r"""Creates a called federated mean.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.
    weight: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the weight or `None`.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  if weight is not None:
    py_typecheck.check_type(
        weight, computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placement_literals.SERVER)
  if weight is not None:
    intrinsic_type = computation_types.FunctionType(
        (value.type_signature, weight.type_signature), result_type)
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri, intrinsic_type)
    values = computation_building_blocks.Tuple((value, weight))
    return computation_building_blocks.Call(intrinsic, values)
  else:
    intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                    result_type)
    intrinsic = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MEAN.uri, intrinsic_type)
    return computation_building_blocks.Call(intrinsic, value)


def create_federated_reduce(value, zero, op):
  r"""Creates a called federated reduce.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp]

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.
    zero: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      initial value.
    op: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      op function.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(op,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(op.type_signature.result,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType((
      value.type_signature,
      zero.type_signature,
      op.type_signature,
  ), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_REDUCE.uri, intrinsic_type)
  values = computation_building_blocks.Tuple((value, zero, op))
  return computation_building_blocks.Call(intrinsic, values)


def create_federated_sum(value):
  r"""Creates a called federated sum.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SUM.uri, intrinsic_type)
  return computation_building_blocks.Call(intrinsic, value)


def create_federated_unzip(value):
  r"""Creates a tuple of called federated maps or applies.

                Block
               /     \
  [value=Comp]        Tuple
                      |
                      [Call,                        Call, ...]
                      /    \                       /    \
             Intrinsic      Tuple         Intrinsic      Tuple
                            |                            |
                [Lambda(arg), Ref(value)]    [Lambda(arg), Ref(value)]
                            \                            \
                             Sel(0)                       Sel(1)
                                   \                            \
                                    Ref(arg)                     Ref(arg)

  This function returns a tuple of federated values given a `value` with a
  federated tuple type signature.

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` with a
      `type_signature` of type `computation_types.NamedTupleType` containing at
      least one element.

  Returns:
    A `computation_building_blocks.Block`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(
      value.type_signature.member)
  length = len(named_type_signatures)
  if length == 0:
    raise ValueError('federated_zip is only supported on non-empty tuples.')
  value_ref = computation_building_blocks.Reference('value',
                                                    value.type_signature)
  elements = []
  fn_ref = computation_building_blocks.Reference('arg', named_type_signatures)
  for index, (name, _) in enumerate(named_type_signatures):
    sel = computation_building_blocks.Selection(fn_ref, index=index)
    fn = computation_building_blocks.Lambda(fn_ref.name, fn_ref.type_signature,
                                            sel)
    intrinsic = create_federated_map_or_apply(fn, value_ref)
    elements.append((name, intrinsic))
  result = computation_building_blocks.Tuple(elements)
  symbols = ((value_ref.name, value),)
  return computation_building_blocks.Block(symbols, result)


def create_federated_value(value, placement):
  r"""Creates a called federated value.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.
    placement: A `placement_literals.PlacementLiteral` to use as the placement.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  if placement is placement_literals.CLIENTS:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri
  elif placement is placement_literals.SERVER:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  result_type = computation_types.FederatedType(value.type_signature, placement,
                                                all_equal=True)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = computation_building_blocks.Intrinsic(uri, intrinsic_type)
  return computation_building_blocks.Call(intrinsic, value)


def create_federated_zip(value):
  r"""Creates a called federated zip.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  This function returns a federated tuple given a `value` with a tuple of
  federated values type signature.

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` with a
      `type_signature` of type `computation_types.NamedTupleType` containing at
      least one element.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length == 0:
    raise ValueError('federated_zip is only supported on non-empty tuples.')
  first_name, first_type_signature = named_type_signatures[0]
  if first_type_signature.placement == placement_literals.CLIENTS:
    map_fn = create_federated_map
  elif first_type_signature.placement == placement_literals.SERVER:
    map_fn = create_federated_apply
  else:
    raise TypeError('Unsupported placement {}.'.format(
        first_type_signature.placement))
  if length == 1:
    ref = computation_building_blocks.Reference('arg',
                                                first_type_signature.member)
    values = computation_building_blocks.Tuple(((first_name, ref),))
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature,
                                            values)
    sel = computation_building_blocks.Selection(value, index=0)
    return map_fn(fn, sel)
  else:
    zipped_args = _create_chain_zipped_values(value)
    append_fn = _create_fn_to_append_chain_zipped_values(value)
    return map_fn(append_fn, zipped_args)


def _create_chain_zipped_values(value):
  r"""Creates a chain of called federated zip with two values.

                Block--------
               /             \
  [value=Tuple]               Call
         |                   /    \
         [Comp1,    Intrinsic      Tuple
          Comp2,                   |
          ...]                     [Call,  Sel(n)]
                                   /    \        \
                          Intrinsic      Tuple    Ref(value)
                                         |
                                         [Sel(0),       Sel(1)]
                                                \             \
                                                 Ref(value)    Ref(value)

  NOTE: This function is intended to be used in conjunction with
  `_create_fn_to_append_chain_zipped_values` and will drop the tuple names. The
  names will be added back to the resulting computation when the zipped values
  are mapped to a function that flattens the chain.

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` with a
      `type_signature` of type `computation_types.NamedTupleType` containing at
      least two elements.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain at least two elements.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length < 2:
    raise ValueError(
        'Expected a value with at least two elements, received {} elements.'
        .format(named_type_signatures))
  first_name, _ = named_type_signatures[0]
  ref = computation_building_blocks.Reference('value', value.type_signature)
  symbols = ((ref.name, value),)
  sel_0 = computation_building_blocks.Selection(ref, index=0)
  result = (first_name, sel_0)
  for i in range(1, length):
    name, _ = named_type_signatures[i]
    sel = computation_building_blocks.Selection(ref, index=i)
    values = computation_building_blocks.Tuple((result, (name, sel)))
    result = _create_zip_two_values(values)
  return computation_building_blocks.Block(symbols, result)


def _create_zip_two_values(value):
  r"""Creates a called federated zip with two values.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp1, Comp2]

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` with a
      `type_signature` of type `computation_types.NamedTupleType` containing
      exactly two elements.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain exactly two elements.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length != 2:
    raise ValueError(
        'Expected a value with exactly two elements, received {} elements.'
        .format(named_type_signatures))
  placement = value[0].type_signature.placement
  if placement is placement_literals.CLIENTS:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri
    all_equal = False
  elif placement is placement_literals.SERVER:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri
    all_equal = True
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  elements = []
  for name, type_signature in named_type_signatures:
    federated_type = computation_types.FederatedType(type_signature.member,
                                                     placement, all_equal)
    elements.append((name, federated_type))
  parameter_type = computation_types.NamedTupleType(elements)
  result_type = computation_types.FederatedType(
      [(n, e.member) for n, e in named_type_signatures], placement, all_equal)
  intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
  intrinsic = computation_building_blocks.Intrinsic(uri, intrinsic_type)
  return computation_building_blocks.Call(intrinsic, value)


def _create_fn_to_append_chain_zipped_values(value):
  r"""Creates a function to append a chain of zipped values.

  Lambda(arg3)
            \
             append([Call,    Sel(1)])
                    /    \            \
        Lambda(arg2)      Sel(0)       Ref(arg3)
                  \             \
                   \             Ref(arg3)
                    \
                     append([Call,    Sel(1)])
                            /    \            \
                Lambda(arg1)      Sel(0)       Ref(arg2)
                            \           \
                             \           Ref(arg2)
                              \
                               Ref(arg1)

  NOTE: This function is intended to be used in conjunction with
  `_create_chain_zipped_values` add will add back the names that were dropped
  when zipping the values.

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` with a
      `type_signature` of type `computation_types.NamedTupleType` containing at
      least two elements.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length < 2:
    raise ValueError(
        'Expected a value with at least two elements, received {} elements.'
        .format(named_type_signatures))
  first_name, first_type_signature = named_type_signatures[0]
  second_name, second_type_signature = named_type_signatures[1]
  ref_type = computation_types.NamedTupleType((
      (first_name, first_type_signature.member),
      (second_name, second_type_signature.member),
  ))
  ref = computation_building_blocks.Reference('arg', ref_type)
  fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
  for name, type_signature in named_type_signatures[2:]:
    ref_type = computation_types.NamedTupleType((
        fn.type_signature.parameter,
        (name, type_signature.member),
    ))
    ref = computation_building_blocks.Reference('arg', ref_type)
    sel_0 = computation_building_blocks.Selection(ref, index=0)
    call = computation_building_blocks.Call(fn, sel_0)
    sel_1 = computation_building_blocks.Selection(ref, index=1)
    result = create_computation_appending(call, (name, sel_1))
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature,
                                            result)
  return fn


def create_sequence_map(fn, arg):
  r"""Creates a called sequence map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.SequenceType(fn.type_signature.result)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.SEQUENCE_MAP.uri, intrinsic_type)
  values = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, values)


def create_sequence_reduce(value, zero, op):
  r"""Creates a called sequence reduce.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp]

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.
    zero: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      initial value.
    op: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      op function.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(op,
                          computation_building_blocks.ComputationBuildingBlock)
  intrinsic_type = computation_types.FunctionType((
      value.type_signature,
      zero.type_signature,
      op.type_signature,
  ), op.type_signature.result)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.SEQUENCE_REDUCE.uri, intrinsic_type)
  values = computation_building_blocks.Tuple((value, zero, op))
  return computation_building_blocks.Call(intrinsic, values)


def create_sequence_sum(value):
  r"""Creates a called sequence sum.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value.

  Returns:
    A `computation_building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  value.type_signature.element)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.SEQUENCE_SUM.uri, intrinsic_type)
  return computation_building_blocks.Call(intrinsic, value)
