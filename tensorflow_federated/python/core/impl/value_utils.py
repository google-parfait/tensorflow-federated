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
"""Utilities file for functions with TFF `Value`s as inputs and outputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


def zip_two_tuple(input_val, context_stack):
  """Helper function to perform 2-tuple at a time zipping.

  Takes 2-tuple of federated values and returns federated 2-tuple of values.

  Args:
    input_val: 2-tuple TFF `Value` of `NamedTuple` type, whose elements must be
      `FederatedTypes` with the same placement.
    context_stack: The context stack to use, as in `impl.value_impl.to_value`.

  Returns:
    TFF `Value` of `FederatedType` with member of 2-tuple `NamedTuple` type.
  """
  py_typecheck.check_type(input_val, value_base.Value)
  py_typecheck.check_type(input_val.type_signature,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(input_val[0].type_signature,
                          computation_types.FederatedType)

  zip_uris = {
      placements.CLIENTS: intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri,
      placements.SERVER: intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri,
  }
  zip_all_equal = {
      placements.CLIENTS: False,
      placements.SERVER: True,
  }
  output_placement = input_val[0].type_signature.placement
  if output_placement not in zip_uris:
    raise TypeError('The argument must have components placed at SERVER or '
                    'CLIENTS')
  output_all_equal_bit = zip_all_equal[output_placement]
  for elem in input_val:
    type_utils.check_federated_value_placement(elem, output_placement)
  num_elements = len(anonymous_tuple.to_elements(input_val.type_signature))
  if num_elements != 2:
    raise ValueError('The argument of zip_two_tuple must be a 2-tuple, '
                     'not an {}-tuple'.format(num_elements))
  result_type = computation_types.FederatedType(
      [(name, e.member)
       for name, e in anonymous_tuple.to_elements(input_val.type_signature)],
      output_placement, output_all_equal_bit)

  def _adjust_all_equal_bit(x):
    return computation_types.FederatedType(x.member, x.placement,
                                           output_all_equal_bit)

  adjusted_input_type = computation_types.NamedTupleType(
      [(k, _adjust_all_equal_bit(v)) if k else _adjust_all_equal_bit(v)
       for k, v in anonymous_tuple.to_elements(input_val.type_signature)])

  intrinsic = value_impl.ValueImpl(
      computation_building_blocks.Intrinsic(
          zip_uris[output_placement],
          computation_types.FunctionType(adjusted_input_type, result_type)),
      context_stack)
  return intrinsic(input_val)


def flatten_first_index(apply_fn, type_to_add, context_stack):
  """Returns a value `(arg -> APPEND(apply_fn(arg[0]), arg[1]))`.

  In the above, `APPEND(a,b)` refers to appending element b to tuple a.

  Constructs a Value of a TFF functional type that:

  1. Takes as argument a 2-element tuple `(x, y)` of TFF type
     `[apply_fn.type_signature.parameter, type_to_add]`.

  2. Transforms the 1st element `x` of this 2-tuple by applying `apply_fn`,
     producing a result `z` that must be a TFF tuple (e.g, as a result of
     flattening `x`).

  3. Leaves the 2nd element `y` of the argument 2-tuple unchanged.

  4. Returns the result of appending the unchanged `y` at the end of the
     tuple `z` returned by `apply_fn`.

  Args:
    apply_fn: TFF `Value` of type_signature `FunctionType`, a function taking
      TFF `Value`s to `Value`s of type `NamedTupleType`.
    type_to_add: 2-tuple specifying name and TFF type of arg[1]. Name can be
      `None` or `string`.
    context_stack: The context stack to use, as in `impl.value_impl.to_value`.

  Returns:
    TFF `Value` of `FunctionType`, taking 2-tuples to N-tuples, which calls
      `apply_fn` on the first index of its argument, appends the second
      index to the resulting (N-1)-tuple, then returns the N-tuple thus created.
  """
  py_typecheck.check_type(apply_fn, value_base.Value)
  py_typecheck.check_type(apply_fn.type_signature,
                          computation_types.FunctionType)
  py_typecheck.check_type(apply_fn.type_signature.result,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(type_to_add, tuple)
  if len(type_to_add) != 2:
    raise ValueError('Please pass a 2-tuple as type_to_add to '
                     'flatten_first_index, with first index name or None '
                     'and second index instance of `computation_types.Type` '
                     'or something convertible to one by '
                     '`computationtypes.to_type`.')
  prev_param_type = apply_fn.type_signature.parameter
  inputs = value_impl.to_value(
      computation_building_blocks.Reference(
          'inputs',
          computation_types.NamedTupleType([prev_param_type, type_to_add])),
      None, context_stack)
  intermediate = apply_fn(inputs[0])
  full_type_spec = anonymous_tuple.to_elements(
      apply_fn.type_signature.result) + [type_to_add]
  named_values = [
      (full_type_spec[k][0], intermediate[k]) for k in range(len(intermediate))
  ] + [(full_type_spec[-1][0], inputs[1])]
  new_elements = value_impl.to_value(
      anonymous_tuple.AnonymousTuple(named_values),
      type_spec=full_type_spec,
      context_stack=context_stack)
  return value_impl.to_value(
      computation_building_blocks.Lambda(
          'inputs', inputs.type_signature,
          value_impl.ValueImpl.get_comp(new_elements)), None, context_stack)


def get_curried(fn):
  """Returns a curried version of function `fn` that takes a parameter tuple.

  For functions `fn` of types <T1,T2,....,Tn> -> U, the result is a function
  of the form T1 -> (T2 -> (T3 -> .... (Tn -> U) ... )).

  NOTE: No attempt is made at avoiding naming conflicts in cases where `fn`
  contains references. The arguments of the curriend function are named `argN`
  with `N` starting at 0.

  Args:
    fn: A value of a functional TFF type.

  Returns:
    A value that represents the curried form of `fn`.
  """
  py_typecheck.check_type(fn, value_base.Value)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  py_typecheck.check_type(fn.type_signature.parameter,
                          computation_types.NamedTupleType)
  param_elements = anonymous_tuple.to_elements(fn.type_signature.parameter)
  references = []
  for idx, (_, elem_type) in enumerate(param_elements):
    references.append(
        computation_building_blocks.Reference('arg{}'.format(idx), elem_type))
  result = computation_building_blocks.Call(
      value_impl.ValueImpl.get_comp(fn),
      computation_building_blocks.Tuple(references))
  for ref in references[::-1]:
    result = computation_building_blocks.Lambda(ref.name, ref.type_signature,
                                                result)
  return value_impl.ValueImpl(result,
                              value_impl.ValueImpl.get_context_stack(fn))
