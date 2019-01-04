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
      `FederatedTypes` with `CLIENT` placements.
    context_stack: The context stack to use, as in `impl.value_impl.to_value`.

  Returns:
    TFF `Value` of `FederatedType` with member of 2-tuple `NamedTuple` type.
  """
  py_typecheck.check_type(input_val, value_base.Value)
  py_typecheck.check_type(input_val.type_signature,
                          computation_types.NamedTupleType)
  for elem in input_val:
    type_utils.check_federated_value_placement(elem, placements.CLIENTS)
  num_elements = len(anonymous_tuple.to_elements(input_val.type_signature))
  if num_elements != 2:
    raise ValueError('The argument of zip_two_tuple must be a 2-tuple, '
                     'not an {}-tuple'.format(num_elements))
  result_type = computation_types.FederatedType(
      [
          e.member
          for _, e in anonymous_tuple.to_elements(input_val.type_signature)
      ], placements.CLIENTS,
      all(e.all_equal
          for _, e in anonymous_tuple.to_elements(input_val.type_signature)))
  intrinsic = value_impl.ValueImpl(
      computation_building_blocks.Intrinsic(
          intrinsic_defs.FEDERATED_ZIP.uri,
          computation_types.FunctionType(input_val.type_signature,
                                         result_type)), context_stack)
  return intrinsic(input_val)


def flatten_first_index(apply_func, type_to_add, context_stack):
  """Returns a value `(arg -> APPEND(apply_func(arg[0]), arg[1]))`.

  In the above, `APPEND(a,b)` refers to appending element b to tuple a.

  Constructs a Value of a TFF functional type that:

  1. Takes as argument a 2-element tuple `(x, y)` of TFF type
     `[apply_func.type_signature.parameter, type_to_add]`.

  2. Transforms the 1st element `x` of this 2-tuple by applying `apply_func`,
     producing a result `z` that must be a TFF tuple (e.g, as a result of
     flattening `x`).

  3. Leaves the 2nd element `y` of the argument 2-tuple unchanged.

  4. Returns the result of appending the unchanged `y` at the end of the
     tuple `z` returned by `apply_func`.

  Args:
    apply_func: TFF `Value` of type_signature `FunctionType`, a function taking
      TFF `Value`s to `Value`s of type `NamedTupleType`.
    type_to_add: Instance of `tff.Type` class, the type expected for arg[1].
    context_stack: The context stack to use, as in `impl.value_impl.to_value`.

  Returns:
    TFF `Value` of `FunctionType`, taking 2-tuples to N-tuples, which calls
      `apply_func` on the first index of its argument, appends the second
      index to the resulting (N-1)-tuple, then returns the N-tuple thus created.
  """
  py_typecheck.check_type(apply_func, value_base.Value)
  py_typecheck.check_type(apply_func.type_signature,
                          computation_types.FunctionType)
  py_typecheck.check_type(apply_func.type_signature.result,
                          computation_types.NamedTupleType)
  prev_param_type = apply_func.type_signature.parameter
  inputs = value_impl.to_value(
      computation_building_blocks.Reference(
          'inputs',
          computation_types.NamedTupleType([prev_param_type, type_to_add])),
      None, context_stack)
  intermediate = apply_func(inputs[0])
  new_elements = value_impl.to_value(
      list(iter(intermediate)) + [inputs[1]],
      type_spec=computation_types.NamedTupleType(
          [x.type_signature for x in iter(intermediate)] +
          [inputs[1].type_signature]),
      context_stack=context_stack)
  return value_impl.to_value(
      computation_building_blocks.Lambda(
          'inputs', inputs.type_signature,
          value_impl.ValueImpl.get_comp(new_elements)), None, context_stack)
