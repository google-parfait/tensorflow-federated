# Lint as: python3
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
"""Utility functions for slicing and dicing intrinsics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import type_constructors
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


def zero_for(type_spec, context_stack):
  """Constructs ZERO intrinsic of TFF type `type_spec`.

  Args:
    type_spec: An instance of `types.Type` or something convertible to it.
      intrinsic.
    context_stack: The context stack to use.

  Returns:
    The `ZERO` intrinsic of the same TFF type as that of `val`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  return value_impl.ValueImpl(
      computation_building_blocks.Intrinsic(intrinsic_defs.GENERIC_ZERO.uri,
                                            type_spec), context_stack)


def plus_for(type_spec, context_stack):
  """Constructs PLUS intrinsic that operates on values of TFF type `type_spec`.

  Args:
    type_spec: An instance of `types.Type` or something convertible to it.
      intrinsic.
    context_stack: The context stack to use.

  Returns:
    The `PLUS` intrinsic of type `<T,T> -> T`, where `T` represents `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  return value_impl.ValueImpl(
      computation_building_blocks.Intrinsic(
          intrinsic_defs.GENERIC_PLUS.uri,
          type_constructors.binary_op(type_spec)), context_stack)


def _construct_lambda_encapsulating_binary_operator(ref_to_arg, operator):
  """Constructs lambda upcasting `ref_to_arg` and applying `operator`."""
  py_typecheck.check_type(ref_to_arg.type_signature,
                          computation_types.NamedTupleType)
  py_typecheck.check_callable(operator)

  def _pack_into_type(to_pack, type_spec):
    """Pack Tensor value `to_pack` into the nested structure `type_spec`."""
    if isinstance(type_spec, computation_types.NamedTupleType):
      elems = anonymous_tuple.to_elements(type_spec)
      packed_elems = [(elem_name, _pack_into_type(to_pack, elem_type))
                      for elem_name, elem_type in elems]
      return computation_building_blocks.Tuple(packed_elems)
    elif isinstance(type_spec, computation_types.TensorType):
      expand_fn = computation_constructing_utils.construct_tensorflow_to_broadcast_scalar(
          to_pack.type_signature.dtype, type_spec.shape)
      return computation_building_blocks.Call(expand_fn, to_pack)

  y_ref = computation_building_blocks.Selection(ref_to_arg, index=1)
  first_arg = computation_building_blocks.Selection(ref_to_arg, index=0)

  if type_utils.are_equivalent_types(first_arg.type_signature,
                                     y_ref.type_signature):
    second_arg = y_ref
  else:
    second_arg = _pack_into_type(y_ref, first_arg.type_signature)

  fn = computation_constructing_utils.construct_tensorflow_binary_operator(
      first_arg.type_signature, operator)
  packed = computation_building_blocks.Tuple([first_arg, second_arg])
  operated = computation_building_blocks.Call(fn, packed)
  lambda_encapsulating_op = computation_building_blocks.Lambda(
      ref_to_arg.name, ref_to_arg.type_signature, operated)
  return lambda_encapsulating_op


def binary_operator_with_upcast(arg, operator):
  """Constructs result of applying `operator` to `arg` upcasting if appropriate.

  Notice `arg` here must be of federated type, with a named tuple member of
  length 2, or a named tuple type of length 2. If the named tuple type of `arg`
  satisfies certain conditions (that is, there is only a single tensor dtype in
  the first element of `arg`, and the second element represents a scalar of
  this dtype), the second element will be upcast to match the first. Here this
  means it will be pushed into a nested structure matching the structure of the
  first element of `arg`. For example, it makes perfect sense to divide a model
  of type `<a=float32[784],b=float32[10]>` by a scalar of type `float32`, but
  the binary operator constructors we have implemented only take arguments of
  type `<T, T>`. Therefore in this case we would broadcast the `float` argument
  to the `tuple` type, before constructing a biary operator which divides
  pointwise.

  Args:
    arg: `computation_building_blocks.ComputationBuildingBlock` of federated
      type whose `member` attribute is a named tuple type of length 2, or named
      tuple type of length 2.
    operator: Callable representing binary operator to apply to the 2-tuple
      represented by the federated `arg`.

  Returns:
    Instance of `computation_building_blocks.ComputationBuildingBlock`
    encapsulating the result of formally applying `operator` to
    `arg[0], `arg[1]`, upcasting `arg[1]` in the condition described above.

  Raises:
    TypeError: If the types don't match.
  """

  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_callable(operator)
  if not type_utils.type_tree_contains_only(
      arg.type_signature,
      (computation_types.FederatedType, computation_types.NamedTupleType,
       computation_types.TensorType)):
    raise TypeError(
        'Generic operators are only implemented for '
        'arguments both containing only federated, tuple and '
        'tensor types; you have passed an argument of type {} '.format(
            arg.type_signature))
  if isinstance(arg.type_signature, computation_types.FederatedType):
    py_typecheck.check_type(arg.type_signature.member,
                            computation_types.NamedTupleType)
    tuple_type = arg.type_signature.member
  elif isinstance(arg.type_signature, computation_types.NamedTupleType):
    tuple_type = arg.type_signature
  else:
    raise TypeError(
        'Generic binary operators are only implemented for '
        'federated tuple and unplaced tuples; you have passed {}.'.format(
            arg.type_signature))
  if len(tuple_type) != 2:
    raise TypeError('We have passed a non 2-tuple to '
                    '`binary_operator_with_upcast`, the type {}.'.format(
                        arg.type_signature.member))
  if not type_utils.is_binary_op_with_upcast_compatible_pair(
      tuple_type[0], tuple_type[1]):
    raise TypeError('The two-tuple you have passed in is incompatible with '
                    'upcasted binary operators. You have passed the tuple '
                    'type {}, which fails the check that the two members of '
                    'the tuple are either the same type, or the second is a '
                    'scalar with the same dtype as the leaves of the first. '
                    'See `type_utils.is_binary_op_with_upcast_compatible_pair` '
                    'for more details.'.format(tuple_type))
  ref_to_arg = computation_building_blocks.Reference('tuple', tuple_type)
  lambda_encapsulating_op = _construct_lambda_encapsulating_binary_operator(
      ref_to_arg, operator)

  if isinstance(arg.type_signature, computation_types.FederatedType):
    called = computation_constructing_utils.create_federated_map_or_apply(
        lambda_encapsulating_op, arg)
  else:
    called = computation_building_blocks.Call(lambda_encapsulating_op, arg)

  return called
