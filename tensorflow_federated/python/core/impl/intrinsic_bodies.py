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
"""Bodies of intrinsics to be added as replacements by the compiler pipleine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import intrinsic_factory
from tensorflow_federated.python.core.impl import intrinsic_utils
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


def get_intrinsic_bodies(context_stack):
  """Returns a dictionary of intrinsic bodies.

  Args:
    context_stack: The context stack to use.
  """
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  intrinsics = intrinsic_factory.IntrinsicFactory(context_stack)

  # TODO(b/122728050): Implement reductions that follow roughly the following
  # breakdown in order to minimize the number of intrinsics that backends need
  # to support and maximize opportunities for merging processing logic to keep
  # the number of communication phases as small as it is practical. Perform
  # these reductions before FEDERATED_SUM (more reductions documented below).
  #
  # - FEDERATED_AGGREGATE(x, zero, accu, merge, report) :=
  #     GENERIC_MAP(
  #       GENERIC_REDUCE(
  #         GENERIC_PARTIAL_REDUCE(x, zero, accu, INTERMEDIATE_AGGREGATORS),
  #         zero, merge, SERVER),
  #       report)
  #
  # - FEDERATED_APPLY(f, x) := GENERIC_APPLY(f, x)
  #
  # - FEDERATED_BROADCAST(x) := GENERIC_BROADCAST(x, CLIENTS)
  #
  # - FEDERATED_COLLECT(x) := GENERIC_COLLECT(x, SERVER)
  #
  # - FEDERATED_MAP(f, x) := GENERIC_MAP(f, x)
  #
  # - FEDERATED_VALUE_AT_CLIENTS(x) := GENERIC_PLACE(x, CLIENTS)
  #
  # - FEDERATED_VALUE_AT_SERVER(x) := GENERIC_PLACE(x, SERVER)
  #
  # - FEDERATED_MEAN(x) := FEDERATED_WEIGHTED_MEAN(
  #     x, FEDERATED_VALUE_AT_CLIENTS(GENERIC_ONE))
  #
  # - FEDERATED_WEIGHTED_MEAN(x, w) :=
  #     GENERIC_DIVIDE(FEDERATED_SUM(GENERIC_MULTIPLY(x, w)), w)

  def _pack_binary_operator_args(x, y):
    """Packs arguments to binary operator into a single arg."""

    def _only_tuple_or_tensor(value):
      return type_utils.type_tree_contains_only(
          value.type_signature,
          (computation_types.NamedTupleType, computation_types.TensorType))

    if _only_tuple_or_tensor(x) and _only_tuple_or_tensor(y):
      arg = value_impl.ValueImpl(
          computation_building_blocks.Tuple([
              value_impl.ValueImpl.get_comp(x),
              value_impl.ValueImpl.get_comp(y)
          ]), context_stack)
    elif (isinstance(x.type_signature, computation_types.FederatedType) and
          isinstance(y.type_signature, computation_types.FederatedType) and
          x.type_signature.placement == y.type_signature.placement):
      if not type_utils.is_binary_op_with_upcast_compatible_pair(
          x.type_signature.member, y.type_signature.member):
        raise TypeError(
            'The members of the federated types {} and {} are not division '
            'compatible; see `type_utils.is_binary_op_with_upcast_compatible_pair` '
            'for more details.'.format(x.type_signature, y.type_signature))
      packed_arg = value_impl.ValueImpl(
          computation_building_blocks.Tuple([
              value_impl.ValueImpl.get_comp(x),
              value_impl.ValueImpl.get_comp(y)
          ]), context_stack)
      arg = intrinsics.federated_zip(packed_arg)
    else:
      raise TypeError
    return arg

  def _check_top_level_compatibility_with_generic_operators(x, y, op_name):
    """Performs non-recursive check on the types of `x` and `y`."""
    x_compatible = type_utils.type_tree_contains_only(
        x.type_signature,
        (computation_types.NamedTupleType, computation_types.TensorType,
         computation_types.FederatedType))
    y_compatible = type_utils.type_tree_contains_only(
        y.type_signature,
        (computation_types.NamedTupleType, computation_types.TensorType,
         computation_types.FederatedType))

    def _make_bad_type_tree_string(index, type_spec):
      return ('{} is only implemented for pairs of '
              'arguments both containing only federated, tuple and '
              'tensor types; you have passed argument at index {} of type {} '
              .format(op_name, index, type_spec))

    if not (x_compatible and y_compatible):
      if y_compatible:
        raise TypeError(_make_bad_type_tree_string(0, x.type_signature))
      elif x_compatible:
        raise TypeError(_make_bad_type_tree_string(1, y.type_signature))
      else:
        raise TypeError(
            '{} is only implemented for pairs of '
            'arguments both containing only federated, tuple and '
            'tensor types; both your arguments fail this condition. '
            'You have passed first argument of type {} '
            'and second argument of type {}.'.format(op_name, x.type_signature,
                                                     y.type_signature))

    top_level_mismatch_string = (
        '{} does not accept arguments of type {} and '
        '{}, as they are mismatched at the top level.'.format(
            x.type_signature, y.type_signature, op_name))
    if isinstance(x.type_signature, computation_types.FederatedType):
      if (not isinstance(y.type_signature, computation_types.FederatedType) or
          x.type_signature.placement != y.type_signature.placement or
          x.type_signature.all_equal != y.type_signature.all_equal or
          not type_utils.is_binary_op_with_upcast_compatible_pair(
              x.type_signature.member, y.type_signature.member)):
        raise TypeError(top_level_mismatch_string)
    if isinstance(x.type_signature, computation_types.NamedTupleType):
      if not isinstance(y.type_signature,
                        computation_types.NamedTupleType) or dir(
                            x.type_signature) != dir(x.type_signature):
        raise TypeError(top_level_mismatch_string)

  def generic_divide(x, y):
    """Divides two arguments when possible."""
    _check_top_level_compatibility_with_generic_operators(
        x, y, 'Generic divide')
    if isinstance(x.type_signature, computation_types.NamedTupleType):
      # This case is needed if federated types are nested deeply.
      names = [t[0] for t in anonymous_tuple.to_elements(x.type_signature)]
      divided = [
          value_impl.ValueImpl.get_comp(generic_divide(x[i], y[i]))
          for i in range(len(names))
      ]
      named_divided = computation_constructing_utils.create_named_tuple(
          computation_building_blocks.Tuple(divided), names)
      return value_impl.ValueImpl(named_divided, context_stack)
    arg = _pack_binary_operator_args(x, y)
    arg_comp = value_impl.ValueImpl.get_comp(arg)
    divided = intrinsic_utils.binary_operator_with_upcast(arg_comp, tf.divide)
    return value_impl.ValueImpl(divided, context_stack)

  def generic_multiply(x, y):
    """Multiplies two arguments when possible."""
    _check_top_level_compatibility_with_generic_operators(
        x, y, 'Generic multiply')
    if isinstance(x.type_signature, computation_types.NamedTupleType):
      # This case is needed if federated types are nested deeply.
      names = [t[0] for t in anonymous_tuple.to_elements(x.type_signature)]
      multiplied = [
          value_impl.ValueImpl.get_comp(generic_multiply(x[i], y[i]))
          for i in range(len(names))
      ]
      named_multiplied = computation_constructing_utils.create_named_tuple(
          computation_building_blocks.Tuple(multiplied), names)
      return value_impl.ValueImpl(named_multiplied, context_stack)
    arg = _pack_binary_operator_args(x, y)
    arg_comp = value_impl.ValueImpl.get_comp(arg)
    multiplied = intrinsic_utils.binary_operator_with_upcast(
        arg_comp, tf.multiply)
    return value_impl.ValueImpl(multiplied, context_stack)

  def federated_weighted_mean(arg):
    x = arg[0]
    w = arg[1]
    multiplied = generic_multiply(x, w)
    summed = federated_sum(intrinsics.federated_zip([multiplied, w]))
    return generic_divide(summed[0], summed[1])

  def federated_sum(x):
    zero = intrinsic_utils.zero_for(x.type_signature.member, context_stack)
    plus = intrinsic_utils.plus_for(x.type_signature.member, context_stack)
    return federated_reduce([x, zero, plus])

  def federated_reduce(arg):
    x = arg[0]
    zero = arg[1]
    op = arg[2]
    identity = computation_constructing_utils.construct_compiled_identity(
        op.type_signature.result)
    return intrinsics.federated_aggregate(x, zero, op, op, identity)

  # - FEDERATED_ZIP(x, y) := GENERIC_ZIP(x, y)
  #
  # - GENERIC_AVERAGE(x: {T}@p, q: placement) :=
  #     GENERIC_WEIGHTED_AVERAGE(x, GENERIC_ONE, q)
  #
  # - GENERIC_WEIGHTED_AVERAGE(x: {T}@p, w: {U}@p, q: placement) :=
  #     GENERIC_MAP(GENERIC_DIVIDE, GENERIC_SUM(
  #       GENERIC_MAP(GENERIC_MULTIPLY, GENERIC_ZIP(x, w)), p))
  #
  #     NOTE: The above formula does not account for type casting issues that
  #     arise due to the interplay betwen the types of values and weights and
  #     how they relate to types of products and ratios, and either the formula
  #     or the type signatures may need to be tweaked.
  #
  # - GENERIC_SUM(x: {T}@p, q: placement) :=
  #     GENERIC_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS, q)
  #
  # - GENERIC_PARTIAL_SUM(x: {T}@p, q: placement) :=
  #     GENERIC_PARTIAL_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS, q)
  #
  # - GENERIC_AGGREGATE(
  #     x: {T}@p, zero: U, accu: <U,T>->U, merge: <U,U>=>U, report: U->R,
  #     q: placement) :=
  #     GENERIC_MAP(report, GENERIC_REDUCE(x, zero, accu, q))
  #
  # - GENERIC_REDUCE(x: {T}@p, zero: U, op: <U,T>->U, q: placement) :=
  #     GENERIC_MAP((a -> SEQUENCE_REDUCE(a, zero, op)), GENERIC_COLLECT(x, q))
  #
  # - GENERIC_PARTIAL_REDUCE(x: {T}@p, zero: U, op: <U,T>->U, q: placement) :=
  #     GENERIC_MAP(
  #       (a -> SEQUENCE_REDUCE(a, zero, op)), GENERIC_PARTIAL_COLLECT(x, q))
  #
  # - SEQUENCE_SUM(x: T*) :=
  #     SEQUENCE_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS)
  #
  # After performing the full set of reductions, we should only see instances
  # of the following intrinsics in the result, all of which are currently
  # considered non-reducible, and intrinsics such as GENERIC_PLUS should apply
  # only to non-federated, non-sequence types (with the appropriate calls to
  # GENERIC_MAP or SEQUENCE_MAP injected).
  #
  # - GENERIC_APPLY
  # - GENERIC_BROADCAST
  # - GENERIC_COLLECT
  # - GENERIC_DIVIDE
  # - GENERIC_MAP
  # - GENERIC_MULTIPLY
  # - GENERIC_ONE
  # - GENERIC_ONLY
  # - GENERIC_PARTIAL_COLLECT
  # - GENERIC_PLACE
  # - GENERIC_PLUS
  # - GENERIC_ZERO
  # - GENERIC_ZIP
  # - SEQUENCE_MAP
  # - SEQUENCE_REDUCE

  return collections.OrderedDict([
      (intrinsic_defs.FEDERATED_SUM.uri, federated_sum),
      (intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri, federated_weighted_mean),
      (intrinsic_defs.GENERIC_DIVIDE.uri, generic_divide),
      (intrinsic_defs.GENERIC_MULTIPLY.uri, generic_multiply),
  ])
