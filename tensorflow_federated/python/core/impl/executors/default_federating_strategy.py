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
"""A strategy for resolving federated types and intrinsics."""
import asyncio

import absl.logging as logging
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory


class DefaultFederatingStrategy(federating_executor.FederatingStrategy):
  """Strategy for parameter server style execution of intrinsics.

  This is the default intrinsic strategy for FederatingExecutor, in which all
  intrinsics are coordinated and executed between tff.SERVER and tff.CLIENTS.
  It is commonly known as the parameter server strategy, with tff.SERVER
  assuming the role of the parameter server.
  """

  def __init__(self, federating_executor):
    super().__init__(federating_executor)

  @classmethod
  def validate_executor_placements(cls, executor_placements):
    py_typecheck.check_type(executor_placements, dict)
    for k, v in executor_placements.items():
      if k is not None:
        py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      py_typecheck.check_type(v, (list, executor_base.Executor))
      if isinstance(v, list):
        for e in v:
          py_typecheck.check_type(e, executor_base.Executor)
      for pl in [None, placement_literals.SERVER]:
        if pl in executor_placements:
          ex = executor_placements[pl]
          if isinstance(ex, list):
            pl_cardinality = len(ex)
            if pl_cardinality != 1:
              raise ValueError(
                  'Unsupported cardinality for placement "{}": {}.'.format(
                      pl, pl_cardinality))
  
  def _get_child_executors(self, placement, index=None):
    child_executors = self.federating_executor._target_executors[placement]
    if index is not None:
      return child_executors[index]
    return child_executors

  @classmethod
  def _check_arg_is_anonymous_tuple(cls, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)

  @tracing.trace
  async def _eval(self, arg, placement, all_equal):
    py_typecheck.check_type(arg.type_signature, computation_types.FunctionType)
    py_typecheck.check_none(arg.type_signature.parameter)
    py_typecheck.check_type(arg.internal_representation, pb.Computation)
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    fn = arg.internal_representation
    fn_type = arg.type_signature
    children = self._get_child_executors(placement)

    async def call(child):
      return await child.create_call(await child.create_value(fn, fn_type))

    results = await asyncio.gather(*[call(child) for child in children])
    return federating_executor.FederatingExecutorValue(
        results,
        computation_types.FederatedType(
            fn_type.result, placement, all_equal=all_equal))

  @tracing.trace
  async def _map(self, arg, all_equal=None):
    self._check_arg_is_anonymous_tuple(arg)
    py_typecheck.check_len(arg.internal_representation, 2)
    fn_type = arg.type_signature[0]
    py_typecheck.check_type(fn_type, computation_types.FunctionType)
    val_type = arg.type_signature[1]
    py_typecheck.check_type(val_type, computation_types.FederatedType)
    if all_equal is None:
      all_equal = val_type.all_equal
    elif all_equal and not val_type.all_equal:
      raise ValueError(
          'Cannot map a non-all_equal argument into an all_equal result.')
    fn = arg.internal_representation[0]
    py_typecheck.check_type(fn, pb.Computation)
    val = arg.internal_representation[1]
    py_typecheck.check_type(val, list)
    for v in val:
      py_typecheck.check_type(v, executor_value_base.ExecutorValue)
    children = self._get_child_executors(val_type.placement)
    fns = await asyncio.gather(*[c.create_value(fn, fn_type) for c in children])
    results = await asyncio.gather(*[
        c.create_call(f, v) for c, (f, v) in zip(children, list(zip(fns, val)))
    ])
    return federating_executor.FederatingExecutorValue(
        results,
        computation_types.FederatedType(
            fn_type.result, val_type.placement, all_equal=all_equal))

  @tracing.trace
  async def _zip(self, arg, placement, all_equal):
    self._check_arg_is_anonymous_tuple(arg)
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    children = self._get_child_executors(placement)
    cardinality = len(children)
    elements = anonymous_tuple.to_elements(arg.internal_representation)
    for _, v in elements:
      py_typecheck.check_type(v, list)
      if len(v) != cardinality:
        raise RuntimeError('Expected {} items, found {}.'.format(
            cardinality, len(v)))
    new_vals = []
    for idx in range(cardinality):
      new_vals.append(
          anonymous_tuple.AnonymousTuple([(k, v[idx]) for k, v in elements]))
    new_vals = await asyncio.gather(
        *[c.create_tuple(x) for c, x in zip(children, new_vals)])
    return federating_executor.FederatingExecutorValue(
        new_vals,
        computation_types.FederatedType(
            computation_types.NamedTupleType((
                (k, v.member) if k else v.member
                for k, v in anonymous_tuple.iter_elements(arg.type_signature))),
            placement,
            all_equal=all_equal))

  async def federated_value_at_clients(self, arg):
    return await executor_utils.compute_intrinsic_federated_value(
        self.federating_executor, arg, placement_literals.CLIENTS)

  async def federated_value_at_server(self, arg):
    return await executor_utils.compute_intrinsic_federated_value(
        self.federating_executor, arg, placement_literals.SERVER)

  async def federated_eval_at_server(self, arg):
    return await self._eval(arg, placement_literals.SERVER, True)

  async def federated_eval_at_clients(self, arg):
    return await self._eval(arg, placement_literals.CLIENTS, False)

  async def federated_apply(self, arg):
    return await self._map(arg)

  async def federated_map(self, arg):
    return await self._map(arg, all_equal=False)

  async def federated_map_all_equal(self, arg):
    return await self._map(arg, all_equal=True)

  async def federated_broadcast(self, arg):
    py_typecheck.check_type(arg.internal_representation, list)
    if len(arg.internal_representation) != 1:
      raise ValueError(
          'Federated broadcast expects a value with a single representation, '
          'found {}.'.format(len(arg.internal_representation)))
    return await executor_utils.compute_intrinsic_federated_broadcast(
        self.federating_executor, arg)

  async def federated_zip_at_server(self, arg):
    return await self._zip(arg, placement_literals.SERVER, all_equal=True)

  async def federated_zip_at_clients(self, arg):
    return await self._zip(arg, placement_literals.CLIENTS, all_equal=False)

  async def federated_reduce(self, arg):
    self._check_arg_is_anonymous_tuple(arg)
    if len(arg.internal_representation) != 3:
      raise ValueError(
          'Expected 3 elements in the `federated_reduce()` argument tuple, '
          'found {}.'.format(len(arg.internal_representation)))

    val_type = arg.type_signature[0]
    py_typecheck.check_type(val_type, computation_types.FederatedType)
    item_type = val_type.member
    zero_type = arg.type_signature[1]
    op_type = arg.type_signature[2]
    type_analysis.check_equivalent_types(
        op_type, type_factory.reduction_op(zero_type, item_type))

    val = arg.internal_representation[0]
    py_typecheck.check_type(val, list)
    child = self._get_child_executors(placement_literals.SERVER, index=0)

    async def _move(v):
      return await child.create_value(await v.compute(), item_type)

    items = await asyncio.gather(*[_move(v) for v in val])

    zero = await child.create_value(
        await (await
               self.federating_executor.create_selection(arg,
                                                         index=1)).compute(),
        zero_type)
    op = await child.create_value(arg.internal_representation[2], op_type)

    result = zero
    for item in items:
      result = await child.create_call(
          op, await child.create_tuple(
              anonymous_tuple.AnonymousTuple([(None, result), (None, item)])))
    return federating_executor.FederatingExecutorValue([result],
                                   computation_types.FederatedType(
                                       result.type_signature,
                                       placement_literals.SERVER,
                                       all_equal=True))

  async def federated_aggregate(self, arg):
    val_type, zero_type, accumulate_type, _, report_type = (
        executor_utils.parse_federated_aggregate_argument_types(
            arg.type_signature))
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    py_typecheck.check_len(arg.internal_representation, 5)

    # Note: This is a simple initial implementation that simply forwards this
    # to `federated_reduce()`. The more complete implementation would be able
    # to take advantage of the parallelism afforded by `merge` to reduce the
    # cost from liner (with respect to the number of clients) to sub-linear.

    # TODO(b/134543154): Expand this implementation to take advantage of the
    # parallelism afforded by `merge`.
    fed_ex = self.federating_executor

    val = arg.internal_representation[0]
    zero = arg.internal_representation[1]
    accumulate = arg.internal_representation[2]
    pre_report = await fed_ex._compute_intrinsic_federated_reduce(
        federating_executor.FederatingExecutorValue(
            anonymous_tuple.AnonymousTuple([(None, val), (None, zero),
                                            (None, accumulate)]),
            computation_types.NamedTupleType(
                (val_type, zero_type, accumulate_type))))

    py_typecheck.check_type(pre_report.type_signature,
                            computation_types.FederatedType)
    type_analysis.check_equivalent_types(pre_report.type_signature.member,
                                         report_type.parameter)

    report = arg.internal_representation[4]
    return await fed_ex._compute_intrinsic_federated_apply(
        federating_executor.FederatingExecutorValue(
            anonymous_tuple.AnonymousTuple([
                (None, report), (None, pre_report.internal_representation)
            ]),
            computation_types.NamedTupleType(
                (report_type, pre_report.type_signature))))

  async def federated_sum(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    zero, plus = await asyncio.gather(
        executor_utils.embed_tf_scalar_constant(self.federating_executor,
                                                arg.type_signature.member,
                                                0),
        executor_utils.embed_tf_binary_operator(self.federating_executor,
                                                arg.type_signature.member,
                                                tf.add))
    return await self.federating_executor._compute_intrinsic_federated_reduce(
        federating_executor.FederatingExecutorValue(
            anonymous_tuple.AnonymousTuple([
                (None, arg.internal_representation),
                (None, zero.internal_representation),
                (None, plus.internal_representation)
            ]),
            computation_types.NamedTupleType(
                (arg.type_signature, zero.type_signature, plus.type_signature)))
    )

  async def federated_mean(self, arg):
    arg_sum = await self.federating_executor._compute_intrinsic_federated_sum(
        arg)
    member_type = arg_sum.type_signature.member
    count = float(len(arg.internal_representation))
    if count < 1.0:
      raise RuntimeError('Cannot compute a federated mean over an empty group.')
    child = self._get_child_executors(placement_literals.SERVER, index=0)
    factor, multiply = await asyncio.gather(
        executor_utils.embed_tf_scalar_constant(child, member_type,
                                                float(1.0 / count)),
        executor_utils.embed_tf_binary_operator(child, member_type,
                                                tf.multiply))
    multiply_arg = await child.create_tuple(
        anonymous_tuple.AnonymousTuple([(None,
                                         arg_sum.internal_representation[0]),
                                        (None, factor)]))
    result = await child.create_call(multiply, multiply_arg)
    return federating_executor.FederatingExecutorValue(
        [result], arg_sum.type_signature)

  async def federated_weighted_mean(self, arg):
    return await executor_utils.compute_intrinsic_federated_weighted_mean(
        self.federating_executor, arg)

  async def federated_collect(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    type_analysis.check_federated_type(
        arg.type_signature, placement=placement_literals.CLIENTS)
    val = arg.internal_representation
    py_typecheck.check_type(val, list)
    member_type = arg.type_signature.member
    child = self._get_child_executors(placement_literals.SERVER, index=0)
    collected_items = await child.create_value(
        await asyncio.gather(*[v.compute() for v in val]),
        computation_types.SequenceType(member_type))
    return federating_executor.FederatingExecutorValue(
        [collected_items],
        computation_types.FederatedType(
            computation_types.SequenceType(member_type),
            placement_literals.SERVER,
            all_equal=True))

  async def federated_secure_sum(self, arg):
    raise NotImplementedError('The secure sum intrinsic is not implemented.')
