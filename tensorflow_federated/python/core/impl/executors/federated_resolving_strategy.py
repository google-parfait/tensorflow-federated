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
"""A strategy for resolving federated types and intrinsics.

                +------------+
                | Federating |   +----------+
            +-->+ Executor   +-->+ unplaced |
            |   +--+---------+   | executor |
            |      |             +----------+
  +---------+--+   |
  | Federated  +<--+
  | Resolving  |
  | Strategy   |
  +--+---------+
     |
     |   +-----------+
     +-->+ target    |
         | executors |
         +-----------+

  A `FederatedResolvingStrategy`:

  * Implements the logic for resolving federated types and intrinsics, while
    delegating unplaced computations to the target executor(s) associated with
    the placement of the federated type or intrinsic.

    Note: The default strategy never delegates federated types and intrinsics.

  * Delegates handling unplaced types, computations, and processing back to the
    `FederatingExecutor`. DO_NOT_SBUMIT: Provide an example of this.
"""

import asyncio
from typing import Any, Dict

import absl.logging as logging
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory


class FederatedResolvingStrategyValue(executor_value_base.ExecutorValue):
  """A value embedded in a `FederatedExecutor`."""

  def __init__(self, value, type_signature):
    """Creates a `FederatedResolvingStrategyValue` embedding the given `value`.

    Args:
      value: An object to embed in the executor, one of the supported types
        defined by the `FederatingExecutor`.
      type_signature: A `tff.Type` or something convertible to instance of
        `tff.Type` via `tff.to_type`, the type of `value`.
    """
    self._value = value
    self._type_signature = computation_types.to_type(type_signature)

  @property
  def internal_representation(self):
    return self._value

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature

  @tracing.trace
  async def compute(self):
    """Returns the result of computing the embedded value.

    Raises:
      TypeError: If the embedded value is composed of values that are not
        embedded in the executor.
      RuntimeError: If the embedded value is not a kind supported by the
        `FederatingExecutor`.
    """
    if isinstance(self._value, executor_value_base.ExecutorValue):
      return await self._value.compute()
    elif isinstance(self._value, structure.Struct):
      results = await asyncio.gather(*[
          FederatedResolvingStrategyValue(v, t).compute()
          for v, t in zip(self._value, self._type_signature)
      ])
      element_types = structure.iter_elements(self._type_signature)
      return structure.Struct(
          (n, v) for (n, _), v in zip(element_types, results))
    elif isinstance(self._value, list):
      py_typecheck.check_type(self._type_signature,
                              computation_types.FederatedType)
      for value in self._value:
        py_typecheck.check_type(value, executor_value_base.ExecutorValue)
      if self._type_signature.all_equal:
        return await self._value[0].compute()
      else:
        return await asyncio.gather(*[v.compute() for v in self._value])
    else:
      raise RuntimeError(
          'Computing values of type {} represented as {} is not supported in '
          'this executor.'.format(self._type_signature,
                                  py_typecheck.type_string(type(self._value))))


class FederatedResolvingStrategy(federating_executor.FederatingStrategy):
  """A strategy for resolving federated types and intrinsics.

  This strategy implements the `federating_executor.FederatingStrategy`
  interface and provides logic for resolving federated types and federated
  intrinsics on an underlying collection of target executor(s) associated with
  individual placements.

  This strategy only supports the placements:

  * `tff.SERVER`
  * `tff.CLIENTS`

  Note that this strategy does not have a built-in concept of intermediate
  aggregation, partitioning placements, clustering clients, etc.
  """

  @classmethod
  def factory(cls, target_executors: Dict[str, executor_base.Executor]):
    return lambda executor: cls(executor, target_executors)

  def __init__(self, executor: federating_executor.FederatingExecutor,
               target_executors: Dict[str, executor_base.Executor]):
    """Creates a `FederatedResolvingStrategy`.

    Args:
      executor: A `federating_executor.FederatingExecutor` to use to handle
        unplaced types, computations, and processing.
      target_executors: A `dict` mapping placements to a collection of target
        executors associated with individual placements. The keys in this
        dictionary are placement literals. The values can be either single
        executors (if there only is a single participant associated with that
        placement, e.g. `tff.SERVER`) or lists of executors.

    Raises:
      TypeError: If `target_executors` is not a `dict`, where each key is a
        `placement_literals.PlacementLiteral` and each value is either an
        `executor_base.Executor` or a list of `executor_base.Executor`s.
      ValueError: If `target_executors` contains a
        `placement_literals.PlacementLiteral` key that is not a kind supported
        by the `FederatedResolvingStrategy`.
    """
    super().__init__(executor)
    py_typecheck.check_type(target_executors, dict)
    self._target_executors = {}
    for k, v in target_executors.items():
      if k is not None:
        py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      py_typecheck.check_type(v, (list, executor_base.Executor))
      if isinstance(v, executor_base.Executor):
        self._target_executors[k] = [v]
      else:
        for e in v:
          py_typecheck.check_type(e, executor_base.Executor)
        self._target_executors[k] = v.copy()
    for pl in [None, placement_literals.SERVER]:
      if pl in self._target_executors:
        pl_cardinality = len(self._target_executors[pl])
        if pl_cardinality != 1:
          raise ValueError(
              'Unsupported cardinality for placement "{}": {}.'.format(
                  pl, pl_cardinality))

  def close(self):
    for p, v in self._target_executors.items():
      for e in v:
        logging.debug('Closing child executor for placement: %s', p)
        e.close()

  def _check_arg_is_structure(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    py_typecheck.check_type(arg.internal_representation, structure.Struct)

  def _check_strategy_compatible_with_placement(self, placement):
    """Tests that this executor is compatible with the given `placement`.

    This function checks that `value` is compatible with the configuration of
    this executor for the given `placement`.

    Args:
      placement: A placement literal, the placement to test.

    Raises:
      ValueError: If `value` is not compatible.
    """
    children = self._target_executors.get(placement)
    if not children:
      # TODO(b/154328996): This executor does not have the context to know that
      # the suggested solution is reasonable; the suggestion is here because it
      # is probably the correct soltuion. We should establish a pattern for
      # raising errors to a level in the stack where the appropriate context
      # exists.
      raise ValueError(
          'Expected at least one participant for the \'{}\' placement, found '
          'none. It is possible that the inferred number of clients is 0, you '
          'can explicitly pass `num_clients` when constructing the execution '
          'stack'.format(placement))

  def _check_value_compatible_with_placement(self, value, placement, all_equal):
    """Tests that `value` is compatible with the given `placement`.

    Args:
      value: A value to test.
      placement: A placement literal indicating the placement of `value`.
      all_equal: A `bool` indicating whether all elements of `value` are equal.

    Raises:
      ValueError: If `value` is not compatible.
    """
    if not all_equal:
      py_typecheck.check_type(value, (list, tuple, set, frozenset))
      children = self._target_executors.get(placement)
      if len(value) != len(children):
        raise ValueError(
            'Expected a value that contains one element for each participant '
            'for the given placement, found a value with {elements} elements '
            'and this executor is configured to have {participants} '
            'participants for the \'{placement}\' placement.'.format(
                elements=len(value),
                placement=placement,
                participants=len(children)))

  def ingest_value(
      self, value: Any, type_signature: computation_types.Type
  ) -> executor_value_base.ExecutorValue:
    if type_signature is not None:
      if type_signature.is_federated():
        self._check_strategy_compatible_with_placement(type_signature.placement)
      elif type_signature.is_function() and type_signature.result.is_federated(
      ):
        self._check_strategy_compatible_with_placement(
            type_signature.result.placement)
    return FederatedResolvingStrategyValue(value, type_signature)

  async def compute_federated_value(
      self, value: Any, type_signature: computation_types.Type
  ) -> FederatedResolvingStrategyValue:
    self._check_strategy_compatible_with_placement(type_signature.placement)
    children = self._target_executors[type_signature.placement]
    self._check_value_compatible_with_placement(value, type_signature.placement,
                                                type_signature.all_equal)
    if type_signature.all_equal:
      value = [value for _ in children]
    result = await asyncio.gather(*[
        c.create_value(v, type_signature.member)
        for v, c in zip(value, children)
    ])
    return FederatedResolvingStrategyValue(result, type_signature)

  @tracing.trace
  async def _eval(self, arg, placement, all_equal):
    py_typecheck.check_type(arg.type_signature, computation_types.FunctionType)
    py_typecheck.check_none(arg.type_signature.parameter)
    py_typecheck.check_type(arg.internal_representation, pb.Computation)
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    fn = arg.internal_representation
    fn_type = arg.type_signature
    self._check_strategy_compatible_with_placement(placement)
    children = self._target_executors[placement]

    async def call(child):
      return await child.create_call(await child.create_value(fn, fn_type))

    results = await asyncio.gather(*[call(child) for child in children])
    return FederatedResolvingStrategyValue(
        results,
        computation_types.FederatedType(
            fn_type.result, placement, all_equal=all_equal))

  @tracing.trace
  async def _map(self, arg, all_equal=None):
    self._check_arg_is_structure(arg)
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
    self._check_strategy_compatible_with_placement(val_type.placement)
    children = self._target_executors[val_type.placement]
    fns = await asyncio.gather(*[c.create_value(fn, fn_type) for c in children])
    results = await asyncio.gather(*[
        c.create_call(f, v) for c, (f, v) in zip(children, list(zip(fns, val)))
    ])
    return FederatedResolvingStrategyValue(
        results,
        computation_types.FederatedType(
            fn_type.result, val_type.placement, all_equal=all_equal))

  @tracing.trace
  async def _zip(self, arg, placement, all_equal):
    self._check_arg_is_structure(arg)
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    self._check_strategy_compatible_with_placement(placement)
    children = self._target_executors[placement]
    cardinality = len(children)
    elements = structure.to_elements(arg.internal_representation)
    for _, v in elements:
      py_typecheck.check_type(v, list)
      if len(v) != cardinality:
        raise RuntimeError('Expected {} items, found {}.'.format(
            cardinality, len(v)))
    new_vals = []
    for idx in range(cardinality):
      new_vals.append(structure.Struct([(k, v[idx]) for k, v in elements]))
    new_vals = await asyncio.gather(
        *[c.create_struct(x) for c, x in zip(children, new_vals)])
    return FederatedResolvingStrategyValue(
        new_vals,
        computation_types.FederatedType(
            computation_types.StructType(
                ((k, v.member) if k else v.member
                 for k, v in structure.iter_elements(arg.type_signature))),
            placement,
            all_equal=all_equal))

  @tracing.trace
  async def compute_federated_aggregate(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    val_type, zero_type, accumulate_type, _, report_type = (
        executor_utils.parse_federated_aggregate_argument_types(
            arg.type_signature))
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    py_typecheck.check_len(arg.internal_representation, 5)

    # Note: This is a simple initial implementation that simply forwards this
    # to `federated_reduce()`. The more complete implementation would be able
    # to take advantage of the parallelism afforded by `merge` to reduce the
    # cost from liner (with respect to the number of clients) to sub-linear.

    # TODO(b/134543154): Expand this implementation to take advantage of the
    # parallelism afforded by `merge`.

    val = arg.internal_representation[0]
    zero = arg.internal_representation[1]
    accumulate = arg.internal_representation[2]
    pre_report = await self.compute_federated_reduce(
        FederatedResolvingStrategyValue(
            structure.Struct([(None, val), (None, zero), (None, accumulate)]),
            computation_types.StructType(
                (val_type, zero_type, accumulate_type))))

    py_typecheck.check_type(pre_report.type_signature,
                            computation_types.FederatedType)
    pre_report.type_signature.member.check_equivalent_to(report_type.parameter)

    report = arg.internal_representation[4]
    return await self.compute_federated_apply(
        FederatedResolvingStrategyValue(
            structure.Struct([(None, report),
                              (None, pre_report.internal_representation)]),
            computation_types.StructType(
                (report_type, pre_report.type_signature))))

  @tracing.trace
  async def compute_federated_apply(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._map(arg)

  @tracing.trace
  async def compute_federated_broadcast(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    py_typecheck.check_type(arg.internal_representation, list)
    if len(arg.internal_representation) != 1:
      raise ValueError(
          'Federated broadcast expects a value with a single representation, '
          'found {}.'.format(len(arg.internal_representation)))
    return await executor_utils.compute_intrinsic_federated_broadcast(
        self._executor, arg)

  @tracing.trace
  async def compute_federated_collect(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    type_analysis.check_federated_type(
        arg.type_signature, placement=placement_literals.CLIENTS)
    val = arg.internal_representation
    py_typecheck.check_type(val, list)
    member_type = arg.type_signature.member
    child = self._target_executors[placement_literals.SERVER][0]
    collected_items = await child.create_value(
        await asyncio.gather(*[v.compute() for v in val]),
        computation_types.SequenceType(member_type))
    return FederatedResolvingStrategyValue(
        [collected_items],
        computation_types.FederatedType(
            computation_types.SequenceType(member_type),
            placement_literals.SERVER,
            all_equal=True))

  @tracing.trace
  async def compute_federated_eval_at_clients(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._eval(arg, placement_literals.CLIENTS, False)

  @tracing.trace
  async def compute_federated_eval_at_server(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._eval(arg, placement_literals.SERVER, True)

  @tracing.trace
  async def compute_federated_map(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._map(arg, all_equal=False)

  @tracing.trace
  async def compute_federated_map_all_equal(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._map(arg, all_equal=True)

  @tracing.trace
  async def compute_federated_mean(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    arg_sum = await self.compute_federated_sum(arg)
    member_type = arg_sum.type_signature.member
    count = float(len(arg.internal_representation))
    if count < 1.0:
      raise RuntimeError('Cannot compute a federated mean over an empty group.')
    child = self._target_executors[placement_literals.SERVER][0]
    factor, multiply = await asyncio.gather(
        executor_utils.embed_tf_scalar_constant(child, member_type,
                                                float(1.0 / count)),
        executor_utils.embed_tf_binary_operator(child, member_type,
                                                tf.multiply))
    multiply_arg = await child.create_struct(
        structure.Struct([(None, arg_sum.internal_representation[0]),
                          (None, factor)]))
    result = await child.create_call(multiply, multiply_arg)
    return FederatedResolvingStrategyValue([result], arg_sum.type_signature)

  @tracing.trace
  async def compute_federated_reduce(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    self._check_arg_is_structure(arg)
    if len(arg.internal_representation) != 3:
      raise ValueError(
          'Expected 3 elements in the `federated_reduce()` argument tuple, '
          'found {}.'.format(len(arg.internal_representation)))

    val_type = arg.type_signature[0]
    py_typecheck.check_type(val_type, computation_types.FederatedType)
    item_type = val_type.member
    zero_type = arg.type_signature[1]
    op_type = arg.type_signature[2]
    op_type.check_equivalent_to(type_factory.reduction_op(zero_type, item_type))

    val = arg.internal_representation[0]
    py_typecheck.check_type(val, list)
    child = self._target_executors[placement_literals.SERVER][0]

    async def _move(v):
      return await child.create_value(await v.compute(), item_type)

    items = await asyncio.gather(*[_move(v) for v in val])

    zero = await child.create_value(
        await (await self._executor.create_selection(arg, index=1)).compute(),
        zero_type)
    op = await child.create_value(arg.internal_representation[2], op_type)

    result = zero
    for item in items:
      result = await child.create_call(
          op, await
          child.create_struct(structure.Struct([(None, result), (None, item)])))
    return FederatedResolvingStrategyValue([result],
                                           computation_types.FederatedType(
                                               result.type_signature,
                                               placement_literals.SERVER,
                                               all_equal=True))

  @tracing.trace
  async def compute_federated_secure_sum(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    raise NotImplementedError('The secure sum intrinsic is not implemented.')

  @tracing.trace
  async def compute_federated_sum(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    zero, plus = await asyncio.gather(
        executor_utils.embed_tf_scalar_constant(self._executor,
                                                arg.type_signature.member, 0),
        executor_utils.embed_tf_binary_operator(self._executor,
                                                arg.type_signature.member,
                                                tf.add))
    return await self.compute_federated_reduce(
        FederatedResolvingStrategyValue(
            structure.Struct([(None, arg.internal_representation),
                              (None, zero.internal_representation),
                              (None, plus.internal_representation)]),
            computation_types.StructType(
                (arg.type_signature, zero.type_signature, plus.type_signature)))
    )

  @tracing.trace
  async def compute_federated_value_at_clients(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_value(
        self._executor, arg, placement_literals.CLIENTS)

  @tracing.trace
  async def compute_federated_value_at_server(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_value(
        self._executor, arg, placement_literals.SERVER)

  @tracing.trace
  async def compute_federated_weighted_mean(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_weighted_mean(
        self._executor, arg)

  @tracing.trace
  async def compute_federated_zip_at_clients(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._zip(arg, placement_literals.CLIENTS, all_equal=False)

  @tracing.trace
  async def compute_federated_zip_at_server(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._zip(arg, placement_literals.SERVER, all_equal=True)
