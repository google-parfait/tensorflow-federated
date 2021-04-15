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
from typing import Any, Dict, List

import absl.logging as logging
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis


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
  def factory(cls,
              target_executors: Dict[str, executor_base.Executor],
              local_computation_factory: local_computation_factory_base
              .LocalComputationFactory = tensorflow_computation_factory
              .TensorFlowComputationFactory()):
    # pylint:disable=g-long-lambda
    return lambda executor: cls(
        executor,
        target_executors,
        local_computation_factory=local_computation_factory)
    # pylint:enable=g-long-lambda

  def __init__(self,
               executor: federating_executor.FederatingExecutor,
               target_executors: Dict[str, executor_base.Executor],
               local_computation_factory: local_computation_factory_base
               .LocalComputationFactory = tensorflow_computation_factory
               .TensorFlowComputationFactory()):
    """Creates a `FederatedResolvingStrategy`.

    Args:
      executor: A `federating_executor.FederatingExecutor` to use to handle
        unplaced types, computations, and processing.
      target_executors: A `dict` mapping placements to a collection of target
        executors associated with individual placements. The keys in this
        dictionary are placement literals. The values can be either single
        executors (if there only is a single participant associated with that
        placement, e.g. `tff.SERVER`) or lists of executors.
      local_computation_factory: An instance of `LocalComputationFactory` to use
        to construct local computations used as parameters in certain federated
        operators (such as `tff.federated_sum`, etc.). Defaults to a TensorFlow
        computation factory that generates TensorFlow code.

    Raises:
      TypeError: If `target_executors` is not a `dict`, where each key is a
        `placements.PlacementLiteral` and each value is either an
        `executor_base.Executor` or a list of `executor_base.Executor`s.
      ValueError: If `target_executors` contains a
        `placements.PlacementLiteral` key that is not a kind supported
        by the `FederatedResolvingStrategy`.
    """
    super().__init__(executor)
    py_typecheck.check_type(target_executors, dict)
    py_typecheck.check_type(
        local_computation_factory,
        local_computation_factory_base.LocalComputationFactory)
    self._target_executors = {}
    self._local_computation_factory = local_computation_factory
    for k, v in target_executors.items():
      if k is not None:
        py_typecheck.check_type(k, placements.PlacementLiteral)
      py_typecheck.check_type(v, (list, executor_base.Executor))
      if isinstance(v, executor_base.Executor):
        self._target_executors[k] = [v]
      else:
        for e in v:
          py_typecheck.check_type(e, executor_base.Executor)
        self._target_executors[k] = v.copy()
    for pl in [None, placements.SERVER]:
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
    py_typecheck.check_type(placement, placements.PlacementLiteral)
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

    async def _map_child(fn, fn_type, value, child):
      fn_at_child = await child.create_value(fn, fn_type)
      return await child.create_call(fn_at_child, value)

    results = await asyncio.gather(*[
        _map_child(fn, fn_type, value, child)
        for (value, child) in zip(val, children)
    ])
    return FederatedResolvingStrategyValue(
        results,
        computation_types.FederatedType(
            fn_type.result, val_type.placement, all_equal=all_equal))

  @tracing.trace
  async def _zip(self, arg, placement, all_equal):
    self._check_arg_is_structure(arg)
    py_typecheck.check_type(placement, placements.PlacementLiteral)
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
    val_type, zero_type, accumulate_type, merge_type, report_type = (
        executor_utils.parse_federated_aggregate_argument_types(
            arg.type_signature))
    del val_type, merge_type
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    py_typecheck.check_len(arg.internal_representation, 5)
    val, zero, accumulate, merge, report = arg.internal_representation

    # Discard `merge`. Since all aggregation happens on a single executor,
    # there's no need for this additional layer.
    del merge

    # Re-wrap `zero` in a `FederatingResolvingStrategyValue` to ensure that it
    # is an `ExecutorValue` rather than a `Struct` (since the internal
    # representation can include embedded values, lists of embedded values
    # (in the case of federated values), or `Struct`s.
    zero = FederatedResolvingStrategyValue(zero, zero_type)
    pre_report = await self.reduce(val, zero, accumulate, accumulate_type)

    py_typecheck.check_type(pre_report.type_signature,
                            computation_types.FederatedType)
    pre_report.type_signature.member.check_equivalent_to(report_type.parameter)

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
        arg.type_signature, placement=placements.CLIENTS)
    val = arg.internal_representation
    py_typecheck.check_type(val, list)
    member_type = arg.type_signature.member
    child = self._target_executors[placements.SERVER][0]
    collected_items = await child.create_value(
        await asyncio.gather(*[v.compute() for v in val]),
        computation_types.SequenceType(member_type))
    return FederatedResolvingStrategyValue(
        [collected_items],
        computation_types.FederatedType(
            computation_types.SequenceType(member_type),
            placements.SERVER,
            all_equal=True))

  @tracing.trace
  async def compute_federated_eval_at_clients(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._eval(arg, placements.CLIENTS, False)

  @tracing.trace
  async def compute_federated_eval_at_server(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._eval(arg, placements.SERVER, True)

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
    child = self._target_executors[placements.SERVER][0]
    factor, multiply = await asyncio.gather(
        executor_utils.embed_constant(
            child,
            member_type,
            float(1.0 / count),
            local_computation_factory=self._local_computation_factory),
        executor_utils.embed_multiply_operator(
            child,
            member_type,
            local_computation_factory=self._local_computation_factory))
    multiply_arg = await child.create_struct(
        structure.Struct([(None, arg_sum.internal_representation[0]),
                          (None, factor)]))
    result = await child.create_call(multiply, multiply_arg)
    return FederatedResolvingStrategyValue([result], arg_sum.type_signature)

  @tracing.trace
  async def reduce(
      self,
      val: List[executor_value_base.ExecutorValue],
      zero: executor_value_base.ExecutorValue,
      op: pb.Computation,
      op_type: computation_types.FunctionType,
  ) -> FederatedResolvingStrategyValue:
    server = self._target_executors[placements.SERVER][0]

    async def _move(v):
      return await server.create_value(await v.compute(), v.type_signature)

    item_futures = asyncio.as_completed([_move(v) for v in val])
    zero_at_server = await server.create_value(await zero.compute(),
                                               zero.type_signature)
    op_at_server = await server.create_value(op, op_type)

    result = zero_at_server
    for item_future in item_futures:
      item = await item_future
      result = await server.create_call(
          op_at_server, await server.create_struct(
              structure.Struct([(None, result), (None, item)])))
    return FederatedResolvingStrategyValue([result],
                                           computation_types.FederatedType(
                                               result.type_signature,
                                               placements.SERVER,
                                               all_equal=True))

  @tracing.trace
  async def compute_federated_secure_sum(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    raise NotImplementedError(
        '`tff.federated_secure_sum()` is not implemented in this executor. '
        'For a fake implementation of `federated_secure_sum` suitable for '
        'testing, consider using the test executor context by adding the '
        'following during initialization: '
        '`tff.backends.test.set_test_execution_context()`')

  @tracing.trace
  async def compute_federated_secure_select(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    raise NotImplementedError(
        '`tff.federated_secure_select()` is not implemented in this executor. '
        'For a fake implementation of `federated_secure_select` suitable for '
        'testing, consider using the test executor context by adding the '
        'following during initialization: '
        '`tff.backends.test.set_test_execution_context()`')

  @tracing.trace
  async def compute_federated_select(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    client_keys_type, max_key_type, server_val_type, select_fn_type = (
        arg.type_signature)
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    client_keys, max_key, server_val, select_fn = arg.internal_representation
    # We slice up the value as-needed, so `max_key` is not used.
    del max_key, max_key_type
    del server_val_type  # unused
    py_typecheck.check_type(client_keys, list)
    py_typecheck.check_type(server_val, list)
    server_val_at_server = server_val[0]
    py_typecheck.check_type(server_val_at_server,
                            executor_value_base.ExecutorValue)
    py_typecheck.check_type(select_fn, pb.Computation)
    server = self._target_executors[placements.SERVER][0]
    clients = self._target_executors[placements.CLIENTS]
    single_key_type = computation_types.TensorType(tf.int32)
    client_keys_type.member.check_tensor()
    if (client_keys_type.member.dtype != tf.int32 or
        client_keys_type.member.shape.rank != 1):
      raise TypeError(f'Unexpected `client_keys_type`: {client_keys_type}')
    num_keys_per_client: int = client_keys_type.member.shape.dims[0].value
    unplaced_result_type = computation_types.SequenceType(select_fn_type.result)
    select_fn_at_server = await server.create_value(select_fn, select_fn_type)
    index_fn_at_server = await executor_utils.embed_indexing_operator(
        server, client_keys_type.member, single_key_type)

    async def select_single_key(keys_at_server, key_index):
      # Grab the `key_index`th key from the keys tensor.
      index_arg = await server.create_struct(
          structure.Struct([
              (None, keys_at_server),
              (None, await server.create_value(key_index, single_key_type)),
          ]))
      key_at_server = await server.create_call(index_fn_at_server, index_arg)
      select_fn_arg = await server.create_struct(
          structure.Struct([
              (None, server_val_at_server),
              (None, key_at_server),
          ]))
      selected = await server.create_call(select_fn_at_server, select_fn_arg)
      return await selected.compute()

    async def select_single_client(client, keys_at_client):
      keys_at_server = await server.create_value(await keys_at_client.compute(),
                                                 client_keys_type.member)
      unplaced_values = await asyncio.gather(*[
          select_single_key(keys_at_server, i)
          for i in range(num_keys_per_client)
      ])
      return await client.create_value(unplaced_values, unplaced_result_type)

    return FederatedResolvingStrategyValue(
        list(await asyncio.gather(*[
            select_single_client(client, keys_at_client)
            for client, keys_at_client in zip(clients, client_keys)
        ])), computation_types.at_clients(unplaced_result_type))

  @tracing.trace
  async def compute_federated_sum(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    zero, plus = await asyncio.gather(
        executor_utils.embed_constant(
            self._executor,
            arg.type_signature.member,
            0,
            local_computation_factory=self._local_computation_factory),
        executor_utils.embed_plus_operator(
            self._executor,
            arg.type_signature.member,
            local_computation_factory=self._local_computation_factory))
    return await self.reduce(arg.internal_representation, zero,
                             plus.internal_representation, plus.type_signature)

  @tracing.trace
  async def compute_federated_value_at_clients(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_value(
        self._executor, arg, placements.CLIENTS)

  @tracing.trace
  async def compute_federated_value_at_server(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_value(
        self._executor, arg, placements.SERVER)

  @tracing.trace
  async def compute_federated_weighted_mean(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_weighted_mean(
        self._executor,
        arg,
        local_computation_factory=self._local_computation_factory)

  @tracing.trace
  async def compute_federated_zip_at_clients(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._zip(arg, placements.CLIENTS, all_equal=False)

  @tracing.trace
  async def compute_federated_zip_at_server(
      self,
      arg: FederatedResolvingStrategyValue) -> FederatedResolvingStrategyValue:
    return await self._zip(arg, placements.SERVER, all_equal=True)
