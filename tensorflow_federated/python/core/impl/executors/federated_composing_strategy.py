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
"""A strategy for composing federated types and intrinsics in disjoint scopes.

                +------------+
                | Federating |   +----------+
            +-->+ Executor   +-->+ unplaced |
            |   +--+---------+   | executor |
            |      |             +----------+
  +---------+--+   |
  | Federated  +<--+
  | Composing  |
  | Strategy   |
  +--+--+------+
     |  |
     |  |   +----------+
     |  +-->+ server   |
     |      | executor |
     |      +----------+
     |
     |   +-----------+
     +-->+ target    |
         | executors |
         +-----------+

  This strategy:

  * Delegates handling unplaced types, computations, and processing back to the
    `FederatingExecutor`.
  * Delegates handling server-side processing to the `server_executor`.
  * Delegates handling federating types and intrinsics to the target
    executor(s).
"""

import asyncio
from typing import Any, List

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_transformations


class FederatedComposingStrategyValue(executor_value_base.ExecutorValue):
  """A value embedded in a `FederatedExecutor`."""

  def __init__(self, value, type_signature):
    """Creates a `FederatedComposingStrategyValue` embedding the given `value`.

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
          FederatedComposingStrategyValue(v, t).compute()
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
        result = []
        values = await asyncio.gather(*[v.compute() for v in self._value])
        for value in values:
          py_typecheck.check_type(value, list)
          result.extend(value)
        return result
    else:
      raise RuntimeError(
          'Computing values of type {} represented as {} is not supported in '
          'this executor.'.format(self._type_signature,
                                  py_typecheck.type_string(type(self._value))))


class FederatedComposingStrategy(federating_executor.FederatingStrategy):
  """A strategy for composing federated types and intrinsics in disjoint scopes.

  This strategy implements the `FederatingStrategy` interface and provides
  logic for handling federated types and federated intrinsics on an underlying
  collection of target executors. This strategy can be used in hierarchical
  aggregation structures with federating executors handling disjoint scopes at
  the leafs.

  This strategy only supports the placements:

  * `tff.SERVER`
  * `tff.CLIENTS`
  """

  @classmethod
  def factory(cls,
              server_executor: executor_base.Executor,
              target_executors: List[executor_base.Executor],
              local_computation_factory: local_computation_factory_base
              .LocalComputationFactory = tensorflow_computation_factory
              .TensorFlowComputationFactory()):
    # pylint:disable=g-long-lambda
    return lambda executor: cls(
        executor,
        server_executor,
        target_executors,
        local_computation_factory=local_computation_factory)
    # pylint:enable=g-long-lambda

  def __init__(self,
               executor: federating_executor.FederatingExecutor,
               server_executor: executor_base.Executor,
               target_executors: List[executor_base.Executor],
               local_computation_factory: local_computation_factory_base
               .LocalComputationFactory = tensorflow_computation_factory
               .TensorFlowComputationFactory()):
    """Creates a `FederatedComposingStrategy`.

    Args:
      executor: A `federating_executor.FederatingExecutor` to use to handle
        unplaced types, computations, and processing.
      server_executor: The parent executor to use for all processing at the
        parent, such as combining values from child executors, unplaced and
        server-side processing, etc.
      target_executors: The list of executors that manage disjoint scopes to
        combine in this executor, delegate to and collect or aggregate from.
      local_computation_factory: An instance of `LocalComputationFactory` to use
        to construct local computations used as parameters in certain federated
        operators (such as `tff.federated_sum`, etc.). Defaults to a TensorFlow
        computation factory that generates TensorFlow code.

    Raises:
      TypeError: If `server_executor` is not an `executor_base.Executor` or if
        `target_executors` is not a `list` of `executor_base.Executor`s.
    """
    super().__init__(executor)
    py_typecheck.check_type(server_executor, executor_base.Executor)
    py_typecheck.check_type(target_executors, list)
    py_typecheck.check_type(
        local_computation_factory,
        local_computation_factory_base.LocalComputationFactory)
    self._local_computation_factory = local_computation_factory
    for e in target_executors:
      py_typecheck.check_type(e, executor_base.Executor)
    self._server_executor = server_executor
    self._target_executors = target_executors

  def close(self):
    self._server_executor.close()
    for e in self._target_executors:
      e.close()

  def ingest_value(
      self, value: Any, type_signature: computation_types.Type
  ) -> executor_value_base.ExecutorValue:
    return FederatedComposingStrategyValue(value, type_signature)

  async def _get_cardinalities(self):
    """Returns information about the number of clients in the child executors.

    Returns:
      A `list` with one element for each element in `self._target_executors`;
      each of these elements is an integer representing the total number of
      clients located in the corresponding child executor.
    """

    async def _num_clients(executor):
      """Returns the number of clients for the given `executor`."""
      # We implement num_clients as a federated_aggregate to allow for federated
      # op resolving strategies which implement only the minimal set of
      # intrinsics.
      int_at_clients_type = computation_types.at_clients(tf.int32)
      zero_type = tf.int32
      accumulate_type = computation_types.FunctionType(
          computation_types.StructType([tf.int32, tf.int32]), tf.int32)
      merge_type = accumulate_type
      report_type = computation_types.FunctionType(tf.int32, tf.int32)

      intrinsic_type = computation_types.FunctionType(
          computation_types.StructType([
              int_at_clients_type, zero_type, accumulate_type, merge_type,
              report_type
          ]), computation_types.at_server(tf.int32))
      intrinsic = executor_utils.create_intrinsic_comp(
          intrinsic_defs.FEDERATED_AGGREGATE, intrinsic_type)
      add_comp = building_block_factory.create_tensorflow_binary_operator_with_upcast(
          tf.add, computation_types.StructType([tf.int32, tf.int32])).proto
      identity_comp = building_block_factory.create_compiled_identity(
          computation_types.TensorType(tf.int32)).proto
      fn, client_data, zero_value, add_value, identity_value = await asyncio.gather(
          executor.create_value(intrinsic, intrinsic_type),
          executor.create_value(
              1, computation_types.at_clients(tf.int32, all_equal=True)),
          executor.create_value(0, tf.int32), executor.create_value(add_comp),
          executor.create_value(identity_comp))
      arg = await executor.create_struct(
          [client_data, zero_value, add_value, add_value, identity_value])
      call = await executor.create_call(fn, arg)
      result = await call.compute()
      if isinstance(result, tf.Tensor):
        return result.numpy()
      else:
        return result

    return await asyncio.gather(
        *[_num_clients(c) for c in self._target_executors])

  async def compute_federated_value(
      self, value: Any, type_signature: computation_types.Type
  ) -> FederatedComposingStrategyValue:
    if type_signature.placement == placements.SERVER:
      if not type_signature.all_equal:
        raise ValueError(
            'Expected an all equal value at the `SERVER` placement, '
            'found {}.'.format(type_signature))
      results = await self._server_executor.create_value(
          value, type_signature.member)
      return FederatedComposingStrategyValue(results, type_signature)
    elif type_signature.placement == placements.CLIENTS:
      if type_signature.all_equal:
        results = await asyncio.gather(*[
            c.create_value(value, type_signature)
            for c in self._target_executors
        ])
        return FederatedComposingStrategyValue(results, type_signature)
      else:
        py_typecheck.check_type(value, list)
        cardinalities = await self._get_cardinalities()
        total_clients = sum(cardinalities)
        py_typecheck.check_len(value, total_clients)
        results = []
        offset = 0
        for child, num_clients in zip(self._target_executors, cardinalities):
          new_offset = offset + num_clients
          result = child.create_value(value[offset:new_offset], type_signature)
          results.append(result)
          offset = new_offset
        return FederatedComposingStrategyValue(await asyncio.gather(*results),
                                               type_signature)
    else:
      raise ValueError('Unexpected placement {}.'.format(
          type_signature.placement))

  @tracing.trace
  async def compute_federated_aggregate(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    value_type, zero_type, accumulate_type, merge_type, report_type = (
        executor_utils.parse_federated_aggregate_argument_types(
            arg.type_signature))
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    py_typecheck.check_len(arg.internal_representation, 5)
    val = arg.internal_representation[0]
    py_typecheck.check_type(val, list)
    py_typecheck.check_len(val, len(self._target_executors))
    identity_report, identity_report_type = tensorflow_computation_factory.create_identity(
        zero_type)
    aggr_type = computation_types.FunctionType(
        computation_types.StructType([
            value_type, zero_type, accumulate_type, merge_type,
            identity_report_type
        ]), computation_types.at_server(zero_type))
    aggr_comp = executor_utils.create_intrinsic_comp(
        intrinsic_defs.FEDERATED_AGGREGATE, aggr_type)
    zero = await (await self._executor.create_selection(arg, 1)).compute()
    accumulate = arg.internal_representation[2]
    merge = arg.internal_representation[3]
    report = arg.internal_representation[4]

    async def _child_fn(ex, v):
      py_typecheck.check_type(v, executor_value_base.ExecutorValue)
      arg_values = [
          ex.create_value(zero, zero_type),
          ex.create_value(accumulate, accumulate_type),
          ex.create_value(merge, merge_type),
          ex.create_value(identity_report, identity_report_type)
      ]
      aggr_func, aggr_args = await asyncio.gather(
          ex.create_value(aggr_comp, aggr_type),
          ex.create_struct([v] + list(await asyncio.gather(*arg_values))))
      child_result = await (await ex.create_call(aggr_func,
                                                 aggr_args)).compute()
      result_at_server = await self._server_executor.create_value(
          child_result, zero_type)
      return result_at_server

    parent_merge, parent_report = await asyncio.gather(
        self._server_executor.create_value(merge, merge_type),
        self._server_executor.create_value(report, report_type))

    if self._target_executors:
      val_futures = asyncio.as_completed(
          [_child_fn(c, v) for c, v in zip(self._target_executors, val)])
      merge_result = await next(val_futures)
      for next_val_future in val_futures:
        next_val = await next_val_future
        merge_arg = await self._server_executor.create_struct(
            [merge_result, next_val])
        merge_result = await self._server_executor.create_call(
            parent_merge, merge_arg)
    else:
      merge_result = await self._server_executor.create_value(zero, zero_type)

    report_result = await self._server_executor.create_call(
        parent_report, merge_result)
    return FederatedComposingStrategyValue(
        report_result, computation_types.at_server(report_type.result))

  @tracing.trace
  async def compute_federated_apply(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    py_typecheck.check_len(arg.internal_representation, 2)
    fn_type = arg.type_signature[0]
    py_typecheck.check_type(fn_type, computation_types.FunctionType)
    val_type = arg.type_signature[1]
    type_analysis.check_federated_type(
        val_type, fn_type.parameter, placements.SERVER, all_equal=True)
    fn = arg.internal_representation[0]
    py_typecheck.check_type(fn, pb.Computation)
    val = arg.internal_representation[1]
    py_typecheck.check_type(val, executor_value_base.ExecutorValue)
    return FederatedComposingStrategyValue(
        await self._server_executor.create_call(
            await self._server_executor.create_value(fn, fn_type), val),
        computation_types.at_server(fn_type.result))

  @tracing.trace
  async def compute_federated_broadcast(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_broadcast(
        self._executor, arg)

  @tracing.trace
  async def _eval(self, arg, intrinsic, placement, all_equal):
    py_typecheck.check_type(arg.type_signature, computation_types.FunctionType)
    py_typecheck.check_type(arg.internal_representation, pb.Computation)
    py_typecheck.check_type(placement, placements.PlacementLiteral)
    fn = arg.internal_representation
    fn_type = arg.type_signature
    eval_type = computation_types.FunctionType(
        fn_type,
        computation_types.FederatedType(
            fn_type.result, placement, all_equal=all_equal))
    eval_comp = executor_utils.create_intrinsic_comp(intrinsic, eval_type)

    async def _child_fn(ex):
      py_typecheck.check_type(ex, executor_base.Executor)
      create_eval = ex.create_value(eval_comp, eval_type)
      create_fn = ex.create_value(fn, fn_type)
      eval_val, fn_val = await asyncio.gather(create_eval, create_fn)
      return await ex.create_call(eval_val, fn_val)

    result_vals = await asyncio.gather(
        *[_child_fn(c) for c in self._target_executors])

    result_type = computation_types.FederatedType(
        fn_type.result, placement, all_equal=all_equal)
    return FederatedComposingStrategyValue(result_vals, result_type)

  @tracing.trace
  async def compute_federated_eval_at_clients(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    return await self._eval(arg, intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS,
                            placements.CLIENTS, False)

  @tracing.trace
  async def compute_federated_eval_at_server(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    py_typecheck.check_type(arg.type_signature, computation_types.FunctionType)
    py_typecheck.check_type(arg.internal_representation, pb.Computation)
    fn_type = arg.type_signature
    embedded_fn = await self._server_executor.create_value(
        arg.internal_representation, fn_type)
    embedded_call = await self._server_executor.create_call(embedded_fn)
    return FederatedComposingStrategyValue(
        embedded_call, computation_types.at_server(fn_type.result))

  @tracing.trace
  async def _map(self, arg, all_equal=None):
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
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

    map_type = computation_types.FunctionType(
        [fn_type, computation_types.at_clients(fn_type.parameter)],
        computation_types.at_clients(fn_type.result))
    map_comp = executor_utils.create_intrinsic_comp(
        intrinsic_defs.FEDERATED_MAP, map_type)

    async def _child_fn(ex, v):
      py_typecheck.check_type(v, executor_value_base.ExecutorValue)
      fn_val = await ex.create_value(fn, fn_type)
      map_val, map_arg = await asyncio.gather(
          ex.create_value(map_comp, map_type), ex.create_struct([fn_val, v]))
      return await ex.create_call(map_val, map_arg)

    result_vals = await asyncio.gather(
        *[_child_fn(c, v) for c, v in zip(self._target_executors, val)])
    federated_type = computation_types.FederatedType(
        fn_type.result, val_type.placement, all_equal=all_equal)
    return FederatedComposingStrategyValue(result_vals, federated_type)

  @tracing.trace
  async def compute_federated_map(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    return await self._map(arg, all_equal=False)

  @tracing.trace
  async def compute_federated_map_all_equal(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    return await self._map(arg, all_equal=True)

  @tracing.trace
  async def compute_federated_mean(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    type_analysis.check_federated_type(
        arg.type_signature, placement=placements.CLIENTS)
    member_type = arg.type_signature.member

    async def _create_total():
      total = await self.compute_federated_sum(arg)
      total = await total.compute()
      return await self._server_executor.create_value(total, member_type)

    async def _create_factor():
      cardinalities = await self._get_cardinalities()
      count = sum(cardinalities)
      return await executor_utils.embed_constant(
          self._server_executor,
          member_type,
          float(1.0 / count),
          local_computation_factory=self._local_computation_factory)

    async def _create_multiply_arg():
      total, factor = await asyncio.gather(_create_total(), _create_factor())
      return await self._server_executor.create_struct([total, factor])

    multiply_fn, multiply_arg = await asyncio.gather(
        executor_utils.embed_multiply_operator(
            self._server_executor,
            member_type,
            local_computation_factory=self._local_computation_factory),
        _create_multiply_arg())
    result = await self._server_executor.create_call(multiply_fn, multiply_arg)
    type_signature = computation_types.at_server(member_type)
    return FederatedComposingStrategyValue(result, type_signature)

  @tracing.trace
  async def compute_federated_sum(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    type_analysis.check_federated_type(
        arg.type_signature, placement=placements.CLIENTS)
    id_comp, id_type = tensorflow_computation_factory.create_identity(
        arg.type_signature.member)
    zero, plus, identity = await asyncio.gather(
        executor_utils.embed_constant(
            self._executor,
            arg.type_signature.member,
            0,
            local_computation_factory=self._local_computation_factory),
        executor_utils.embed_plus_operator(
            self._executor,
            arg.type_signature.member,
            local_computation_factory=self._local_computation_factory),
        self._executor.create_value(id_comp, id_type))
    aggregate_args = await self._executor.create_struct(
        [arg, zero, plus, plus, identity])
    return await self.compute_federated_aggregate(aggregate_args)

  @tracing.trace
  async def compute_federated_secure_sum_bitwidth(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    raise NotImplementedError('The secure sum intrinsic is not implemented.')

  @tracing.trace
  async def compute_federated_secure_select(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    raise NotImplementedError('The secure select intrinsic is not implemented.')

  @tracing.trace
  async def compute_federated_select(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    client_keys_type, max_key_type, server_val_type, select_fn_type = (
        arg.type_signature)
    del client_keys_type  # Unused
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    client_keys, max_key, server_val, select_fn = arg.internal_representation
    py_typecheck.check_type(client_keys, list)
    py_typecheck.check_len(client_keys, len(self._target_executors))
    py_typecheck.check_type(max_key, executor_value_base.ExecutorValue)
    py_typecheck.check_type(server_val, executor_value_base.ExecutorValue)
    py_typecheck.check_type(select_fn, pb.Computation)
    unplaced_server_val, unplaced_max_key = await asyncio.gather(
        server_val.compute(), max_key.compute())
    select_type = computation_types.FunctionType(
        arg.type_signature,
        computation_types.at_clients(
            computation_types.SequenceType(select_fn_type.result)))
    select_pb = executor_utils.create_intrinsic_comp(
        intrinsic_defs.FEDERATED_SELECT, select_type)

    async def child_fn(child, child_client_keys):
      child_max_key_fut = child.create_value(unplaced_max_key, max_key_type)
      child_server_val_fut = child.create_value(unplaced_server_val,
                                                server_val_type)
      child_select_fn_fut = child.create_value(select_fn, select_fn_type)
      child_max_key, child_server_val, child_select_fn = await asyncio.gather(
          child_max_key_fut, child_server_val_fut, child_select_fn_fut)
      child_fn_fut = child.create_value(select_pb, select_type)
      child_arg_fut = child.create_struct(
          structure.Struct([(None, child_client_keys), (None, child_max_key),
                            (None, child_server_val), (None, child_select_fn)]))
      child_fn, child_arg = await asyncio.gather(child_fn_fut, child_arg_fut)
      return await child.create_call(child_fn, child_arg)

    return FederatedComposingStrategyValue(
        list(await asyncio.gather(*[
            child_fn(ex, ex_keys)
            for (ex, ex_keys) in zip(self._target_executors, client_keys)
        ])),
        computation_types.at_clients(
            computation_types.SequenceType(select_fn_type.result)))

  @tracing.trace
  async def compute_federated_value_at_clients(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_value(
        self._executor, arg, placements.CLIENTS)

  @tracing.trace
  async def compute_federated_value_at_server(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_value(
        self._executor, arg, placements.SERVER)

  @tracing.trace
  async def compute_federated_weighted_mean(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    return await executor_utils.compute_intrinsic_federated_weighted_mean(
        self._executor,
        arg,
        local_computation_factory=self._local_computation_factory)

  async def _zip_struct_into_child(self, child, child_index, value, value_type):
    """Embeds elements of `value` at `child_index` into `child`."""
    if value_type.is_federated():
      py_typecheck.check_type(value, list)
      return value[child_index]
    py_typecheck.check_type(value, structure.Struct)
    new_elements = await asyncio.gather(*[
        self._zip_struct_into_child(child, child_index, element, element_type)
        for element, element_type in zip(value, value_type)
    ])
    named_elements = structure.Struct(
        list(zip(structure.name_list_with_nones(value), new_elements)))
    return await child.create_struct(named_elements)

  @tracing.trace
  async def compute_federated_zip_at_clients(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    result_type = computation_types.at_clients(
        type_transformations.strip_placement(arg.type_signature))
    zip_type = computation_types.FunctionType(arg.type_signature, result_type)
    zip_comp = executor_utils.create_intrinsic_comp(
        intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS, zip_type)

    async def _child_fn(child, child_index):
      struct_value = await self._zip_struct_into_child(
          child, child_index, arg.internal_representation, arg.type_signature)
      return await child.create_call(
          await child.create_value(zip_comp, zip_type), struct_value)

    result = await asyncio.gather(
        *[_child_fn(c, i) for (i, c) in enumerate(self._target_executors)])
    return FederatedComposingStrategyValue(result, result_type)

  async def _zip_struct_into_server(self, value, value_type):
    """Embeds `value` into `self._server_executor`."""
    if value_type.is_federated():
      py_typecheck.check_type(value, executor_value_base.ExecutorValue)
      return value
    py_typecheck.check_type(value, structure.Struct)
    new_elements = await asyncio.gather(*[
        self._zip_struct_into_server(element, element_type)
        for element, element_type in zip(value, value_type)
    ])
    named_elements = structure.Struct(
        list(zip(structure.name_list_with_nones(value), new_elements)))
    return await self._server_executor.create_struct(named_elements)

  @tracing.trace
  async def compute_federated_zip_at_server(
      self,
      arg: FederatedComposingStrategyValue) -> FederatedComposingStrategyValue:
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    result_type = computation_types.at_server(
        type_transformations.strip_placement(arg.type_signature))
    return FederatedComposingStrategyValue(
        await self._zip_struct_into_server(arg.internal_representation,
                                           arg.type_signature), result_type)
