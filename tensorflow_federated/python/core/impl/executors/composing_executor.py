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
"""An executor composed of subordinate executors that manage disjoint scopes."""

import asyncio
from typing import List

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory

# TODO(b/139903095): Factor out more commonalities with federated executor code
# into the `executor_utils.py`.


class CompositeValue(executor_value_base.ExecutorValue):
  """Represents a value embedded in the composite executor."""

  def __init__(self, value, type_signature):
    """Creates an instance of `CompositeValue`.

    The kinds of supported internal representations (`value`) and types are as
    follows:

    * An instance of `intrinsic_defs.IntrinsicDef` in case of a federated
      operator (to be interpreted by this executor upon invocation).

    * An instance of `pb.Computation` in an unparsed form (to be relayed to one
      of the subordinate executors), which must be of a functional type.

    * A single `ExecutorValue` embedded in the parent executor.

    * An ordinary Python `list` with values embedded in the child executors.

    * An instance of `anonymous_tuple.AnonymousTuple` with values being one of
      the supported types listed above.

    Args:
      value: An internal value representation (of one of the allowed types, as
        defined above).
      type_signature: An instance of `tff.Type` or something convertible to it
        that is compatible with `value` (as defined above).
    """
    py_typecheck.check_type(type_signature, computation_types.Type)
    self._value = value
    self._type_signature = computation_types.to_type(type_signature)

  @property
  def internal_representation(self):
    return self._value

  @property
  def type_signature(self):
    return self._type_signature

  @tracing.trace
  async def compute(self):
    if isinstance(self._value, executor_value_base.ExecutorValue):
      return await self._value.compute()
    elif isinstance(self._value, anonymous_tuple.AnonymousTuple):
      results = await asyncio.gather(*[
          CompositeValue(v, t).compute()
          for v, t in zip(self._value, self._type_signature)
      ])
      element_types = anonymous_tuple.iter_elements(self._type_signature)
      return anonymous_tuple.AnonymousTuple(
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


class ComposingExecutor(executor_base.Executor):
  """An executor composed of subordinate executors that manage disjoint scopes.

  This executor can be used to construct multi-level hierarchical aggregation
  structures with federated executors managing disjoint subsets of clients at
  the leaf level.

  The intrinsics currently implemented include:
  - federated_aggregate
  - federated_apply
  - federated_broadcast
  - federated_eval
  - federated_map
  - federated_mean
  - federated_sum
  - federated_value
  - federated_weighted_mean
  - federated_zip
  """

  # TODO(b/139129100): Implement the remaining operators (collect, reduce, etc.)
  # for feature parity with the reference executor.

  def __init__(self, parent_executor: executor_base.Executor,
               child_executors: List[executor_base.Executor]):
    """Creates a composite executor from a collection of subordinate executors.

    Args:
      parent_executor: The parent executor to use for all processing at the
        parent, such as combining values from child executors, unplaced and
        server-side processing, etc.
      child_executors: The list of executors that manage disjoint scopes to
        combine in this executor, delegate to and collect or aggregate from.

    Raises:
      ValueError: If `parent_executor` is not an `Executor` instance, or
        `child_executors` is not a list of `Exector` instances.
    """
    py_typecheck.check_type(parent_executor, executor_base.Executor)
    py_typecheck.check_type(child_executors, list)
    for e in child_executors:
      py_typecheck.check_type(e, executor_base.Executor)
    self._parent_executor = parent_executor
    self._child_executors = child_executors
    self._cardinalities_task = None

  def close(self):
    self._parent_executor.close()
    for e in self._child_executors:
      e.close()

  async def _get_cardinalities(self):
    """Returns information about the number of clients in the child executors.

    Returns:
      A `list` with one element for each element in `self._child_executors`;
      each of these elements is an integer representing the total number of
      clients located in the corresponding child executor.
    """

    # This helper function and the logic to cache the `_cardinalities_task` is
    # is required because `functools.lru_cache` is not compatible with async
    # coroutines. See https://bugs.python.org/issue35040 for more information.
    async def _get_cardinalities_helper():

      async def _num_clients(executor):
        """Returns the number of clients for the given `executor`."""
        intrinsic_type = computation_types.FunctionType(
            type_factory.at_clients(tf.int32), type_factory.at_server(tf.int32))
        intrinsic = executor_utils.create_intrinsic_comp(
            intrinsic_defs.FEDERATED_SUM, intrinsic_type)
        arg_type = type_factory.at_clients(tf.int32, all_equal=True)
        fn, arg = await asyncio.gather(
            executor.create_value(intrinsic, intrinsic_type),
            executor.create_value(1, arg_type))
        call = await executor.create_call(fn, arg)
        result = await call.compute()
        if isinstance(result, tf.Tensor):
          return result.numpy()
        else:
          return result

      return await asyncio.gather(
          *[_num_clients(c) for c in self._child_executors])

    if self._cardinalities_task is None:
      self._cardinalities_task = asyncio.ensure_future(
          _get_cardinalities_helper())
    return await self._cardinalities_task

  @tracing.trace(span=True, stats=False)
  async def create_value(self, value, type_spec=None):
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      if not type_analysis.is_concrete_instance_of(type_spec,
                                                   value.type_signature):  # pytype: disable=attribute-error
        raise TypeError('Incompatible type {} used with intrinsic {}.'.format(
            type_spec, value.uri))
      return CompositeValue(value, type_spec)
    elif isinstance(value, pb.Computation):
      which_computation = value.WhichOneof('computation')
      if which_computation in ['lambda', 'tensorflow']:
        return CompositeValue(value, type_spec)
      elif which_computation == 'intrinsic':
        intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(value.intrinsic.uri)
        if intrinsic_def is None:
          raise ValueError('Encountered an unrecognized intrinsic "{}".'.format(
              value.intrinsic.uri))
        return await self.create_value(intrinsic_def, type_spec)
      else:
        raise NotImplementedError(
            'Unimplemented computation type {}.'.format(which_computation))
    elif isinstance(type_spec, computation_types.FederatedType):
      if type_spec.placement == placement_literals.SERVER:
        if not type_spec.all_equal:
          raise ValueError(
              'Expected an all equal value at the `SERVER` placement, '
              'found {}.'.format(type_spec))
        results = await self._parent_executor.create_value(
            value, type_spec.member)
        return CompositeValue(results, type_spec)
      elif type_spec.placement == placement_literals.CLIENTS:
        if type_spec.all_equal:
          results = await asyncio.gather(*[
              c.create_value(value, type_spec) for c in self._child_executors
          ])
          return CompositeValue(results, type_spec)
        else:
          py_typecheck.check_type(value, list)
          cardinalities = await self._get_cardinalities()
          total_clients = sum(cardinalities)
          py_typecheck.check_len(value, total_clients)
          results = []
          offset = 0
          for child, num_clients in zip(self._child_executors, cardinalities):
            new_offset = offset + num_clients
            result = child.create_value(value[offset:new_offset], type_spec)
            results.append(result)
            offset = new_offset
          return CompositeValue(await asyncio.gather(*results), type_spec)
      else:
        raise ValueError('Unexpected placement {}.'.format(type_spec.placement))
    else:
      return CompositeValue(
          await self._parent_executor.create_value(value, type_spec), type_spec)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, CompositeValue)
    if arg is not None:
      py_typecheck.check_type(arg, CompositeValue)
      py_typecheck.check_type(comp.type_signature,
                              computation_types.FunctionType)
      param_type = comp.type_signature.parameter
      param_type.check_assignable_from(arg.type_signature)
      arg = CompositeValue(arg.internal_representation, param_type)
    if isinstance(comp.internal_representation, pb.Computation):
      which_computation = comp.internal_representation.WhichOneof('computation')
      if which_computation == 'tensorflow':
        child = self._parent_executor
        embedded_comp = await child.create_value(comp.internal_representation,
                                                 comp.type_signature)
        if arg is not None:
          embedded_arg = await executor_utils.delegate_entirely_to_executor(
              arg.internal_representation, arg.type_signature, child)
        else:
          embedded_arg = None
        result = await child.create_call(embedded_comp, embedded_arg)
        return CompositeValue(result, result.type_signature)
      else:
        raise ValueError(
            'Directly calling computations of type {} is unsupported.'.format(
                which_computation))
    elif isinstance(comp.internal_representation, intrinsic_defs.IntrinsicDef):
      coro = getattr(
          self,
          '_compute_intrinsic_{}'.format(comp.internal_representation.uri),
          None)
      if coro is not None:
        return await coro(arg)  # pylint: disable=not-callable
      else:
        raise NotImplementedError(
            'Support for intrinsic "{}" has not been implemented yet.'.format(
                comp.internal_representation.uri))
    else:
      raise ValueError('Calling objects of type {} is unsupported.'.format(
          py_typecheck.type_string(type(comp.internal_representation))))

  @tracing.trace
  async def create_tuple(self, elements):
    element_values = []
    element_types = []
    for name, value in anonymous_tuple.iter_elements(
        anonymous_tuple.from_container(elements)):
      py_typecheck.check_type(value, CompositeValue)
      element_values.append((name, value.internal_representation))
      if name is not None:
        element_types.append((name, value.type_signature))
      else:
        element_types.append(value.type_signature)
    value = anonymous_tuple.AnonymousTuple(element_values)
    type_signature = computation_types.NamedTupleType(element_types)
    return CompositeValue(value, type_signature)

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, CompositeValue)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    if index is None and name is None:
      raise ValueError(
          'Expected either `index` or `name` to be specificed, found both are '
          '`None`.')
    if isinstance(source.internal_representation,
                  executor_value_base.ExecutorValue):
      child = self._parent_executor
      value = await child.create_selection(
          source.internal_representation, index=index, name=name)
      return CompositeValue(value, value.type_signature)
    elif isinstance(source.internal_representation,
                    anonymous_tuple.AnonymousTuple):
      if name is not None:
        value = source.internal_representation[name]
        type_signature = source.type_signature[name]
      else:
        value = source.internal_representation[index]
        type_signature = source.type_signature[index]
      return CompositeValue(value, type_signature)
    else:
      raise ValueError(
          'Unexpected internal representation while creating selection. '
          'Expected one of `AnonymousTuple` or value embedded in target '
          'executor, received {}'.format(source.internal_representation))

  @tracing.trace
  async def _compute_intrinsic_federated_aggregate(self, arg):
    value_type, zero_type, accumulate_type, merge_type, report_type = (
        executor_utils.parse_federated_aggregate_argument_types(
            arg.type_signature))
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    py_typecheck.check_len(arg.internal_representation, 5)
    val = arg.internal_representation[0]
    py_typecheck.check_type(val, list)
    py_typecheck.check_len(val, len(self._child_executors))
    identity_report = computation_factory.create_lambda_identity(zero_type)
    identity_report_type = type_factory.unary_op(zero_type)
    aggr_type = computation_types.FunctionType(
        computation_types.NamedTupleType([
            value_type, zero_type, accumulate_type, merge_type,
            identity_report_type
        ]), type_factory.at_server(zero_type))
    aggr_comp = executor_utils.create_intrinsic_comp(
        intrinsic_defs.FEDERATED_AGGREGATE, aggr_type)
    zero = await (await self.create_selection(arg, index=1)).compute()
    accumulate = arg.internal_representation[2]
    merge = arg.internal_representation[3]
    report = arg.internal_representation[4]

    async def _child_fn(ex, v):
      py_typecheck.check_type(v, executor_value_base.ExecutorValue)
      aggr_func, aggr_args = await asyncio.gather(
          ex.create_value(aggr_comp, aggr_type),
          ex.create_tuple([v] + list(await asyncio.gather(
              ex.create_value(zero, zero_type),
              ex.create_value(accumulate, accumulate_type),
              ex.create_value(merge, merge_type),
              ex.create_value(identity_report, identity_report_type)))))
      return await (await ex.create_call(aggr_func, aggr_args)).compute()

    vals = await asyncio.gather(
        *[_child_fn(c, v) for c, v in zip(self._child_executors, val)])
    parent_vals = await asyncio.gather(
        *[self._parent_executor.create_value(v, zero_type) for v in vals])
    parent_merge, parent_report = await asyncio.gather(
        self._parent_executor.create_value(merge, merge_type),
        self._parent_executor.create_value(report, report_type))
    merge_result = parent_vals[0]
    for next_val in parent_vals[1:]:
      merge_result = await self._parent_executor.create_call(
          parent_merge, await
          self._parent_executor.create_tuple([merge_result, next_val]))
    return CompositeValue(
        await self._parent_executor.create_call(parent_report, merge_result),
        type_factory.at_server(report_type.result))

  @tracing.trace
  async def _compute_intrinsic_federated_apply(self, arg):
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    py_typecheck.check_len(arg.internal_representation, 2)
    fn_type = arg.type_signature[0]
    py_typecheck.check_type(fn_type, computation_types.FunctionType)
    val_type = arg.type_signature[1]
    type_analysis.check_federated_type(
        val_type, fn_type.parameter, placement_literals.SERVER, all_equal=True)
    fn = arg.internal_representation[0]
    py_typecheck.check_type(fn, pb.Computation)
    val = arg.internal_representation[1]
    py_typecheck.check_type(val, executor_value_base.ExecutorValue)
    return CompositeValue(
        await self._parent_executor.create_call(
            await self._parent_executor.create_value(fn, fn_type), val),
        type_factory.at_server(fn_type.result))

  @tracing.trace
  async def _compute_intrinsic_federated_broadcast(self, arg):
    return await executor_utils.compute_intrinsic_federated_broadcast(self, arg)

  @tracing.trace
  async def _eval(self, arg, intrinsic, placement, all_equal):
    py_typecheck.check_type(arg.type_signature, computation_types.FunctionType)
    py_typecheck.check_type(arg.internal_representation, pb.Computation)
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
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
        *[_child_fn(c) for c in self._child_executors])

    result_type = computation_types.FederatedType(
        fn_type.result, placement, all_equal=all_equal)
    return CompositeValue(result_vals, result_type)

  @tracing.trace
  async def _compute_intrinsic_federated_eval_at_clients(self, arg):
    return await self._eval(arg, intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS,
                            placement_literals.CLIENTS, False)

  @tracing.trace
  async def _compute_intrinsic_federated_eval_at_server(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.FunctionType)
    py_typecheck.check_type(arg.internal_representation, pb.Computation)
    fn_type = arg.type_signature
    embedded_fn = await self._parent_executor.create_value(
        arg.internal_representation, fn_type)
    evaled_at_parent = await self._parent_executor.create_call(embedded_fn)
    return CompositeValue(evaled_at_parent,
                          type_factory.at_server(fn_type.result))

  @tracing.trace
  async def _map(self, arg, all_equal=None):
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
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
        [fn_type, type_factory.at_clients(fn_type.parameter)],
        type_factory.at_clients(fn_type.result))
    map_comp = executor_utils.create_intrinsic_comp(
        intrinsic_defs.FEDERATED_MAP, map_type)

    async def _child_fn(ex, v):
      py_typecheck.check_type(v, executor_value_base.ExecutorValue)
      fn_val = await ex.create_value(fn, fn_type)
      map_val, map_arg = await asyncio.gather(
          ex.create_value(map_comp, map_type), ex.create_tuple([fn_val, v]))
      return await ex.create_call(map_val, map_arg)

    result_vals = await asyncio.gather(
        *[_child_fn(c, v) for c, v in zip(self._child_executors, val)])
    federated_type = computation_types.FederatedType(
        fn_type.result, val_type.placement, all_equal=all_equal)
    return CompositeValue(result_vals, federated_type)

  @tracing.trace
  async def _compute_intrinsic_federated_map(self, arg):
    return await self._map(arg, all_equal=False)

  @tracing.trace
  async def _compute_intrinsic_federated_map_all_equal(self, arg):
    return await self._map(arg, all_equal=True)

  @tracing.trace
  async def _compute_intrinsic_federated_mean(self, arg):
    type_analysis.check_federated_type(
        arg.type_signature, placement=placement_literals.CLIENTS)
    member_type = arg.type_signature.member

    async def _compute_total():
      total = await self._compute_intrinsic_federated_sum(arg)
      total = await total.compute()
      return await self._parent_executor.create_value(total, member_type)

    async def _compute_factor():
      cardinalities = await self._get_cardinalities()
      count = sum(cardinalities)
      return await executor_utils.embed_tf_scalar_constant(
          self._parent_executor, member_type, float(1.0 / count))

    async def _compute_multiply_arg():
      total, factor = await asyncio.gather(_compute_total(), _compute_factor())
      return await self._parent_executor.create_tuple([total, factor])

    multiply_fn, multiply_arg = await asyncio.gather(
        executor_utils.embed_tf_binary_operator(self._parent_executor,
                                                member_type, tf.multiply),
        _compute_multiply_arg())
    result = await self._parent_executor.create_call(multiply_fn, multiply_arg)
    type_signature = type_factory.at_server(member_type)
    return CompositeValue(result, type_signature)

  @tracing.trace
  async def _compute_intrinsic_federated_sum(self, arg):
    type_analysis.check_federated_type(
        arg.type_signature, placement=placement_literals.CLIENTS)
    zero, plus, identity = await asyncio.gather(
        executor_utils.embed_tf_scalar_constant(self, arg.type_signature.member,
                                                0),
        executor_utils.embed_tf_binary_operator(self, arg.type_signature.member,
                                                tf.add),
        self.create_value(
            computation_factory.create_lambda_identity(
                arg.type_signature.member),
            type_factory.unary_op(arg.type_signature.member)))
    aggregate_args = await self.create_tuple([arg, zero, plus, plus, identity])
    return await self._compute_intrinsic_federated_aggregate(aggregate_args)

  @tracing.trace
  async def _compute_intrinsic_federated_secure_sum(self, arg):
    raise NotImplementedError('The secure sum intrinsic is not implemented.')

  @tracing.trace
  async def _compute_intrinsic_federated_value_at_clients(self, arg):
    return await executor_utils.compute_intrinsic_federated_value(
        self, arg, placement_literals.CLIENTS)

  @tracing.trace
  async def _compute_intrinsic_federated_value_at_server(self, arg):
    return await executor_utils.compute_intrinsic_federated_value(
        self, arg, placement_literals.SERVER)

  @tracing.trace
  async def _compute_intrinsic_federated_weighted_mean(self, arg):
    return await executor_utils.compute_intrinsic_federated_weighted_mean(
        self, arg)

  @tracing.trace
  async def _compute_intrinsic_federated_zip_at_clients(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_len(arg.type_signature, 2)
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    py_typecheck.check_len(arg.internal_representation, 2)
    keys = [k for k, _ in anonymous_tuple.to_elements(arg.type_signature)]
    vals = [arg.internal_representation[n] for n in [0, 1]]
    types = [arg.type_signature[n] for n in [0, 1]]
    for n in [0, 1]:
      type_analysis.check_federated_type(
          types[n], placement=placement_literals.CLIENTS)
      types[n] = type_factory.at_clients(types[n].member)
      py_typecheck.check_type(vals[n], list)
      py_typecheck.check_len(vals[n], len(self._child_executors))
    item_type = computation_types.NamedTupleType([
        ((keys[n], types[n].member) if keys[n] else types[n].member)
        for n in [0, 1]
    ])
    result_type = type_factory.at_clients(item_type)
    zip_type = computation_types.FunctionType(
        computation_types.NamedTupleType([
            ((keys[n], types[n]) if keys[n] else types[n]) for n in [0, 1]
        ]), result_type)
    zip_comp = executor_utils.create_intrinsic_comp(
        intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS, zip_type)

    async def _child_fn(ex, x, y):
      py_typecheck.check_type(x, executor_value_base.ExecutorValue)
      py_typecheck.check_type(y, executor_value_base.ExecutorValue)
      return await ex.create_call(
          await ex.create_value(zip_comp, zip_type), await ex.create_tuple(
              anonymous_tuple.AnonymousTuple([(keys[0], x), (keys[1], y)])))

    result = await asyncio.gather(*[
        _child_fn(c, x, y)
        for c, x, y in zip(self._child_executors, vals[0], vals[1])
    ])
    return CompositeValue(result, result_type)

  @tracing.trace
  async def _compute_intrinsic_federated_zip_at_server(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_len(arg.type_signature, 2)
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    py_typecheck.check_len(arg.internal_representation, 2)
    for n in [0, 1]:
      type_analysis.check_federated_type(
          arg.type_signature[n],
          placement=placement_literals.SERVER,
          all_equal=True)
    return CompositeValue(
        await self._parent_executor.create_tuple(
            [arg.internal_representation[n] for n in [0, 1]]),
        type_factory.at_server(
            computation_types.NamedTupleType(
                [arg.type_signature[0].member, arg.type_signature[1].member])))
