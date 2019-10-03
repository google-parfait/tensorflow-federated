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
"""An executor that handles federated types and federated operators."""

import asyncio

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_utils
from tensorflow_federated.python.core.impl import executor_value_base
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization


class FederatedExecutorValue(executor_value_base.ExecutorValue):
  """Represents a value embedded in the federated executor."""

  def __init__(self, value, type_spec):
    """Creates an embedded instance of a value in this executor.

    The kinds of supported internal representations (`value`) and types are as
    follows:

    * An instance of `intrinsic_defs.IntrinsicDef` in case of a federated
      operator (to be interpreted by this executor upon invocation).

    * An instance of `placement_literals.PlacementLiteral`.

    * An instance of `pb.Computation` in an unparsed form (to be relayed to one
      of the executors responsible for the given placement later on), which
      must be of one of the following varieties: tensorflow, lambda.

    * An ordinary Python `list` with values embedded in subordinate executors
      in case `type_spec` is a federated type. The list representation is used
      even if the value is of an `all_equal` type or there's only a single
      participant associated with the given placement.

    * A single value embedded in a subordinate executor in case `type_spec` is
      of a non-federated non-functional type.

    * An instance of `anonymous_tuple.AnonymousTuple` with values being one of
      the supported types listed above.

    This constructor does not perform any verification, however.

    Args:
      value: An internal value representation (of one of the allowed types, as
        defined above).
      type_spec: An instance of `tff.Type` or something convertible to it that
        is compatible with `value` (as defined above).
    """
    self._value = value
    self._type_signature = computation_types.to_type(type_spec)
    py_typecheck.check_type(type_spec, computation_types.Type)

  @property
  def internal_representation(self):
    return self._value

  @property
  def type_signature(self):
    return self._type_signature

  async def compute(self):
    if isinstance(self._value, executor_value_base.ExecutorValue):
      return await self._value.compute()
    elif isinstance(self._type_signature, computation_types.FederatedType):
      py_typecheck.check_type(self._value, list)
      if self._type_signature.all_equal:
        vals = [self._value[0]]
      else:
        vals = self._value
      results = []
      for v in vals:
        py_typecheck.check_type(v, executor_value_base.ExecutorValue)
        results.append(v.compute())
      results = await asyncio.gather(*results)
      if self._type_signature.all_equal:
        return results[0]
      else:
        return results
    elif isinstance(self._value, anonymous_tuple.AnonymousTuple):
      value_elements = anonymous_tuple.to_elements(self._value)
      type_elements = anonymous_tuple.to_elements(self._type_signature)
      gathered_values = await asyncio.gather(*[
          FederatedExecutorValue(v, t).compute()
          for (_, v), (_, t) in zip(value_elements, type_elements)
      ])
      return anonymous_tuple.AnonymousTuple([
          (k, v) for (k, _), v in zip(type_elements, gathered_values)
      ])
    else:
      raise RuntimeError(
          'Computing values of type {} represented as {} is not supported in '
          'this executor.'.format(self._type_signature,
                                  py_typecheck.type_string(type(self._value))))


class FederatedExecutor(executor_base.Executor):
  """The federated executor orchestrates federated computations.

  NOTE: This component is only available in Python 3.

  The intrinsics currently implemented include:
  - federated_aggregate
  - federated_apply
  - federated_broadcast
  - federated_map
  - federated_mean
  - federated_reduce
  - federated_sum
  - federated_value
  - federated_weighted_mean
  - federated_zip

  This executor is only responsible for handling federated types and federated
  operators, and a delegation of work to an underlying collection of target
  executors associated with individual system participants. This executor does
  not interpret lambda calculus and compositional constructs (blocks, etc.).
  It understands placements, selected intrinsics (federated operators), it can
  handle tuples, selections, and calls in a limited way (to the extent that it
  deals with intrinsics or lambda expressions it can delegate).

  The initial implementation of the executor only supports the two basic types
  of placements (SERVER and CLIENTS), and does not have a built-in concept of
  intermediate aggregation, partitioning placements, clustering clients, etc.

  The initial implementation also does not attempt at performing optimizations
  in case when the constituents of this executor are either located on the same
  machine (where marshaling/unmarshaling could be avoided), or when they have
  the `all_equal` property (and a single value could be shared by them all).
  """

  # TODO(b/134543154): Extend this executor to support intermediate aggregation
  # and other optimizations hinted above.

  # TODO(b/134543154): Add support for `data` as a building block.

  # TODO(b/134543154): Implement the commonly used aggregation intrinsics so we
  # can begin to use this executor in integration tests.

  def __init__(self, target_executors):
    """Creates a federated executor backed by a collection of target executors.

    Args:
      target_executors: A dictionary mapping placements to executors or lists of
        executors associated with these placements. The keys in this dictionary
        can be either placement literals, or `None` to specify the executor for
        unplaced computations. The values can be either single executors (if
        there only is a single participant associated with that placement, as
        would typically be the case with `tff.SERVER`) or lists of target
        executors.

    Raises:
      ValueError: If the value is unrecognized (e.g., a nonexistent intrinsic).
    """
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
        self._target_executors[k] = v
    for pl in [None, placement_literals.SERVER]:
      if pl in self._target_executors:
        pl_cardinality = len(self._target_executors[pl])
        if pl_cardinality != 1:
          raise ValueError(
              'Unsupported cardinality for placement "{}": {}.'.format(
                  pl, pl_cardinality))

  async def create_value(self, value, type_spec=None):
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      if not type_utils.is_concrete_instance_of(type_spec,
                                                value.type_signature):
        raise TypeError('Incompatible type {} used with intrinsic {}.'.format(
            type_spec, value.uri))
      else:
        return FederatedExecutorValue(value, type_spec)
    if isinstance(value, placement_literals.PlacementLiteral):
      if type_spec is not None:
        py_typecheck.check_type(type_spec, computation_types.PlacementType)
      return FederatedExecutorValue(value, computation_types.PlacementType())
    elif isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          type_utils.reconcile_value_with_type_spec(value, type_spec))
    elif isinstance(value, pb.Computation):
      if type_spec is None:
        type_spec = type_serialization.deserialize_type(value.type)
      which_computation = value.WhichOneof('computation')
      if which_computation in ['tensorflow', 'lambda']:
        return FederatedExecutorValue(value, type_spec)
      elif which_computation == 'reference':
        raise ValueError(
            'Encountered an unexpected unbound references "{}".'.format(
                value.reference.name))
      elif which_computation == 'intrinsic':
        intr = intrinsic_defs.uri_to_intrinsic_def(value.intrinsic.uri)
        if intr is None:
          raise ValueError('Encountered an unrecognized intrinsic "{}".'.format(
              value.intrinsic.uri))
        py_typecheck.check_type(intr, intrinsic_defs.IntrinsicDef)
        return await self.create_value(intr, type_spec)
      elif which_computation == 'placement':
        return await self.create_value(
            placement_literals.uri_to_placement_literal(value.placement.uri),
            type_spec)
      elif which_computation == 'call':
        parts = [value.call.function]
        if value.call.argument.WhichOneof('computation'):
          parts.append(value.call.argument)
        parts = await asyncio.gather(*[self.create_value(x) for x in parts])
        return await self.create_call(parts[0],
                                      parts[1] if len(parts) > 1 else None)
      elif which_computation == 'tuple':
        element_values = await asyncio.gather(
            *[self.create_value(x.value) for x in value.tuple.element])
        return await self.create_tuple(
            anonymous_tuple.AnonymousTuple([
                (e.name if e.name else None, v)
                for e, v in zip(value.tuple.element, element_values)
            ]))
      elif which_computation == 'selection':
        which_selection = value.selection.WhichOneof('selection')
        if which_selection == 'name':
          name = value.selection.name
          index = None
        elif which_selection != 'index':
          raise ValueError(
              'Unrecognized selection type: "{}".'.format(which_selection))
        else:
          index = value.selection.index
          name = None
        return await self.create_selection(
            await self.create_value(value.selection.source),
            index=index,
            name=name)
      else:
        raise ValueError(
            'Unsupported computation building block of type "{}".'.format(
                which_computation))
    else:
      py_typecheck.check_type(type_spec, computation_types.Type)
      if isinstance(type_spec, computation_types.FunctionType):
        raise ValueError(
            'Encountered a value of a functional TFF type {} and Python type '
            '{} that is not of one of the recognized representations.'.format(
                type_spec, py_typecheck.type_string(type(value))))
      elif isinstance(type_spec, computation_types.FederatedType):
        children = self._target_executors.get(type_spec.placement)
        if not children:
          raise ValueError(
              'Placement "{}" is not configured in this executor.'.format(
                  type_spec.placement))
        py_typecheck.check_type(children, list)
        if not type_spec.all_equal:
          py_typecheck.check_type(value, (list, tuple, set, frozenset))
          if not isinstance(value, list):
            value = list(value)
        elif isinstance(value, list):
          raise ValueError(
              'An all_equal value should be passed directly, not as a list.')
        else:
          value = [value for _ in children]
        if len(value) != len(children):
          raise ValueError(
              'Federated value contains {} items, but the placement {} in this '
              'executor is configured with {} participants.'.format(
                  len(value), type_spec.placement, len(children)))
        child_vals = await asyncio.gather(*[
            c.create_value(v, type_spec.member)
            for v, c in zip(value, children)
        ])
        return FederatedExecutorValue(child_vals, type_spec)
      else:
        child = self._target_executors.get(None)
        if not child or len(child) > 1:
          raise RuntimeError('Executor is not configured for unplaced values.')
        else:
          return FederatedExecutorValue(
              await child[0].create_value(value, type_spec), type_spec)

  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, FederatedExecutorValue)
    if arg is not None:
      py_typecheck.check_type(arg, FederatedExecutorValue)
      py_typecheck.check_type(comp.type_signature,
                              computation_types.FunctionType)
      param_type = comp.type_signature.parameter
      type_utils.check_assignable_from(param_type, arg.type_signature)
      arg = FederatedExecutorValue(arg.internal_representation, param_type)
    if isinstance(comp.internal_representation, pb.Computation):
      which_computation = comp.internal_representation.WhichOneof('computation')
      if which_computation == 'tensorflow':
        child = self._target_executors[None][0]
        embedded_comp = await child.create_value(comp.internal_representation,
                                                 comp.type_signature)
        if arg is not None:
          embedded_arg = await executor_utils.delegate_entirely_to_executor(
              arg.internal_representation, arg.type_signature, child)
        else:
          embedded_arg = None
        result = await child.create_call(embedded_comp, embedded_arg)
        return FederatedExecutorValue(result, result.type_signature)
      else:
        raise ValueError(
            'Directly calling computations of type {} is unsupported.'.format(
                which_computation))
    elif isinstance(comp.internal_representation, intrinsic_defs.IntrinsicDef):
      coro = getattr(
          self,
          '_compute_intrinsic_{}'.format(comp.internal_representation.uri))
      if coro is not None:
        return await coro(arg)
      else:
        raise NotImplementedError(
            'Support for intrinsic "{}" has not been implemented yet.'.format(
                comp.internal_representation.uri))
    else:
      raise ValueError('Calling objects of type {} is unsupported.'.format(
          py_typecheck.type_string(type(comp.internal_representation))))

  async def create_tuple(self, elements):
    elem = anonymous_tuple.to_elements(anonymous_tuple.from_container(elements))
    for _, v in elem:
      py_typecheck.check_type(v, FederatedExecutorValue)
    return FederatedExecutorValue(
        anonymous_tuple.AnonymousTuple([
            (k, v.internal_representation) for k, v in elem
        ]),
        computation_types.NamedTupleType([
            (k, v.type_signature) if k else v.type_signature for k, v in elem
        ]))

  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, FederatedExecutorValue)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    if name is not None:
      name_to_index = dict((n, i) for i, (
          n,
          t) in enumerate(anonymous_tuple.to_elements(source.type_signature)))
      index = name_to_index[name]
    if isinstance(source.internal_representation,
                  anonymous_tuple.AnonymousTuple):
      val = source.internal_representation
      selected = val[index]
      return FederatedExecutorValue(selected, source.type_signature[index])
    elif isinstance(source.internal_representation,
                    executor_value_base.ExecutorValue):
      if type_utils.type_tree_contains_types(source.type_signature,
                                             computation_types.FederatedType):
        raise ValueError('FederatedExecutorValue {} has violated its contract; '
                         'it is embedded in another executor and yet its type '
                         'has placement. The embedded value is {}, with type '
                         'signature {}.'.format(source,
                                                source.internal_representation,
                                                source.type_signature))
      val = source.internal_representation
      child = self._target_executors[None][0]
      return FederatedExecutorValue(
          await child.create_selection(val, index=index),
          source.type_signature[index])
    else:
      raise ValueError('Unexpected internal representation while creating '
                       'selection. Expected one of `AnonymousTuple` or value '
                       'embedded in target executor, received {}'.format(
                           source.internal_representation))

  async def _place(self, arg, placement):
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    children = self._target_executors[placement]
    val = await arg.internal_representation.compute()
    return FederatedExecutorValue(
        await asyncio.gather(
            *[c.create_value(val, arg.type_signature) for c in children]),
        computation_types.FederatedType(
            arg.type_signature, placement, all_equal=True))

  async def _map(self, arg, all_equal=None):
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    if len(arg.internal_representation) != 2:
      raise ValueError('Expected 2 elements in the tuple, found {}.'.format(
          len(arg)))
    fn_type = arg.type_signature[0]
    val_type = arg.type_signature[1]
    py_typecheck.check_type(val_type, computation_types.FederatedType)
    if all_equal is None:
      all_equal = val_type.all_equal
    elif all_equal and not val_type.all_equal:
      raise ValueError(
          'Cannot map a non-all_equal argument into an all_equal result.')
    fn = arg.internal_representation[0]
    val = arg.internal_representation[1]
    py_typecheck.check_type(fn, pb.Computation)
    py_typecheck.check_type(val, list)
    for v in val:
      py_typecheck.check_type(v, executor_value_base.ExecutorValue)
    children = self._target_executors[val_type.placement]
    fns = await asyncio.gather(*[c.create_value(fn, fn_type) for c in children])
    results = await asyncio.gather(*[
        c.create_call(f, v) for c, (f, v) in zip(children, list(zip(fns, val)))
    ])
    return FederatedExecutorValue(
        results,
        computation_types.FederatedType(
            fn_type.result, val_type.placement, all_equal=all_equal))

  async def _zip(self, arg, placement, all_equal):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    cardinality = len(self._target_executors[placement])
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
    children = self._target_executors[placement]
    new_vals = await asyncio.gather(
        *[c.create_tuple(x) for c, x in zip(children, new_vals)])
    return FederatedExecutorValue(
        new_vals,
        computation_types.FederatedType(
            computation_types.NamedTupleType([
                (k, v.member) if k else v.member
                for k, v in anonymous_tuple.iter_elements(arg.type_signature)
            ]),
            placement,
            all_equal=all_equal))

  async def _compute_intrinsic_federated_value_at_server(self, arg):
    return await self._place(arg, placement_literals.SERVER)

  async def _compute_intrinsic_federated_value_at_clients(self, arg):
    return await self._place(arg, placement_literals.CLIENTS)

  async def _compute_intrinsic_federated_apply(self, arg):
    return await self._map(arg)

  async def _compute_intrinsic_federated_map(self, arg):
    return await self._map(arg, all_equal=False)

  async def _compute_intrinsic_federated_broadcast(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    py_typecheck.check_type(arg.internal_representation, list)
    if not arg.type_signature.all_equal:
      raise ValueError('Cannot broadcast a non all_equal value.')
    if len(arg.internal_representation) != 1:
      raise ValueError(
          'Cannot broadcast a with a non-singleton representation.')
    val = await arg.internal_representation[0].compute()
    return FederatedExecutorValue(
        await asyncio.gather(*[
            c.create_value(val, arg.type_signature.member)
            for c in self._target_executors[placement_literals.CLIENTS]
        ]), type_factory.at_clients(arg.type_signature.member, all_equal=True))

  async def _compute_intrinsic_federated_zip_at_server(self, arg):
    return await self._zip(arg, placement_literals.SERVER, all_equal=True)

  async def _compute_intrinsic_federated_zip_at_clients(self, arg):
    return await self._zip(arg, placement_literals.CLIENTS, all_equal=False)

  async def _compute_intrinsic_federated_reduce(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    if len(arg.internal_representation) != 3:
      raise ValueError(
          'Expected 3 elements in the `federated_reduce()` argument tuple, '
          'found {}.'.format(len(arg.internal_representation)))

    val_type = arg.type_signature[0]
    py_typecheck.check_type(val_type, computation_types.FederatedType)
    item_type = val_type.member
    zero_type = arg.type_signature[1]
    op_type = arg.type_signature[2]
    type_utils.check_equivalent_types(
        op_type, type_factory.reduction_op(zero_type, item_type))

    val = arg.internal_representation[0]
    py_typecheck.check_type(val, list)
    child = self._target_executors[placement_literals.SERVER][0]

    async def _move(v):
      return await child.create_value(await v.compute(), item_type)

    items = await asyncio.gather(*[_move(v) for v in val])

    zero = await child.create_value(
        await (await self.create_selection(arg, index=1)).compute(), zero_type)
    op = await child.create_value(arg.internal_representation[2], op_type)

    result = zero
    for item in items:
      result = await child.create_call(
          op, await child.create_tuple(
              anonymous_tuple.AnonymousTuple([(None, result), (None, item)])))
    return FederatedExecutorValue([result],
                                  computation_types.FederatedType(
                                      result.type_signature,
                                      placement_literals.SERVER,
                                      all_equal=True))

  async def _compute_intrinsic_federated_aggregate(self, arg):
    val_type, zero_type, accumulate_type, _, report_type = (
        executor_utils.parse_federated_aggregate_argument_types(
            arg.type_signature))
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    py_typecheck.check_len(arg.internal_representation, 5)

    # NOTE: This is a simple initial implementation that simply forwards this
    # to `federated_reduce()`. The more complete implementation would be able
    # to take advantage of the parallelism afforded by `merge` to reduce the
    # cost from liner (with respect to the number of clients) to sub-linear.

    # TODO(b/134543154): Expand this implementation to take advantage of the
    # parallelism afforded by `merge`.

    val = arg.internal_representation[0]
    zero = arg.internal_representation[1]
    accumulate = arg.internal_representation[2]
    pre_report = await self._compute_intrinsic_federated_reduce(
        FederatedExecutorValue(
            anonymous_tuple.AnonymousTuple([(None, val), (None, zero),
                                            (None, accumulate)]),
            computation_types.NamedTupleType(
                [val_type, zero_type, accumulate_type])))

    py_typecheck.check_type(pre_report.type_signature,
                            computation_types.FederatedType)
    type_utils.check_equivalent_types(pre_report.type_signature.member,
                                      report_type.parameter)

    report = arg.internal_representation[4]
    return await self._compute_intrinsic_federated_apply(
        FederatedExecutorValue(
            anonymous_tuple.AnonymousTuple([
                (None, report), (None, pre_report.internal_representation)
            ]),
            computation_types.NamedTupleType(
                [report_type, pre_report.type_signature])))

  async def _compute_intrinsic_federated_sum(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    zero, plus = tuple(await
                       asyncio.gather(*[
                           executor_utils.embed_tf_scalar_constant(
                               self, arg.type_signature.member, 0),
                           executor_utils.embed_tf_binary_operator(
                               self, arg.type_signature.member, tf.add)
                       ]))
    return await self._compute_intrinsic_federated_reduce(
        FederatedExecutorValue(
            anonymous_tuple.AnonymousTuple([
                (None, arg.internal_representation),
                (None, zero.internal_representation),
                (None, plus.internal_representation)
            ]),
            computation_types.NamedTupleType(
                [arg.type_signature, zero.type_signature,
                 plus.type_signature])))

  async def _compute_intrinsic_federated_mean(self, arg):
    arg_sum = await self._compute_intrinsic_federated_sum(arg)
    member_type = arg_sum.type_signature.member
    count = float(len(arg.internal_representation))
    if count < 1.0:
      raise RuntimeError('Cannot compute a federated mean over an empty group.')
    child = self._target_executors[placement_literals.SERVER][0]
    factor, multiply = tuple(await asyncio.gather(*[
        executor_utils.embed_tf_scalar_constant(child, member_type,
                                                float(1.0 / count)),
        executor_utils.embed_tf_binary_operator(child, member_type, tf.multiply)
    ]))
    multiply_arg = await child.create_tuple(
        anonymous_tuple.AnonymousTuple([(None,
                                         arg_sum.internal_representation[0]),
                                        (None, factor)]))
    result = await child.create_call(multiply, multiply_arg)
    return FederatedExecutorValue([result], arg_sum.type_signature)

  async def _compute_intrinsic_federated_weighted_mean(self, arg):
    return await executor_utils.compute_federated_weighted_mean(self, arg)

  async def _compute_intrinsic_federated_collect(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    type_utils.check_federated_type(
        arg.type_signature, placement=placement_literals.CLIENTS)
    val = arg.internal_representation
    py_typecheck.check_type(val, list)
    member_type = arg.type_signature.member
    child = self._target_executors[placement_literals.SERVER][0]
    collected_items = await child.create_value(
        await asyncio.gather(*[v.compute() for v in val]),
        computation_types.SequenceType(member_type))
    return FederatedExecutorValue(
        [collected_items],
        computation_types.FederatedType(
            computation_types.SequenceType(member_type),
            placement_literals.SERVER))
