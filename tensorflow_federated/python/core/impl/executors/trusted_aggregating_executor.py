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

import absl.logging as logging
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base


class Aggregator():
  def __init__(self):
    self._uri = 'AGGREGATOR'
  
  @property
  def uri(self):
    return self._uri


AGGREGATOR = Aggregator()


class TrustedAggregatingExecutorValue(executor_value_base.ExecutorValue):
  """A value embedded in TrustedAggregatingExecutor."""

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

  @tracing.trace
  async def compute(self):
    import pdb; pdb.set_trace()
    if isinstance(self._value, executor_value_base.ExecutorValue):
      return await self._value.compute()
    elif isinstance(self._type_signature, computation_types.FederatedType):
      py_typecheck.check_type(self._value, list)
      if self._type_signature.all_equal:
        if not self._value:
          # TODO(b/145936344): this happens when the executor has inferred the
          # cardinality of clients as 0, which can happen in tff.Computation
          # that only do a tff.federated_broadcast. This probably should be
          # handled elsewhere.
          raise RuntimeError('Arrived at a computation that inferred there are '
                             '0 clients. Try explicity passing `num_clients` '
                             'parameter when constructor the executor.')
      vals = self._value
      results = []
      for v in vals:
        py_typecheck.check_type(v, executor_value_base.ExecutorValue)
        results.append(v.compute())
      results = await asyncio.gather(*results)
      return results
    elif isinstance(self._value, anonymous_tuple.AnonymousTuple):
      gathered_values = await asyncio.gather(*[
          TrustedAggregatingExecutorValue(v, t).compute()
          for v, t in zip(self._value, self._type_signature)
      ])
      type_elements_iter = anonymous_tuple.iter_elements(self._type_signature)
      return anonymous_tuple.AnonymousTuple(
          (k, v) for (k, _), v in zip(type_elements_iter, gathered_values))
    else:
      raise RuntimeError(
          'Computing values of type {} represented as {} is not supported in '
          'this executor.'.format(self._type_signature,
                                  py_typecheck.type_string(type(self._value))))

class TrustedAggregatingExecutor(executor_base.Executor):

  def __init__(self, target_executors):
    """Creates a trusted executor for Aggregating.

    Args:
      target_executors: A dictionary mapping placements to executors or lists of
        executors associated with these placements. The keys in this dictionary
        can be either placement literals, or `None` to specify the executor for
        unplaced computations. The values can be either single executors (if
        there only is a single participant associated with that placement, as
        would typically be the case with `tff.SERVER`) or lists of target
        executors. This dictionary must contain an 'AGGREGATOR' key-value pair
        targeting the aggregating executor stack.

    Raises:
      ValueError: If the value is unrecognized (e.g., a nonexistent intrinsic).
    """
    assert AGGREGATOR in target_executors

    py_typecheck.check_type(target_executors, dict)
    self._target_executors = {}
    for k, v in target_executors.items():
      if k is not None and k != AGGREGATOR:
        py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      py_typecheck.check_type(v, (list, executor_base.Executor))
      if isinstance(v, executor_base.Executor):
        self._target_executors[k] = [v]
      else:
        for e in v:
          py_typecheck.check_type(e, executor_base.Executor)
        self._target_executors[k] = v.copy()
    for pl in [None, AGGREGATOR, placement_literals.SERVER]:
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

  @tracing.trace(stats=False)
  async def create_value(self, value, type_spec=None):
    # import pdb; pdb.set_trace()
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      if not type_utils.is_concrete_instance_of(type_spec,
                                                value.type_signature):
        raise TypeError('Incompatible type {} used with intrinsic {}.'.format(
            type_spec, value.uri))
      else:
        return TrustedAggregatingExecutorValue(value, type_spec)
    if isinstance(value, placement_literals.PlacementLiteral):
      if type_spec is not None:
        py_typecheck.check_type(type_spec, computation_types.PlacementType)
      return TrustedAggregatingExecutorValue(value, computation_types.PlacementType())
    elif isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          type_utils.reconcile_value_with_type_spec(value, type_spec))
    elif isinstance(value, pb.Computation):
      if type_spec is None:
        type_spec = type_serialization.deserialize_type(value.type)
      which_computation = value.WhichOneof('computation')
      if which_computation in ['tensorflow', 'lambda']:
        return TrustedAggregatingExecutorValue(value, type_spec)
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
            anonymous_tuple.AnonymousTuple(
                (e.name if e.name else None, v)
                for e, v in zip(value.tuple.element, element_values)))
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
        return TrustedAggregatingExecutorValue(child_vals, type_spec)
      else:
        child = self._target_executors.get(None)
        if not child or len(child) > 1:
          raise RuntimeError('Executor is not configured for unplaced values.')
        else:
          return TrustedAggregatingExecutorValue(
              await child[0].create_value(value, type_spec), type_spec)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    # import pdb; pdb.set_trace()
    py_typecheck.check_type(comp, TrustedAggregatingExecutorValue)
    if arg is not None:
      py_typecheck.check_type(arg, TrustedAggregatingExecutorValue)
      py_typecheck.check_type(comp.type_signature,
                              computation_types.FunctionType)
      param_type = comp.type_signature.parameter
      type_utils.check_assignable_from(param_type, arg.type_signature)
      arg = TrustedAggregatingExecutorValue(arg.internal_representation, param_type)
    if isinstance(comp.internal_representation, pb.Computation):
      which_computation = comp.internal_representation.WhichOneof('computation')
      comp_type_signature = comp.type_signature
      if which_computation == 'lambda':
        # Pull the inner computation out of called no-arg lambdas.
        if comp.type_signature.parameter is not None:
          raise ValueError(
              'Directly calling lambdas with arguments is unsupported. '
              'Found call to lambda with type {}.'.format(comp.type_signature))
        return await self.create_value(
            getattr(comp.internal_representation, 'lambda').result,
            comp.type_signature.result)
      elif which_computation == 'tensorflow':
        # Run tensorflow computations.
        child = self._target_executors[None][0]
        embedded_comp = await child.create_value(comp.internal_representation,
                                                 comp_type_signature)
        if arg is not None:
          embedded_arg = await executor_utils.delegate_entirely_to_executor(
              arg.internal_representation, arg.type_signature, child)
        else:
          embedded_arg = None
        result = await child.create_call(embedded_comp, embedded_arg)
        return TrustedAggregatingExecutorValue(result, result.type_signature)
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
        return await coro(arg)
      else:
        child_coro = getattr(
            self._target_executors[None][0],
            '_compute_intrinsic_{}'.format(comp.internal_representation.uri),
            None)
        if child_coro is not None:
          import pdb; pdb.set_trace()
          result = await child_coro(arg)
          import pdb; pdb.set_trace()
          return TrustedAggregatingExecutorValue([result], result.type_signature)
        else:
          raise NotImplementedError(
              'Support for intrinsic "{}" has not been implemented yet.'.format(
                  comp.internal_representation.uri))
    else:
      raise ValueError('Calling objects of type {} is unsupported.'.format(
          py_typecheck.type_string(type(comp.internal_representation))))

  @tracing.trace
  async def create_tuple(self, elements):
    # import pdb; pdb.set_trace()
    elem = anonymous_tuple.to_elements(anonymous_tuple.from_container(elements))
    for _, v in elem:
      py_typecheck.check_type(v, TrustedAggregatingExecutorValue)
    return TrustedAggregatingExecutorValue(
        anonymous_tuple.AnonymousTuple(
            (k, v.internal_representation) for k, v in elem),
        computation_types.NamedTupleType(
            (k, v.type_signature) if k else v.type_signature for k, v in elem))

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    # import pdb; pdb.set_trace()
    py_typecheck.check_type(source, TrustedAggregatingExecutorValue)
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
      return TrustedAggregatingExecutorValue(selected, source.type_signature[index])
    elif isinstance(source.internal_representation,
                    executor_value_base.ExecutorValue):
      if type_utils.type_tree_contains_types(source.type_signature,
                                             computation_types.FederatedType):
        raise ValueError(
            'TrustedAggregatingExecutorValue {} has violated its contract; '
            'it is embedded in another executor and yet its type '
            'has placement. The embedded value is {}, with type '
            'signature {}.'.format(source, source.internal_representation,
                                   source.type_signature))
      val = source.internal_representation
      child = self._target_executors[None][0]
      return TrustedAggregatingExecutorValue(
          await child.create_selection(val, index=index),
          source.type_signature[index])
    else:
      raise ValueError('Unexpected internal representation while creating '
                       'selection. Expected one of `AnonymousTuple` or value '
                       'embedded in target executor, received {}'.format(
                           source.internal_representation))

  def _check_arg_is_anonymous_tuple(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_type(arg.internal_representation,
                            anonymous_tuple.AnonymousTuple)

  @tracing.trace
  async def _compute_intrinsic_federated_reduce(self, arg):
    # import pdb; pdb.set_trace()
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
    type_utils.check_equivalent_types(
        op_type, type_factory.reduction_op(zero_type, item_type))

    import pdb; pdb.set_trace()
    val = arg.internal_representation[0]
    py_typecheck.check_type(val, list)
    aggregator_child = self._target_executors[AGGREGATOR][0]
    server_child = self._target_executors[placement_literals.SERVER][0]

    async def _move(v, target):
      return await target.create_value(await v.compute(), item_type)

    # move reduce arguments to aggregator
    item = _move(val[0], aggregator_child)

    # zero = await aggregator_child.create_value(
    #     await (await self.create_selection(arg, index=1)).compute(), zero_type)
    # op = await aggregator_child.create_value(arg.internal_representation[2], op_type)

    # result = zero
    # for item in items:
    #   # compute result on aggregator
    #   result = await aggregator_child.create_call(
    #       op, await aggregator_child.create_tuple(
    #           anonymous_tuple.AnonymousTuple([(None, result), (None, item)])))

    # # move result to SERVER
    # server_result = await asyncio.gather(_move(result, server_child))
    server_result = await asyncio.gather(item)

    # create the server's federated value
    return TrustedAggregatingExecutorValue(
        server_result,
        computation_types.FederatedType(
            server_result[0].type_signature,
            placement_literals.SERVER,
            all_equal=True))
