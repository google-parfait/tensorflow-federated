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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import asyncio

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_value_base
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_constructors
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils


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
    if isinstance(self._value, executor_value_base.Value):
      return await self._value.compute()
    elif isinstance(self._type_signature, computation_types.FederatedType):
      py_typecheck.check_type(self._value, list)
      results = []
      for v in self._value:
        py_typecheck.check_type(v, executor_value_base.Value)
        results.append(v.compute())
      return await asyncio.gather(*results)
    else:
      raise RuntimeError('Computing values of type {} represented as {} is not '
                         'supported in this executor.'.format(
                             str(self._type_signature),
                             py_typecheck.type_string(type(self._value))))


class FederatedExecutor(executor_base.Executor):
  """The federated executor orchestrates federated computations.

  WARNING: This executor is not functional enough to use it. The intrinsics
  currently implemented include:
  - federated_apply
  - federated_broadcast
  - federated_map
  - federated_value_at_clients
  - federated_value_at_server
  - federated_zip_at_clients
  - federated_zip_at_server

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
                  str(pl), str(pl_cardinality)))

  async def create_value(self, value, type_spec=None):
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      if not type_utils.is_concrete_instance_of(type_spec,
                                                value.type_signature):
        raise TypeError('Incompatible type {} used with intrinsic {}.'.format(
            str(type_spec), value.uri))
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
            'Uncountered a value of a functional TFF type {} and Python type '
            '{} that is not of one of the recognized representations.'.format(
                str(type_spec), py_typecheck.type_string(type(value))))
      elif isinstance(type_spec, computation_types.FederatedType):
        children = self._target_executors.get(type_spec.placement)
        if not children:
          raise ValueError(
              'Placement "{}" is not configured in this executor.'.format(
                  str(type_spec.placement)))
        py_typecheck.check_type(children, list)
        if not type_spec.all_equal:
          py_typecheck.check_type(value, list)
        elif isinstance(value, list):
          raise ValueError(
              'An all_equal value should be passed directly, not as a list.')
        else:
          value = [value for _ in children]
        if len(value) != len(children):
          raise ValueError(
              'Federated value contains {} items, but the placement {} in this '
              'executor is configured with {} participants.'.format(
                  len(value), str(type_spec.placement), len(children)))
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
    if isinstance(comp.internal_representation, pb.Computation):
      which_computation = comp.internal_representation.WhichOneof('computation')
      if which_computation == 'tensorflow':
        child = self._target_executors[None][0]
        embedded_comp = await child.create_value(comp.internal_representation,
                                                 comp.type_signature)
        result = await child.create_call(
            embedded_comp,
            arg.internal_representation if arg is not None else None)
        return FederatedExecutorValue(result, result.type_signature)
      else:
        raise ValueError(
            'Directly calling computations of type {} is unsupported.'.format(
                which_computation))
    if isinstance(comp.internal_representation, intrinsic_defs.IntrinsicDef):
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
    raise NotImplementedError

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
    results = await asyncio.gather(
        *[c.create_call(f, v) for c, (f, v) in zip(children, zip(fns, val))])
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
    return FederatedExecutorValue(
        new_vals,
        computation_types.FederatedType(
            computation_types.NamedTupleType([
                (k, v.member) if k else v.member
                for k, v in anonymous_tuple.to_elements(arg.type_signature)
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
        ]),
        type_constructors.at_clients(arg.type_signature.member, all_equal=True))

  async def _compute_intrinsic_federated_zip_at_server(self, arg):
    return await self._zip(arg, placement_literals.SERVER, all_equal=True)

  async def _compute_intrinsic_federated_zip_at_clients(self, arg):
    return await self._zip(arg, placement_literals.CLIENTS, all_equal=False)
