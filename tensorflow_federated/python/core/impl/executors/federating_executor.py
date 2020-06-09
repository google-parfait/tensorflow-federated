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

import abc
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
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization


class FederatingExecutorValue(executor_value_base.ExecutorValue):
  """A value embedded in a `FederatedExecutor`."""

  def __init__(self, value, type_signature):
    """Creates an embedded instance of a value in this executor.

    The kinds of supported `value`s or internal representations and
    `type_signature` are:

    * An instance of `intrinsic_defs.IntrinsicDef` in case of a federated
      operator (to be interpreted by this executor upon invocation).

    * An instance of `placement_literals.PlacementLiteral`.

    * An instance of `pb.Computation` in an unparsed form (to be relayed to one
      of the executors responsible for the given placement later on), which
      must be of one of the following varieties: call, intrinsic, lambda,
      placement,  selection, tensorflow, and tuple.

    * A Python `list` of values embedded in subordinate executors in the case
      where `type_signature` is a federated type. The `value` must be a list
      even if it is of an `all_equal` type or if there is only a single
      participant associated with the given placement.

    * A single value embedded in a subordinate executor in the case where
      `type_signature` is of a non-federated and non-functional type.

    * An instance of `anonymous_tuple.AnonymousTuple` with values being one of
      the supported types listed above.

    Note: This constructor does not check that the `value` and `type_signature`
    are of a kind supported by the `FederatingExecutor`.

    Args:
      value: An internal value representation (one of the allowed types,
        defined above).
      type_signature: An instance of `tff.Type` or something convertible to it
        that is compatible with `value` (as defined above).

    Raises:
      TypeError: If `type_spec` is not a `tff.Type`.
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
          FederatingExecutorValue(v, t).compute()
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
        return await asyncio.gather(*[v.compute() for v in self._value])
    else:
      raise RuntimeError(
          'Computing values of type {} represented as {} is not supported in '
          'this executor.'.format(self._type_signature,
                                  py_typecheck.type_string(type(self._value))))


class FederatingStrategy(abc.ABC):
  """Base class for intrinsic strategies.

  Defines the contract between FederatingStrategy implementations and
  FederatingExecutor. Includes some basic functionality that should be common
  to all intrinsic strategies, e.g. placement and mapping functions.
  See concrete implementations for more details.
  """

  def __init__(self, federating_executor):
    self.federating_executor = federating_executor

  @classmethod
  @abc.abstractmethod
  def validate_executor_placements(cls, executor_placements):
    pass


class FederatingExecutor(executor_base.Executor):
  """The federated executor orchestrates federated computations.

  The intrinsics currently implemented include:

  * federated_aggregate
  * federated_apply
  * federated_broadcast
  * federated_collect
  * federated_eval
  * federated_map
  * federated_mean
  * federated_reduce
  * federated_sum
  * federated_value
  * federated_weighted_mean
  * federated_zip

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

  def __init__(self,
               target_executors,
               strategy):
    """Creates a federated executor backed by a collection of target executors.

    Args:
      target_executors: A dictionary mapping placements to executors or lists of
        executors associated with these placements. The keys in this dictionary
        can be either placement literals, or `None` to specify the executor for
        unplaced computations. The values can be either single executors (if
        there only is a single participant associated with that placement, as
        would typically be the case with `tff.SERVER`) or lists of target
        executors.
      strategy: A callable mapping the current executor instance to
        an instantiation of a FederatingStrategy implementation.

    Raises:
      ValueError: If the target_executors are improper for the given
        strategy.
    """
    py_typecheck.check_callable(strategy)
    self._strategy = strategy(self)
    py_typecheck.check_type(self._strategy, FederatingStrategy)
    self._strategy.validate_executor_placements(target_executors)

    self._target_executors = {}
    for k, v in target_executors.items():
      # v is either an Executor or a list of Executors
      if isinstance(v, executor_base.Executor):
        self._target_executors[k] = [v]
      else:
        self._target_executors[k] = v.copy()

  def close(self):
    for p, v in self._target_executors.items():
      for e in v:
        logging.debug('Closing child executor for placement: %s', p)
        e.close()

  @tracing.trace(stats=False)
  async def create_value(self, value, type_spec=None):
    """A coroutine that creates embedded value from `value` of type `type_spec`.

    See the `FederatingExecutorValue` for detailed information about the
    `value`s and `type_spec`s that can be embedded using `create_value`.

    Args:
      value: An object that represents the value to embed within the executor.
      type_spec: An optional `tff.Type` of the value represented by this object,
        or something convertible to it.

    Returns:
      An instance of `FederatingExecutorValue` that represents the embedded
      value.

    Raises:
      TypeError: If the `value` and `type_spec` do not match.
      ValueError: If `value` is not a kind recognized by the
        `FederatingExecutor`.
    """
    type_spec = computation_types.to_type(type_spec)
    if isinstance(type_spec, computation_types.FederatedType):
      self._check_executor_compatible_with_placement(type_spec.placement)
    elif (isinstance(type_spec, computation_types.FunctionType) and
          isinstance(type_spec.result, computation_types.FederatedType)):
      self._check_executor_compatible_with_placement(type_spec.result.placement)
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      if not type_analysis.is_concrete_instance_of(type_spec,
                                                   value.type_signature):
        raise TypeError('Incompatible type {} used with intrinsic {}.'.format(
            type_spec, value.uri))
      return FederatingExecutorValue(value, type_spec)
    elif isinstance(value, placement_literals.PlacementLiteral):
      if type_spec is None:
        type_spec = computation_types.PlacementType()
      else:
        py_typecheck.check_type(type_spec, computation_types.PlacementType)
      return FederatingExecutorValue(value, type_spec)
    elif isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          type_utils.reconcile_value_with_type_spec(value, type_spec))
    elif isinstance(value, pb.Computation):
      deserialized_type = type_serialization.deserialize_type(value.type)
      if type_spec is None:
        type_spec = deserialized_type
      else:
        type_analysis.check_assignable_from(type_spec, deserialized_type)
      which_computation = value.WhichOneof('computation')
      if which_computation in ['lambda', 'tensorflow']:
        return FederatingExecutorValue(value, type_spec)
      elif which_computation == 'intrinsic':
        intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(value.intrinsic.uri)
        if intrinsic_def is None:
          raise ValueError('Encountered an unrecognized intrinsic "{}".'.format(
              value.intrinsic.uri))
        return await self.create_value(intrinsic_def, type_spec)
      else:
        raise ValueError(
            'Unsupported computation building block of type "{}".'.format(
                which_computation))
    elif isinstance(type_spec, computation_types.FederatedType):
      self._check_value_compatible_with_placement(value, type_spec.placement,
                                                  type_spec.all_equal)
      children = self._target_executors[type_spec.placement]
      if type_spec.all_equal:
        value = [value for _ in children]
      results = await asyncio.gather(*[
          c.create_value(v, type_spec.member) for v, c in zip(value, children)
      ])
      return FederatingExecutorValue(results, type_spec)
    else:
      child = self._target_executors[None][0]
      return FederatingExecutorValue(await child.create_value(value, type_spec),
                                     type_spec)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    """A coroutine that creates a call to `comp` with optional argument `arg`.

    Args:
      comp: The computation to invoke.
      arg: An optional argument of the call, or `None` if no argument was
        supplied.

    Returns:
      An instance of `FederatingExecutorValue` that represents the constructed
      call.

    Raises:
      TypeError: If `comp` or `arg` are not embedded in the executor, before
        calling `create_call` or if the `type_signature` of `arg` do not match
        the expected `type_signature` of the parameter to `comp`.
      ValueError: If `comp` is not a functional kind recognized by
        `FederatingExecutor` or if `comp` is a lambda with an argument.
      NotImplementedError: If `comp` is an intrinsic and it has not been
        implemented by the `FederatingExecutor`.
    """
    py_typecheck.check_type(comp, FederatingExecutorValue)
    if arg is not None:
      py_typecheck.check_type(arg, FederatingExecutorValue)
      py_typecheck.check_type(comp.type_signature,
                              computation_types.FunctionType)
      param_type = comp.type_signature.parameter
      type_analysis.check_assignable_from(param_type, arg.type_signature)
      arg = FederatingExecutorValue(arg.internal_representation, param_type)
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
        return FederatingExecutorValue(result, result.type_signature)
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
    """A coroutine that creates a tuple of `elements`.

    Args:
      elements: A collection of `FederatingExecutorValue`s to create a tuple
        from.

    Returns:
      An instance of `FederatingExecutorValue` that represents the constructed
      tuple.

    Raises:
      TypeError: If the `elements` are not embedded in the executor, before
        calling `create_call`.
    """
    element_values = []
    element_types = []
    for name, value in anonymous_tuple.iter_elements(
        anonymous_tuple.from_container(elements)):
      py_typecheck.check_type(value, FederatingExecutorValue)
      element_values.append((name, value.internal_representation))
      if name is not None:
        element_types.append((name, value.type_signature))
      else:
        element_types.append(value.type_signature)
    value = anonymous_tuple.AnonymousTuple(element_values)
    type_signature = computation_types.NamedTupleType(element_types)
    return FederatingExecutorValue(value, type_signature)

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    """A coroutine that creates a selection from `source`.

    Args:
      source: The source to select from.
      index: An optional integer index. Either this, or `name` must be present.
      name: An optional string name. Either this, or `index` must be present.

    Returns:
      An instance of `FederatingExecutorValue` that represents the constructed
      selection.

    Raises:
      TypeError: If `source` is not embedded in the executor, before calling
        `create_call` or if `source` is not a `tff.NamedTupleType`.
      ValueError: If both `index` and `name` are `None` of if `source` is not a
        kind recognized by `FederatingExecutor`.
    """
    py_typecheck.check_type(source, FederatingExecutorValue)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    if index is None and name is None:
      raise ValueError(
          'Expected either `index` or `name` to be specificed, found both are '
          '`None`.')
    if isinstance(source.internal_representation,
                  executor_value_base.ExecutorValue):
      child = self._target_executors[None][0]
      value = await child.create_selection(
          source.internal_representation, index=index, name=name)
      return FederatingExecutorValue(value, value.type_signature)
    elif isinstance(source.internal_representation,
                    anonymous_tuple.AnonymousTuple):
      if name is not None:
        value = source.internal_representation[name]
        type_signature = source.type_signature[name]
      else:
        value = source.internal_representation[index]
        type_signature = source.type_signature[index]
      return FederatingExecutorValue(value, type_signature)
    else:
      raise ValueError(
          'Unexpected internal representation while creating selection. '
          'Expected one of `AnonymousTuple` or value embedded in target '
          'executor, received {}'.format(source.internal_representation))

  def _check_executor_compatible_with_placement(self, placement):
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
    if all_equal:
      if isinstance(value, list):
        raise ValueError(
            'Expected a single value when \'all_equal\' is \'True\', found a '
            'list.')
    else:
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

  @tracing.trace
  async def _compute_intrinsic_federated_aggregate(self, arg):
    return await self._strategy.federated_aggregate(arg)
  
  @tracing.trace
  async def _compute_intrinsic_federated_apply(self, arg):
    return await self._strategy.federated_apply(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_broadcast(self, arg):
    return await self._strategy.federated_broadcast(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_collect(self, arg):
    return await self._strategy.federated_collect(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_eval_at_clients(self, arg):
    return await self._strategy.federated_eval_at_clients(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_eval_at_server(self, arg):
    return await self._strategy.federated_eval_at_server(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_map(self, arg):
    return await self._strategy.federated_map(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_map_all_equal(self, arg):
    return await self._strategy.federated_map_all_equal(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_mean(self, arg):
    return await self._strategy.federated_mean(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_reduce(self, arg):
    return await self._strategy.federated_reduce(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_secure_sum(self, arg):
    return await self._strategy.federated_secure_sum(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_sum(self, arg):
    return await self._strategy.federated_sum(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_value_at_clients(self, arg):
    return await self._strategy.federated_value_at_clients(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_value_at_server(self, arg):
    return await self._strategy.federated_value_at_server(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_weighted_mean(self, arg):
    return await self._strategy.federated_weighted_mean(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_zip_at_clients(self, arg):
    return await self._strategy.federated_zip_at_clients(arg)

  @tracing.trace
  async def _compute_intrinsic_federated_zip_at_server(self, arg):
    return await self._strategy.federated_zip_at_server(arg)
