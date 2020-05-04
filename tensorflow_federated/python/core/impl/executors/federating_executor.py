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
"""An executor that handles federated types and federated operators.

                +------------+
                | Federating |   +----------+
            +-->+ Executor   +-->+ unplaced |
       weak |   +--+---------+   | executor |
            |      |             +----------+
  +---------+--+   | strong
  | Federating +<--+
  | Strategy   |
  +------------+

  A `FederatingExecutor`:

  * Only implements the `executor_base.Executor` API and therefore creates
    values, calls, tuples, and selections, and does not implement the logic for
    resolving federated types and intrinsics
  * Delegates resolving unplaced types, computations, and processing to the
    `unplaced_executor`.
  * Delegates resolving federated types and intrinsics to a
    `FederatingStrategy`.

  A `FederatingStrategy`:

  * Only implements the logic for resolving federated types and intrinsics, and
    does not implement the `executor_base.Executor` API.
  * Delegates resolving unplaced types, computations, and processing back to the
    `FederatingExecutor`. For example, in order to resolve some federated
    intrinics it might be useful to create a tuple or selection, this unplaced
    processing is an example of something that is delegated back to the
    `FederatingExecutor`.

  Note: Neither the `FederatingExecutor` nor the `FederatingStrategy` handle
  resolving non-federated intrinsics.
"""

import abc
import weakref

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
from tensorflow_federated.python.core.impl.types import type_serialization


class FederatingExecutor(executor_base.Executor):
  """An executor for resolving federated types and intrinsics.

  This executor is responsible for resolving federated types and federated
  intrinsics, and delegating work to an underlying collection of target
  executors associated with individual placements.

  Much of this responsiblity is delegate to the provided `strategy`:

  * which federated intrinsics are implemented
  * how federated types and federated intrinsics are implemented
  * how federated values are represented in the execution stack
  * how work is delegated to the target executors
  * which placements are implemented

  This executor integrates a `strategy` with an execution stack, providing a
  way to inject logic for resolving federated types and intrinsics into the
  execution stack.

  Additionally, this executor delegates resolving unplaced values and unplaced
  work to the given `unplaced_executor` and requires that lambdas; compositional
  constructs (blocks, etc.); and non-federated intrinscis have already been
  resolved.
  """

  def __init__(self, strategy, unplaced_executor):
    """Creates a `FederatingExecutor` backed by a `FederatingStrategy`.

    Args:
      strategy: A `FederatingStrategy` to use to resolve federated types and
        intrinsics.
      unplaced_executor: An `executor_base.Executor` to use to resolve unplaced
        types, computations, and processing.
    """
    # py_typecheck.check_type(strategy, FederatingStrategy)
    py_typecheck.check_type(unplaced_executor, executor_base.Executor)
    self._strategy = strategy(self)
    self._unplaced_executor = unplaced_executor

  def close(self):
    self._strategy.close()
    self._unplaced_executor.close()

  @tracing.trace(stats=False)
  async def create_value(self, value, type_spec=None):
    """Creates an embedded value from the given `value` and `type_spec`.

    The kinds of supported `value`s are:

    * An instance of `intrinsic_defs.IntrinsicDef`.

    * An instance of `placement_literals.PlacementLiteral`.

    * An instance of `pb.Computation` if of one of the following kinds:
      intrinsic, lambda, and tensorflow.

    * A Python `list` if `type_spec` is a federated type.

      Note: The `value` must be a list even if it is of an `all_equal` type or
      if there is only a single participant associated with the given placement.

    * A Python value if `type_spec` is a non-functional, non-federated type.

    Args:
      value: An object to embed in the executor, one of the supported types
        defined by above.
      type_spec: An optional type convertible to instance of `tff.Type` via
        `tff.to_type`, the type of `value`.

    Returns:
      An instance of `executor_value_base.ExecutorValue` representing the
      embedded value.

    Raises:
      TypeError: If the `value` and `type_spec` do not match.
      ValueError: If `value` is not a kind supported by the
        `FederatingExecutor`.
    """
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      if not type_analysis.is_concrete_instance_of(type_spec,
                                                   value.type_signature):
        raise TypeError('Incompatible type {} used with intrinsic {}.'.format(
            type_spec, value.uri))
      return self._strategy.value_type(value, type_spec)
    elif isinstance(value, placement_literals.PlacementLiteral):
      if type_spec is None:
        type_spec = computation_types.PlacementType()
      else:
        py_typecheck.check_type(type_spec, computation_types.PlacementType)
      return self._strategy.value_type(value, type_spec)
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
        return self._strategy.value_type(value, type_spec)
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
      return await self._strategy.create_federated_value(value, type_spec)
    else:
      result = await self._unplaced_executor.create_value(value, type_spec)
      return self._strategy.value_type(result, type_spec)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    """Creates an embedded call for the given `comp` and optional `arg`.

    The kinds of supported `comp`s are:

    * An instance of `pb.Computation` if of one of the following kinds:
      tensorflow.
    * An instance of `intrinsic_defs.IntrinsicDef`.

    Args:
      comp: An embedded computation with a functional type signature
        representing the function of the call.
      arg: An optional embedded argument of the call, or `None` if no argument
        is supplied.

    Returns:
      An instance of `executor_value_base.ExecutorValue` representing the
      embedded call.

    Raises:
      TypeError: If `comp` or `arg` are not embedded in the executor; if the
        `type_signature` of `comp` is not `tff.FunctionType`; or if the
        `type_signature`s of `comp` and `arg` are not compatible.
      ValueError: If `comp` is not a kind supported by the `FederatingExecutor`.
    """
    py_typecheck.check_type(comp, self._strategy.value_type)
    if arg is not None:
      py_typecheck.check_type(arg, self._strategy.value_type)
      py_typecheck.check_type(comp.type_signature,
                              computation_types.FunctionType)
      param_type = comp.type_signature.parameter
      type_analysis.check_assignable_from(param_type, arg.type_signature)
      arg = self._strategy.value_type(arg.internal_representation, param_type)
    if isinstance(comp.internal_representation, pb.Computation):
      which_computation = comp.internal_representation.WhichOneof('computation')
      if which_computation == 'tensorflow':
        embedded_comp = await self._unplaced_executor.create_value(
            comp.internal_representation, comp.type_signature)
        if arg is not None:
          embedded_arg = await executor_utils.delegate_entirely_to_executor(
              arg.internal_representation, arg.type_signature,
              self._unplaced_executor)
        else:
          embedded_arg = None
        result = await self._unplaced_executor.create_call(
            embedded_comp, embedded_arg)
        return self._strategy.value_type(result, result.type_signature)
      else:
        raise ValueError(
            'Directly calling computations of type {} is unsupported.'.format(
                which_computation))
    elif isinstance(comp.internal_representation, intrinsic_defs.IntrinsicDef):
      return await self._strategy.create_federated_intrinsic(
          comp.internal_representation.uri, arg)
    else:
      raise ValueError('Calling objects of type {} is unsupported.'.format(
          py_typecheck.type_string(type(comp.internal_representation))))

  @tracing.trace
  async def create_tuple(self, elements):
    """Creates an embedded tuple of the given `elements`.

    Args:
      elements: A collection of embedded values.

    Returns:
      An instance of `executor_value_base.ExecutorValue` representing the
      embedded tuple.

    Raises:
      TypeError: If the `elements` are not embedded in the executor.
    """
    element_values = []
    element_types = []
    for name, value in anonymous_tuple.iter_elements(
        anonymous_tuple.from_container(elements)):
      py_typecheck.check_type(value, self._strategy.value_type)
      element_values.append((name, value.internal_representation))
      if name is not None:
        element_types.append((name, value.type_signature))
      else:
        element_types.append(value.type_signature)
    value = anonymous_tuple.AnonymousTuple(element_values)
    type_signature = computation_types.NamedTupleType(element_types)
    return self._strategy.value_type(value, type_signature)

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    """Creates an embedded selection from the given `source`.

    The kinds of supported `source`s are:

    * An embedded value.
    * An instance of `anonymous_tuple.AnonymousTuple`.

    Args:
      source: An embedded computation with a tuple type signature representing
        the source from which to make a selection.
      index: An optional integer index. Either this, or `name` must be present.
      name: An optional string name. Either this, or `index` must be present.

    Returns:
      An instance of `executor_value_base.ExecutorValue` representing the
      embedded selection.

    Raises:
      TypeError: If `source` is not embedded in the executor or if the
        `type_signature` of `source` is not a `tff.NamedTupleType`.
      ValueError: If both `index` and `name` are `None` or if `source` is not a
        kind supported by the `FederatingExecutor`.
    """
    py_typecheck.check_type(source, self._strategy.value_type)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    if index is None and name is None:
      raise ValueError(
          'Expected either `index` or `name` to be specificed, found both are '
          '`None`.')
    if isinstance(source.internal_representation,
                  executor_value_base.ExecutorValue):
      result = await self._unplaced_executor.create_selection(
          source.internal_representation, index=index, name=name)
      return self._strategy.value_type(result, result.type_signature)
    elif isinstance(source.internal_representation,
                    anonymous_tuple.AnonymousTuple):
      if name is not None:
        value = source.internal_representation[name]
        type_signature = source.type_signature[name]
      else:
        value = source.internal_representation[index]
        type_signature = source.type_signature[index]
      return self._strategy.value_type(value, type_signature)
    else:
      raise ValueError(
          'Unexpected internal representation while creating selection. '
          'Expected one of `AnonymousTuple` or value embedded in target '
          'executor, received {}'.format(source.internal_representation))


class FederatingStrategy(abc.ABC):
  """The abstract interface federating strategies must implement.

  A federting strategy defines the logic for how a `FederatingExecutor` resolves
  federated types and federated intrinsics, specificially:

  * which federated intrinsics are implemented
  * how federated types and federated intrinsics are implemented
  * how federated values are represented in the execution stack
  * how work is delegated to the target executors
  * which placements are implemented
  """

  def __init__(self, executor):
    """Creates a `FederatingStrategy`.

    Args:
      executor: A weak reference to an `executor_base.Executor` to use to
        resolve unplaced types, computations, and processing.

    Raises:
      TypeError: If `executor` is not a `executor_base.Executor`.
    """
    py_typecheck.check_type(executor, executor_base.Executor)
    self._executor = weakref.proxy(executor)

  @property
  @abc.abstractmethod
  def value_type(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def close(self):
    """Release resources associated with this strategy, if any."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_value(self, value, type_spec):
    """Creates an embedded value for a federated type.

    Args:
      value: An object to embed in the executor.
      type_spec: A type convertible to instance of `tff.Type` via `tff.to_type`,
        the type of `value`.

    Returns:
      An instance of `executor_value_base.ExecutorValue` representing the
      embedded value.
    """
    raise NotImplementedError()

  async def create_federated_intrinsic(self, uri, arg=None):
    """Creates an embedded call for a federated intrinsic.

    Args:
      uri: The URI of an intrinsic to embed.
      arg: An optional embedded argument of the call, or `None` if no argument
        is supplied.

    Returns:
      An instance of `executor_value_base.ExecutorValue` representing the
      embedded call.
    """
    # Note: Relying on the names of the methods in order to select the function
    # responsible for resolving the given URI is safe because this abstract
    # interface forces subclasses to explicitly implement all of the intrinsics
    # with specificly named methods. In other words, this coneience is safe
    # because this abstract interface owns the names of the methods.
    fn = getattr(self, 'create_{}'.format(uri), None)
    if fn is not None:
      return await fn(arg)  # pylint: disable=not-callable
    else:
      raise NotImplementedError(
          'Support for intrinsic \'{}\' has not been implemented yet.'.format(
              uri))

  @abc.abstractmethod
  async def create_federated_aggregate(self, arg):
    """Returns an embedded call for a federated aggregate."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_apply(self, arg):
    """Returns an embedded call for a federated apply."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_broadcast(self, arg):
    """Returns an embedded call for a federated broadcast."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_collect(self, arg):
    """Returns an embedded call for a federated collect."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_eval_at_clients(self, arg):
    """Returns an embedded call for a federated eval at clients."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_eval_at_server(self, arg):
    """Returns an embedded call for a federated eval at server."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_map(self, arg):
    """Returns an embedded call for a federated map."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_map_all_equal(self, arg):
    """Returns an embedded call for a federated map all equal."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_mean(self, arg):
    """Returns an embedded call for a federated mean."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_reduce(self, arg):
    """Returns an embedded call for a federated reduce."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_secure_sum(self, arg):
    """Returns an embedded call for a federated secure sum."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_sum(self, arg):
    """Returns an embedded call for a federated sum."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_value_at_clients(self, arg):
    """Returns an embedded call for a federated value at clients."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_value_at_server(self, arg):
    """Returns an embedded call for a federated value at server."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_weighted_mean(self, arg):
    """Returns an embedded call for a federated weighted mean."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_zip_at_clients(self, arg):
    """Returns an embedded call for a federated zip at clients."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def create_federated_zip_at_server(self, arg):
    """Returns an embedded call for a federated zip at server."""
    raise NotImplementedError()
