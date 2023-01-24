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
          |   +--+---------+   | executor |
          |      |             +----------+
+---------+--+   |
| Federating +<--+
| Strategy   |
+------------+

A `FederatingExecutor`:

* Only implements the `executor_base.Executor` API and therefore creates
  values, calls, tuples, and selections, and does not implement the logic for
  handling federated types and intrinsics.
* Delegates handling unplaced types, computations, and processing to the
  `unplaced_executor`.
* Delegates handling federated types and intrinsics to a
  `FederatingStrategy`; for the purposes of reasoning about the relationships
  of the objects, the `FederatingExecutor` owns or is the parent to the
  `FederatingStrategy`.

A `FederatingStrategy`:

* Only implements the logic for handling federated types and intrinsics, and
  does not implement the `executor_base.Executor` API.
* Delegates handling unplaced types, computations, and processing back to the
  `FederatingExecutor`. For example, in order to handle some federated
  intrinics it might be useful to create a tuple or selection, this unplaced
  processing is an example of something that is delegated back to the
  `FederatingExecutor`.

Note: Neither the `FederatingExecutor` nor the `FederatingStrategy` handle
handling non-federated intrinsics.
"""

import abc
from collections.abc import Callable
from typing import Any, Optional

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_serialization


class FederatingStrategy(abc.ABC):
  """The abstract interface federating strategies must implement.

  A `FederatingStrategy` defines the logic for how a `FederatingExecutor`
  handles federated types and federated intrinsics, specificially:

  * which federated intrinsics are implemented
  * how federated types and federated intrinsics are implemented
  * how federated values are represented in the execution stack
  * how work is delegated to the target executors
  * which placements are implemented
  """

  def __init__(self, executor: 'FederatingExecutor'):
    """Creates a `FederatingStrategy`.

    Args:
      executor: A `FederatingExecutor` to use to handle unplaced types,
        computations, and processing.

    Raises:
      TypeError: If `executor` is not a `FederatingExecutor`.
    """
    py_typecheck.check_type(executor, FederatingExecutor)
    self._executor = executor

  @abc.abstractmethod
  def close(self):
    """Release resources associated with this strategy, if any."""
    raise NotImplementedError()

  @abc.abstractmethod
  def ingest_value(
      self, value: Any, type_signature: computation_types.Type
  ) -> executor_value_base.ExecutorValue:
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_value(
      self, value: Any, type_signature: computation_types.Type
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded value for a federated type.

    Args:
      value: An object to embed in the executor.
      type_signature: A `tff.Type`, the type of `value`.
    """
    raise NotImplementedError()

  async def compute_federated_intrinsic(
      self, uri: str, arg: Optional[executor_value_base.ExecutorValue] = None
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated intrinsic.

    Args:
      uri: The URI of an intrinsic to embed.
      arg: An optional embedded argument of the call, or `None` if no argument
        is supplied.
    """
    # Note: Relying on the names of the methods in order to select the function
    # responsible for handling the given URI is safe because this abstract
    # interface forces subclasses to explicitly implement all of the intrinsics
    # with specificly named methods. In other words, this convenience is safe
    # because this abstract interface owns the names of the methods.
    fn = getattr(self, 'compute_{}'.format(uri), None)
    if fn is not None:
      return await fn(arg)  # pylint: disable=not-callable
    else:
      raise NotImplementedError(
          "The intrinsic '{}' is not implemented.".format(uri)
      )

  @abc.abstractmethod
  async def compute_federated_aggregate(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated aggregate."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_apply(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated apply."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_broadcast(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated broadcast."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_eval_at_clients(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated eval at clients."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_eval_at_server(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated eval at server."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_map(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated map."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_map_all_equal(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated map all equal."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_mean(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated mean."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_secure_sum_bitwidth(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated secure sum."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_secure_select(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated secure select."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_select(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated select."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_sum(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated sum."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_value_at_clients(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated value at clients."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_value_at_server(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated value at server."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_weighted_mean(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated weighted mean."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_zip_at_clients(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated zip at clients."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def compute_federated_zip_at_server(
      self, arg: executor_value_base.ExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Returns an embedded call for a federated zip at server."""
    raise NotImplementedError()


class FederatingExecutor(executor_base.Executor):
  """An executor for handling federated types and intrinsics.

  At a high-level, this executor is responsible for handling federated types
  and federated intrinsics and delegating work to an underlying collection of
  target executors associated with individual placements.

  The responsibility for handling federated types and federated intrinsics is
  delegated to a `FederatingStrategy`, specifically:

  * which federated intrinsics are implemented
  * how federated types and federated intrinsics are implemented
  * how federated values are represented in the execution stack
  * how work is delegated to the target executors
  * which placements are implemented

  The reponsibility for handling unplaced types, computations, and processing
  is delegated to an `unplaced_executor`.

  Expressed in a different way, this executor integrates a `FederatedStrategy`
  with an execution stack, providing a way to inject the logic for handling
  federated types and intrinsics into the execution stack.
  """

  _FORWARDED_INTRINSICS = [
      intrinsic_defs.SEQUENCE_MAP.uri,
      intrinsic_defs.SEQUENCE_REDUCE.uri,
      intrinsic_defs.SEQUENCE_SUM.uri,
  ]

  def __init__(
      self,
      strategy_factory: Callable[['FederatingExecutor'], FederatingStrategy],
      unplaced_executor: executor_base.Executor,
  ):
    """Creates a `FederatingExecutor` backed by a `FederatingStrategy`.

    Args:
      strategy_factory: A function that accepts an instance of a
        `FederatingExecutor` and returns a newly constructed
        `FederatingStrategy` to use to handle federated types and intrinsics.
      unplaced_executor: An `executor_base.Executor` to use to handle unplaced
        types, computations, and processing.
    """
    py_typecheck.check_type(unplaced_executor, executor_base.Executor)
    self._strategy = strategy_factory(self)
    self._unplaced_executor = unplaced_executor

  def close(self):
    self._strategy.close()
    self._unplaced_executor.close()

  @tracing.trace(stats=False)
  async def create_value(
      self, value: Any, type_spec: Any = None
  ) -> executor_value_base.ExecutorValue:
    """Creates an embedded value from the given `value` and `type_spec`.

    The kinds of supported `value`s are:

    * An instance of `intrinsic_defs.IntrinsicDef`.

    * An instance of `placements.PlacementLiteral`.

    * An instance of `pb.Computation` if of one of the following kinds:
      intrinsic, lambda, tensorflow, xla, or data.

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
      An instance of `executor_value_base.ExecutorValue` representing a value
      embedded in the `FederatingExecutor` using a particular
      `FederatingStrategy`.

    Raises:
      TypeError: If the `value` and `type_spec` do not match.
      ValueError: If `value` is not a kind supported by the
        `FederatingExecutor`.
    """
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      type_analysis.check_concrete_instance_of(type_spec, value.type_signature)
      return self._strategy.ingest_value(value, type_spec)
    elif isinstance(value, placements.PlacementLiteral):
      if type_spec is None:
        type_spec = computation_types.PlacementType()
      type_spec.check_placement()
      return self._strategy.ingest_value(value, type_spec)
    elif isinstance(value, computation_impl.ConcreteComputation):
      return await self.create_value(
          computation_impl.ConcreteComputation.get_proto(value),
          executor_utils.reconcile_value_with_type_spec(value, type_spec),
      )
    elif isinstance(value, pb.Computation):
      deserialized_type = type_serialization.deserialize_type(value.type)
      if type_spec is None:
        type_spec = deserialized_type
      else:
        type_spec.check_assignable_from(deserialized_type)
      which_computation = value.WhichOneof('computation')
      if which_computation in ['lambda', 'tensorflow', 'xla', 'data']:
        return self._strategy.ingest_value(value, type_spec)
      elif which_computation == 'intrinsic':
        if value.intrinsic.uri in FederatingExecutor._FORWARDED_INTRINSICS:
          return self._strategy.ingest_value(value, type_spec)
        intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(value.intrinsic.uri)
        if intrinsic_def is None:
          raise ValueError(
              'Encountered an unrecognized intrinsic "{}".'.format(
                  value.intrinsic.uri
              )
          )
        return await self.create_value(intrinsic_def, type_spec)
      else:
        raise ValueError(
            'Unsupported computation building block of type "{}".'.format(
                which_computation
            )
        )
    elif type_spec is not None and type_spec.is_federated():
      return await self._strategy.compute_federated_value(value, type_spec)
    else:
      result = await self._unplaced_executor.create_value(value, type_spec)
      return self._strategy.ingest_value(result, type_spec)

  @tracing.trace
  async def create_call(
      self,
      comp: executor_value_base.ExecutorValue,
      arg: Optional[executor_value_base.ExecutorValue] = None,
  ) -> executor_value_base.ExecutorValue:
    """Creates an embedded call for the given `comp` and optional `arg`.

    The kinds of supported `comp`s are:

    * An instance of `pb.Computation` if of one of the following kinds:
      tensorflow, xla.
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
    py_typecheck.check_type(comp, executor_value_base.ExecutorValue)
    if arg is not None:
      py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
      py_typecheck.check_type(
          comp.type_signature, computation_types.FunctionType
      )
      param_type = comp.type_signature.parameter
      param_type.check_assignable_from(arg.type_signature)
      arg = self._strategy.ingest_value(arg.reference, param_type)
    if isinstance(comp.reference, pb.Computation):
      which_computation = comp.reference.WhichOneof('computation')
      if (which_computation in ['tensorflow', 'xla', 'intrinsic']) or (
          (which_computation == 'intrinsic')
          and (
              comp.reference.intrinsic.uri
              in FederatingExecutor._FORWARDED_INTRINSICS
          )
      ):
        embedded_comp = await self._unplaced_executor.create_value(
            comp.reference, comp.type_signature
        )
        if arg is not None:
          embedded_arg = await executor_utils.delegate_entirely_to_executor(
              arg.reference, arg.type_signature, self._unplaced_executor
          )
        else:
          embedded_arg = None
        result = await self._unplaced_executor.create_call(
            embedded_comp, embedded_arg
        )
        return self._strategy.ingest_value(result, result.type_signature)
      else:
        raise ValueError(
            'Directly calling computations of type {} is unsupported.'.format(
                which_computation
            )
        )
    elif isinstance(comp.reference, intrinsic_defs.IntrinsicDef):
      return await self._strategy.compute_federated_intrinsic(
          comp.reference.uri, arg
      )
    else:
      raise ValueError(
          'Calling objects of type {} is unsupported.'.format(
              py_typecheck.type_string(type(comp.reference))
          )
      )

  @tracing.trace
  async def create_struct(
      self, elements: list[executor_value_base.ExecutorValue]
  ) -> executor_value_base.ExecutorValue:
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
    for name, value in structure.iter_elements(
        structure.from_container(elements)
    ):
      py_typecheck.check_type(value, executor_value_base.ExecutorValue)
      element_values.append((name, value.reference))
      if name is not None:
        element_types.append((name, value.type_signature))
      else:
        element_types.append(value.type_signature)
    value = structure.Struct(element_values)
    type_signature = computation_types.StructType(element_types)
    return self._strategy.ingest_value(value, type_signature)

  @tracing.trace
  async def create_selection(
      self, source: executor_value_base.ExecutorValue, index: int
  ) -> executor_value_base.ExecutorValue:
    """Creates an embedded selection from the given `source`.

    The kinds of supported `source`s are:

    * An embedded value.
    * An instance of `structure.Struct`.

    Args:
      source: An embedded computation with a tuple type signature representing
        the source from which to make a selection.
      index: An integer index.

    Returns:
      An instance of `executor_value_base.ExecutorValue` representing the
      embedded selection.

    Raises:
      TypeError: If `source` is not embedded in the executor or if the
        `type_signature` of `source` is not a `tff.StructType`.
      ValueError: If both `index` and `name` are `None` or if `source` is not a
        kind supported by the `FederatingExecutor`.
    """
    py_typecheck.check_type(source, executor_value_base.ExecutorValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    if isinstance(source.reference, executor_value_base.ExecutorValue):
      result = await self._unplaced_executor.create_selection(
          source.reference, index
      )
      return self._strategy.ingest_value(result, result.type_signature)
    elif isinstance(source.reference, structure.Struct):
      value = source.reference[index]
      type_signature = source.type_signature[index]
      return self._strategy.ingest_value(value, type_signature)
    else:
      raise ValueError(
          'Unexpected internal representation while creating selection. '
          'Expected one of `Struct` or value embedded in target '
          'executor, received {}'.format(source.reference)
      )
