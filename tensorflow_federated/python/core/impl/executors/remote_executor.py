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
"""A local proxy for a remote executor service hosted on a separate machine."""

import asyncio
from collections.abc import Mapping
import weakref

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import remote_executor_stub
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class RemoteValue(executor_value_base.ExecutorValue):
  """A reference to a value embedded in a remotely deployed executor service."""

  def __init__(
      self,
      value_ref: executor_pb2.ValueRef,
      type_spec,
      executor,
      dispose_at_exit: bool = True,
  ):
    """Creates the value.

    Args:
      value_ref: An instance of `executor_pb2.ValueRef` returned by the remote
        executor service.
      type_spec: An instance of `computation_types.Type`.
      executor: The executor that created this value.
      dispose_at_exit: The flag to disable calling dispose on the object at
        deletion.
    """
    py_typecheck.check_type(value_ref, executor_pb2.ValueRef)
    py_typecheck.check_type(type_spec, computation_types.Type)
    py_typecheck.check_type(executor, RemoteExecutor)
    self._value_ref = value_ref
    self._type_signature = type_spec
    self._executor = executor

    # Clean up the value and the memory associated with it on the remote
    # worker when no references to it remain.
    def finalizer(value_ref, executor, executor_id):
      executor._dispose(value_ref, executor_id)  # pylint: disable=protected-access

    f = weakref.finalize(
        self, finalizer, value_ref, executor, executor._executor_id
    )
    f.atexit = dispose_at_exit

  @property
  def type_signature(self):
    return self._type_signature

  @property
  def reference(self):
    return self._value_ref

  @tracing.trace(span=True)
  async def compute(self):
    return await self._executor._compute(self._value_ref, self._type_signature)  # pylint: disable=protected-access


class RemoteExecutor(executor_base.Executor):
  """The remote executor is a local proxy for a remote executor instance."""

  # TODO: b/134543154 - Switch to using an asynchronous gRPC client so we don't
  # have to block on all those calls.

  def __init__(
      self,
      stub: remote_executor_stub.RemoteExecutorStub,
      dispose_batch_size=20,
      stream_structs: bool = False,
  ):
    """Creates a remote executor.

    Args:
      stub: An instance of stub used for communication with the remote executor
        service.
      dispose_batch_size: The batch size for requests to dispose of remote
        worker values. Lower values will result in more requests to the remote
        worker, but will result in values being cleaned up sooner and therefore
        may result in lower memory usage on the remote worker.
      stream_structs: The flag to enable decomposing and streaming struct
        values.
    """

    py_typecheck.check_type(dispose_batch_size, int)

    logging.debug('Creating new ExecutorStub')

    # We need to keep a reference to the channel around to prevent the Python
    # object from being GC'ed and the callback above from no-op'ing.
    self._stub = stub
    self._executor_id = None
    self._dispose_request = None
    self._dispose_batch_size = dispose_batch_size
    self._stream_structs = stream_structs

  def close(self):
    logging.debug('Clearing executor state on server.')
    self._clear_executor()

  def _check_has_executor_id(self):
    if self._executor_id is None:
      raise ValueError(
          'Attempted to use a `RemoteExecutor` without first calling '
          '`set_cardinalities` after creation or after a call to `close()`.'
      )

  def _dispose(
      self,
      value_ref: executor_pb2.ValueRef,
      value_executor_id: executor_pb2.ExecutorId,
  ):
    """Disposes of the remote value stored on the worker service."""
    if value_executor_id != self._executor_id:
      # The executor this value corresponds to was already disposed, so we can
      # skip disposing this value.
      return

    assert self._dispose_request is not None
    self._dispose_request.value_ref.append(value_ref)
    if len(self._dispose_request.value_ref) < self._dispose_batch_size:
      return
    dispose_request = self._dispose_request
    self._dispose_request = executor_pb2.DisposeRequest(
        executor=self._executor_id
    )
    self._stub.dispose(dispose_request)

  @tracing.trace(span=True)
  def set_cardinalities(
      self, cardinalities: Mapping[placements.PlacementLiteral, int]
  ):
    if self._executor_id is not None:
      self._clear_executor()
    serialized_cardinalities = value_serialization.serialize_cardinalities(
        cardinalities
    )
    request = executor_pb2.GetExecutorRequest(
        cardinalities=serialized_cardinalities
    )
    self._executor_id = self._stub.get_executor(request).executor
    self._dispose_request = executor_pb2.DisposeRequest(
        executor=self._executor_id
    )

  @tracing.trace(span=True)
  def _clear_executor(self):
    if self._executor_id is None:
      return
    request = executor_pb2.DisposeExecutorRequest(executor=self._executor_id)
    try:
      self._stub.dispose_executor(request)
    except (grpc.RpcError, executors_errors.RetryableError):
      logging.debug(
          'RPC error caught during attempt to clear state on the '
          'server; this likely indicates a broken connection, and '
          'therefore there is no state to clear.'
      )
    self._executor_id = None
    self._dispose_request = None
    return

  @tracing.trace(span=True)
  async def create_value_stream_structs(
      self, value, type_spec: computation_types.StructType
  ):
    value = structure.from_container(value)
    if len(value) != len(type_spec):
      raise TypeError(
          'Value {} does not match type {}: mismatching tuple length.'.format(
              value, type_spec
          )
      )

    value_refs = []
    for (value_elem_name, value_elem), (type_elem_name, type_elem) in zip(
        structure.iter_elements(value), structure.iter_elements(type_spec)
    ):
      if value_elem_name not in [type_elem_name, None]:
        raise TypeError(
            'Value {} does not match type {}: mismatching tuple element '
            'names {} vs. {}.'.format(
                value, type_spec, value_elem_name, type_elem_name
            )
        )
      value_refs.append(self.create_value(value_elem, type_elem))
    value_refs = await asyncio.gather(*value_refs)
    return await self.create_struct(value_refs)

  @tracing.trace(span=True)
  async def create_value(self, value, type_spec=None):
    self._check_has_executor_id()

    @tracing.trace
    def serialize_value():
      return value_serialization.serialize_value(value, type_spec)

    if self._stream_structs and isinstance(
        type_spec, computation_types.StructType
    ):
      return await self.create_value_stream_structs(value, type_spec)

    value_proto, type_spec = serialize_value()
    create_value_request = executor_pb2.CreateValueRequest(
        executor=self._executor_id, value=value_proto
    )
    response = self._stub.create_value(create_value_request)
    py_typecheck.check_type(response, executor_pb2.CreateValueResponse)
    return RemoteValue(response.value_ref, type_spec, self)

  @tracing.trace(span=True)
  async def create_call(self, comp, arg=None):
    self._check_has_executor_id()
    py_typecheck.check_type(comp, RemoteValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    if arg is not None:
      py_typecheck.check_type(arg, RemoteValue)
    create_call_request = executor_pb2.CreateCallRequest(
        executor=self._executor_id,
        function_ref=comp.reference,
        argument_ref=(arg.reference if arg is not None else None),
    )
    response = self._stub.create_call(create_call_request)
    py_typecheck.check_type(response, executor_pb2.CreateCallResponse)
    return RemoteValue(response.value_ref, comp.type_signature.result, self)

  @tracing.trace(span=True)
  async def create_struct(self, elements):
    self._check_has_executor_id()
    constructed_anon_tuple = structure.from_container(elements)
    proto_elem = []
    type_elem = []
    for k, v in structure.iter_elements(constructed_anon_tuple):
      py_typecheck.check_type(v, RemoteValue)
      proto_elem.append(
          executor_pb2.CreateStructRequest.Element(
              name=(k if k else None), value_ref=v.reference
          )
      )
      type_elem.append((k, v.type_signature) if k else v.type_signature)
    result_type = computation_types.StructType(type_elem)
    request = executor_pb2.CreateStructRequest(
        executor=self._executor_id, element=proto_elem
    )
    response = self._stub.create_struct(request)
    py_typecheck.check_type(response, executor_pb2.CreateStructResponse)
    return RemoteValue(response.value_ref, result_type, self)

  @tracing.trace(span=True)
  async def create_selection(self, source, index):
    self._check_has_executor_id()
    py_typecheck.check_type(source, RemoteValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    py_typecheck.check_type(index, int)
    result_type = source.type_signature[index]
    request = executor_pb2.CreateSelectionRequest(
        executor=self._executor_id, source_ref=source.reference, index=index
    )
    response = self._stub.create_selection(request)
    py_typecheck.check_type(response, executor_pb2.CreateSelectionResponse)
    return RemoteValue(response.value_ref, result_type, self)

  @tracing.trace(span=True)
  async def _compute_stream_structs(
      self, value_ref, type_spec: computation_types.StructType
  ):
    py_typecheck.check_type(value_ref, executor_pb2.ValueRef)
    values = []
    source = RemoteValue(value_ref, type_spec, self, False)

    async def per_element(source, index, element_spec):
      select_response = await self.create_selection(source, index)
      value = await self._compute(select_response.reference, element_spec)
      return value

    for index, (_, element_spec) in enumerate(
        structure.iter_elements(type_spec)
    ):
      values.append(per_element(source, index, element_spec))

    values = await asyncio.gather(*values)
    structure.name_list_with_nones(type_spec)
    return structure.Struct(
        zip(structure.name_list_with_nones(type_spec), values)
    )

  @tracing.trace(span=True)
  async def _compute(self, value_ref, type_spec):
    self._check_has_executor_id()
    py_typecheck.check_type(value_ref, executor_pb2.ValueRef)

    if self._stream_structs and isinstance(
        type_spec, computation_types.StructType
    ):
      return await self._compute_stream_structs(value_ref, type_spec)

    request = executor_pb2.ComputeRequest(
        executor=self._executor_id, value_ref=value_ref
    )
    response = self._stub.compute(request)
    py_typecheck.check_type(response, executor_pb2.ComputeResponse)
    value, _ = value_serialization.deserialize_value(response.value, type_spec)
    return value
