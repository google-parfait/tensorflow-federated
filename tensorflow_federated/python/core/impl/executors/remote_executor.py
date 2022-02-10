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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A local proxy for a remote executor service hosted on a separate machine."""

from typing import Mapping
import weakref

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_serialization
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import remote_executor_stub
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements

_STREAM_CLOSE_WAIT_SECONDS = 10


class RemoteValue(executor_value_base.ExecutorValue):
  """A reference to a value embedded in a remotely deployed executor service."""

  def __init__(self, value_ref: executor_pb2.ValueRef, type_spec, executor):
    """Creates the value.

    Args:
      value_ref: An instance of `executor_pb2.ValueRef` returned by the remote
        executor service.
      type_spec: An instance of `computation_types.Type`.
      executor: The executor that created this value.
    """
    py_typecheck.check_type(value_ref, executor_pb2.ValueRef)
    py_typecheck.check_type(type_spec, computation_types.Type)
    py_typecheck.check_type(executor, RemoteExecutor)
    self._value_ref = value_ref
    self._type_signature = type_spec
    self._executor = executor

    # Clean up the value and the memory associated with it on the remote
    # worker when no references to it remain.
    def finalizer(value_ref, executor):
      executor._dispose(value_ref)  # pylint: disable=protected-access

    weakref.finalize(self, finalizer, value_ref, executor)

  @property
  def type_signature(self):
    return self._type_signature

  @tracing.trace(span=True)
  async def compute(self):
    return await self._executor._compute(self._value_ref)  # pylint: disable=protected-access

  @property
  def value_ref(self):
    return self._value_ref


class RemoteExecutor(executor_base.Executor):
  """The remote executor is a local proxy for a remote executor instance."""

  # TODO(b/134543154): Switch to using an asynchronous gRPC client so we don't
  # have to block on all those calls.

  def __init__(self,
               stub: remote_executor_stub.RemoteExecutorStub,
               thread_pool_executor=None,
               dispose_batch_size=20):
    """Creates a remote executor.

    Args:
      stub: An instance of stub used for communication with the remote executor
        service.
      thread_pool_executor: Optional concurrent.futures.Executor used to wait
        for the reply to a streaming RPC message. Uses the default Executor if
        not specified.
      dispose_batch_size: The batch size for requests to dispose of remote
        worker values. Lower values will result in more requests to the remote
        worker, but will result in values being cleaned up sooner and therefore
        may result in lower memory usage on the remote worker.
    """

    py_typecheck.check_type(dispose_batch_size, int)

    logging.debug('Creating new ExecutorStub')

    # We need to keep a reference to the channel around to prevent the Python
    # object from being GC'ed and the callback above from no-op'ing.
    self._stub = stub
    self._dispose_batch_size = dispose_batch_size
    self._dispose_request = executor_pb2.DisposeRequest()

  @property
  def is_ready(self) -> bool:
    return self._stub.is_ready

  def close(self):
    logging.debug('Clearing executor state on server.')
    self._clear_executor()

  def _dispose(self, value_ref: executor_pb2.ValueRef):
    """Disposes of the remote value stored on the worker service."""
    self._dispose_request.value_ref.append(value_ref)
    if len(self._dispose_request.value_ref) < self._dispose_batch_size:
      return
    dispose_request = self._dispose_request
    self._dispose_request = executor_pb2.DisposeRequest()
    self._stub.dispose(dispose_request)

  @tracing.trace(span=True)
  def set_cardinalities(self,
                        cardinalities: Mapping[placements.PlacementLiteral,
                                               int]):
    serialized_cardinalities = executor_serialization.serialize_cardinalities(
        cardinalities)
    request = executor_pb2.SetCardinalitiesRequest(
        cardinalities=serialized_cardinalities)

    self._stub.set_cardinalities(request)

  @tracing.trace(span=True)
  def _clear_executor(self):
    request = executor_pb2.ClearExecutorRequest()
    try:
      self._stub.clear_executor(request)
    except (grpc.RpcError, executors_errors.RetryableError):
      logging.debug('RPC error caught during attempt to clear state on the '
                    'server; this likely indicates a broken connection, and '
                    'therefore there is no state to clear.')
    return

  @tracing.trace(span=True)
  async def create_value(self, value, type_spec=None):

    @tracing.trace
    def serialize_value():
      return executor_serialization.serialize_value(value, type_spec)

    value_proto, type_spec = serialize_value()
    create_value_request = executor_pb2.CreateValueRequest(value=value_proto)
    response = self._stub.create_value(create_value_request)
    py_typecheck.check_type(response, executor_pb2.CreateValueResponse)
    return RemoteValue(response.value_ref, type_spec, self)

  @tracing.trace(span=True)
  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, RemoteValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    if arg is not None:
      py_typecheck.check_type(arg, RemoteValue)
    create_call_request = executor_pb2.CreateCallRequest(
        function_ref=comp.value_ref,
        argument_ref=(arg.value_ref if arg is not None else None))
    response = self._stub.create_call(create_call_request)
    py_typecheck.check_type(response, executor_pb2.CreateCallResponse)
    return RemoteValue(response.value_ref, comp.type_signature.result, self)

  @tracing.trace(span=True)
  async def create_struct(self, elements):
    constructed_anon_tuple = structure.from_container(elements)
    proto_elem = []
    type_elem = []
    for k, v in structure.iter_elements(constructed_anon_tuple):
      py_typecheck.check_type(v, RemoteValue)
      proto_elem.append(
          executor_pb2.CreateStructRequest.Element(
              name=(k if k else None), value_ref=v.value_ref))
      type_elem.append((k, v.type_signature) if k else v.type_signature)
    result_type = computation_types.StructType(type_elem)
    request = executor_pb2.CreateStructRequest(element=proto_elem)
    response = self._stub.create_struct(request)
    py_typecheck.check_type(response, executor_pb2.CreateStructResponse)
    return RemoteValue(response.value_ref, result_type, self)

  @tracing.trace(span=True)
  async def create_selection(self, source, index):
    py_typecheck.check_type(source, RemoteValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    py_typecheck.check_type(index, int)
    result_type = source.type_signature[index]
    request = executor_pb2.CreateSelectionRequest(
        source_ref=source.value_ref, index=index)
    response = self._stub.create_selection(request)
    py_typecheck.check_type(response, executor_pb2.CreateSelectionResponse)
    return RemoteValue(response.value_ref, result_type, self)

  @tracing.trace(span=True)
  async def _compute(self, value_ref):
    py_typecheck.check_type(value_ref, executor_pb2.ValueRef)
    request = executor_pb2.ComputeRequest(value_ref=value_ref)
    response = self._stub.compute(request)
    py_typecheck.check_type(response, executor_pb2.ComputeResponse)
    value, _ = executor_serialization.deserialize_value(response.value)
    return value
