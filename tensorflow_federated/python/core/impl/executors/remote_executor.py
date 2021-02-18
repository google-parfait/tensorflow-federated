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
import queue
import threading
from typing import Mapping
import warnings
import weakref

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_serialization
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import placement_literals

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


class _BidiStream:
  """A bidi stream connection to the Executor service's Execute method."""

  def __init__(self, stub, thread_pool_executor):
    self._stub = stub
    self._thread_pool_executor = thread_pool_executor
    self._is_initialized = False

  def _lazy_init(self):
    """Lazily initialize the underlying gRPC stream."""
    if self._is_initialized:
      return

    logging.debug('Initializing bidi stream')

    self._request_queue = queue.Queue()
    # Lock preventing concurrent mutation of the `response_event_dict`.
    self._response_event_lock = threading.Lock()
    self._response_event_dict = {}
    self._stream_closed_event = threading.Event()
    self._stream_error = None
    self._request_num = 0

    def request_iter():
      """Iterator that blocks on the request Queue."""
      request = self._request_queue.get()
      while request is not None:
        yield request
        request = self._request_queue.get()

    response_iter = self._stub.Execute(request_iter())

    def response_thread_fn():
      """Consumes response iter and exposes the value on corresponding Event."""
      try:
        logging.debug('Response thread: blocking for next response')
        for response in response_iter:
          if response.WhichOneof('response') is None:
            # TODO(b/175927125): We currently pass an empty response in some
            # error cases and pass a GRPC error back via the ServicerContext in
            # some others. Unify this error-passing.
            raise execution_context.RetryableError(
                'Unknown error on the service side.')
          logging.debug(
              'Response thread: processing response of type %s, seq_no %s',
              response.WhichOneof('response'), response.sequence_number)
          with self._response_event_lock:
            # Get the corresponding response Event
            response_event = self._response_event_dict[response.sequence_number]
            # Attach the response as an attribute on the Event
            response_event.response = response
            response_event.set()
        # Set the event indicating the stream has been closed
        self._stream_closed_event.set()
      except Exception as error:  # pylint: disable=broad-except
        logging.exception('Error calling remote executor: %s', error)
        if _is_retryable_grpc_error(error):
          logging.exception('gRPC error is retryable')
          error = execution_context.RetryableError(error)
        with self._response_event_lock:
          self._stream_error = error
          for _, response_event in self._response_event_dict.items():
            if not response_event.isSet():
              response_event.response = error
              response_event.set()
        self._stream_closed_event.set()

    response_thread = threading.Thread(target=response_thread_fn)
    response_thread.daemon = True
    response_thread.start()

    self._is_initialized = True

  @tracing.trace(span=True)
  async def send_request(self, request):
    """Send a request on the bidi stream."""
    self._lazy_init()

    py_typecheck.check_type(request, executor_pb2.ExecuteRequest)
    request_type = request.WhichOneof('request')
    response_event = threading.Event()

    py_typecheck.check_type(request, executor_pb2.ExecuteRequest)
    py_typecheck.check_type(response_event, threading.Event)

    seq = self._request_num
    self._request_num += 1

    with self._response_event_lock:
      if self._stream_error is not None:
        logging.debug('Stream failed before msg enqueued')
        logging.debug('%s', self._stream_error)
        response_event.response = self._stream_error
        response_event.set()
      else:
        request.sequence_number = seq
        logging.debug(
            'Request thread: processing request of type %s, seq_no %s',
            request.WhichOneof('request'), seq)
        self._response_event_dict[seq] = response_event

        # Enqueue a tuple of request and an Event used to return the response
        self._request_queue.put(request)

    await asyncio.get_event_loop().run_in_executor(self._thread_pool_executor,
                                                   response_event.wait)
    response = response_event.response
    if isinstance(response, Exception):
      raise response

    py_typecheck.check_type(response, executor_pb2.ExecuteResponse)
    response_type = response.WhichOneof('response')
    if response_type != request_type:
      raise ValueError('Request had type: {} but response had type: {}'.format(
          request_type, response_type))
    return response

  def close(self):
    if self._is_initialized:
      logging.debug('Closing bidi stream')

      self._request_queue.put(None)
      # Wait for the stream to be closed
      self._stream_closed_event.wait(_STREAM_CLOSE_WAIT_SECONDS)
    else:
      logging.debug('Closing unused bidi stream')
    self._is_initialized = False


@tracing.trace(span=True)
def _request(rpc_func, request):
  """Populates trace context and reraises gRPC errors with retryable info."""
  with tracing.wrap_rpc_in_trace_context():
    try:
      return rpc_func(request)
    except grpc.RpcError as e:
      if _is_retryable_grpc_error(e):
        logging.info('Received retryable gRPC error: %s', e)
        raise execution_context.RetryableError(e)
      else:
        raise


def _is_retryable_grpc_error(error):
  """Predicate defining what is a retryable gRPC error."""
  non_retryable_errors = {
      grpc.StatusCode.INVALID_ARGUMENT,
      grpc.StatusCode.NOT_FOUND,
      grpc.StatusCode.ALREADY_EXISTS,
      grpc.StatusCode.PERMISSION_DENIED,
      grpc.StatusCode.FAILED_PRECONDITION,
      grpc.StatusCode.ABORTED,
      grpc.StatusCode.OUT_OF_RANGE,
      grpc.StatusCode.UNIMPLEMENTED,
      grpc.StatusCode.DATA_LOSS,
      grpc.StatusCode.UNAUTHENTICATED,
  }
  return (isinstance(error, grpc.RpcError) and
          error.code() not in non_retryable_errors)


class RemoteExecutor(executor_base.Executor):
  """The remote executor is a local proxy for a remote executor instance."""

  # TODO(b/134543154): Switch to using an asynchronous gRPC client so we don't
  # have to block on all those calls.

  def __init__(self,
               channel,
               rpc_mode='REQUEST_REPLY',
               thread_pool_executor=None,
               dispose_batch_size=20):
    """Creates a remote executor.

    Args:
      channel: An instance of `grpc.Channel` to use for communication with the
        remote executor service.
      rpc_mode: Optional mode of calling the remote executor. Must be either
        'REQUEST_REPLY' or 'STREAMING' (defaults to 'REQUEST_REPLY'). This
        option will be removed after the request-reply interface is deprecated.
      thread_pool_executor: Optional concurrent.futures.Executor used to wait
        for the reply to a streaming RPC message. Uses the default Executor if
        not specified.
      dispose_batch_size: The batch size for requests to dispose of remote
        worker values. Lower values will result in more requests to the remote
        worker, but will result in values being cleaned up sooner and therefore
        may result in lower memory usage on the remote worker.
    """

    py_typecheck.check_type(channel, grpc.Channel)
    py_typecheck.check_type(rpc_mode, str)
    py_typecheck.check_type(dispose_batch_size, int)
    if rpc_mode not in ['REQUEST_REPLY', 'STREAMING']:
      raise ValueError('Invalid rpc_mode: {}'.format(rpc_mode))

    if rpc_mode == 'STREAMING':
      warnings.warn('TFF streaming RPCs are deprecated and being removed; '
                    'falling back to request-reply behavior.')
      rpc_mode = 'REQUEST_REPLY'

    logging.debug('Creating new ExecutorStub with RPC_MODE=%s', rpc_mode)

    self._channel_status = False

    def _channel_status_callback(
        channel_connectivity: grpc.ChannelConnectivity):
      self._channel_status = channel_connectivity

    channel.subscribe(_channel_status_callback, try_to_connect=True)

    # We need to keep a reference to the channel around to prevent the Python
    # object from being GC'ed and the callback above from no-op'ing.
    self._channel = channel
    self._stub = executor_pb2_grpc.ExecutorStub(channel)
    self._bidi_stream = None
    self._dispose_batch_size = dispose_batch_size
    self._dispose_request = executor_pb2.DisposeRequest()
    if rpc_mode == 'STREAMING':
      logging.debug('Creating Bidi stream')
      self._bidi_stream = _BidiStream(self._stub, thread_pool_executor)

  @property
  def is_ready(self) -> bool:
    return self._channel_status == grpc.ChannelConnectivity.READY

  def close(self):
    if self._bidi_stream is not None:
      logging.debug('Closing bidi stream')
      self._bidi_stream.close()
    logging.debug('Clearing executor state on server.')
    self._clear_executor()

  def _dispose(self, value_ref: executor_pb2.ValueRef):
    """Disposes of the remote value stored on the worker service."""
    self._dispose_request.value_ref.append(value_ref)
    if len(self._dispose_request.value_ref) < self._dispose_batch_size:
      return
    dispose_request = self._dispose_request
    self._dispose_request = executor_pb2.DisposeRequest()
    if self._bidi_stream is None:
      _request(self._stub.Dispose, dispose_request)
    else:
      send_request_fut = self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(dispose=dispose_request))
      # We don't care about the response, and so don't bother to await it.
      # Just start it as a task so that it runs at some point.
      asyncio.get_event_loop().create_task(send_request_fut)

  @tracing.trace(span=True)
  async def set_cardinalities(
      self, cardinalities: Mapping[placement_literals.PlacementLiteral, int]):
    serialized_cardinalities = executor_serialization.serialize_cardinalities(
        cardinalities)
    request = executor_pb2.SetCardinalitiesRequest(
        cardinalities=serialized_cardinalities)

    if self._bidi_stream is None:
      _request(self._stub.SetCardinalities, request)
    else:
      await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(set_cardinalities=request))
    return

  @tracing.trace(span=True)
  def _clear_executor(self):
    request = executor_pb2.ClearExecutorRequest()
    try:
      _request(self._stub.ClearExecutor, request)
    except (grpc.RpcError, execution_context.RetryableError):
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
    if self._bidi_stream is None:
      response = _request(self._stub.CreateValue, create_value_request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(create_value=create_value_request)
      )).create_value
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
    if self._bidi_stream is None:
      response = _request(self._stub.CreateCall, create_call_request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(create_call=create_call_request)
      )).create_call
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
    if self._bidi_stream is None:
      response = _request(self._stub.CreateStruct, request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(create_struct=request))).create_struct
    py_typecheck.check_type(response, executor_pb2.CreateStructResponse)
    return RemoteValue(response.value_ref, result_type, self)

  @tracing.trace(span=True)
  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, RemoteValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    if index is not None:
      py_typecheck.check_type(index, int)
      py_typecheck.check_none(name)
      result_type = source.type_signature[index]
    else:
      py_typecheck.check_type(name, str)
      result_type = getattr(source.type_signature, name)
    request = executor_pb2.CreateSelectionRequest(
        source_ref=source.value_ref, name=name, index=index)
    if self._bidi_stream is None:
      response = _request(self._stub.CreateSelection, request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(create_selection=request)
      )).create_selection
    py_typecheck.check_type(response, executor_pb2.CreateSelectionResponse)
    return RemoteValue(response.value_ref, result_type, self)

  @tracing.trace(span=True)
  async def _compute(self, value_ref):
    py_typecheck.check_type(value_ref, executor_pb2.ValueRef)
    request = executor_pb2.ComputeRequest(value_ref=value_ref)
    if self._bidi_stream is None:
      response = _request(self._stub.Compute, request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(compute=request))).compute
    py_typecheck.check_type(response, executor_pb2.ComputeResponse)
    value, _ = executor_serialization.deserialize_value(response.value)
    return value
