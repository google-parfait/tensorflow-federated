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
"""A local proxy for a remote executor service hosted on a separate machine."""

import asyncio
import logging
import queue
import threading

import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_service_utils
from tensorflow_federated.python.core.impl import executor_value_base


_STREAM_CLOSE_WAIT_SECONDS = 10


class RemoteValue(executor_value_base.ExecutorValue):
  """A reference to a value embedded in a remotely deployed executor service."""

  def __init__(self, value_ref, type_spec, executor):
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

  @property
  def type_signature(self):
    return self._type_signature

  async def compute(self):
    return await self._executor._compute(self._value_ref)  # pylint: disable=protected-access

  @property
  def value_ref(self):
    return self._value_ref


class _BidiStream:
  """A bidi stream connection to the Executor service's Execute method."""

  def __init__(self, stub, thread_pool_executor):
    self._request_queue = queue.Queue()
    self._response_event_queue = queue.Queue()
    self._stream_closed_event = threading.Event()
    self._thread_pool_executor = thread_pool_executor

    def request_iter():
      """Iterator that blocks on the request Queue."""
      while True:
        logging.debug('request_iter: waiting for request')
        val = self._request_queue.get()
        if val:
          py_typecheck.check_type(val[0], executor_pb2.ExecuteRequest)
          py_typecheck.check_type(val[1], threading.Event)
          logging.debug('request_iter: got request of type %s',
                        val[0].WhichOneof('request'))
          self._response_event_queue.put_nowait(val[1])
          yield val[0]
        else:
          logging.debug('request_iter: got None request')
          # None means we are done processing
          return

    response_iter = stub.Execute(request_iter())

    def response_thread_fn():
      """Consumes response iter and exposes the value on corresponding Event."""
      try:
        logging.debug('response_thread_fn: waiting for response')
        for response in response_iter:
          logging.debug('response_thread_fn: got response of type %s',
                        response.WhichOneof('response'))
          # Get the corresponding response Event from the queue
          response_event = self._response_event_queue.get_nowait()
          # Attach the response as an attribute on the Event
          response_event.response = response
          response_event.set()
        # Set the event indicating the stream has been closed
        self._stream_closed_event.set()
      except grpc.RpcError as error:
        self._response_event_queue.put(error)

    response_thread = threading.Thread(target=response_thread_fn)
    response_thread.daemon = True
    response_thread.start()

  async def send_request(self, request):
    """Send a request on the bidi stream."""
    py_typecheck.check_type(request, executor_pb2.ExecuteRequest)
    request_type = request.WhichOneof('request')
    response_event = threading.Event()
    # Enqueue a tuple of request and an Event used to return the response
    self._request_queue.put((request, response_event))
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
    self._request_queue.put(None)
    # Wait for the stream to be closed
    self._stream_closed_event.wait(_STREAM_CLOSE_WAIT_SECONDS)


class RemoteExecutor(executor_base.Executor):
  """The remote executor is a local proxy for a remote executor instance.

  NOTE: This component is only available in Python 3.
  """

  # TODO(b/134543154): Switch to using an asynchronous gRPC client so we don't
  # have to block on all those calls.

  def __init__(self,
               channel,
               rpc_mode='REQUEST_REPLY',
               thread_pool_executor=None):
    """Creates a remote executor.

    Args:
      channel: An instance of `grpc.Channel` to use for communication with the
        remote executor service.
      rpc_mode: Optional mode of calling the remote executor. Must be either
        'REQUEST_REPLY' or 'STREAMING' (defaults to 'REQUEST_REPLY'). This
        option will be removed after the request-reply interface is deprecated.
      thread_pool_executor: Optional concurrent.futures.Executor used to wait
        for the reply to a streaming RPC message. Uses the default Executor
        if not specified.
    """
    py_typecheck.check_type(channel, grpc.Channel)
    py_typecheck.check_type(rpc_mode, str)
    if rpc_mode not in ['REQUEST_REPLY', 'STREAMING']:
      raise ValueError('Invalid rpc_mode: {}'.format(rpc_mode))

    self._stub = executor_pb2_grpc.ExecutorStub(channel)
    self._bidi_stream = None
    if rpc_mode == 'STREAMING':
      self._bidi_stream = _BidiStream(self._stub, thread_pool_executor)

  def __del__(self):
    if self._bidi_stream:
      self._bidi_stream.close()
      del self._bidi_stream

  async def create_value(self, value, type_spec=None):
    value_proto, type_spec = (
        executor_service_utils.serialize_value(value, type_spec))
    create_value_request = executor_pb2.CreateValueRequest(value=value_proto)
    if not self._bidi_stream:
      response = self._stub.CreateValue(create_value_request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(create_value=create_value_request)
      )).create_value
    py_typecheck.check_type(response, executor_pb2.CreateValueResponse)
    return RemoteValue(response.value_ref, type_spec, self)

  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, RemoteValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    if arg is not None:
      py_typecheck.check_type(arg, RemoteValue)
    create_call_request = executor_pb2.CreateCallRequest(
        function_ref=comp.value_ref,
        argument_ref=(arg.value_ref if arg is not None else None))
    if not self._bidi_stream:
      response = self._stub.CreateCall(create_call_request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(create_call=create_call_request)
      )).create_call
    py_typecheck.check_type(response, executor_pb2.CreateCallResponse)
    return RemoteValue(response.value_ref, comp.type_signature.result, self)

  async def create_tuple(self, elements):
    elem = anonymous_tuple.to_elements(anonymous_tuple.from_container(elements))
    proto_elem = []
    type_elem = []
    for k, v in elem:
      py_typecheck.check_type(v, RemoteValue)
      proto_elem.append(
          executor_pb2.CreateTupleRequest.Element(
              name=(k if k else None), value_ref=v.value_ref))
      type_elem.append((k, v.type_signature) if k else v.type_signature)
    result_type = computation_types.NamedTupleType(type_elem)
    request = executor_pb2.CreateTupleRequest(element=proto_elem)
    if not self._bidi_stream:
      response = self._stub.CreateTuple(request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(create_tuple=request))).create_tuple
    py_typecheck.check_type(response, executor_pb2.CreateTupleResponse)
    return RemoteValue(response.value_ref, result_type, self)

  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, RemoteValue)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    if index is not None:
      py_typecheck.check_type(index, int)
      py_typecheck.check_none(name)
      result_type = source.type_signature[index]
    else:
      py_typecheck.check_type(name, str)
      result_type = getattr(source.type_signature, name)
    response = self._stub.CreateSelection(
        executor_pb2.CreateSelectionRequest(
            source_ref=source.value_ref, name=name, index=index))
    py_typecheck.check_type(response, executor_pb2.CreateSelectionResponse)
    return RemoteValue(response.value_ref, result_type, self)

  async def _compute(self, value_ref):
    py_typecheck.check_type(value_ref, executor_pb2.ValueRef)
    request = executor_pb2.ComputeRequest(value_ref=value_ref)
    if not self._bidi_stream:
      response = self._stub.Compute(request)
    else:
      response = (await self._bidi_stream.send_request(
          executor_pb2.ExecuteRequest(compute=request))).compute
    py_typecheck.check_type(response, executor_pb2.ComputeResponse)
    value, _ = executor_service_utils.deserialize_value(response.value)
    return value
