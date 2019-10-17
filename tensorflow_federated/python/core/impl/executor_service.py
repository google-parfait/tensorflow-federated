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
"""A service wrapper around an executor that makes it accessible over gRPC."""

import asyncio
import functools
import queue
import sys
import threading
import traceback
import uuid
import weakref

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_service_utils


class ExecutorService(executor_pb2_grpc.ExecutorServicer):
  """A wrapper around a target executor that makes it into a gRPC service.

  NOTE: This component is only available in Python 3.
  """

  def __init__(self, executor, *args, **kwargs):
    py_typecheck.check_type(executor, executor_base.Executor)
    super(ExecutorService, self).__init__(*args, **kwargs)
    self._executor = executor
    self._lock = threading.Lock()

    # The keys in this dictionary are value ids (the same as what we return
    # in the gRPC responses), and the values are `concurrent.futures.Future`
    # instances (this may, and probably will change as we flesh out the rest
    # of this implementation).
    self._values = {}

    def run_loop(loop):
      loop.run_forever()
      loop.close()

    self._event_loop = asyncio.new_event_loop()
    self._thread = threading.Thread(
        target=functools.partial(run_loop, self._event_loop), daemon=True)
    self._thread.start()

    def finalize(loop, thread):
      loop.call_soon_threadsafe(loop.stop)
      thread.join()

    weakref.finalize(self, finalize, self._event_loop, self._thread)

  async def _HandleRequest(self, req, context, response_queue):
    which = req.WhichOneof('request')
    if not which:
      raise RuntimeError('Must set a request type')
    if which == 'create_value':
      response = executor_pb2.ExecuteResponse(
          create_value=self.CreateValue(req.create_value, context))
    elif which == 'create_call':
      response = executor_pb2.ExecuteResponse(
          create_call=self.CreateCall(req.create_call, context))
    elif which == 'create_tuple':
      response = executor_pb2.ExecuteResponse(
          create_tuple=self.CreateTuple(req.create_tuple, context))
    elif which == 'create_selection':
      response = executor_pb2.ExecuteResponse(
          create_selection=self.CreateSelection(req.create_selection, context))
    elif which == 'compute':
      response = executor_pb2.ExecuteResponse(
          compute=await self._Compute(req.compute, context))
    response.sequence_number = req.sequence_number
    response_queue.put_nowait(response)

  def Execute(self, request_iter, context):
    response_queue = queue.Queue()

    class RequestIterFinished:
      """Marker object indicating how many requests were received."""

      def __init__(self, n_reqs):
        self._n_reqs = n_reqs

      def get_n_reqs(self):
        return self._n_reqs

    def request_thread_fn():
      n_reqs = 0
      for req in request_iter:
        n_reqs += 1
        asyncio.run_coroutine_threadsafe(
            self._HandleRequest(req, context, response_queue), self._event_loop)
      response_queue.put_nowait(RequestIterFinished(n_reqs))

    threading.Thread(target=request_thread_fn).start()

    # This generator is finished when the request iterator is finished and we
    # have yielded a response for every request.
    n_responses = 0
    target_responses = sys.maxsize
    while n_responses < target_responses:
      response = response_queue.get()
      if isinstance(response, executor_pb2.ExecuteResponse):
        n_responses += 1
        logging.debug('Yielding response of type %s with sequence no. %s',
                      response.WhichOneof('response'), response.sequence_number)
        yield response
      elif isinstance(response, RequestIterFinished):
        target_responses = response.get_n_reqs()
      else:
        raise ValueError('Illegal response object: {}'.format(response))

  def CreateValue(self, request, context):
    """Creates a value embedded in the executor.

    Args:
      request: An instance of `executor_pb2.CreateValueRequest`.
      context: An instance of `grpc.ServicerContext`.

    Returns:
      An instance of `executor_pb2.CreateValueResponse`.
    """
    py_typecheck.check_type(request, executor_pb2.CreateValueRequest)
    try:
      value, value_type = (
          executor_service_utils.deserialize_value(request.value))
      value_id = str(uuid.uuid4())
      future_val = asyncio.run_coroutine_threadsafe(
          self._executor.create_value(value, value_type), self._event_loop)
      with self._lock:
        self._values[value_id] = future_val
      return executor_pb2.CreateValueResponse(
          value_ref=executor_pb2.ValueRef(id=value_id))
    except (ValueError, TypeError) as err:
      logging.error(traceback.format_exc())
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details(str(err))
      return executor_pb2.CreateValueResponse()

  def CreateCall(self, request, context):
    """Creates a call embedded in the executor.

    Args:
      request: An instance of `executor_pb2.CreateCallRequest`.
      context: An instance of `grpc.ServicerContext`.

    Returns:
      An instance of `executor_pb2.CreateCallResponse`.
    """
    py_typecheck.check_type(request, executor_pb2.CreateCallRequest)
    try:
      function_id = str(request.function_ref.id)
      argument_id = str(request.argument_ref.id)
      with self._lock:
        function_val = self._values[function_id]
        argument_val = self._values[argument_id] if argument_id else None

      async def _processing():
        function = await asyncio.wrap_future(function_val)
        argument = await asyncio.wrap_future(
            argument_val) if argument_val is not None else None
        return await self._executor.create_call(function, argument)

      result_fut = self._event_loop.create_task(_processing())
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateCallResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
      logging.error(traceback.format_exc())
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details(str(err))
      return executor_pb2.CreateCallResponse()

  def CreateTuple(self, request, context):
    """Creates a tuple embedded in the executor.

    Args:
      request: An instance of `executor_pb2.CreateTupleRequest`.
      context: An instance of `grpc.ServicerContext`.

    Returns:
      An instance of `executor_pb2.CreateTupleResponse`.
    """
    py_typecheck.check_type(request, executor_pb2.CreateTupleRequest)
    try:
      with self._lock:
        elem_futures = [self._values[e.value_ref.id] for e in request.element]
      elem_names = [
          str(elem.name) if elem.name else None for elem in request.element
      ]

      async def _processing():
        elem_values = await asyncio.gather(
            *[asyncio.wrap_future(v) for v in elem_futures])
        elements = list(zip(elem_names, elem_values))
        anon_tuple = anonymous_tuple.AnonymousTuple(elements)
        return await self._executor.create_tuple(anon_tuple)

      result_fut = self._event_loop.create_task(_processing())
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateTupleResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
      logging.error(traceback.format_exc())
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details(str(err))
      return executor_pb2.CreateTupleResponse()

  def CreateSelection(self, request, context):
    """Creates a selection embedded in the executor.

    Args:
      request: An instance of `executor_pb2.CreateSelectionRequest`.
      context: An instance of `grpc.ServicerContext`.

    Returns:
      An instance of `executor_pb2.CreateSelectionResponse`.
    """
    py_typecheck.check_type(request, executor_pb2.CreateSelectionRequest)
    try:
      with self._lock:
        source_fut = self._values[request.source_ref.id]

      async def _processing():
        source = await asyncio.wrap_future(source_fut)
        which_selection = request.WhichOneof('selection')
        if which_selection == 'name':
          coro = self._executor.create_selection(source, name=request.name)
        else:
          coro = self._executor.create_selection(source, index=request.index)
        return await coro

      result_fut = self._event_loop.create_task(_processing())
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateSelectionResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
      logging.error(traceback.format_exc())
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details(str(err))
      return executor_pb2.CreateSelectionResponse()

  def Compute(self, request, context):
    """Computes a value embedded in the executor.

    Args:
      request: An instance of `executor_pb2.ComputeRequest`.
      context: An instance of `grpc.ServicerContext`.

    Returns:
      An instance of `executor_pb2.ComputeResponse`.
    """
    return asyncio.run_coroutine_threadsafe(
        self._Compute(request, context), self._event_loop).result()

  async def _Compute(self, request, context):
    """Computes a value embedded in the executor.

    Args:
      request: An instance of `executor_pb2.ComputeRequest`.
      context: An instance of `grpc.ServicerContext`.

    Returns:
      An instance of `executor_pb2.ComputeResponse`.
    """
    py_typecheck.check_type(request, executor_pb2.ComputeRequest)
    try:
      value_id = str(request.value_ref.id)
      with self._lock:
        future_val = asyncio.wrap_future(self._values[value_id])
      val = await future_val
      result_val = await val.compute()
      val_type = val.type_signature
      value_proto, _ = executor_service_utils.serialize_value(
          result_val, val_type)
      return executor_pb2.ComputeResponse(value=value_proto)
    except (ValueError, TypeError) as err:
      logging.error(traceback.format_exc())
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details(str(err))
      return executor_pb2.ComputeResponse()
