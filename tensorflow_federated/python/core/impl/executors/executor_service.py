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
import threading
import traceback
import uuid
import weakref

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_serialization


def _set_invalid_arg_err(context: grpc.ServicerContext, err):
  logging.error(traceback.format_exc())
  context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
  context.set_details(str(err))


def _set_unknown_err(context: grpc.ServicerContext, err):
  logging.error(traceback.format_exc())
  context.set_code(grpc.StatusCode.UNKNOWN)
  context.set_details(str(err))


class ExecutorService(executor_pb2_grpc.ExecutorServicer):
  """A wrapper around a target executor that makes it into a gRPC service."""

  def __init__(self, ex_factory: executor_factory.ExecutorFactory, *args,
               **kwargs):
    py_typecheck.check_type(ex_factory, executor_factory.ExecutorFactory)
    super().__init__(*args, **kwargs)
    self._ex_factory = ex_factory
    self._executor = None
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
    self._event_loop.set_task_factory(
        tracing.propagate_trace_context_task_factory)
    self._thread = threading.Thread(
        target=functools.partial(run_loop, self._event_loop), daemon=True)
    self._thread.start()

    def finalize(loop, thread):
      loop.call_soon_threadsafe(loop.stop)
      thread.join()

    weakref.finalize(self, finalize, self._event_loop, self._thread)

  def _run_coro_threadsafe_with_tracing(self, coro):
    """Runs `coro` on `self._event_loop` inside the current trace spans."""
    with tracing.with_trace_context_from_rpc():
      return asyncio.run_coroutine_threadsafe(
          tracing.wrap_coroutine_in_current_trace_context(coro),
          self._event_loop)

  @property
  def executor(self):
    if self._executor is None:
      raise RuntimeError('The executor service has not yet been configured '
                         'with cardinalities and cannot execute any '
                         'concrete requests.')
    return self._executor

  def SetCardinalities(
      self,
      request: executor_pb2.SetCardinalitiesRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.SetCardinalitiesResponse:
    """Sets the cardinality for the executor service."""
    py_typecheck.check_type(request, executor_pb2.SetCardinalitiesRequest)
    try:
      cardinalities_dict = executor_serialization.deserialize_cardinalities(
          request.cardinalities)
      self._executor = self._ex_factory.create_executor(cardinalities_dict)
      return executor_pb2.SetCardinalitiesResponse()
    except (ValueError, TypeError) as err:
      _set_invalid_arg_err(context, err)
      return executor_pb2.SetCardinalitiesResponse()

  def ClearExecutor(
      self,
      request: executor_pb2.ClearExecutorRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.ClearExecutorResponse:
    """Clears the service Executor-related state."""
    py_typecheck.check_type(request, executor_pb2.ClearExecutorRequest)
    self._executor = None
    self._ex_factory.clean_up_executors()
    return executor_pb2.ClearExecutorResponse()

  def CreateValue(
      self,
      request: executor_pb2.CreateValueRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateValueResponse:
    """Creates a value embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateValueRequest)
    try:
      with tracing.span('ExecutorService.CreateValue', 'deserialize_value'):
        value, value_type = (
            executor_serialization.deserialize_value(request.value))
      value_id = str(uuid.uuid4())
      coro = self.executor.create_value(value, value_type)
      future_val = self._run_coro_threadsafe_with_tracing(coro)
      with self._lock:
        self._values[value_id] = future_val
      return executor_pb2.CreateValueResponse(
          value_ref=executor_pb2.ValueRef(id=value_id))
    except (ValueError, TypeError) as err:
      _set_invalid_arg_err(context, err)
      return executor_pb2.CreateValueResponse()

  def CreateCall(
      self,
      request: executor_pb2.CreateCallRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateCallResponse:
    """Creates a call embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateCallRequest)
    try:
      function_id = str(request.function_ref.id)
      argument_id = str(request.argument_ref.id)
      with self._lock:
        function_val = self._values[function_id]
        argument_val = self._values[argument_id] if argument_id else None

      async def _process_create_call():
        function = await asyncio.wrap_future(function_val)
        argument = await asyncio.wrap_future(
            argument_val) if argument_val is not None else None
        return await self.executor.create_call(function, argument)

      coro = _process_create_call()
      result_fut = self._run_coro_threadsafe_with_tracing(coro)
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateCallResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
      _set_invalid_arg_err(context, err)
      return executor_pb2.CreateCallResponse()

  def CreateStruct(
      self,
      request: executor_pb2.CreateStructRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateStructResponse:
    """Creates a struct embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateStructRequest)
    try:
      with self._lock:
        elem_futures = [self._values[e.value_ref.id] for e in request.element]
      elem_names = [
          str(elem.name) if elem.name else None for elem in request.element
      ]

      async def _process_create_struct():
        elem_values = await asyncio.gather(
            *[asyncio.wrap_future(v) for v in elem_futures])
        elements = list(zip(elem_names, elem_values))
        struct = structure.Struct(elements)
        return await self.executor.create_struct(struct)

      result_fut = self._run_coro_threadsafe_with_tracing(
          _process_create_struct())
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateStructResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
      _set_invalid_arg_err(context, err)
      return executor_pb2.CreateStructResponse()

  def CreateSelection(
      self,
      request: executor_pb2.CreateSelectionRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateSelectionResponse:
    """Creates a selection embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateSelectionRequest)
    try:
      with self._lock:
        source_fut = self._values[request.source_ref.id]

      async def _process_create_selection():
        source = await asyncio.wrap_future(source_fut)
        return await self.executor.create_selection(source, request.index)

      result_fut = self._run_coro_threadsafe_with_tracing(
          _process_create_selection())
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateSelectionResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
      _set_invalid_arg_err(context, err)
      return executor_pb2.CreateSelectionResponse()

  def Compute(
      self,
      request: executor_pb2.ComputeRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.ComputeResponse:
    """Computes a value embedded in the executor."""
    return self._run_coro_threadsafe_with_tracing(
        self._Compute(request, context)).result()

  async def _Compute(
      self,
      request: executor_pb2.ComputeRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.ComputeResponse:
    """Asynchronous implemention of `Compute`."""
    py_typecheck.check_type(request, executor_pb2.ComputeRequest)
    try:
      value_id = str(request.value_ref.id)
      with self._lock:
        future_val = asyncio.wrap_future(self._values[value_id])
      val = await future_val
      result_val = await val.compute()
      val_type = val.type_signature
      value_proto, _ = executor_serialization.serialize_value(
          result_val, val_type)
      return executor_pb2.ComputeResponse(value=value_proto)
    except (ValueError, TypeError) as err:
      _set_invalid_arg_err(context, err)
      return executor_pb2.ComputeResponse()

  def Dispose(
      self,
      request: executor_pb2.DisposeRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.DisposeResponse:
    """Disposes of a value, making it no longer available for future calls."""
    py_typecheck.check_type(request, executor_pb2.DisposeRequest)
    try:
      with self._lock:
        for value_ref in request.value_ref:
          del self._values[value_ref.id]
    except KeyError as err:
      _set_invalid_arg_err(context, err)
    return executor_pb2.DisposeResponse()
