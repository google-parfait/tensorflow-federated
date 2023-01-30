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
import collections
import contextlib
import functools
import threading
import traceback
from typing import Any
import uuid
import weakref

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import value_serialization


def _set_invalid_arg_error(context: grpc.ServicerContext, error: Exception):
  logging.error(traceback.format_exc())
  context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
  context.set_details(str(error))


def _set_unavailable_error(context: grpc.ServicerContext, error: Exception):
  logging.error(traceback.format_exc())
  context.set_code(grpc.StatusCode.UNAVAILABLE)
  context.set_details(str(error))


def _propagate_grpc_code_error(context: grpc.ServicerContext, error: grpc.Call):
  logging.error(traceback.format_exc())
  context.set_code(error.code())
  context.set_details(str(error))


def _get_hashable_key(cardinalities: executor_factory.CardinalitiesType) -> str:
  return str(tuple(sorted((str(k), v) for k, v in cardinalities.items())))


class ExecutorService(executor_pb2_grpc.ExecutorGroupServicer):
  """A wrapper around a target executor that makes it into a gRPC service."""

  def __init__(
      self, ex_factory: executor_factory.ExecutorFactory, *args, **kwargs
  ):
    py_typecheck.check_type(ex_factory, executor_factory.ExecutorFactory)
    super().__init__(*args, **kwargs)
    self._ex_factory = ex_factory
    self._executors = {}
    self._executor_ref_counts = collections.defaultdict(lambda: 0)
    self._ids_to_cardinalities = {}
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
        tracing.propagate_trace_context_task_factory
    )
    self._thread = threading.Thread(
        target=functools.partial(run_loop, self._event_loop), daemon=True
    )
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
          self._event_loop,
      )

  def executor(
      self, request: Any, context: grpc.ServicerContext
  ) -> executor_base.Executor:
    """Returns the executor which should be used to handle `request`."""
    with self._lock:
      executor = self._executors.get(request.executor.id)
      if executor is None:
        message = f'No executor found for executor id: {request.executor.id}'
        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
        context.set_details(message)
        raise RuntimeError(message)
      return executor

  def GetExecutor(
      self,
      request: executor_pb2.GetExecutorRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.GetExecutorResponse:
    """Returns an identifier for an executor with the provided requirements."""
    py_typecheck.check_type(request, executor_pb2.GetExecutorRequest)
    with self._try_handle_request_context(
        request, context, executor_pb2.GetExecutorResponse
    ):
      cardinalities = value_serialization.deserialize_cardinalities(
          request.cardinalities
      )
      key = _get_hashable_key(cardinalities)
      with self._lock:
        if key not in self._executors:
          self._executors[key] = self._ex_factory.create_executor(cardinalities)
        self._executor_ref_counts[key] += 1
        self._ids_to_cardinalities[key] = cardinalities
      return executor_pb2.GetExecutorResponse(
          executor=executor_pb2.ExecutorId(id=key)
      )

  def DestroyExecutor(self, request):
    with self._lock:
      key = request.executor.id
      if self._executors.get(key):
        del self._executors[key]
        cardinalities = self._ids_to_cardinalities[key]
        self._ex_factory.clean_up_executor(cardinalities)
        del self._ids_to_cardinalities[key]
        self._executor_ref_counts[key] = 0

  def DisposeExecutor(
      self,
      request: executor_pb2.DisposeExecutorRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.DisposeExecutorResponse:
    """Clears the service Executor-related state."""
    py_typecheck.check_type(request, executor_pb2.DisposeExecutorRequest)
    key = request.executor.id
    with self._lock:
      self._executor_ref_counts[key] -= 1
      if self._executor_ref_counts[key] == 0:
        del self._executors[key]
        del self._executor_ref_counts[key]
        cardinalities = self._ids_to_cardinalities[key]
        self._ex_factory.clean_up_executor(cardinalities)
        del self._ids_to_cardinalities[key]
    return executor_pb2.DisposeExecutorResponse()

  @contextlib.contextmanager
  def _try_handle_request_context(self, request, context, blank_response_fn):
    try:
      yield
    except grpc.RpcError as e:
      if (
          isinstance(e, grpc.Call)
          and e.code() in executors_errors.get_grpc_retryable_error_codes()
      ):
        # If an RPC error is raised to us directly, ensure we propagate the
        # proper code to the other side of the service.
        _propagate_grpc_code_error(context, e)
        logging.info('Raised an RPC error')
        if e.code() is grpc.StatusCode.FAILED_PRECONDITION:
          # Raised if a worker needs to be reconfigured.
          logging.info(
              'Executor underneath service raised FailedPrecondition; '
              'invalidating references to this executor.'
          )
          self.DestroyExecutor(request)
        elif e.code() is grpc.StatusCode.UNAVAILABLE:
          # Raised if a worker goes down during invocation.
          logging.info(
              'Executor underneath service unavailable; preemptively '
              'invalidating references to this executor.'
          )
          self.DestroyExecutor(request)
        return blank_response_fn()
      else:
        # Unknown; just reraise, see if anyone else can handle.
        raise e
    except executors_errors.RetryableError as e:
      # Raised if no workers are available during executor construction.
      _set_unavailable_error(context, e)
      return blank_response_fn()
    except (ValueError, TypeError) as e:
      _set_invalid_arg_error(context, e)
      return blank_response_fn()

  def CreateValue(
      self,
      request: executor_pb2.CreateValueRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateValueResponse:
    """Creates a value embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateValueRequest)
    with self._try_handle_request_context(
        request, context, executor_pb2.CreateValueResponse
    ):
      with tracing.span('ExecutorService.CreateValue', 'deserialize_value'):
        value, value_type = value_serialization.deserialize_value(request.value)
      value_id = str(uuid.uuid4())
      coro = self.executor(request, context).create_value(value, value_type)
      future_val = self._run_coro_threadsafe_with_tracing(coro)
      with self._lock:
        self._values[value_id] = future_val
      return executor_pb2.CreateValueResponse(
          value_ref=executor_pb2.ValueRef(id=value_id)
      )

  def CreateCall(
      self,
      request: executor_pb2.CreateCallRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateCallResponse:
    """Creates a call embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateCallRequest)
    with self._try_handle_request_context(
        request, context, executor_pb2.CreateCallResponse
    ):
      function_id = str(request.function_ref.id)
      argument_id = str(request.argument_ref.id)
      with self._lock:
        function_val = self._values[function_id]
        argument_val = self._values[argument_id] if argument_id else None

      async def _process_create_call():
        function = await asyncio.wrap_future(function_val)
        argument = (
            await asyncio.wrap_future(argument_val)
            if argument_val is not None
            else None
        )
        return await self.executor(request, context).create_call(
            function, argument
        )

      coro = _process_create_call()
      result_fut = self._run_coro_threadsafe_with_tracing(coro)
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateCallResponse(
          value_ref=executor_pb2.ValueRef(id=result_id)
      )

  def CreateStruct(
      self,
      request: executor_pb2.CreateStructRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateStructResponse:
    """Creates a struct embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateStructRequest)
    with self._try_handle_request_context(
        request, context, executor_pb2.CreateStructResponse
    ):
      with self._lock:
        elem_futures = [self._values[e.value_ref.id] for e in request.element]
      elem_names = [
          str(elem.name) if elem.name else None for elem in request.element
      ]

      async def _process_create_struct():
        elem_values = await asyncio.gather(
            *[asyncio.wrap_future(v) for v in elem_futures]
        )
        elements = list(zip(elem_names, elem_values))
        struct = structure.Struct(elements)
        return await self.executor(request, context).create_struct(struct)

      result_fut = self._run_coro_threadsafe_with_tracing(
          _process_create_struct()
      )
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateStructResponse(
          value_ref=executor_pb2.ValueRef(id=result_id)
      )

  def CreateSelection(
      self,
      request: executor_pb2.CreateSelectionRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.CreateSelectionResponse:
    """Creates a selection embedded in the executor."""
    py_typecheck.check_type(request, executor_pb2.CreateSelectionRequest)
    with self._try_handle_request_context(
        request, context, executor_pb2.CreateSelectionResponse
    ):
      with self._lock:
        source_fut = self._values[request.source_ref.id]

      async def _process_create_selection():
        source = await asyncio.wrap_future(source_fut)
        return await self.executor(request, context).create_selection(
            source, request.index
        )

      result_fut = self._run_coro_threadsafe_with_tracing(
          _process_create_selection()
      )
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_fut
      return executor_pb2.CreateSelectionResponse(
          value_ref=executor_pb2.ValueRef(id=result_id)
      )

  def Compute(
      self,
      request: executor_pb2.ComputeRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.ComputeResponse:
    """Computes a value embedded in the executor."""
    return self._run_coro_threadsafe_with_tracing(
        self._Compute(request, context)
    ).result()

  async def _Compute(
      self,
      request: executor_pb2.ComputeRequest,
      context: grpc.ServicerContext,
  ) -> executor_pb2.ComputeResponse:
    """Asynchronous implemention of `Compute`."""
    py_typecheck.check_type(request, executor_pb2.ComputeRequest)
    with self._try_handle_request_context(
        request, context, executor_pb2.ComputeResponse
    ):
      value_id = str(request.value_ref.id)
      with self._lock:
        future_val = asyncio.wrap_future(self._values[value_id])
      val = await future_val
      result_val = await val.compute()
      val_type = val.type_signature
      value_proto, _ = value_serialization.serialize_value(result_val, val_type)
      return executor_pb2.ComputeResponse(value=value_proto)

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
    except KeyError as e:
      _set_invalid_arg_error(context, e)
    return executor_pb2.DisposeResponse()
