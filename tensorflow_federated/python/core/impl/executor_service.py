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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import asyncio
import functools
import threading
import uuid
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_service_utils
from tensorflow_federated.python.core.impl import executor_value_base


class ExecutorService(executor_pb2_grpc.ExecutorServicer):
  """A wrapper around a target executor that makes it into a gRPC service."""

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
        target=functools.partial(run_loop, self._event_loop))
    self._thread.start()

  def __del__(self):
    self._event_loop.call_soon_threadsafe(self._event_loop.stop)
    self._thread.join()

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
      function = function_val.result()
      argument = argument_val.result() if argument_val is not None else None
      result_val = asyncio.run_coroutine_threadsafe(
          self._executor.create_call(function, argument), self._event_loop)
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_val
      return executor_pb2.CreateCallResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
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
        element_vals = [self._values[e.value_ref.id] for e in request.element]
      elements = []
      for idx, elem in enumerate(request.element):
        elements.append((str(elem.name), element_vals[idx].result()))
      anon_tuple = anonymous_tuple.AnonymousTuple(elements)
      result_val = asyncio.run_coroutine_threadsafe(
          self._executor.create_tuple(anon_tuple), self._event_loop)
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_val
      return executor_pb2.CreateTupleResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
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
        source_val = self._values[request.source_ref.id]
      source = source_val.result()
      which_selection = request.WhichOneof('selection')
      if which_selection == 'name':
        coro = self._executor.create_selection(source, name=request.name)
      else:
        coro = self._executor.create_selection(source, index=request.index)
      result_val = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
      result_id = str(uuid.uuid4())
      with self._lock:
        self._values[result_id] = result_val
      return executor_pb2.CreateSelectionResponse(
          value_ref=executor_pb2.ValueRef(id=result_id))
    except (ValueError, TypeError) as err:
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
    py_typecheck.check_type(request, executor_pb2.ComputeRequest)
    try:
      value_id = str(request.value_ref.id)
      with self._lock:
        future_val = self._values[value_id]
      val = future_val.result()
      py_typecheck.check_type(val, executor_value_base.ExecutorValue)
      result = asyncio.run_coroutine_threadsafe(val.compute(), self._event_loop)
      result_val = result.result()
      val_type = val.type_signature
      value_proto, _ = executor_service_utils.serialize_value(
          result_val, val_type)
      return executor_pb2.ComputeResponse(value=value_proto)
    except (ValueError, TypeError) as err:
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details(str(err))
      return executor_pb2.ComputeResponse()
