# Copyright 2022, The TensorFlow Federated Authors.
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
"""A stub connects to a remote executor over gRPC."""

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import remote_executor_stub


@tracing.trace(span=True)
def _request(rpc_func, request):
  """Populates trace context and reraises gRPC errors with retryable info."""
  with tracing.wrap_rpc_in_trace_context():
    try:
      return rpc_func(request)
    except grpc.RpcError as e:
      if (
          isinstance(e, grpc.Call)
          and e.code() in executors_errors.get_grpc_retryable_error_codes()
      ):
        logging.info('Received retryable gRPC error: %s', e)
        raise executors_errors.RetryableGRPCError(e)
      else:
        raise e


class RemoteExecutorGrpcStub(remote_executor_stub.RemoteExecutorStub):
  """A stub connects to a remote executor service over gRPC."""

  def __init__(self, channel: grpc.Channel):
    """Initialize the stub by establishing the connection.

    Args:
      channel: An instance of `grpc.Channel` to use for communication with a
        remote executor service.
    """

    def _channel_status_callback(
        channel_connectivity: grpc.ChannelConnectivity,
    ):
      self._channel_status = channel_connectivity

    self._channel_status = False

    channel.subscribe(_channel_status_callback, try_to_connect=True)
    self._channel = channel
    self._stub = executor_pb2_grpc.ExecutorGroupStub(channel)

  def get_executor(
      self, request: executor_pb2.GetExecutorRequest
  ) -> executor_pb2.GetExecutorResponse:
    """Dispatches a GetExecutor gRPC."""
    return _request(self._stub.GetExecutor, request)

  def create_value(
      self, request: executor_pb2.CreateValueRequest
  ) -> executor_pb2.CreateValueResponse:
    """Dispatches a CreateValue gRPC."""
    return _request(self._stub.CreateValue, request)

  def create_struct(
      self, request: executor_pb2.CreateStructRequest
  ) -> executor_pb2.CreateStructResponse:
    """Dispatches a CreateStruct gRPC."""
    return _request(self._stub.CreateStruct, request)

  def create_call(
      self, request: executor_pb2.CreateCallRequest
  ) -> executor_pb2.CreateCallResponse:
    """Dispatches a CreateCall gRPC."""
    return _request(self._stub.CreateCall, request)

  def create_selection(
      self, request: executor_pb2.CreateSelectionRequest
  ) -> executor_pb2.CreateSelectionResponse:
    """Dispatches a CreateSelection gRPC."""
    return _request(self._stub.CreateSelection, request)

  def compute(
      self, request: executor_pb2.ComputeRequest
  ) -> executor_pb2.ComputeResponse:
    """Dispatches a Compute gRPC."""
    return _request(self._stub.Compute, request)

  def dispose(
      self, request: executor_pb2.DisposeRequest
  ) -> executor_pb2.DisposeResponse:
    """Dispatches a Dispose gRPC."""
    return _request(self._stub.Dispose, request)

  def dispose_executor(
      self, request: executor_pb2.DisposeExecutorRequest
  ) -> executor_pb2.DisposeExecutorResponse:
    return _request(self._stub.DisposeExecutor, request)

  @property
  def is_ready(self) -> bool:
    """True if the gRPC connection is ready."""
    return self._channel_status == grpc.ChannelConnectivity.READY
