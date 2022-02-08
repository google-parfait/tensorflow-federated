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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A stub connects to a remote executor over gRPC."""

from absl import logging
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import remote_executor_stub

_RETRYABLE_GRPC_ERRORS = frozenset([
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
])


def _is_retryable_grpc_error(error):
  """Predicate defining what is a retryable gRPC error."""
  return (isinstance(error, grpc.RpcError) and
          error.code() not in _RETRYABLE_GRPC_ERRORS)


@tracing.trace(span=True)
def _request(rpc_func, request):
  """Populates trace context and reraises gRPC errors with retryable info."""
  with tracing.wrap_rpc_in_trace_context():
    try:
      return rpc_func(request)
    except grpc.RpcError as e:
      if _is_retryable_grpc_error(e):
        logging.info("Received retryable gRPC error: %s", e)
        raise executors_errors.RetryableError(e)
      else:
        raise


class RemoteExecutorGrpcStub(remote_executor_stub.RemoteExecutorStub):
  """A stub connects to a remote executor service over gRPC."""

  def __init__(self, channel: grpc.Channel):
    """Initialize the stub by establishing the connection.

    Args:
      channel: An instance of `grpc.Channel` to use for communication with a
        remote executor service.
    """

    def _channel_status_callback(
        channel_connectivity: grpc.ChannelConnectivity):
      self._channel_status = channel_connectivity

    self._channel_status = False

    channel.subscribe(_channel_status_callback, try_to_connect=True)
    self._channel = channel
    self._stub = executor_pb2_grpc.ExecutorStub(channel)

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
      self,
      request: executor_pb2.ComputeRequest) -> executor_pb2.ComputeResponse:
    """Dispatches a Compute gRPC."""
    return _request(self._stub.Compute, request)

  def set_cardinalities(
      self, request: executor_pb2.SetCardinalitiesRequest
  ) -> executor_pb2.SetCardinalitiesResponse:
    """Dispatches a SetCardinalities gRPC."""
    return _request(self._stub.SetCardinalities, request)

  def dispose(
      self,
      request: executor_pb2.DisposeRequest) -> executor_pb2.DisposeResponse:
    """Dispatches a Dispose gRPC."""
    return _request(self._stub.Dispose, request)

  def clear_executor(
      self, request: executor_pb2.ClearExecutorRequest
  ) -> executor_pb2.ClearExecutorResponse:
    """Dispatches a ClearExecutor gRPC."""
    return _request(self._stub.ClearExecutor, request)

  @property
  def is_ready(self) -> bool:
    """True if the gRPC connection is ready."""
    return self._channel_status == grpc.ChannelConnectivity.READY
