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
"""A base Python interface for all stubs handles remote executions."""

import abc

from tensorflow_federated.proto.v0 import executor_pb2


class RemoteExecutorStub(abc.ABC):
  """Represents the interface of stubs call to an remote executor instance."""

  @abc.abstractmethod
  def get_executor(
      self, request: executor_pb2.GetExecutorRequest
  ) -> executor_pb2.GetExecutorResponse:
    """Invokes GetExecutor remotely."""
    raise NotImplementedError

  @abc.abstractmethod
  def create_value(
      self, request: executor_pb2.CreateValueRequest
  ) -> executor_pb2.CreateValueResponse:
    """Invokes CreateValue remotely.

    Args:
      request: CreateValueRequest.

    Returns:
      CreateValueResponse.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def create_struct(
      self, request: executor_pb2.CreateStructRequest
  ) -> executor_pb2.CreateStructResponse:
    """Invokes CreateStruct in a remote TFF runtime.

    Args:
      request: CreateStructRequest.

    Returns:
      CreateStructResponse.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def create_call(
      self, request: executor_pb2.CreateCallRequest
  ) -> executor_pb2.CreateCallResponse:
    """Invokes CreateCall in a remote TFF runtime.

    Args:
      request: CreateCallRequest.

    Returns:
      CreateCallResponse.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def create_selection(
      self, request: executor_pb2.CreateSelectionRequest
  ) -> executor_pb2.CreateSelectionResponse:
    """Invokes CreateSelection in a remote TFF runtime.

    Args:
      request: CreateSelectionRequest.

    Returns:
      CreateSelectionResponse.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def compute(
      self, request: executor_pb2.ComputeRequest
  ) -> executor_pb2.ComputeResponse:
    """Invokes Compute in a remote TFF runtime.

    Args:
      request: ComputeRequest.

    Returns:
      ComputeResponse.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def dispose(
      self, request: executor_pb2.DisposeRequest
  ) -> executor_pb2.DisposeResponse:
    """Invokes Dispose in a remote TFF runtime.

    Args:
      request: DisposeRequest.

    Returns:
      DisposeResponse.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def dispose_executor(
      self, request: executor_pb2.DisposeExecutorRequest
  ) -> executor_pb2.DisposeExecutorResponse:
    """Invokes `DisposeExecutor` in a remote TFF runtime."""
    raise NotImplementedError

  @property
  def is_ready(self) -> bool:
    """Tells if the connection to remote is established."""
    raise NotImplementedError
