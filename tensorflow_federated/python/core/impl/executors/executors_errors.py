# Copyright 2021, The TensorFlow Federated Authors.
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
"""Custom exceptions and symbols for TFF executors."""

import typing
from typing import Any

import grpc


class RetryableError(Exception):
  """Raised when execution fails and can be retried."""


class RetryableGRPCError(RetryableError, grpc.RpcError, grpc.Call):
  """Raised when execution fails across a gRPC connection and can be retried."""

  def __init__(self, e: grpc.Call):
    self._grpc_error = e
    super().__init__(e)

  def code(self) -> grpc.StatusCode:
    return self._grpc_error.code()

  def details(self) -> str:
    return self._grpc_error.details()

  def initial_metadata(self) -> Any:
    return self._grpc_error.initial_metadata()

  def trailing_metadata(self) -> Any:
    return self._grpc_error.trailing_metadata()


def get_grpc_retryable_error_codes() -> set[grpc.StatusCode]:
  """Returns gRPC retryable error codes."""
  return set([
      grpc.StatusCode.UNAVAILABLE,
      grpc.StatusCode.FAILED_PRECONDITION,
  ])


class RetryableAbslStatusError(RetryableError):
  """Raised when execution fails with an absl status error and can be retried."""


def get_absl_status_retryable_error_codes() -> set[int]:
  """Returns Absl retryable error codes."""
  return set([
      14,
      9,
  ])


def is_absl_status_retryable_error(exception: Exception) -> bool:
  """Checks if the exception is an absl status error that can be retried."""
  if (not hasattr(exception, 'status') or
      not hasattr(exception.status, 'code_int')):
    return False
  code = exception.status.code_int()
  return code in get_absl_status_retryable_error_codes()


class CardinalityError(Exception):
  """Raised when a value in a stack does not match the stack's cardinality."""
