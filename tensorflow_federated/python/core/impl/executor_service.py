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

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc


class ExecutorService(executor_pb2_grpc.ExecutorServicer):
  """A wrapper around a target executor that makes it into a gRPC service."""

  # pylint: disable=useless-super-delegation
  def __init__(self, *args, **kwargs):
    super(ExecutorService, self).__init__(*args, **kwargs)

  def CreateValue(self, request, context):
    """Creates a value embedded in the executor.

    Args:
      request: An instance of `executor_pb2.CreateValueRequest`.
      context: An instance of `grpc.ServicerContext`.

    Returns:
      An instance of `executor_pb2.CreateValueResponse`.
    """
    return executor_pb2.CreateValueResponse()
