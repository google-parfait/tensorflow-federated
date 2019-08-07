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

import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_service_utils
from tensorflow_federated.python.core.impl import executor_value_base


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


class RemoteExecutor(executor_base.Executor):
  """The remote executor is a local proxy for a remote executor instance.

  NOTE: This component is only available in Python 3.
  """

  # TODO(b/134543154): Switch to using an asynchronous gRPC client so we don't
  # have to block on all those calls.

  def __init__(self, channel):
    """Creates a remote executor.

    Args:
      channel: An instance of `grpc.Channel` to use for communication with the
        remote executor service.
    """
    py_typecheck.check_type(channel, grpc.Channel)
    self._stub = executor_pb2_grpc.ExecutorStub(channel)

  async def create_value(self, value, type_spec=None):
    value_proto, type_spec = (
        executor_service_utils.serialize_value(value, type_spec))
    response = self._stub.CreateValue(
        executor_pb2.CreateValueRequest(value=value_proto))
    py_typecheck.check_type(response, executor_pb2.CreateValueResponse)
    return RemoteValue(response.value_ref, type_spec, self)

  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, RemoteValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    if arg is not None:
      py_typecheck.check_type(arg, RemoteValue)
    response = self._stub.CreateCall(
        executor_pb2.CreateCallRequest(
            function_ref=comp.value_ref,
            argument_ref=(arg.value_ref if arg is not None else None)))
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
    response = self._stub.CreateTuple(
        executor_pb2.CreateTupleRequest(element=proto_elem))
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
    response = self._stub.Compute(request)
    py_typecheck.check_type(response, executor_pb2.ComputeResponse)
    value, _ = executor_service_utils.deserialize_value(response.value)
    return value
