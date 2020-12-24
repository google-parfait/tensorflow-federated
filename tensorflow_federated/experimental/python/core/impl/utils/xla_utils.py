# Copyright 2020, The TensorFlow Federated Authors.
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
"""Experimental utilities for serializing and deserializing XLA code."""

from jax.lib.xla_bridge import xla_client

from google.protobuf import any_pb2
from tensorflow_federated.python.common_libs import py_typecheck

_HLO_MODULE_PROTO_URI = 'type.googleapis.com/xla.HloModuleProto'


def pack_xla_computation(xla_computation):
  """Pack a `XlaComputation` into `Any` proto with a HLO module proto payload.

  Args:
    xla_computation: An instance of `xla_client.XlaComputation` to pack.

  Returns:
    A `google.protobuf.Any` protocol buffer message containing this
    computation's `HloModuleProto` in a binary-serialized form.

  Raises:
    TypeError: if `xla_computation` is not an `xla_client.XlaComputation`.
  """
  py_typecheck.check_type(xla_computation, xla_client.XlaComputation)
  return any_pb2.Any(
      type_url=_HLO_MODULE_PROTO_URI,
      value=xla_computation.as_serialized_hlo_module_proto())


def unpack_xla_computation(any_pb):
  """Unpacks an `Any` proto to an `XlaComputation`.

  Args:
    any_pb: An instance of `google.protobuf.Any` to unpack.

  Returns:
    The unpacked instance of `xla_client.XlaComputation`.

  Raises:
    TypeError: if `any_pb` is not an `Any` protocol buffer message.
    ValueError: if the object packed into `any_pb` cannot be unpacked.
  """
  py_typecheck.check_type(any_pb, any_pb2.Any)
  if any_pb.type_url != _HLO_MODULE_PROTO_URI:
    raise ValueError('Not a serialized `HloModuleProto`.')
  return xla_client.XlaComputation(any_pb.value)
