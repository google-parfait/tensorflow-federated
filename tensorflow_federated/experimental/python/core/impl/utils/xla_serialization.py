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

from typing import Optional

from jax.lib.xla_bridge import xla_client

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization

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


def _make_xla_binding_for_type(
    type_spec: Optional[computation_types.Type]) -> Optional[pb.Xla.Binding]:
  """Generates an XLA binding for TFF type `type_spec`.

  In the generated binding, tensors are assigned indexes in consecutive order
  of DFS traversal.

  Args:
    type_spec: The type to generate the binding for. Must be either an instance
      of `computation_types.Type`, or `None`.

  Returns:
    The generated binding (either `pb.Xla.Binding` or `None`).
  """
  if type_spec is None:
    return None

  py_typecheck.check_type(type_spec, computation_types.Type)

  def _make_starting_at_index(type_spec, idx):
    if isinstance(type_spec, computation_types.TensorType):
      return pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=idx)), idx + 1

    if isinstance(type_spec, computation_types.StructType):
      elements = []
      for _, v in structure.iter_elements(type_spec):
        binding, idx = _make_starting_at_index(v, idx)
        elements.append(binding)
      return pb.Xla.Binding(struct=pb.Xla.StructBinding(element=elements)), idx

    raise NotImplementedError('XLA bindings for {} are unsupported'.format(
        str(type_spec)))

  binding, _ = _make_starting_at_index(type_spec, 0)
  return binding


def create_xla_tff_computation(xla_computation, type_spec):
  """Creates an XLA TFF computation.

  Args:
    xla_computation: An instance of `xla_client.XlaComputation`.
    type_spec: The TFF type of the computation to be constructed.

  Returns:
    An instance of `pb.Computation`.
  """
  py_typecheck.check_type(xla_computation, xla_client.XlaComputation)
  py_typecheck.check_type(type_spec, computation_types.FunctionType)
  return pb.Computation(
      type=type_serialization.serialize_type(type_spec),
      xla=pb.Xla(
          hlo_module=pack_xla_computation(xla_computation),
          parameter=_make_xla_binding_for_type(type_spec.parameter),
          result=_make_xla_binding_for_type(type_spec.result)))
