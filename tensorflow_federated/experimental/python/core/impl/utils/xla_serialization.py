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

from typing import List, Optional

from jax.lib.xla_bridge import xla_client
import numpy as np

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
    raise ValueError('Not a serialized `HloModuleProto`: {}.'.format(
        str(any_pb.type_url)))
  return xla_client.XlaComputation(any_pb.value)


def _make_xla_binding_for_type(
    tensor_indexes: List[int],
    type_spec: Optional[computation_types.Type]) -> Optional[pb.Xla.Binding]:
  """Generates an XLA binding for TFF type `type_spec`.

  In the generated binding, tensors are assigned indexes in consecutive order
  of DFS traversal.

  Args:
    tensor_indexes: The list of tensor indexes to use in the binding, in the
      order matching the order of flattened `type_spec`.
    type_spec: The type to generate the binding for. Must be either an instance
      of `computation_types.Type`, or `None`.

  Returns:
    The generated binding (either `pb.Xla.Binding` or `None`).
  """
  if type_spec is None:
    return None

  py_typecheck.check_type(type_spec, computation_types.Type)
  py_typecheck.check_type(tensor_indexes, list)

  def _make_starting_at_index(type_spec, idx):
    if isinstance(type_spec, computation_types.TensorType):
      return pb.Xla.Binding(
          tensor=pb.Xla.TensorBinding(index=tensor_indexes[idx])), idx + 1

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


def _remove_struct_element_names_from_tff_type(type_spec):
  """Removes names of struct elements from `type_spec`.

  Args:
    type_spec: An instance of `computation_types.Type` that must be a tensor, a
      (possibly) nested structure of tensors, or a function.

  Returns:
    A modified version of `type_spec` with element names in stuctures removed.

  Raises:
    TypeError: if arg is of the wrong type.
  """
  if type_spec is None:
    return None
  if isinstance(type_spec, computation_types.FunctionType):
    return computation_types.FunctionType(
        _remove_struct_element_names_from_tff_type(type_spec.parameter),
        _remove_struct_element_names_from_tff_type(type_spec.result))
  if isinstance(type_spec, computation_types.TensorType):
    return type_spec
  py_typecheck.check_type(type_spec, computation_types.StructType)
  return computation_types.StructType([
      (None, _remove_struct_element_names_from_tff_type(v))
      for _, v in structure.iter_elements(type_spec)
  ])


def create_xla_tff_computation(xla_computation, tensor_indexes, type_spec):
  """Creates an XLA TFF computation.

  Args:
    xla_computation: An instance of `xla_client.XlaComputation`.
    tensor_indexes: The list of tensor indexes to use in the parameter binding,
      in the order matching the order of flattened parameter in `type_spec`.
    type_spec: The TFF type of the computation to be constructed.

  Returns:
    An instance of `pb.Computation`.

  Raises:
    ValueError: if the arguments are invalid or incompatible with each other,
      e.g., because the TFF types mismatch.
  """
  py_typecheck.check_type(xla_computation, xla_client.XlaComputation)
  py_typecheck.check_type(tensor_indexes, list)
  py_typecheck.check_type(type_spec, computation_types.FunctionType)
  parameter_binding = _make_xla_binding_for_type(tensor_indexes,
                                                 type_spec.parameter)
  result_binding = _make_xla_binding_for_type(
      list(range(len(structure.flatten(type_spec.result)))), type_spec.result)
  reconstructed_type = xla_computation_and_bindings_to_tff_type(
      xla_computation, parameter_binding, result_binding)
  py_typecheck.check_type(reconstructed_type, computation_types.FunctionType)
  expected_type = _remove_struct_element_names_from_tff_type(type_spec)
  if not reconstructed_type.is_equivalent_to(expected_type):
    raise ValueError(
        'The TFF type of the XLA computation {} does not match the expected '
        'TFF type {}.'.format(str(reconstructed_type), str(expected_type)))
  return pb.Computation(
      type=type_serialization.serialize_type(type_spec),
      xla=pb.Xla(
          hlo_module=pack_xla_computation(xla_computation),
          parameter=parameter_binding,
          result=result_binding))


def xla_computation_and_bindings_to_tff_type(xla_computation, parameter_binding,
                                             result_binding):
  """Constructs the TFF type from an `xla_client.XlaComputation` and bindings.

  NOTE: This is a helper function, primarily intended for use in checking the
  well-formedness of TFF computations during serialization and deserialization,
  and for serialization testing/debugging purposes.

  Args:
    xla_computation: An instance of `xla_client.XlaComputation` to get type for.
    parameter_binding: An instance of `pb.Xla.Binding` for the parameter.
    result_binding: An instance of `pb.Xla.Binding` for the result.

  Returns:
    An instance of `computation_types.Type`.
  """
  py_typecheck.check_type(xla_computation, xla_client.XlaComputation)
  program_shape = xla_computation.program_shape()
  return computation_types.FunctionType(
      xla_shapes_and_binding_to_tff_type(program_shape.parameter_shapes(),
                                         parameter_binding),
      xla_shapes_and_binding_to_tff_type([program_shape.result_shape()],
                                         result_binding))


def xla_shapes_and_binding_to_tff_type(xla_shapes, binding):
  """Constructs the TFF type from a list of `xla_client.Shape` and a binding.

  Args:
    xla_shapes: A list of `xla_client.Shape` instances.
    binding: An instance of `pb.Xla.Binding` (or `None` if there's none).

  Returns:
    An instance of `computation_types.Type` (or `None`).
  """
  py_typecheck.check_type(xla_shapes, list)
  if binding is not None:
    py_typecheck.check_type(binding, pb.Xla.Binding)
  tensor_shapes = []
  for shape in xla_shapes:
    tensor_shapes += flatten_xla_shape(shape)
  unused_shape_indexes = set(range(len(tensor_shapes)))

  def _get_type(binding):
    if binding is None:
      return None
    kind = binding.WhichOneof('binding')
    if kind == 'tensor':
      index = binding.tensor.index
      if (index < 0) or (index >= len(tensor_shapes)):
        raise ValueError(
            'Binding refers to an inexistent index {}.'.format(index))
      if index not in unused_shape_indexes:
        raise ValueError(
            'Duplicate bindings referring to index {}.'.format(index))
      unused_shape_indexes.remove(index)
      shape = tensor_shapes[index]
      return computation_types.TensorType(shape.numpy_dtype(),
                                          shape.dimensions())
    if kind == 'struct':
      return computation_types.StructType([
          (None, _get_type(x)) for x in binding.struct.element
      ])
    if kind is None:
      return None
    raise ValueError('Unrecognized binding type {}.'.format(kind))

  tff_type = _get_type(binding)
  if unused_shape_indexes:
    raise ValueError(
        'Binding fails to capture tensors {}.'.format(unused_shape_indexes))
  return tff_type


def flatten_xla_shape(xla_shape):
  """Flattens a possibly nested tuple XLA shape into a list of tensor shapes.

  Args:
    xla_shape: An instance of `xla_client.Shape` (could be a nested structure).

  Returns:
    A Python list of `xla_client.Shape` instances representing tensors.
  """
  py_typecheck.check_type(xla_shape, xla_client.Shape)
  if xla_shape.is_tuple():
    tensor_shapes = []
    for shape in xla_shape.tuple_shapes():
      tensor_shapes += flatten_xla_shape(shape)
    return tensor_shapes
  else:
    # Must be a tensor (array) type; verify this by probing for dimensions and
    # element_type, since there's no explicit way to check otherwise.
    py_typecheck.check_type(xla_shape.element_type(), np.dtype)
    py_typecheck.check_type(xla_shape.dimensions(), tuple)
    return [xla_shape]
