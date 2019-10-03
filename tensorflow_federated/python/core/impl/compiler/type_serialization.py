# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""Utilities for serializing and deserializing TFF computation_types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import placement_literals


def _to_tensor_type_proto(tensor_type):
  py_typecheck.check_type(tensor_type, computation_types.TensorType)
  shape = tensor_type.shape
  if shape.dims is None:
    dims = None
  else:
    dims = [d.value if d.value is not None else -1 for d in shape.dims]
  return pb.TensorType(
      dtype=tensor_type.dtype.base_dtype.as_datatype_enum,
      dims=dims,
      unknown_rank=dims is None)


def _to_tensor_shape(tensor_type_proto):
  py_typecheck.check_type(tensor_type_proto, pb.TensorType)
  if not hasattr(tensor_type_proto, 'dims'):
    if tensor_type_proto.unknown_rank:
      return tf.TensorShape(None)
    else:
      return tf.TensorShape([])
  dims = [dim if dim >= 0 else None for dim in tensor_type_proto.dims]
  return tf.TensorShape(dims)


def serialize_type(type_spec):
  """Serializes 'type_spec' as a pb.Type.

  NOTE: Currently only serialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_spec: Either an instance of computation_types.Type, or something
      convertible to it by computation_types.to_type(), or None.

  Returns:
    The corresponding instance of `pb.Type`, or `None` if the argument was
      `None`.

  Raises:
    TypeError: if the argument is of the wrong type.
    NotImplementedError: for type variants for which serialization is not
      implemented.
  """
  # TODO(b/113112885): Implement serialization of the remaining types.
  if type_spec is None:
    return None
  target = computation_types.to_type(type_spec)
  py_typecheck.check_type(target, computation_types.Type)
  if isinstance(target, computation_types.TensorType):
    return pb.Type(tensor=_to_tensor_type_proto(target))
  elif isinstance(target, computation_types.SequenceType):
    return pb.Type(
        sequence=pb.SequenceType(element=serialize_type(target.element)))
  elif isinstance(target, computation_types.NamedTupleType):
    return pb.Type(
        tuple=pb.NamedTupleType(element=[
            pb.NamedTupleType.Element(name=e[0], value=serialize_type(e[1]))
            for e in anonymous_tuple.iter_elements(target)
        ]))
  elif isinstance(target, computation_types.FunctionType):
    return pb.Type(
        function=pb.FunctionType(
            parameter=serialize_type(target.parameter),
            result=serialize_type(target.result)))
  elif isinstance(target, computation_types.PlacementType):
    return pb.Type(placement=pb.PlacementType())
  elif isinstance(target, computation_types.FederatedType):
    if isinstance(target.placement, placement_literals.PlacementLiteral):
      return pb.Type(
          federated=pb.FederatedType(
              member=serialize_type(target.member),
              placement=pb.PlacementSpec(
                  value=pb.Placement(uri=target.placement.uri)),
              all_equal=target.all_equal))
    else:
      raise NotImplementedError(
          'Serialization of federated types with placements specifications '
          'of type {} is not currently implemented yet.'.format(
              type(target.placement)))
  else:
    raise NotImplementedError


def deserialize_type(type_proto):
  """Deserializes 'type_proto' as a computation_types.Type.

  NOTE: Currently only deserialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_proto: An instance of pb.Type or None.

  Returns:
    The corresponding instance of computation_types.Type (or None if the
    argument was None).

  Raises:
    TypeError: if the argument is of the wrong type.
    NotImplementedError: for type variants for which deserialization is not
      implemented.
  """
  # TODO(b/113112885): Implement deserialization of the remaining types.
  if type_proto is None:
    return None
  py_typecheck.check_type(type_proto, pb.Type)
  type_variant = type_proto.WhichOneof('type')
  if type_variant is None:
    return None
  elif type_variant == 'tensor':
    tensor_proto = type_proto.tensor
    return computation_types.TensorType(
        dtype=tf.DType(tensor_proto.dtype),
        shape=_to_tensor_shape(tensor_proto))
  elif type_variant == 'sequence':
    return computation_types.SequenceType(
        deserialize_type(type_proto.sequence.element))
  elif type_variant == 'tuple':
    return computation_types.NamedTupleType([
        (lambda k, v: (k, v) if k else v)(e.name, deserialize_type(e.value))
        for e in type_proto.tuple.element
    ])
  elif type_variant == 'function':
    return computation_types.FunctionType(
        parameter=deserialize_type(type_proto.function.parameter),
        result=deserialize_type(type_proto.function.result))
  elif type_variant == 'placement':
    return computation_types.PlacementType()
  elif type_variant == 'federated':
    placement_oneof = type_proto.federated.placement.WhichOneof('placement')
    if placement_oneof == 'value':
      return computation_types.FederatedType(
          member=deserialize_type(type_proto.federated.member),
          placement=placement_literals.uri_to_placement_literal(
              type_proto.federated.placement.value.uri),
          all_equal=type_proto.federated.all_equal)
    else:
      raise NotImplementedError(
          'Deserialization of federated types with placement spec as {} '
          'is not currently implemented yet.'.format(placement_oneof))
  else:
    raise NotImplementedError('Unknown type variant {}.'.format(type_variant))
