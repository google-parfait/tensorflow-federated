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
"""Utilities for serializing and deserializing TFF types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import placement_literals


def serialize_type(type_spec):
  """Serializes 'type_spec' as a pb.Type.

  NOTE: Currently only serialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_spec: Either an instance of types.Type, or something convertible to it
      by types.to_type(), or None.

  Returns:
    The corresponding instance of pb.Type, or None if the argument was None.

  Raises:
    TypeError: if the argument is of the wrong type.
    NotImplementedError: for type variants for which serialization is not
      implemented.
  """
  # TODO(b/113112885): Implement serialization of the remaining types.
  if type_spec is None:
    return None
  target = types.to_type(type_spec)
  py_typecheck.check_type(target, types.Type)
  if isinstance(target, types.TensorType):
    return pb.Type(
        tensor=pb.TensorType(
            dtype=target.dtype.as_datatype_enum, shape=target.shape.as_proto()))
  elif isinstance(target, types.SequenceType):
    return pb.Type(
        sequence=pb.SequenceType(element=serialize_type(target.element)))
  elif isinstance(target, types.NamedTupleType):
    return pb.Type(
        tuple=pb.NamedTupleType(element=[
            pb.NamedTupleType.Element(name=e[0], value=serialize_type(e[1]))
            for e in target.elements
        ]))
  elif isinstance(target, types.FunctionType):
    return pb.Type(
        function=pb.FunctionType(
            parameter=serialize_type(target.parameter),
            result=serialize_type(target.result)))
  elif isinstance(target, types.PlacementType):
    return pb.Type(placement=pb.PlacementType())
  elif isinstance(target, types.FederatedType):
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
  """Deserializes 'type_proto' as a types.Type.

  NOTE: Currently only deserialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_proto: An instance of pb.Type or None.

  Returns:
    The corresponding instance of types.Type (or None if the argument was None).

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
    return types.TensorType(
        dtype=tf.DType(type_proto.tensor.dtype),
        shape=tf.TensorShape(type_proto.tensor.shape))
  elif type_variant == 'sequence':
    return types.SequenceType(deserialize_type(type_proto.sequence.element))
  elif type_variant == 'tuple':
    return types.NamedTupleType([(lambda k, v: (k, v) if k else v)(
        e.name, deserialize_type(e.value)) for e in type_proto.tuple.element])
  elif type_variant == 'function':
    return types.FunctionType(
        parameter=deserialize_type(type_proto.function.parameter),
        result=deserialize_type(type_proto.function.result))
  elif type_variant == 'placement':
    return types.PlacementType()
  elif type_variant == 'federated':
    placement_oneof = type_proto.federated.placement.WhichOneof('placement')
    if placement_oneof == 'value':
      return types.FederatedType(
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
