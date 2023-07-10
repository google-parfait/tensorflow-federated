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
"""A library of (de)serialization functions for computation types."""

import weakref

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _to_tensor_type_proto(
    tensor_type: computation_types.TensorType,
) -> pb.TensorType:
  shape = tensor_type.shape
  if shape.dims is None:
    dims = None
    unknown_rank = True
  else:
    dims = [d.value if d.value is not None else -1 for d in shape.dims]
    unknown_rank = False
  return pb.TensorType(
      dtype=tensor_type.dtype.base_dtype.as_datatype_enum,
      dims=dims,
      unknown_rank=unknown_rank,
  )


def _to_tensor_shape(tensor_type_proto: pb.TensorType) -> tf.TensorShape:
  if tensor_type_proto.unknown_rank:
    return tf.TensorShape(None)
  elif not hasattr(tensor_type_proto, 'dims'):
    return tf.TensorShape([])
  dims = [dim if dim >= 0 else None for dim in tensor_type_proto.dims]
  return tf.TensorShape(dims)


# Manual cache used rather than `cachetools.cached` due to incompatibility
# with `WeakKeyDictionary`. We want to use a `WeakKeyDictionary` so that
# cache entries are destroyed once the types they index no longer exist.
_type_serialization_cache = weakref.WeakKeyDictionary({})


def serialize_type(type_spec: computation_types.Type) -> pb.Type:
  """Serializes 'type_spec' as a pb.Type.

  Note: Currently only serialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    The corresponding instance of `pb.Type`.

  Raises:
    TypeError: if the argument is of the wrong type.
    NotImplementedError: for type variants for which serialization is not
      implemented.
  """
  cached_proto = _type_serialization_cache.get(type_spec, None)
  if cached_proto is not None:
    return cached_proto
  if isinstance(type_spec, computation_types.TensorType):
    proto = pb.Type(tensor=_to_tensor_type_proto(type_spec))
  elif isinstance(type_spec, computation_types.SequenceType):
    proto = pb.Type(
        sequence=pb.SequenceType(element=serialize_type(type_spec.element))
    )
  elif isinstance(type_spec, computation_types.StructType):
    proto = pb.Type(
        struct=pb.StructType(
            element=[
                pb.StructType.Element(name=e[0], value=serialize_type(e[1]))
                for e in structure.iter_elements(type_spec)
            ]
        )
    )
  elif isinstance(type_spec, computation_types.FunctionType):
    if type_spec.parameter is not None:
      serialized_parameter = serialize_type(type_spec.parameter)
    else:
      serialized_parameter = None
    proto = pb.Type(
        function=pb.FunctionType(
            parameter=serialized_parameter,
            result=serialize_type(type_spec.result),
        )
    )
  elif isinstance(type_spec, computation_types.PlacementType):
    proto = pb.Type(placement=pb.PlacementType())
  elif isinstance(type_spec, computation_types.FederatedType):
    proto = pb.Type(
        federated=pb.FederatedType(
            member=serialize_type(type_spec.member),
            placement=pb.PlacementSpec(
                value=pb.Placement(uri=type_spec.placement.uri)
            ),
            all_equal=type_spec.all_equal,
        )
    )
  else:
    raise NotImplementedError

  _type_serialization_cache[type_spec] = proto
  return proto


def deserialize_type(type_proto: pb.Type) -> computation_types.Type:
  """Deserializes 'type_proto' as a `tff.Type`.

  Note: Currently only deserialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_proto: A `pb.Type` to deserialize.

  Returns:
    The corresponding instance of `tff.Type`.

  Raises:
    TypeError: If the argument is of the wrong type.
    NotImplementedError: For type variants for which deserialization is not
      implemented.
  """
  type_variant = type_proto.WhichOneof('type')
  if type_variant == 'tensor':
    tensor_proto = type_proto.tensor
    return computation_types.TensorType(
        dtype=tf.dtypes.as_dtype(tensor_proto.dtype),
        shape=_to_tensor_shape(tensor_proto),
    )
  elif type_variant == 'sequence':
    return computation_types.SequenceType(
        deserialize_type(type_proto.sequence.element)
    )
  elif type_variant == 'struct':

    def empty_str_to_none(s):
      if not s:
        return None
      return s

    return computation_types.StructType(
        [
            (empty_str_to_none(e.name), deserialize_type(e.value))
            for e in type_proto.struct.element
        ],
        convert=False,
    )
  elif type_variant == 'function':
    if type_proto.function.HasField('parameter'):
      parameter_type = deserialize_type(type_proto.function.parameter)
    else:
      parameter_type = None
    result_type = deserialize_type(type_proto.function.result)
    return computation_types.FunctionType(
        parameter=parameter_type, result=result_type
    )
  elif type_variant == 'placement':
    return computation_types.PlacementType()
  elif type_variant == 'federated':
    placement_oneof = type_proto.federated.placement.WhichOneof('placement')
    if placement_oneof == 'value':
      return computation_types.FederatedType(
          member=deserialize_type(type_proto.federated.member),
          placement=placements.uri_to_placement_literal(
              type_proto.federated.placement.value.uri
          ),
          all_equal=type_proto.federated.all_equal,
      )
    else:
      raise NotImplementedError(
          'Deserialization of federated types with placement spec as {} '
          'is not currently implemented yet.'.format(placement_oneof)
      )
  else:
    raise NotImplementedError('Unknown type variant {}.'.format(type_variant))
