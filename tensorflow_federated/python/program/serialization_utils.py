# Copyright 2023, The TensorFlow Federated Authors.
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
"""Utilities for packing and unpacking bytes.

This library primarily uses `struct` to pack and unpack bytes. All the format
strings in this library use the network byte order, the assumption is that this
should be safe as long as both the pack and unpack functions use the same byte
order. See
https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment for
more information.

Important: This library only uses `pickle` to serialize Python containers (e.g.
`Sequence`, `Mapping`, `NamedTuple`, etc) and does not use `pickle` to serialize
the values held in those containers.
"""

from collections.abc import Sequence
import importlib
import pickle
import struct
from typing import Protocol, TypeVar

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import serializable as serializable_lib
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.program import structure_utils


# The maximum size allowed for serialized `tf.data.Dataset`s.
_MAX_SERIALIZED_DATASET_SIZE = 100 * (1024**2)  # 100 MB


_T = TypeVar('_T')


class PackFn(Protocol[_T]):

  def __call__(self, _: _T) -> bytes:
    ...


class UnpackFn(Protocol[_T]):

  def __call__(self, buffer: bytes, offset: int = 0) -> tuple[_T, int]:
    ...


def _pack_length(buffer: bytes) -> bytes:
  """Packs the length of `buffer` as bytes."""
  length = len(buffer)
  length_bytes = struct.pack('!Q', length)
  return length_bytes


def _unpack_length_from(buffer: bytes, offset: int = 0) -> tuple[int, int]:
  """Unpacks a length from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked length and the packed bytes size.
  """
  length_size = struct.calcsize('!Q')
  length, *_ = struct.unpack_from('!Q', buffer, offset=offset)
  return length, length_size


def pack_str(value: str) -> bytes:
  """Packs a `str` as bytes."""
  str_bytes = value.encode('utf-8')
  length_bytes = _pack_length(str_bytes)
  return length_bytes + str_bytes


def unpack_str_from(buffer: bytes, offset: int = 0) -> tuple[str, int]:
  """Unpacks a `str` from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `str` and the packed bytes size.
  """
  length, length_size = _unpack_length_from(buffer, offset=offset)
  offset += length_size
  str_bytes, *_ = struct.unpack_from(f'!{length}s', buffer, offset=offset)
  value = str_bytes.decode('utf-8')
  return value, length_size + length


def pack_sequence(fn: PackFn[_T], sequence: Sequence[_T]) -> bytes:
  """Packs a `Sequence` as bytes using `fn` to pack each item."""
  sequence_bytes = bytearray()
  for item in sequence:
    item_bytes = fn(item)
    sequence_bytes.extend(item_bytes)
  length_bytes = _pack_length(sequence_bytes)
  return length_bytes + sequence_bytes


def unpack_sequence_from(
    fn: UnpackFn[_T], buffer: bytes, offset: int = 0
) -> tuple[Sequence[_T], int]:
  """Unpacks a `Sequence` from bytes using `fn` to unpack each item.

  Args:
    fn: The `UnpackFn` to use to unpack each item.
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `Sequence` and the packed bytes size.
  """
  sequence = []
  length, length_size = _unpack_length_from(buffer, offset=offset)
  offset += length_size
  item_offset = 0
  while item_offset < length:
    item, item_size = fn(buffer, offset=offset + item_offset)
    item_offset += item_size
    sequence.append(item)
  return sequence, length_size + length


def pack_serializable(serializable: serializable_lib.Serializable) -> bytes:
  """Packs a `tff.Serializable` as bytes."""
  module_name_bytes = pack_str(type(serializable).__module__)
  class_name_bytes = pack_str(type(serializable).__name__)
  serializable_bytes = serializable.to_bytes()
  serializable_length_bytes = _pack_length(serializable_bytes)
  return (
      module_name_bytes
      + class_name_bytes
      + serializable_length_bytes
      + serializable_bytes
  )


def unpack_serializable_from(
    buffer: bytes, offset: int = 0
) -> tuple[serializable_lib.Serializable, int]:
  """Unpacks a `tff.Serializable` from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `tff.Serializable` and the packed bytes
    size.
  """
  module_name, module_name_size = unpack_str_from(buffer, offset=offset)
  offset += module_name_size
  class_name, class_name_size = unpack_str_from(buffer, offset=offset)
  offset += class_name_size
  serializable_length, serializable_length_size = _unpack_length_from(
      buffer, offset=offset
  )
  offset += serializable_length_size
  serializable_bytes, *_ = struct.unpack_from(
      f'!{serializable_length}s', buffer, offset=offset
  )
  module = importlib.import_module(module_name)
  cls = getattr(module, class_name)
  value = cls.from_bytes(serializable_bytes)
  return value, (
      module_name_size
      + class_name_size
      + serializable_length_size
      + serializable_length
  )


def pack_type_spec(type_spec: computation_types.Type) -> bytes:
  """Packs a `tff.Type` as bytes."""
  proto = type_serialization.serialize_type(type_spec)
  type_bytes = proto.SerializeToString()
  length_bytes = _pack_length(type_bytes)
  return length_bytes + type_bytes


def unpack_type_spec_from(
    buffer: bytes, offset: int = 0
) -> tuple[computation_types.Type, int]:
  """Unpacks a `tff.Type` from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `tff.Type` and the packed bytes size.
  """
  length, length_size = _unpack_length_from(buffer, offset=offset)
  offset += length_size
  type_spec_bytes, *_ = struct.unpack_from(f'!{length}s', buffer, offset=offset)
  proto = computation_pb2.Type.FromString(type_spec_bytes)
  type_spec = type_serialization.deserialize_type(proto)
  return type_spec, length_size + length


def pack_element_spec(
    element_spec: structure_utils.Structure[tf.TypeSpec],
) -> bytes:
  """Packs a structure of `tf.TypeSpec`s as bytes."""

  def _serialize_tensor_spec(tensor_spec: tf.TensorSpec) -> bytes:
    proto = tensor_spec.experimental_as_proto()
    return proto.SerializeToString()

  partial_bytes = structure_utils.map_structure(
      _serialize_tensor_spec, element_spec
  )
  element_spec_bytes = pickle.dumps(partial_bytes)
  length_bytes = _pack_length(element_spec_bytes)
  return length_bytes + element_spec_bytes


def unpack_element_spec_from(
    buffer: bytes, offset: int = 0
) -> tuple[structure_utils.Structure[tf.TypeSpec], int]:
  """Unpacks a structure of `tf.TypeSpec`s from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked structure of `tf.TypeSpec`s and the packed
    bytes size.
  """
  length, length_size = _unpack_length_from(buffer, offset=offset)
  offset += length_size
  element_spec_bytes, *_ = struct.unpack_from(
      f'!{length}s', buffer, offset=offset
  )
  partial_bytes = pickle.loads(element_spec_bytes)

  def _deserialize_tensor_spec(buffer: bytes) -> tf.TensorSpec:
    proto = tf.TensorSpec.experimental_type_proto().FromString(buffer)
    return tf.TensorSpec.experimental_from_proto(proto)

  element_spec = structure_utils.map_structure(
      _deserialize_tensor_spec, partial_bytes
  )
  return element_spec, length_size + length


class SerializedDatasetSizeExceededError(Exception):
  """Raised when the size of the serialized `tf.data.Dataset` is too large."""

  def __init__(self, size: int):
    super().__init__(
        'Expected the size of the serialized `tf.data.Dataset` to be less than '
        f'{_MAX_SERIALIZED_DATASET_SIZE} bytes, found {size} bytes. '
        '`tf.data.Dataset` serialized to a size larger than '
        f'{_MAX_SERIALIZED_DATASET_SIZE} bytes will result in errors. This '
        'likely occurs when the `tf.data.Dataset` is dependent on variables '
        'from the graph.'
    )


def pack_dataset(dataset: tf.data.Dataset) -> bytes:
  """Packs a `tf.data.Dataset` as bytes."""
  element_spec_bytes = pack_element_spec(dataset.element_spec)
  variant = tf.data.experimental.to_variant(dataset)
  graph = tf.raw_ops.DatasetToGraphV2(input_dataset=variant)
  graph_bytes = graph.numpy()
  if len(graph_bytes) > _MAX_SERIALIZED_DATASET_SIZE:
    raise SerializedDatasetSizeExceededError(len(graph_bytes))
  graph_length_bytes = _pack_length(graph_bytes)
  return element_spec_bytes + graph_length_bytes + graph_bytes


def unpack_dataset_from(
    buffer: bytes, offset: int = 0
) -> tuple[tf.data.Dataset, int]:
  """Unpacks a `tf.data.Dataset` from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `tf.data.Dataset` and the packed bytes
    size.
  """
  element_spec, element_spec_size = unpack_element_spec_from(
      buffer, offset=offset
  )
  offset += element_spec_size
  graph_length, graph_length_size = _unpack_length_from(buffer, offset=offset)
  offset += graph_length_size
  graph_bytes, *_ = struct.unpack_from(
      f'!{graph_length}s', buffer, offset=offset
  )
  variant = tf.raw_ops.DatasetFromGraph(graph_def=graph_bytes)
  dataset = tf.data.experimental.from_variant(variant, element_spec)
  return dataset, element_spec_size + graph_length_size + graph_length
