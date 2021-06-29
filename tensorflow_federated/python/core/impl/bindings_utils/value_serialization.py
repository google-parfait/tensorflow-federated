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
"""A set of utility methods for serializing Value protos using pybind11 bindings."""

import collections
import os
import os.path
import tempfile
from typing import Any, Collection, Optional, List, Mapping, Tuple, Union
import warnings
import zipfile

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import serialization_bindings
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_transformations
from tensorflow_federated.python.core.impl.utils import tensorflow_utils

_SerializeReturnType = Tuple[serialization_bindings.Value,
                             computation_types.Type]
_DeserializeReturnType = Tuple[Any, computation_types.Type]

# The maximum size allowed for serialized sequence values. Sequence that
# serialize to values larger than this will result in errors being raised.  This
# likely occurs when the sequence is dependent on, and thus pulling in, many of
# variables from the graph.
_DEFAULT_MAX_SERIALIZED_SEQUENCE_SIZE_BYTES = 20 * (1024**2)  # 20 MB


class DatasetSerializationError(Exception):
  """Error raised during Dataset serialization or deserialization."""
  pass


@tracing.trace
def _serialize_computation(
    comp: computation_pb2.Computation,
    type_spec: Optional[computation_types.Type]) -> _SerializeReturnType:
  """Serializes a TFF computation."""
  type_spec = executor_utils.reconcile_value_type_with_type_spec(
      type_serialization.deserialize_type(comp.type), type_spec)
  return serialization_bindings.Value(computation=comp), type_spec


@tracing.trace
def _serialize_tensor_value(
    value_proto: serialization_bindings.Value, value: Any,
    type_spec: computation_types.TensorType) -> computation_types.TensorType:
  """Serializes a tensor value into `serialization_bindings.Value`.

  Args:
    value_proto: A `serialization_bindings.Value` proto to serialize the `value`
      into.
    value: A Numpy array or other object understood by `tf.make_tensor_proto`.
    type_spec: A `tff.TensorType`.

  Returns:
    A tuple `(value_proto, ret_type_spec)` in which `value_proto` is an instance
    of `serialization_bindings.Value` with the serialized content of `value`,
    and `ret_type_spec` is the type of the serialized value. The `ret_type_spec`
    is the same as the argument `type_spec` if that argument was not `None`. If
    the argument was `None`, `ret_type_spec` is a type determined from `value`.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  if not tf.is_tensor(value):
    value = tf.convert_to_tensor(value, dtype=type_spec.dtype)
  if not value.shape.is_compatible_with(type_spec.shape):
    raise TypeError(f'Cannot serialize tensor with shape {value.shape} to '
                    f'shape {type_spec.shape}.')
  if value.dtype != type_spec.dtype:
    value = tf.cast(value, type_spec.dtype)
  return serialization_bindings.serialize_tensor_value(value,
                                                       value_proto), type_spec


def _serialize_dataset(
    dataset,
    max_serialized_size_bytes=_DEFAULT_MAX_SERIALIZED_SEQUENCE_SIZE_BYTES):
  """Serializes a `tf.data.Dataset` value into a `bytes` object.

  Args:
    dataset: A `tf.data.Dataset`.
    max_serialized_size_bytes: An `int` size in bytes designating the threshold
      on when to raise an error if the resulting serialization is too big.

  Returns:
    A `bytes` object that can be sent to
  `tensorflow_serialization.deserialize_dataset` to recover the original
  `tf.data.Dataset`.

  Raises:
    SerializationError: if there was an error in TensorFlow during
      serialization.
  """
  py_typecheck.check_type(dataset,
                          type_conversions.TF_DATASET_REPRESENTATION_TYPES)
  dataset_graph_def_bytes = tf.raw_ops.DatasetToGraphV2(
      input_dataset=tf.data.experimental.to_variant(dataset)).numpy()
  if len(dataset_graph_def_bytes) > max_serialized_size_bytes:
    raise ValueError('Serialized size of Dataset ({:d} bytes) exceeds maximum '
                     'allowed ({:d} bytes)'.format(
                         len(dataset_graph_def_bytes),
                         max_serialized_size_bytes))
  return dataset_graph_def_bytes


@tracing.trace
def _serialize_sequence_value(
    value_proto: serialization_bindings.Value,
    value: Union[type_conversions.TF_DATASET_REPRESENTATION_TYPES],
    type_spec: computation_types.SequenceType
) -> computation_types.SequenceType:
  """Serializes a `tf.data.Dataset` value into `serialization_bindings.Value`.

  Args:
    value_proto: A `serialization_bindings.Value` proto to serialize the `value`
      into.
    value: A `tf.data.Dataset`, or equivalent.
    type_spec: A `computation_types.Type` specifying the TFF sequence type of
      `value.`

  Returns:
    A tuple `(value_proto, type_spec)` in which `value_proto` is an instance
    of `serialization_bindings.Value` with the serialized content of `value`,
    and `type_spec` is the type of the serialized value.
  """
  if not isinstance(value, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
    raise TypeError(
        'Cannot serialize Python type {!s} as TFF type {!s}.'.format(
            py_typecheck.type_string(type(value)),
            type_spec if type_spec is not None else 'unknown'))
  element_type = computation_types.to_type(value.element_spec)
  value_type = computation_types.SequenceType(element_type)
  if not type_spec.is_assignable_from(value_type):
    raise TypeError(
        'Cannot serialize dataset with elements of type {!s} as TFF type {!s}.'
        .format(value_type, type_spec if type_spec is not None else 'unknown'))
  # TFF must store the type spec here because TF will lose the ordering of the
  # names for `tf.data.Dataset` that return elements of
  # `collections.abc.Mapping` type. This allows TFF to preserve and restore the
  # key ordering upon deserialization.
  value_proto.sequence.serialized_graph_def = _serialize_dataset(value)
  value_proto.sequence.element_type.CopyFrom(
      type_serialization.serialize_type(element_type))
  return value_proto, type_spec


@tracing.trace
def _serialize_struct_type(
    value_proto: serialization_bindings.Value,
    struct_typed_value: Any,
    type_spec: computation_types.StructType,
) -> computation_types.StructType:
  """Serializes a value of tuple type."""
  value_structure = structure.from_container(struct_typed_value)
  if len(value_structure) != len(type_spec):
    raise TypeError('Cannot serialize a struct value of '
                    f'{len(value_structure)} elements to a struct type '
                    f'requiring {len(type_spec)} elements. Trying to serialize'
                    f'\n{struct_typed_value!r}\nto\n{type_spec}.')
  type_elem_iter = structure.iter_elements(type_spec)
  val_elem_iter = structure.iter_elements(value_structure)
  struct_proto = value_proto.struct
  for (e_name, e_type), (_, e_val) in zip(type_elem_iter, val_elem_iter):
    element_proto = struct_proto.element.add()
    serialize_value(e_val, e_type, value_proto=element_proto.value)
    if e_name:
      element_proto.name = e_name
  return value_proto, type_spec


@tracing.trace
def _serialize_federated_value(
    value_proto: serialization_bindings.Value, federated_value: Any,
    type_spec: computation_types.FederatedType
) -> computation_types.FederatedType:
  """Serializes a value of federated type."""
  if type_spec.all_equal:
    value = [federated_value]
  else:
    value = federated_value
  py_typecheck.check_type(value, list)
  for v in value:
    federated_value_proto = value_proto.federated.value.add()
    _, it_type = serialize_value(
        v, type_spec.member, value_proto=federated_value_proto)
    type_spec.member.check_assignable_from(it_type)
  value_proto.federated.type.CopyFrom(
      type_serialization.serialize_type(type_spec).federated)
  return value_proto, type_spec


@tracing.trace
def serialize_value(
    value: Any,
    type_spec: Optional[computation_types.Type] = None,
    *,
    value_proto: Optional[serialization_bindings.Value] = None
) -> computation_types.Type:
  """Serializes a value into `serialization_bindings.Value`.

  We use a switch/function pattern in the body here (and in `deserialize_value`
  below in order to persist more information in traces and profiling.

  Args:
    value: A value to be serialized.
    type_spec: Optional type spec, a `tff.Type` or something convertible to it.
    value_proto: The protobuf instance to serialize into. `value_proto` will
      first be cleared before serializing.

  Returns:
    An instance of `tff.Type` that represents the TFF type of the serialized
    value.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  if value_proto is None:
    value_proto = serialization_bindings.Value()
  type_spec = computation_types.to_type(type_spec)
  if isinstance(value, computation_pb2.Computation):
    return _serialize_computation(value, type_spec)
  elif isinstance(value, computation_impl.ComputationImpl):
    return _serialize_computation(
        computation_impl.ComputationImpl.get_proto(value),
        executor_utils.reconcile_value_with_type_spec(value, type_spec))
  elif type_spec is None:
    raise TypeError('A type hint is required when serializing a value which '
                    'is not a TFF computation. Asked to serialized value {v} '
                    ' of type {t} with None type spec.'.format(
                        v=value, t=type(value)))
  elif type_spec.is_tensor():
    return _serialize_tensor_value(value_proto, value, type_spec)
  elif type_spec.is_sequence():
    return _serialize_sequence_value(value_proto, value, type_spec)
  elif type_spec.is_struct():
    return _serialize_struct_type(value_proto, value, type_spec)
  elif type_spec.is_federated():
    proto, type_spec = _serialize_federated_value(value_proto, value, type_spec)
    return proto, type_spec
  else:
    raise ValueError(
        'Unable to serialize value with Python type {} and {} TFF type.'.format(
            str(py_typecheck.type_string(type(value))),
            str(type_spec) if type_spec is not None else 'unknown'))


@tracing.trace
def _deserialize_computation(
    value_proto: serialization_bindings.Value) -> _DeserializeReturnType:
  """Deserializes a TFF computation."""
  return (value_proto.computation,
          type_serialization.deserialize_type(value_proto.computation.type))


@tracing.trace
def _deserialize_tensor_value(
    value_proto: serialization_bindings.Value) -> _DeserializeReturnType:
  """Deserializes a tensor value from `serialization_bindings.Value`.

  Args:
    value_proto: An instance of `serialization_bindings.Value`.

  Returns:
    A tuple `(value, type_spec)`, where `value` is a Numpy array that represents
    the deserialized value, and `type_spec` is an instance of `tff.TensorType`
    that represents its type.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  value = serialization_bindings.deserialize_tensor_value(value_proto)
  value_type = computation_types.TensorType(
      dtype=value.dtype, shape=value.shape)
  return value, value_type


def _deserialize_dataset_from_zipped_saved_model(serialized_bytes):
  """Deserializes a zipped SavedModel `bytes` object to a `tf.data.Dataset`.

  DEPRECATED: this method is deprecated and replaced by
  `_deserialize_dataset_from_graph_def`.

  Args:
    serialized_bytes: `bytes` object produced by older versions of
      `tensorflow_serialization.serialize_dataset` that produced zipped
      SavedModel `bytes` strings.

  Returns:
    A `tf.data.Dataset` instance.

  Raises:
    SerializationError: if there was an error in TensorFlow during
      serialization.
  """
  py_typecheck.check_type(serialized_bytes, bytes)
  temp_dir = tempfile.mkdtemp('dataset')
  fd, temp_zip = tempfile.mkstemp('zip')
  os.close(fd)
  try:
    with open(temp_zip, 'wb') as f:
      f.write(serialized_bytes)
    with zipfile.ZipFile(temp_zip, 'r') as z:
      z.extractall(path=temp_dir)
    loaded = tf.saved_model.load(temp_dir)
    # TODO(b/156302055): Follow up here when bug is resolved, either remove
    # if this function call stops failing by default, or leave if this is
    # working as intended.
    with tf.device('cpu'):
      ds = loaded.dataset_fn()
  except Exception as e:  # pylint: disable=broad-except
    raise DatasetSerializationError(
        'Error deserializing tff.Sequence value. Inner error: {!s}'.format(
            e)) from e
  finally:
    tf.io.gfile.rmtree(temp_dir)
    tf.io.gfile.remove(temp_zip)
  return ds


def _deserialize_dataset_from_graph_def(serialized_graph_def: bytes,
                                        element_type: computation_types.Type):
  """Deserializes a serialized `tf.compat.v1.GraphDef` to a `tf.data.Dataset`.

  Args:
    serialized_graph_def: `bytes` object produced by
      `tensorflow_serialization.serialize_dataset`
    element_type: a `tff.Type` object representing the type structure of the
      elements yielded from the dataset.

  Returns:
    A `tf.data.Dataset` instance.
  """
  py_typecheck.check_type(element_type, computation_types.Type)
  type_analysis.check_tensorflow_compatible_type(element_type)

  def transform_to_tff_known_type(
      type_spec: computation_types.Type) -> Tuple[computation_types.Type, bool]:
    """Transforms `StructType` to `StructWithPythonType`."""
    if type_spec.is_struct() and not type_spec.is_struct_with_python():
      field_is_named = tuple(
          name is not None for name, _ in structure.iter_elements(type_spec))
      has_names = any(field_is_named)
      is_all_named = all(field_is_named)
      if is_all_named:
        return computation_types.StructWithPythonType(
            elements=structure.iter_elements(type_spec),
            container_type=collections.OrderedDict), True
      elif not has_names:
        return computation_types.StructWithPythonType(
            elements=structure.iter_elements(type_spec),
            container_type=tuple), True
      else:
        raise TypeError('Cannot represent TFF type in TF because it contains '
                        f'partially named structures. Type: {type_spec}')
    return type_spec, False

  if element_type.is_struct():
    # TF doesn't suppor `structure.Strut` types, so we must transform the
    # `StructType` into a `StructWithPythonType` for use as the
    # `tf.data.Dataset.element_spec` later.
    tf_compatible_type, _ = type_transformations.transform_type_postorder(
        element_type, transform_to_tff_known_type)
  else:
    # We've checked this is only a struct or tensors, so we know this is a
    # `TensorType` here and will use as-is.
    tf_compatible_type = element_type

  def type_to_tensorspec(t: computation_types.TensorType) -> tf.TensorSpec:
    return tf.TensorSpec(shape=t.shape, dtype=t.dtype)

  element_spec = type_conversions.structure_from_tensor_type_tree(
      type_to_tensorspec, tf_compatible_type)
  ds = tf.data.experimental.from_variant(
      tf.raw_ops.DatasetFromGraph(graph_def=serialized_graph_def),
      structure=element_spec)
  # If a serialized dataset had elements of nested structes of tensors (e.g.
  # `dict`, `OrderedDict`), the deserialized dataset will return `dict`,
  # `tuple`, or `namedtuple` (loses `collections.OrderedDict` in a conversion).
  #
  # Since the dataset will only be used inside TFF, we wrap the dictionary
  # coming from TF in an `OrderedDict` when necessary (a type that both TF and
  # TFF understand), using the field order stored in the TFF type stored during
  # serialization.
  return tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
      ds, tf_compatible_type)


@tracing.trace
def _deserialize_sequence_value(
    sequence_value_proto: serialization_bindings.Sequence,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a `tf.data.Dataset`.

  Args:
    sequence_value_proto: `Sequence` protocol buffer message.
    type_hint: A `computation_types.Type` that hints at what the value type
      should be for executors that only return values. If the
      `sequence_value_proto.element_type` field was not set, the `type_hint` is
      used instead.

  Returns:
    A tuple of `(tf.data.Dataset, tff.Type)`.
  """
  if sequence_value_proto.HasField('element_type'):
    element_type = type_serialization.deserialize_type(
        sequence_value_proto.element_type)
  elif type_hint is not None:
    element_type = type_hint.element
  else:
    raise ValueError(
        'Cannot deserialize a sequence Value proto that without one of '
        '`element_type` proto field or `element_type_hint`')
  which_value = sequence_value_proto.WhichOneof('value')
  if which_value == 'zipped_saved_model':
    warnings.warn(
        'Deserializng a sequence value that was encoded as a zipped SavedModel.'
        ' This is a deprecated path, please update the binary that is '
        'serializing the sequences.', DeprecationWarning)
    ds = _deserialize_dataset_from_zipped_saved_model(
        sequence_value_proto.zipped_saved_model)
    ds = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
        ds, element_type)
  elif which_value == 'serialized_graph_def':
    ds = _deserialize_dataset_from_graph_def(
        sequence_value_proto.serialized_graph_def, element_type)
  else:
    raise NotImplementedError(
        'Deserializing Sequences enocded as {!s} has not been implemented'
        .format(which_value))
  return ds, computation_types.SequenceType(element=element_type)


@tracing.trace
def _deserialize_struct_value(
    value_proto: serialization_bindings.Value,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a value of struct type."""
  val_elems = []
  type_elems = []
  if type_hint is not None:
    element_types = tuple(type_hint)
  else:
    element_types = [None] * len(value_proto.struct.element)
  for e, e_type in zip(value_proto.struct.element, element_types):
    name = e.name if e.name else None
    e_val, e_type = deserialize_value(e.value, e_type)
    val_elems.append((name, e_val))
    type_elems.append((name, e_type) if name else e_type)
  return (structure.Struct(val_elems), computation_types.StructType(type_elems))


def _ensure_deserialized_types_compatible(
    previous_type: Optional[computation_types.Type],
    next_type: computation_types.Type) -> computation_types.Type:
  """Ensures one of `previous_type` or `next_type` is assignable to the other.

  Returns the type which is assignable from the other.

  Args:
    previous_type: Instance of `computation_types.Type` or `None`.
    next_type: Instance of `computation_types.Type`.

  Returns:
    The supertype of `previous_type` and `next_type`.

  Raises:
    TypeError if neither type is assignable from the other.
  """
  if previous_type is None:
    return next_type
  else:
    if next_type.is_assignable_from(previous_type):
      return next_type
    elif previous_type.is_assignable_from(next_type):
      return previous_type
    raise TypeError('Type mismatch checking member assignability under a '
                    'federated value. Deserialized type {} is incompatible '
                    'with previously deserialized {}.'.format(
                        next_type, previous_type))


@tracing.trace
def _deserialize_federated_value(
    value_proto: serialization_bindings.Value,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a value of federated type."""
  all_equal = value_proto.federated.type.all_equal
  placement_uri = value_proto.federated.type.placement.value.uri
  if not value_proto.federated.value:
    raise ValueError('Attempting to deserialize federated value with no data.')
  value = []
  # item_type will represent a supertype of all deserialized member types in the
  # federated value.
  if type_hint is not None:
    item_type_hint = type_hint.member
  else:
    item_type_hint = None
  item_type = None
  for item in value_proto.federated.value:
    item_value, next_item_type = deserialize_value(item, item_type_hint)
    item_type = _ensure_deserialized_types_compatible(item_type, next_item_type)
    value.append(item_value)
  if all_equal:
    if len(value) == 1:
      value = value[0]
    else:
      raise ValueError(
          'Encountered an all_equal value with {} member constituents. '
          'Expected exactly 1.'.format(len(value)))
  type_spec = computation_types.FederatedType(
      item_type,
      placement=placements.uri_to_placement_literal(placement_uri),
      all_equal=all_equal)
  return value, type_spec


@tracing.trace
def deserialize_value(
    value_proto: serialization_bindings.Value,
    type_hint: Optional[computation_types.Type] = None
) -> _DeserializeReturnType:
  """Deserializes a value (of any type) from `serialization_bindings.Value`.

  Args:
    value_proto: An instance of `serialization_bindings.Value`.
    type_hint: A `comptuations_types.Type` that hints at what the value type
      should be for executors that only return values.

  Returns:
    A tuple `(value, type_spec)`, where `value` is a deserialized
    representation of the transmitted value (e.g., Numpy array, or a
    `pb.Computation` instance), and `type_spec` is an instance of
    `tff.TensorType` that represents its type.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  if not hasattr(value_proto, 'WhichOneof'):
    raise TypeError('`value_proto` must be a protocol buffer message with a '
                    '`value` oneof field.')
  which_value = value_proto.WhichOneof('value')
  if which_value == 'tensor':
    return _deserialize_tensor_value(value_proto)
  elif which_value == 'computation':
    return _deserialize_computation(value_proto)
  elif which_value == 'sequence':
    return _deserialize_sequence_value(value_proto.sequence, type_hint)
  elif which_value == 'struct':
    return _deserialize_struct_value(value_proto, type_hint)
  elif which_value == 'federated':
    return _deserialize_federated_value(value_proto, type_hint)
  else:
    raise ValueError(
        'Unable to deserialize a value of type {}.'.format(which_value))


CardinalitiesType = Mapping[placements.PlacementLiteral, int]


def serialize_cardinalities(
    cardinalities: CardinalitiesType
) -> List[serialization_bindings.Cardinality]:
  serialized_cardinalities = []
  for placement, cardinality in cardinalities.items():
    cardinality_message = serialization_bindings.Cardinality(
        placement=computation_pb2.Placement(uri=placement.uri),
        cardinality=cardinality)
    serialized_cardinalities.append(cardinality_message)
  return serialized_cardinalities


def deserialize_cardinalities(
    serialized_cardinalities: Collection[serialization_bindings.Cardinality]
) -> CardinalitiesType:
  cardinalities_dict = {}
  for cardinality_spec in serialized_cardinalities:
    literal = placements.uri_to_placement_literal(
        cardinality_spec.placement.uri)
    cardinalities_dict[literal] = cardinality_spec.cardinality
  return cardinalities_dict
