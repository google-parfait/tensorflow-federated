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
"""A set of utility methods for `executor_service.py` and its clients."""

import os
import os.path
import tempfile
from typing import Any, Collection, Optional, List, Mapping, Tuple, Union
import zipfile

import numpy as np
import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils

_SerializeReturnType = Tuple[executor_pb2.Value, computation_types.Type]
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
  type_spec = type_utils.reconcile_value_type_with_type_spec(
      type_serialization.deserialize_type(comp.type), type_spec)
  return executor_pb2.Value(computation=comp), type_spec


@tracing.trace
def _serialize_tensor_value(
    value: Any,
    type_spec: computation_types.TensorType) -> _SerializeReturnType:
  """Serializes a tensor value into `executor_pb2.Value`.

  Args:
    value: A Numpy array or other object understood by `tf.make_tensor_proto`.
    type_spec: A `tff.TensorType`.

  Returns:
    A tuple `(value_proto, ret_type_spec)` in which `value_proto` is an instance
    of `executor_pb2.Value` with the serialized content of `value`, and
    `ret_type_spec` is the type of the serialized value. The `ret_type_spec` is
    the same as the argument `type_spec` if that argument was not `None`. If
    the argument was `None`, `ret_type_spec` is a type determined from `value`.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  if isinstance(value, tf.Tensor):
    value = value.numpy()
  if isinstance(value, np.ndarray):
    tensor_proto = tf.make_tensor_proto(
        value, dtype=type_spec.dtype, verify_shape=False)
  else:
    tensor_proto = tf.make_tensor_proto(
        value, dtype=type_spec.dtype, shape=type_spec.shape, verify_shape=True)
  type_spec.check_assignable_from(
      computation_types.TensorType(
          dtype=tf.dtypes.as_dtype(tensor_proto.dtype),
          shape=tf.TensorShape(tensor_proto.tensor_shape)))
  any_pb = any_pb2.Any()
  any_pb.Pack(tensor_proto)
  return executor_pb2.Value(tensor=any_pb), type_spec


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
  module = tf.Module()
  module.dataset = dataset
  module.dataset_fn = tf.function(lambda: module.dataset, input_signature=())

  temp_dir = tempfile.mkdtemp('dataset')
  fd, temp_zip = tempfile.mkstemp('zip')
  os.close(fd)
  try:
    tf.saved_model.save(module, temp_dir, signatures={})
    with zipfile.ZipFile(temp_zip, 'w') as z:
      for topdir, _, filenames in tf.io.gfile.walk(temp_dir):
        dest_dir = topdir[len(temp_dir):]
        for filename in filenames:
          z.write(
              os.path.join(topdir, filename), os.path.join(dest_dir, filename))
    with open(temp_zip, 'rb') as z:
      zip_bytes = z.read()
  except Exception as e:  # pylint: disable=broad-except
    raise DatasetSerializationError(
        'Error serializing tff.Sequence value. Inner error: {!s}'.format(
            e)) from e
  finally:
    tf.io.gfile.rmtree(temp_dir)
    tf.io.gfile.remove(temp_zip)

  if len(zip_bytes) > max_serialized_size_bytes:
    raise ValueError('Serialized size of Dataset ({:d} bytes) exceeds maximum '
                     'allowed ({:d} bytes)'.format(
                         len(zip_bytes), max_serialized_size_bytes))
  return zip_bytes


@tracing.trace
def _serialize_sequence_value(
    value: Union[type_conversions.TF_DATASET_REPRESENTATION_TYPES],
    type_spec: computation_types.SequenceType) -> _SerializeReturnType:
  """Serializes a `tf.data.Dataset` value into `executor_pb2.Value`.

  Args:
    value: A `tf.data.Dataset`, or equivalent.
    type_spec: A `computation_types.Type` specifying the TFF sequence type of
      `value.`

  Returns:
    A tuple `(value_proto, type_spec)` in which `value_proto` is an instance
    of `executor_pb2.Value` with the serialized content of `value`, and
    `type_spec` is the type of the serialized value.
  """
  if not isinstance(value, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
    raise TypeError(
        'Cannot serialize Python type {!s} as TFF type {!s}.'.format(
            py_typecheck.type_string(type(value)),
            type_spec if type_spec is not None else 'unknown'))

  value_type = computation_types.SequenceType(
      computation_types.to_type(value.element_spec))
  if not type_spec.is_assignable_from(value_type):
    raise TypeError(
        'Cannot serialize dataset with elements of type {!s} as TFF type {!s}.'
        .format(value_type, type_spec if type_spec is not None else 'unknown'))

  # TFF must store the type spec here because TF will lose the ordering of the
  # names for `tf.data.Dataset` that return elements of
  # `collections.abc.Mapping` type. This allows TFF to preserve and restore the
  # key ordering upon deserialization.
  element_type = computation_types.to_type(value.element_spec)
  return executor_pb2.Value(
      sequence=executor_pb2.Value.Sequence(
          zipped_saved_model=_serialize_dataset(value),
          element_type=type_serialization.serialize_type(
              element_type))), type_spec


@tracing.trace
def _serialize_struct_type(
    struct_typed_value: Any,
    type_spec: computation_types.StructType) -> _SerializeReturnType:
  """Serializes a value of tuple type."""
  type_elem_iter = structure.iter_elements(type_spec)
  val_elem_iter = structure.iter_elements(
      structure.from_container(struct_typed_value))
  tup_elems = []
  for (e_name, e_type), (_, e_val) in zip(type_elem_iter, val_elem_iter):
    e_proto, _ = serialize_value(e_val, e_type)
    tup_elems.append(
        executor_pb2.Value.Struct.Element(
            name=e_name if e_name else None, value=e_proto))
  result_proto = (
      executor_pb2.Value(struct=executor_pb2.Value.Struct(element=tup_elems)))
  return result_proto, type_spec


@tracing.trace
def _serialize_federated_value(
    federated_value: Any,
    type_spec: computation_types.FederatedType) -> _SerializeReturnType:
  """Serializes a value of federated type."""
  if type_spec.all_equal:
    value = [federated_value]
  else:
    value = federated_value
  py_typecheck.check_type(value, list)
  items = []
  for v in value:
    it, it_type = serialize_value(v, type_spec.member)
    type_spec.member.check_assignable_from(it_type)
    items.append(it)
  result_proto = executor_pb2.Value(
      federated=executor_pb2.Value.Federated(
          type=type_serialization.serialize_type(type_spec).federated,
          value=items))
  return result_proto, type_spec


@tracing.trace
def serialize_value(
    value: Any,
    type_spec: Optional[computation_types.Type] = None) -> _SerializeReturnType:
  """Serializes a value into `executor_pb2.Value`.

  We use a switch/function pattern in the body here (and in `deserialize_value`
  below in order to persist more information in traces and profiling.

  Args:
    value: A value to be serialized.
    type_spec: Optional type spec, a `tff.Type` or something convertible to it.

  Returns:
    A tuple `(value_proto, ret_type_spec)` where `value_proto` is an instance
    of `executor_pb2.Value` with the serialized content of `value`, and the
    returned `ret_type_spec` is an instance of `tff.Type` that represents the
    TFF type of the serialized value.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  type_spec = computation_types.to_type(type_spec)
  if isinstance(value, computation_pb2.Computation):
    return _serialize_computation(value, type_spec)
  elif isinstance(value, computation_impl.ComputationImpl):
    return _serialize_computation(
        computation_impl.ComputationImpl.get_proto(value),
        type_utils.reconcile_value_with_type_spec(value, type_spec))
  elif type_spec is None:
    raise TypeError('A type hint is required when serializing a value which '
                    'is not a TFF computation. Asked to serialized value {v} '
                    ' of type {t} with None type spec.'.format(
                        v=value, t=type(value)))
  elif type_spec.is_tensor():
    return _serialize_tensor_value(value, type_spec)
  elif type_spec.is_sequence():
    return _serialize_sequence_value(value, type_spec)
  elif type_spec.is_struct():
    return _serialize_struct_type(value, type_spec)
  elif type_spec.is_federated():
    return _serialize_federated_value(value, type_spec)
  else:
    raise ValueError(
        'Unable to serialize value with Python type {} and {} TFF type.'.format(
            str(py_typecheck.type_string(type(value))),
            str(type_spec) if type_spec is not None else 'unknown'))


@tracing.trace
def _deserialize_computation(
    value_proto: executor_pb2.Value) -> _DeserializeReturnType:
  """Deserializes a TFF computation."""
  return (value_proto.computation,
          type_serialization.deserialize_type(value_proto.computation.type))


@tracing.trace
def _deserialize_tensor_value(
    value_proto: executor_pb2.Value) -> _DeserializeReturnType:
  """Deserializes a tensor value from `executor_pb2.Value`.

  Args:
    value_proto: An instance of `executor_pb2.Value`.

  Returns:
    A tuple `(value, type_spec)`, where `value` is a Numpy array that represents
    the deserialized value, and `type_spec` is an instance of `tff.TensorType`
    that represents its type.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  which_value = value_proto.WhichOneof('value')
  if which_value != 'tensor':
    raise ValueError('Not a tensor value: {}'.format(which_value))

  # TODO(b/134543154): Find some way of creating the `TensorProto` using a
  # proper public interface rather than creating a dummy value that we will
  # overwrite right away.
  tensor_proto = tf.make_tensor_proto(values=0)
  if not value_proto.tensor.Unpack(tensor_proto):
    raise ValueError('Unable to unpack the received tensor value.')

  tensor_value = tf.make_ndarray(tensor_proto)
  value_type = computation_types.TensorType(
      dtype=tf.dtypes.as_dtype(tensor_proto.dtype),
      shape=tf.TensorShape(tensor_proto.tensor_shape))

  return tensor_value, value_type


def _deserialize_dataset(serialized_bytes):
  """Deserializes a `bytes` object to a `tf.data.Dataset`.

  Args:
    serialized_bytes: `bytes` object produced by
      `tensorflow_serialization.serialize_dataset`

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


@tracing.trace
def _deserialize_sequence_value(
    sequence_value_proto: executor_pb2.Value.Sequence
) -> _DeserializeReturnType:
  """Deserializes a `tf.data.Dataset`.

  Args:
    sequence_value_proto: `Sequence` protocol buffer message.

  Returns:
    A tuple of `(tf.data.Dataset, tff.Type)`.
  """
  which_value = sequence_value_proto.WhichOneof('value')
  if which_value == 'zipped_saved_model':
    ds = _deserialize_dataset(sequence_value_proto.zipped_saved_model)
  else:
    raise NotImplementedError(
        'Deserializing Sequences enocded as {!s} has not been implemented'
        .format(which_value))

  element_type = type_serialization.deserialize_type(
      sequence_value_proto.element_type)

  # If a serialized dataset had elements of nested structes of tensors (e.g.
  # `dict`, `OrderedDict`), the deserialized dataset will return `dict`,
  # `tuple`, or `namedtuple` (loses `collections.OrderedDict` in a conversion).
  #
  # Since the dataset will only be used inside TFF, we wrap the dictionary
  # coming from TF in an `OrderedDict` when necessary (a type that both TF and
  # TFF understand), using the field order stored in the TFF type stored during
  # serialization.
  ds = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
      ds, element_type)

  return ds, computation_types.SequenceType(element=element_type)


@tracing.trace
def _deserialize_struct_value(
    value_proto: executor_pb2.Value) -> _DeserializeReturnType:
  """Deserializes a value of struct type."""
  val_elems = []
  type_elems = []
  for e in value_proto.struct.element:
    name = e.name if e.name else None
    e_val, e_type = deserialize_value(e.value)
    val_elems.append((name, e_val))
    type_elems.append((name, e_type) if name else e_type)
  return (structure.Struct(val_elems), computation_types.StructType(type_elems))


@tracing.trace
def _deserialize_federated_value(
    value_proto: executor_pb2.Value) -> _DeserializeReturnType:
  """Deserializes a value of federated type."""
  type_spec = type_serialization.deserialize_type(
      computation_pb2.Type(federated=value_proto.federated.type))
  value = []
  for item in value_proto.federated.value:
    item_value, item_type = deserialize_value(item)
    type_spec.member.check_assignable_from(item_type)
    value.append(item_value)
  if type_spec.all_equal:
    if len(value) == 1:
      value = value[0]
    else:
      raise ValueError(
          'Encountered an all_equal value with {} member constituents. '
          'Expected exactly 1.'.format(len(value)))
  return value, type_spec


@tracing.trace
def deserialize_value(
    value_proto: executor_pb2.Value) -> _DeserializeReturnType:
  """Deserializes a value (of any type) from `executor_pb2.Value`.

  Args:
    value_proto: An instance of `executor_pb2.Value`.

  Returns:
    A tuple `(value, type_spec)`, where `value` is a deserialized
    representation of the transmitted value (e.g., Numpy array, or a
    `pb.Computation` instance), and `type_spec` is an instance of
    `tff.TensorType` that represents its type.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  py_typecheck.check_type(value_proto, executor_pb2.Value)
  which_value = value_proto.WhichOneof('value')
  if which_value == 'tensor':
    return _deserialize_tensor_value(value_proto)
  elif which_value == 'computation':
    return _deserialize_computation(value_proto)
  elif which_value == 'sequence':
    return _deserialize_sequence_value(value_proto.sequence)
  elif which_value == 'struct':
    return _deserialize_struct_value(value_proto)
  elif which_value == 'federated':
    return _deserialize_federated_value(value_proto)
  else:
    raise ValueError(
        'Unable to deserialize a value of type {}.'.format(which_value))


CardinalitiesType = Mapping[placement_literals.PlacementLiteral, int]


def serialize_cardinalities(
    cardinalities: CardinalitiesType
) -> List[executor_pb2.SetCardinalitiesRequest.Cardinality]:
  serialized_cardinalities = []
  for placement, cardinality in cardinalities.items():
    cardinality_message = executor_pb2.SetCardinalitiesRequest.Cardinality(
        placement=computation_pb2.Placement(uri=placement.uri),
        cardinality=cardinality)
    serialized_cardinalities.append(cardinality_message)
  return serialized_cardinalities


def deserialize_cardinalities(
    serialized_cardinalities: Collection[
        executor_pb2.SetCardinalitiesRequest.Cardinality]
) -> CardinalitiesType:
  cardinalities_dict = {}
  for cardinality_spec in serialized_cardinalities:
    literal = placement_literals.uri_to_placement_literal(
        cardinality_spec.placement.uri)
    cardinalities_dict[literal] = cardinality_spec.cardinality
  return cardinalities_dict
