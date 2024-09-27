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

from collections.abc import Collection, Mapping, Sequence
import typing
from typing import Optional

import numpy as np
import tree

from tensorflow_federated.proto.v0 import array_pb2
from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.compiler import array
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.types import array_shape
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import dtype_utils
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization

_SerializeReturnType = tuple[executor_pb2.Value, computation_types.Type]
_DeserializeReturnType = tuple[object, computation_types.Type]


@tracing.trace
def _serialize_computation(
    comp: computation_pb2.Computation,
    type_spec: Optional[computation_types.Type],
) -> _SerializeReturnType:
  """Serializes a TFF computation."""
  type_spec = executor_utils.reconcile_value_type_with_type_spec(
      type_serialization.deserialize_type(comp.type), type_spec
  )
  return executor_pb2.Value(computation=comp), type_spec


@tracing.trace
def _serialize_tensor_value(
    value: object, type_spec: computation_types.TensorType
) -> tuple[executor_pb2.Value, computation_types.TensorType]:
  """Serializes a tensor value into `executor_pb2.Value`.

  Args:
    value: A Numpy array or Python value that can be coerced into an Array.
    type_spec: A `tff.TensorType`.

  Returns:
    A tuple `(value_proto, ret_type_spec)` in which `value_proto` is an instance
    of `executor_pb2.Value` with the serialized content of `value`,
    and `ret_type_spec` is the type of the serialized value. The `ret_type_spec`
    is the same as the argument `type_spec` if that argument was not `None`. If
    the argument was `None`, `ret_type_spec` is a type determined from `value`.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """

  # It is necessary to coerce Python `list` and `tuple` to a numpy value,
  # because these types are not an `array.Array`, but can be serialized as a
  # single `tff.TensorType`. Additionally, it is safe to coerce these kinds of
  # values to a numpy value of type `type_spec.dtype.type` if each element in
  # the sequence is compatible with `type_spec.dtype.type`.
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    if not all(
        array.is_compatible_dtype(x, type_spec.dtype.type)
        for x in tree.flatten(value)
    ):
      raise TypeError(
          f'Failed to serialize the value {value} to dtype'
          f' {type_spec.dtype.type}.'
      )
    value = np.asarray(value, type_spec.dtype.type)
  elif not isinstance(value, typing.get_args(array.Array)):
    raise NotImplementedError(f'Unexpected `value` found: {type(value)}.')
  else:
    # This is required because in Python 3.9 `isinstance` cannot accept a
    # `Union` of types and `pytype` does not parse `typing.get_args`.
    value = typing.cast(array.Array, value)

  if not array.is_compatible_shape(value, type_spec.shape):
    if isinstance(value, (np.ndarray, np.generic)):
      shape = value.shape
    else:
      shape = ()
    raise TypeError(
        f'Failed to serialize the value {value} with shape {shape} to shape'
        f' {type_spec.shape}.'
    )

  if not array.is_compatible_dtype(value, type_spec.dtype.type):
    if isinstance(value, (np.ndarray, np.generic)):
      dtype = value.dtype.type
    else:
      dtype = type(value)
    raise TypeError(
        f'Failed to serialize the value {value} of dtype {dtype} to dtype'
        f' {type_spec.dtype.type}.'
    )

  # Repeated fields are used for strings and constants to maintain compatibility
  # with other external environments.
  if (
      array_shape.is_shape_scalar(type_spec.shape)
      or type_spec.dtype.type is np.str_
  ):
    array_pb = array.to_proto(value, dtype_hint=type_spec.dtype.type)
  else:
    array_pb = array.to_proto_content(value, dtype_hint=type_spec.dtype.type)

  value_pb = executor_pb2.Value(array=array_pb)
  return value_pb, type_spec


@tracing.trace
def _serialize_array(
    value: array.Array,
    type_spec: computation_types.TensorType,
) -> array_pb2.Array:
  value_proto, _ = _serialize_tensor_value(value, type_spec)
  return value_proto.array


@tracing.trace
def _serialize_sequence_value(
    value: Sequence[object],
    type_spec: computation_types.SequenceType,
) -> _SerializeReturnType:
  """Serializes a sequence into `executor_pb2.Value`.

  Args:
    value: A list of values convertible to (potentially structures of) tensors.
    type_spec: A `computation_types.Type` specifying the TFF sequence type of
      `value.`

  Returns:
    A tuple `(value_proto, type_spec)` in which `value_proto` is an instance
    of `executor_pb2.Value` with the serialized content of `value`,
    and `type_spec` is the type of the serialized value.
  """
  element_type = type_spec.element
  if not type_analysis.is_structure_of_tensors(element_type):
    raise ValueError(
        'Expected `element_type` to contain only `tff.StructType` or'
        f' `tff.TensorType`, found {element_type}.'
    )

  def _flatten(value, type_spec):
    """Flatten `value` according to `type_spec`."""
    if isinstance(type_spec, computation_types.StructType):

      if isinstance(value, Mapping):
        value = value.values()
      elif isinstance(value, Sequence):
        pass
      else:
        raise NotImplementedError(
            'Expected `value` to be a `Mapping` or `Sequence`, found:'
            f' {type(value)}.'
        )

      result = []
      for v, (_, t) in zip(value, structure.iter_elements(type_spec)):
        result.extend(_flatten(v, t))
      return result
    else:
      return [(value, type_spec)]

  elements_proto = []
  for element in value:
    flat_element = _flatten(element, element_type)
    flat_element_proto = [_serialize_array(v, t) for v, t in flat_element]
    element_proto = executor_pb2.Value.Sequence.Element(
        flat_value=flat_element_proto
    )
    elements_proto.append(element_proto)

  element_type_proto = type_serialization.serialize_type(element_type)
  sequence_proto = executor_pb2.Value.Sequence(
      element_type=element_type_proto, element=elements_proto
  )
  value_proto = executor_pb2.Value(sequence=sequence_proto)
  return value_proto, type_spec


@tracing.trace
def _serialize_struct_type(
    struct_typed_value: object,
    type_spec: computation_types.StructType,
) -> tuple[executor_pb2.Value, computation_types.StructType]:
  """Serializes a value of tuple type."""
  value_structure = structure.from_container(struct_typed_value)
  if len(value_structure) != len(type_spec):
    raise TypeError(
        'Cannot serialize a struct value of '
        f'{len(value_structure)} elements to a struct type '
        f'requiring {len(type_spec)} elements. Trying to serialize'
        f'\n{struct_typed_value!r}\nto\n{type_spec}.'
    )
  type_elem_iter = structure.iter_elements(type_spec)
  val_elem_iter = structure.iter_elements(value_structure)
  elements = []
  for (e_name, e_type), (_, e_val) in zip(type_elem_iter, val_elem_iter):
    e_value, _ = serialize_value(e_val, e_type)
    if e_name:
      element = executor_pb2.Value.Struct.Element(name=e_name, value=e_value)
    else:
      element = executor_pb2.Value.Struct.Element(value=e_value)
    elements.append(element)
  value_proto = executor_pb2.Value(
      struct=executor_pb2.Value.Struct(element=elements)
  )
  return value_proto, type_spec


@tracing.trace
def _serialize_federated_value(
    federated_value: object, type_spec: computation_types.FederatedType
) -> tuple[executor_pb2.Value, computation_types.FederatedType]:
  """Serializes a value of federated type."""
  if type_spec.all_equal:
    value = [federated_value]
  else:
    value = federated_value
  py_typecheck.check_type(value, list)
  value_proto = executor_pb2.Value()
  for v in value:
    federated_value_proto, it_type = serialize_value(v, type_spec.member)
    type_spec.member.check_assignable_from(it_type)
    value_proto.federated.value.append(federated_value_proto)
  value_proto.federated.type.CopyFrom(
      type_serialization.serialize_type(type_spec).federated
  )
  return value_proto, type_spec


@tracing.trace
def serialize_value(
    value: object,
    type_spec: Optional[computation_types.Type] = None,
) -> _SerializeReturnType:
  """Serializes a value into `executor_pb2.Value`.

  We use a switch/function pattern in the body here (and in `deserialize_value`
  below in order to persist more information in traces and profiling.

  Args:
    value: A value to be serialized.
    type_spec: An optional `tff.Type`.

  Returns:
    A 2-tuple of serialized value and `tff.Type` that represents the TFF type of
    the serialized value.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the value is malformed.
  """
  if isinstance(value, computation_pb2.Computation):
    return _serialize_computation(value, type_spec)
  elif isinstance(value, computation_impl.ConcreteComputation):
    return _serialize_computation(
        computation_impl.ConcreteComputation.get_proto(value),
        executor_utils.reconcile_value_with_type_spec(value, type_spec),
    )
  elif type_spec is None:
    raise TypeError(
        'A type hint is required when serializing a value which '
        'is not a TFF computation. Asked to serialized value {v} '
        ' of type {t} with None type spec.'.format(v=value, t=type(value))
    )
  elif isinstance(type_spec, computation_types.TensorType):
    return _serialize_tensor_value(value, type_spec)
  elif isinstance(type_spec, computation_types.SequenceType):
    return _serialize_sequence_value(value, type_spec)
  elif isinstance(type_spec, computation_types.StructType):
    return _serialize_struct_type(value, type_spec)
  elif isinstance(type_spec, computation_types.FederatedType):
    return _serialize_federated_value(value, type_spec)
  else:
    raise ValueError(
        'Unable to serialize value with Python type {} and {} TFF type.'.format(
            str(py_typecheck.type_string(type(value))),
            str(type_spec) if type_spec is not None else 'unknown',
        )
    )


@tracing.trace
def _deserialize_computation(
    value_proto: executor_pb2.Value,
) -> _DeserializeReturnType:
  """Deserializes a TFF computation."""
  which_value = value_proto.computation.WhichOneof('computation')
  if which_value == 'literal':
    value = array.from_proto(value_proto.computation.literal.value)
  else:
    value = value_proto.computation
  type_spec = type_serialization.deserialize_type(value_proto.computation.type)
  return value, type_spec


@tracing.trace
def _deserialize_tensor_value(
    array_proto: array_pb2.Array,
    type_hint: Optional[computation_types.TensorType] = None,
) -> _DeserializeReturnType:
  """Deserializes a tensor value from `.Value`.

  Args:
    array_proto: A `array_pb2.Array` to deserialize.
    type_hint: An optional `tff.Type` to use when deserializing `array_proto`.

  Returns:
    A tuple `(value, type_spec)`, where `value` is a Numpy array that represents
    the deserialized value, and `type_spec` is an instance of `tff.TensorType`
    that represents its type.
  """
  if type_hint is not None:
    type_spec = type_hint
  else:
    dtype = dtype_utils.from_proto(array_proto.dtype)
    shape = array_shape.from_proto(array_proto.shape)
    type_spec = computation_types.TensorType(dtype, shape)

  # Repeated fields are used for strings and constants to maintain compatibility
  # with other external environments.
  if (
      array_shape.is_shape_scalar(type_spec.shape)
      or type_spec.dtype.type is np.str_
  ):
    value = array.from_proto(array_proto)
  else:
    value = array.from_proto_content(array_proto)

  return value, type_spec


@tracing.trace
def _deserialize_sequence_value(
    sequence_proto: executor_pb2.Value.Sequence,
    type_hint: Optional[computation_types.SequenceType] = None,
) -> _DeserializeReturnType:
  """Deserializes a value of sequence type.

  Args:
    sequence_proto: `Sequence` protocol buffer message.
    type_hint: A `computation_types.Type` that hints at what the value type
      should be for executors that only return values. If the
      `sequence_value_proto.element_type` field was not set, the `type_hint` is
      used instead.

  Returns:
    A tuple of `([Array], tff.Type)`.
  """
  if type_hint is not None:
    element_type = type_hint.element
  else:
    element_type = type_serialization.deserialize_type(
        sequence_proto.element_type
    )

  flat_element_type = structure.flatten(element_type)

  elements = []
  for element_proto in sequence_proto.element:
    flat_element = []
    for array_proto, type_spec in zip(
        element_proto.flat_value, flat_element_type
    ):
      value, _ = _deserialize_tensor_value(array_proto, type_spec)
      flat_element.append(value)

    if isinstance(element_type, computation_types.TensorType):
      if len(flat_element) != 1:
        raise ValueError(
            f'Expected `flat_element` of type {element_type} to have only one'
            f' element, found {len(flat_element)}.'
        )
      element, *_ = flat_element
    elif isinstance(element_type, computation_types.StructType):
      element = structure.pack_sequence_as(element_type, flat_element)
      element = type_conversions.type_to_py_container(element, element_type)
    else:
      raise ValueError(
          'Expected `element_type` to be either a `tff.StructType` or a'
          f' `tff.TensorType`, found {element_type}.'
      )
    elements.append(element)

  type_spec = computation_types.SequenceType(element_type)
  return elements, type_spec


@tracing.trace
def _deserialize_struct_value(
    value_proto: executor_pb2.Value,
    type_hint: Optional[computation_types.Type] = None,
) -> _DeserializeReturnType:
  """Deserializes a value of struct type."""
  val_elems = []
  type_elems = []
  if type_hint is not None:
    element_types = tuple(type_hint)  # pytype: disable=wrong-arg-types
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
    next_type: computation_types.Type,
) -> computation_types.Type:
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
    raise TypeError(
        'Type mismatch checking member assignability under a '
        'federated value. Deserialized type {} is incompatible '
        'with previously deserialized {}.'.format(next_type, previous_type)
    )


@tracing.trace
def _deserialize_federated_value(
    value_proto: executor_pb2.Value,
    type_hint: Optional[computation_types.Type] = None,
) -> _DeserializeReturnType:
  """Deserializes a value of federated type."""
  if not value_proto.federated.value:
    raise ValueError('Attempting to deserialize federated value with no data.')
  # The C++ runtime doesn't use the `all_equal` boolean (and doesn't report it
  # in returned values), however the type_hint on the computation may contain
  # it.
  if type_hint is not None:
    all_equal = type_hint.all_equal  # pytype: disable=attribute-error
  else:
    all_equal = value_proto.federated.type.all_equal
  placement_uri = value_proto.federated.type.placement.value.uri
  # item_type will represent a supertype of all deserialized member types in the
  # federated value. This will be the hint used for deserialize member values.
  if type_hint is not None:
    item_type_hint = type_hint.member  # pytype: disable=attribute-error
  else:
    item_type_hint = None
  item_type = None
  if all_equal:
    # As an optimization, we only deserialize the first value of an
    # `all_equal=True` federated value.
    items = [value_proto.federated.value[0]]
  else:
    items = value_proto.federated.value
  value = []
  for item in items:
    item_value, next_item_type = deserialize_value(item, item_type_hint)
    item_type = _ensure_deserialized_types_compatible(item_type, next_item_type)
    value.append(item_value)
  type_spec = computation_types.FederatedType(
      item_type,
      placement=placements.uri_to_placement_literal(placement_uri),
      all_equal=all_equal,
  )
  if all_equal:
    value = value[0]
  return value, type_spec


@tracing.trace
def deserialize_value(
    value_proto: executor_pb2.Value,
    type_hint: Optional[computation_types.Type] = None,
) -> _DeserializeReturnType:
  """Deserializes a value (of any type) from `executor_pb2.Value`.

  Args:
    value_proto: An instance of `executor_pb2.Value`.
    type_hint: A `tff.Type` that hints at what the value type should be for
      executors that only return values.

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
    raise TypeError(
        '`value_proto` must be a protocol buffer message with a '
        '`value` oneof field.'
    )
  which_value = value_proto.WhichOneof('value')
  if which_value == 'array':
    if type_hint is not None and not isinstance(
        type_hint, computation_types.TensorType
    ):
      raise ValueError(f'Expected a `tff.TensorType`, found {type_hint}.')
    return _deserialize_tensor_value(value_proto.array, type_hint)
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
        'Unable to deserialize a value of type {}.'.format(which_value)
    )


def serialize_cardinalities(
    cardinalities: Mapping[placements.PlacementLiteral, int],
) -> list[executor_pb2.Cardinality]:
  serialized_cardinalities = []
  for placement, cardinality in cardinalities.items():
    cardinality_message = executor_pb2.Cardinality(
        placement=computation_pb2.Placement(uri=placement.uri),
        cardinality=cardinality,
    )
    serialized_cardinalities.append(cardinality_message)
  return serialized_cardinalities


def deserialize_cardinalities(
    serialized_cardinalities: Collection[executor_pb2.Cardinality],
) -> dict[placements.PlacementLiteral, int]:
  cardinalities = {}
  for cardinality_spec in serialized_cardinalities:
    literal = placements.uri_to_placement_literal(
        cardinality_spec.placement.uri
    )
    cardinalities[literal] = cardinality_spec.cardinality
  return cardinalities
