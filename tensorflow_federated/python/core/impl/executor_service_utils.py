# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import any_pb2

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils


def serialize_tensor_value(value, type_spec=None):
  """Serializes a tensor value into `executor_pb2.Value`.

  Args:
    value: A Numpy array or other object understood by `tf.make_tensor_proto`.
    type_spec: An optional type spec, a `tff.TensorType` or something
      convertible to it.

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
    if type_spec is None:
      type_spec = computation_types.TensorType(
          dtype=tf.DType(value.dtype), shape=tf.TensorShape(value.shape))
    value = value.numpy()
  if type_spec is not None:
    type_spec = computation_types.to_type(type_spec)
    py_typecheck.check_type(type_spec, computation_types.TensorType)
    if isinstance(value, np.ndarray):
      tensor_proto = tf.make_tensor_proto(
          value, dtype=type_spec.dtype, verify_shape=False)
      type_utils.check_assignable_from(
          type_spec,
          computation_types.TensorType(
              dtype=tf.DType(tensor_proto.dtype),
              shape=tf.TensorShape(tensor_proto.tensor_shape)))
    else:
      tensor_proto = tf.make_tensor_proto(
          value,
          dtype=type_spec.dtype,
          shape=type_spec.shape,
          verify_shape=True)
  else:
    tensor_proto = tf.make_tensor_proto(value)
    type_spec = computation_types.TensorType(
        dtype=tf.DType(tensor_proto.dtype),
        shape=tf.TensorShape(tensor_proto.tensor_shape))
  any_pb = any_pb2.Any()
  any_pb.Pack(tensor_proto)
  return executor_pb2.Value(tensor=any_pb), type_spec


def deserialize_tensor_value(value_proto):
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
  py_typecheck.check_type(value_proto, executor_pb2.Value)
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
      dtype=tf.DType(tensor_proto.dtype),
      shape=tf.TensorShape(tensor_proto.tensor_shape))

  return tensor_value, value_type


def serialize_value(value, type_spec=None):
  """Serializes a value into `executor_pb2.Value`.

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
    type_spec = type_utils.reconcile_value_type_with_type_spec(
        type_serialization.deserialize_type(value.type), type_spec)
    return executor_pb2.Value(computation=value), type_spec
  elif isinstance(value, computation_impl.ComputationImpl):
    return serialize_value(
        computation_impl.ComputationImpl.get_proto(value),
        type_utils.reconcile_value_with_type_spec(value, type_spec))
  elif isinstance(type_spec, computation_types.TensorType):
    return serialize_tensor_value(value, type_spec)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    type_elements = anonymous_tuple.to_elements(type_spec)
    val_elements = anonymous_tuple.to_elements(
        anonymous_tuple.from_container(value))
    tup_elems = []
    for (e_name, e_type), (_, e_val) in zip(type_elements, val_elements):
      e_proto, _ = serialize_value(e_val, e_type)
      tup_elems.append(
          executor_pb2.Value.Tuple.Element(
              name=e_name if e_name else None, value=e_proto))
    result_proto = (
        executor_pb2.Value(tuple=executor_pb2.Value.Tuple(element=tup_elems)))
    return result_proto, type_spec
  else:
    raise ValueError(
        'Unable to serialize value with Python type {} and {} TFF type.'.format(
            str(py_typecheck.type_string(type(value))),
            str(type_spec) if type_spec is not None else 'unknown'))


def deserialize_value(value_proto):
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
    return deserialize_tensor_value(value_proto)
  elif which_value == 'computation':
    return (value_proto.computation,
            type_serialization.deserialize_type(value_proto.computation.type))
  elif which_value == 'tuple':
    val_elems = []
    type_elems = []
    for e in value_proto.tuple.element:
      name = e.name if e.name else None
      e_val, e_type = deserialize_value(e.value)
      val_elems.append((name, e_val))
      type_elems.append((name, e_type) if name else e_type)
    return (anonymous_tuple.AnonymousTuple(val_elems),
            computation_types.NamedTupleType(type_elems))
  else:
    raise ValueError(
        'Unable to deserialize a value of type {}.'.format(which_value))
