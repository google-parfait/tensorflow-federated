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
"""A library of contruction functions for tensorflow computation structures."""

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_transformations
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def create_broadcast_scalar_to_shape(scalar_type: tf.DType,
                                     shape: tf.TensorShape) -> pb.Computation:
  """Returns a tensorflow computation broacasting a scalar to `shape`.

  The returned computation has the type signature `(T -> U)`, where
  `T` is `scalar_type` and the `U` is a `tff.TensorType` with a dtype of
  `scalar_type` and a `shape`.

  Args:
    scalar_type: A `tf.DType`, the type of the scalar to broadcast.
    shape: A `tf.TensorShape` to broadcast to. Must be fully defined.

  Raises:
    TypeError: If `scalar_type` is not a `tf.DType` or if `shape` is not a
      `tf.TensorShape`.
    ValueError: If `shape` is not fully defined.
  """
  py_typecheck.check_type(scalar_type, tf.DType)
  py_typecheck.check_type(shape, tf.TensorShape)
  shape.assert_is_fully_defined()
  type_spec = computation_types.TensorType(scalar_type, shape=())

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    result = tf.broadcast_to(parameter_value, shape)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(type_spec, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_constant(scalar_value, type_spec) -> pb.Computation:
  """Returns a tensorflow computation returning a constant `scalar_value`.

  The returned computation has the type signature `( -> T)`, where `T` is
  `type_spec`.

  `scalar_value` must be a scalar, and cannot be a float if any of the tensor
  leaves of `type_spec` contain an integer data type. `type_spec` must contain
  only named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    scalar_value: A scalar value to place in all the tensor leaves of
      `type_spec`.
    type_spec: A type convertible to instance of `computation_types.Type` via
      `computation_types.to_type` and whose resulting type tree can only contain
      named tuples and tensors.

  Raises:
    TypeError: If the constraints of `type_spec` are violated.
  """
  type_spec = computation_types.to_type(type_spec)

  if not type_analysis.is_generic_op_compatible_type(type_spec):
    raise TypeError(
        'Type spec {} cannot be constructed as a TensorFlow constant in TFF; '
        ' only nested tuples and tensors are permitted.'.format(type_spec))
  inferred_scalar_value_type = type_conversions.infer_type(scalar_value)
  if (not isinstance(inferred_scalar_value_type, computation_types.TensorType)
      or inferred_scalar_value_type.shape != tf.TensorShape(())):
    raise TypeError(
        'Must pass a scalar value to `create_tensorflow_constant`; encountered '
        'a value {}'.format(scalar_value))
  tensor_dtypes_in_type_spec = []

  def _pack_dtypes(type_signature):
    """Appends dtype of `type_signature` to nonlocal variable."""
    if isinstance(type_signature, computation_types.TensorType):
      tensor_dtypes_in_type_spec.append(type_signature.dtype)
    return type_signature, False

  type_transformations.transform_type_postorder(type_spec, _pack_dtypes)

  if (any(x.is_integer for x in tensor_dtypes_in_type_spec) and
      not inferred_scalar_value_type.dtype.is_integer):
    raise TypeError(
        'Only integers can be used as scalar values if our desired constant '
        'type spec contains any integer tensors; passed scalar {} of dtype {} '
        'for type spec {}.'.format(scalar_value,
                                   inferred_scalar_value_type.dtype, type_spec))

  def _create_result_tensor(type_spec, scalar_value):
    """Packs `scalar_value` into `type_spec` recursively."""
    if isinstance(type_spec, computation_types.TensorType):
      type_spec.shape.assert_is_fully_defined()
      result = tf.constant(
          scalar_value, dtype=type_spec.dtype, shape=type_spec.shape)
    else:
      elements = []
      for _, type_element in anonymous_tuple.iter_elements(type_spec):
        elements.append(_create_result_tensor(type_element, scalar_value))
      result = elements
    return result

  with tf.Graph().as_default() as graph:
    result = _create_result_tensor(type_spec, scalar_value)
    _, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(None, type_spec)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_binary_operator(operator, operand_type) -> pb.Computation:
  """Returns a tensorflow computation representing the binary `operator`.

  The returned computation has the type signature `(<T,T> -> U)`, where `T` is
  `operand_type` and `U` is the result of applying the `operator` to a tuple of
  type `<T,T>`

  Note: If `operand_type` is a `computation_types.NamedTupleType`, then
  `operator` will be applied pointwise. This places the burden on callers of
  this function to construct the correct values to pass into the returned
  function. For example, to divide `[2, 2]` by `2`, first `2` must be packed
  into the data structure `[x, x]`, before the division operator of the
  appropriate type is called.

  Args:
    operator: A callable taking two arguments representing the operation to
      encode For example: `tf.math.add`, `tf.math.multiply`, and
        `tf.math.divide`.
    operand_type: The type of the argument to the constructed binary operator; A
      type convertible to instance of `computation_types.Type` via
      `computation_types.to_type` which can only contain types which are
      compatible with the TFF generic operators (named tuples and tensors).

  Raises:
    TypeError: If the constraints of `operand_type` are violated or `operator`
      is not callable.
  """
  operand_type = computation_types.to_type(operand_type)
  if not type_analysis.is_generic_op_compatible_type(operand_type):
    raise TypeError(
        'The type {} contains a type other than `computation_types.TensorType` '
        'and `computation_types.NamedTupleType`; this is disallowed in the '
        'generic operators.'.format(operand_type))
  py_typecheck.check_callable(operator)
  with tf.Graph().as_default() as graph:
    operand_1_value, operand_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', operand_type, graph)
    operand_2_value, operand_2_binding = tensorflow_utils.stamp_parameter_in_graph(
        'y', operand_type, graph)

    if isinstance(operand_type, computation_types.TensorType):
      result_value = operator(operand_1_value, operand_2_value)
    elif isinstance(operand_type, computation_types.NamedTupleType):
      result_value = anonymous_tuple.map_structure(operator, operand_1_value,
                                                   operand_2_value)
    else:
      raise TypeError(
          'Operand type {} cannot be used in generic operations. The whitelist '
          'in `type_analysis.is_generic_op_compatible_type` has allowed it to '
          'pass, and should be updated.'.format(operand_type))
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph)

  type_signature = computation_types.FunctionType([operand_type, operand_type],
                                                  result_type)
  parameter_binding = pb.TensorFlow.Binding(
      tuple=pb.TensorFlow.NamedTupleBinding(
          element=[operand_1_binding, operand_2_binding]))
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_empty_tuple() -> pb.Computation:
  """Returns a tensorflow computation returning an empty tuple.

  The returned computation has the type signature `( -> <>)`.
  """

  with tf.Graph().as_default() as graph:
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        [], graph)

  type_signature = computation_types.FunctionType(None, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_identity(type_spec) -> pb.Computation:
  """Returns a tensorflow computation representing an identity function.

  The returned computation has the type signature `(T -> T)`, where `T` is
  `type_spec`.

  Args:
    type_spec: A type convertible to instance of `computation_types.Type` via
      `computation_types.to_type`.

  Raises:
    TypeError: If `type_spec` contains any types which cannot appear in
      TensorFlow bindings.
  """
  type_spec = computation_types.to_type(type_spec)
  type_analysis.check_tensorflow_compatible_type(type_spec)

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        parameter_value, graph)

  type_signature = computation_types.FunctionType(type_spec, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_replicate_input(type_spec, count: int) -> pb.Computation:
  """Returns a tensorflow computation which returns `count` clones of an input.

  The returned computation has the type signature `(T -> <T, T, T, ...>)`, where
  `T` is `type_spec` and the length of the result is `count`.

  Args:
    type_spec: A type convertible to instance of `computation_types.Type` via
      `computation_types.to_type`.
    count: An integer, the number of times the input is replicated.

  Raises:
    TypeError: If `type_spec` contains any types which cannot appear in
      TensorFlow bindings or if `which` is not an integer.
  """
  type_spec = computation_types.to_type(type_spec)
  type_analysis.check_tensorflow_compatible_type(type_spec)
  py_typecheck.check_type(count, int)

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    result = [parameter_value] * count
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(type_spec, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)
