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

import types
from typing import Any, Callable, Optional

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_transformations
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def create_broadcast_scalar_to_shape(scalar_type: tf.DType,
                                     shape: tf.TensorShape) -> pb.Computation:
  """Returns a tensorflow computation returning the result of `tf.broadcast_to`.

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
  parameter_type = computation_types.TensorType(scalar_type, shape=())

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', parameter_type, graph)
    result = tf.broadcast_to(parameter_value, shape)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(parameter_type, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_constant(scalar_value,
                    type_spec: computation_types.Type) -> pb.Computation:
  """Returns a tensorflow computation returning a constant `scalar_value`.

  The returned computation has the type signature `( -> T)`, where `T` is
  `type_spec`.

  `scalar_value` must be a scalar, and cannot be a float if any of the tensor
  leaves of `type_spec` contain an integer data type. `type_spec` must contain
  only named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    scalar_value: A scalar value to place in all the tensor leaves of
      `type_spec`.
    type_spec: A `computation_types.Type` to use as the argument to the
      constructed binary operator; must contain only named tuples and tensor
      types.

  Raises:
    TypeError: If the constraints of `type_spec` are violated.
  """
  if not type_analysis.is_generic_op_compatible_type(type_spec):
    raise TypeError(
        'Type spec {} cannot be constructed as a TensorFlow constant in TFF; '
        ' only nested tuples and tensors are permitted.'.format(type_spec))
  inferred_scalar_value_type = type_conversions.infer_type(scalar_value)
  if (not inferred_scalar_value_type.is_tensor() or
      inferred_scalar_value_type.shape != tf.TensorShape(())):
    raise TypeError(
        'Must pass a scalar value to `create_tensorflow_constant`; encountered '
        'a value {}'.format(scalar_value))
  tensor_dtypes_in_type_spec = []

  def _pack_dtypes(type_signature):
    """Appends dtype of `type_signature` to nonlocal variable."""
    if type_signature.is_tensor():
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

  result_type = type_spec

  def _create_result_tensor(type_spec, scalar_value):
    """Packs `scalar_value` into `type_spec` recursively."""
    if type_spec.is_tensor():
      type_spec.shape.assert_is_fully_defined()
      result = tf.constant(
          scalar_value, dtype=type_spec.dtype, shape=type_spec.shape)
    else:
      elements = []
      for _, type_element in structure.iter_elements(type_spec):
        elements.append(_create_result_tensor(type_element, scalar_value))
      result = elements
    return result

  with tf.Graph().as_default() as graph:
    result = _create_result_tensor(result_type, scalar_value)
    _, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(None, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_binary_operator(
    operator, operand_type: computation_types.Type) -> pb.Computation:
  """Returns a tensorflow computation computing a binary operation.

  The returned computation has the type signature `(<T,T> -> U)`, where `T` is
  `operand_type` and `U` is the result of applying the `operator` to a tuple of
  type `<T,T>`

  Note: If `operand_type` is a `computation_types.StructType`, then
  `operator` will be applied pointwise. This places the burden on callers of
  this function to construct the correct values to pass into the returned
  function. For example, to divide `[2, 2]` by `2`, first `2` must be packed
  into the data structure `[x, x]`, before the division operator of the
  appropriate type is called.

  Args:
    operator: A callable taking two arguments representing the operation to
      encode For example: `tf.math.add`, `tf.math.multiply`, and
        `tf.math.divide`.
    operand_type: A `computation_types.Type` to use as the argument to the
      constructed binary operator; must contain only named tuples and tensor
      types.

  Raises:
    TypeError: If the constraints of `operand_type` are violated or `operator`
      is not callable.
  """
  if not type_analysis.is_generic_op_compatible_type(operand_type):
    raise TypeError(
        'The type {} contains a type other than `computation_types.TensorType` '
        'and `computation_types.StructType`; this is disallowed in the '
        'generic operators.'.format(operand_type))
  py_typecheck.check_callable(operator)
  with tf.Graph().as_default() as graph:
    operand_1_value, operand_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', operand_type, graph)
    operand_2_value, operand_2_binding = tensorflow_utils.stamp_parameter_in_graph(
        'y', operand_type, graph)

    if operand_type is not None:
      if operand_type.is_tensor():
        result_value = operator(operand_1_value, operand_2_value)
      elif operand_type.is_struct():
        result_value = structure.map_structure(operator, operand_1_value,
                                               operand_2_value)
    else:
      raise TypeError(
          'Operand type {} cannot be used in generic operations. The call to '
          '`type_analysis.is_generic_op_compatible_type` has allowed it to '
          'pass, and should be updated.'.format(operand_type))
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph)

  type_signature = computation_types.FunctionType(
      computation_types.StructType((operand_type, operand_type)), result_type)
  parameter_binding = pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(
          element=[operand_1_binding, operand_2_binding]))
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_binary_operator_with_upcast(
    type_signature: computation_types.StructType,
    operator: Callable[[Any, Any], Any]) -> pb.Computation:
  """Creates TF computation upcasting its argument and applying `operator`.

  Args:
    type_signature: A `computation_types.StructType` with two elements, both
      of the same type or the second able to be upcast to the first, as
      explained in `apply_binary_operator_with_upcast`, and both containing only
      tuples and tensors in their type tree.
    operator: Callable defining the operator.

  Returns:
    A `building_blocks.CompiledComputation` encapsulating a function which
    upcasts the second element of its argument and applies the binary
    operator.
  """
  py_typecheck.check_type(type_signature, computation_types.StructType)
  py_typecheck.check_callable(operator)
  type_analysis.check_tensorflow_compatible_type(type_signature)
  if not type_signature.is_struct() or len(type_signature) != 2:
    raise TypeError('To apply a binary operator, we must by definition have an '
                    'argument which is a `StructType` with 2 elements; '
                    'asked to create a binary operator for type: {t}'.format(
                        t=type_signature))
  if type_analysis.contains(type_signature, lambda t: t.is_sequence()):
    raise TypeError(
        'Applying binary operators in TensorFlow is only '
        'supported on Tensors and StructTypes; you '
        'passed {t} which contains a SequenceType.'.format(t=type_signature))

  def _pack_into_type(to_pack, type_spec):
    """Pack Tensor value `to_pack` into the nested structure `type_spec`."""
    if type_spec.is_struct():
      elem_iter = structure.iter_elements(type_spec)
      return structure.Struct([(elem_name, _pack_into_type(to_pack, elem_type))
                               for elem_name, elem_type in elem_iter])
    elif type_spec.is_tensor():
      return tf.broadcast_to(to_pack, type_spec.shape)

  with tf.Graph().as_default() as graph:
    first_arg, operand_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_signature[0], graph)
    operand_2_value, operand_2_binding = tensorflow_utils.stamp_parameter_in_graph(
        'y', type_signature[1], graph)
    if type_signature[0].is_equivalent_to(type_signature[1]):
      second_arg = operand_2_value
    else:
      second_arg = _pack_into_type(operand_2_value, type_signature[0])

    if type_signature[0].is_tensor():
      result_value = operator(first_arg, second_arg)
    elif type_signature[0].is_struct():
      result_value = structure.map_structure(operator, first_arg, second_arg)
    else:
      raise TypeError('Encountered unexpected type {t}; can only handle Tensor '
                      'and StructTypes.'.format(t=type_signature[0]))

  result_type, result_binding = tensorflow_utils.capture_result_from_graph(
      result_value, graph)

  type_signature = computation_types.FunctionType(type_signature, result_type)
  parameter_binding = pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(
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


def create_identity(type_signature: computation_types.Type) -> pb.Computation:
  """Returns a tensorflow computation representing an identity function.

  The returned computation has the type signature `(T -> T)`, where `T` is
  `type_signature`.

  Args:
    type_signature: A `computation_types.Type` to use as the parameter type and
      result type of the identity function.

  Raises:
    TypeError: If `type_signature` contains any types which cannot appear in
      TensorFlow bindings.
  """
  type_analysis.check_tensorflow_compatible_type(type_signature)
  parameter_type = type_signature

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', parameter_type, graph)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        parameter_value, graph)

  type_signature = computation_types.FunctionType(parameter_type, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_replicate_input(type_signature: computation_types.Type,
                           count: int) -> pb.Computation:
  """Returns a tensorflow computation returning `count` copies of its argument.

  The returned computation has the type signature `(T -> <T, T, T, ...>)`, where
  `T` is `type_signature` and the length of the result is `count`.

  Args:
    type_signature: A `computation_types.Type` to replicate.
    count: An integer, the number of times the input is replicated.

  Raises:
    TypeError: If `type_signature` contains any types which cannot appear in
      TensorFlow bindings or if `which` is not an integer.
  """
  type_analysis.check_tensorflow_compatible_type(type_signature)
  py_typecheck.check_type(count, int)
  parameter_type = type_signature

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', parameter_type, graph)
    result = [parameter_value] * count
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(parameter_type, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_computation_for_py_fn(
    fn: types.FunctionType,
    parameter_type: Optional[computation_types.Type]) -> pb.Computation:
  """Returns a tensorflow computation returning the result of `fn`.

  The returned computation has the type signature `(T -> U)`, where `T` is
  `parameter_type` and `U` is the type returned by `fn`.

  Args:
    fn: A Python function.
    parameter_type: A `computation_types.Type`.
  """
  py_typecheck.check_type(fn, types.FunctionType)
  if parameter_type is not None:
    py_typecheck.check_type(parameter_type, computation_types.Type)

  with tf.Graph().as_default() as graph:
    if parameter_type is not None:
      parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
          'x', parameter_type, graph)
      result = fn(parameter_value)
    else:
      parameter_binding = None
      result = fn()
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(parameter_type, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)
