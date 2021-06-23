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

import functools
import types
from typing import Any, Callable, Optional

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_transformations
from tensorflow_federated.python.core.impl.utils import tensorflow_utils

# TODO(b/181028772): Move this and similar code to `backends/tensorflow`.

# TODO(b/181131807): Remove independent invocations of the helper methods, and
# replace them with calls to the factory, then inline the bodies of the methods
# within the factory.

ComputationProtoAndType = local_computation_factory_base.ComputationProtoAndType


class TensorFlowComputationFactory(
    local_computation_factory_base.LocalComputationFactory):
  """An implementation of local computation factory for TF computations."""

  def __init__(self):
    pass

  def create_constant_from_scalar(
      self, value,
      type_spec: computation_types.Type) -> ComputationProtoAndType:
    return create_constant(value, type_spec)

  def create_plus_operator(
      self, type_spec: computation_types.Type) -> ComputationProtoAndType:

    def plus(a, b):
      return structure.map_structure(tf.add, a, b)

    return create_binary_operator(plus, type_spec)

  def create_multiply_operator(
      self, type_spec: computation_types.Type) -> ComputationProtoAndType:

    def multiply(a, b):
      return structure.map_structure(tf.multiply, a, b)

    return create_binary_operator(multiply, type_spec)

  def create_scalar_multiply_operator(
      self, operand_type: computation_types.Type,
      scalar_type: computation_types.TensorType) -> ComputationProtoAndType:
    return create_binary_operator_with_upcast(
        computation_types.StructType([(None, operand_type),
                                      (None, scalar_type)]), tf.multiply)

  def create_indexing_operator(
      self,
      operand_type: computation_types.TensorType,
      index_type: computation_types.TensorType,
  ) -> ComputationProtoAndType:
    return create_indexing_operator(operand_type, index_type)


def _tensorflow_comp(
    tensorflow_proto: pb.TensorFlow,
    type_signature: computation_types.Type,
) -> ComputationProtoAndType:
  serialized_type = type_serialization.serialize_type(type_signature)
  comp = pb.Computation(type=serialized_type, tensorflow=tensorflow_proto)
  return (comp, type_signature)


def create_constant(
    value, type_spec: computation_types.Type) -> ComputationProtoAndType:
  """Returns a tensorflow computation returning a constant `value`.

  The returned computation has the type signature `( -> T)`, where `T` is
  `type_spec`.

  `value` must be a value convertible to a tensor or a structure of values, such
  that the dtype and shapes match `type_spec`. `type_spec` must contain only
  named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    value: A value to embed as a constant in the tensorflow graph.
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
  inferred_value_type = type_conversions.infer_type(value)
  if (inferred_value_type.is_struct() and
      not type_spec.is_assignable_from(inferred_value_type)):
    raise TypeError(
        'Must pass a only tensor or structure of tensor values to '
        '`create_tensorflow_constant`; encountered a value {v} with inferred '
        'type {t!r}, but needed {s!r}'.format(
            v=value, t=inferred_value_type, s=type_spec))
  if inferred_value_type.is_struct():
    value = structure.from_container(value, recursive=True)
  tensor_dtypes_in_type_spec = []

  def _pack_dtypes(type_signature):
    """Appends dtype of `type_signature` to nonlocal variable."""
    if type_signature.is_tensor():
      tensor_dtypes_in_type_spec.append(type_signature.dtype)
    return type_signature, False

  type_transformations.transform_type_postorder(type_spec, _pack_dtypes)

  if (any(x.is_integer for x in tensor_dtypes_in_type_spec) and
      (inferred_value_type.is_tensor() and
       not inferred_value_type.dtype.is_integer)):
    raise TypeError(
        'Only integers can be used as scalar values if our desired constant '
        'type spec contains any integer tensors; passed scalar {} of dtype {} '
        'for type spec {}.'.format(value, inferred_value_type.dtype, type_spec))

  result_type = type_spec

  def _create_result_tensor(type_spec, value):
    """Packs `value` into `type_spec` recursively."""
    if type_spec.is_tensor():
      type_spec.shape.assert_is_fully_defined()
      result = tf.constant(value, dtype=type_spec.dtype, shape=type_spec.shape)
    else:
      elements = []
      if inferred_value_type.is_struct():
        # Copy the leaf values according to the type_spec structure.
        for (name, elem_type), value in zip(
            structure.iter_elements(type_spec), value):
          elements.append((name, _create_result_tensor(elem_type, value)))
      else:
        # "Broadcast" the value to each level of the type_spec structure.
        for _, elem_type in structure.iter_elements(type_spec):
          elements.append((None, _create_result_tensor(elem_type, value)))
      result = structure.Struct(elements)
    return result

  with tf.Graph().as_default() as graph:
    result = _create_result_tensor(result_type, value)
    _, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(None, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding)
  return _tensorflow_comp(tensorflow, type_signature)


def create_unary_operator(
    operator, operand_type: computation_types.Type) -> ComputationProtoAndType:
  """Returns a tensorflow computation computing a unary operation.

  The returned computation has the type signature `(T -> U)`, where `T` is
  `operand_type` and `U` is the result of applying the `operator` to a value of
  type `T`

  Args:
    operator: A callable taking one argument representing the operation to
      encode For example: `tf.math.abs`.
    operand_type: A `computation_types.Type` to use as the argument to the
      constructed unary operator; must contain only named tuples and tensor
      types.

  Raises:
    TypeError: If the constraints of `operand_type` are violated or `operator`
      is not callable.
  """
  if (operand_type is None or
      not type_analysis.is_generic_op_compatible_type(operand_type)):
    raise TypeError(
        '`operand_type` contains a type other than '
        '`computation_types.TensorType` and `computation_types.StructType`; '
        f'this is disallowed in the generic operators. Got: {operand_type} ')
  py_typecheck.check_callable(operator)

  with tf.Graph().as_default() as graph:
    operand_value, operand_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', operand_type, graph)
    result_value = operator(operand_value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph)

  type_signature = computation_types.FunctionType(operand_type, result_type)
  parameter_binding = operand_binding
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return _tensorflow_comp(tensorflow, type_signature)


def create_binary_operator(
    operator,
    operand_type: computation_types.Type,
    second_operand_type: Optional[computation_types.Type] = None
) -> ComputationProtoAndType:
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
    second_operand_type: An optional `computation_types.Type` to use as the
      seocnd argument to the constructed binary operator. If `None`, operator
      uses `operand_type` for both arguments. Must contain only named tuples and
      tensor types.

  Raises:
    TypeError: If the constraints of `operand_type` are violated or `operator`
      is not callable.
  """
  if not type_analysis.is_generic_op_compatible_type(operand_type):
    raise TypeError(
        '`operand_type` contains a type other than '
        '`computation_types.TensorType` and `computation_types.StructType`; '
        f'this is disallowed in the generic operators. Got: {operand_type} ')
  if second_operand_type is not None:
    if not type_analysis.is_generic_op_compatible_type(second_operand_type):
      raise TypeError(
          '`second_operand_type` contains a type other than '
          '`computation_types.TensorType` and `computation_types.StructType`; '
          'this is disallowed in the generic operators. '
          f'Got: {second_operand_type} ')
  elif second_operand_type is None:
    second_operand_type = operand_type
  py_typecheck.check_callable(operator)

  with tf.Graph().as_default() as graph:
    operand_1_value, operand_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', operand_type, graph)
    operand_2_value, operand_2_binding = tensorflow_utils.stamp_parameter_in_graph(
        'y', second_operand_type, graph)
    result_value = operator(operand_1_value, operand_2_value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph)

  type_signature = computation_types.FunctionType(
      computation_types.StructType((operand_type, second_operand_type)),
      result_type)
  parameter_binding = pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(
          element=[operand_1_binding, operand_2_binding]))
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return _tensorflow_comp(tensorflow, type_signature)


def create_binary_operator_with_upcast(
    type_signature: computation_types.StructType,
    operator: Callable[[Any, Any], Any]) -> ComputationProtoAndType:
  """Creates TF computation upcasting its argument and applying `operator`.

  Args:
    type_signature: A `computation_types.StructType` with two elements, both
      only containing structs or tensors in their type tree. The first and
      second element must match in structure, or the second element may be a
      single tensor type that is broadcasted (upcast) to the leaves of the
      structure of the first type.
    operator: Callable defining the operator.

  Returns:
    Same as `create_binary_operator()`.
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

    if type_signature[0].is_struct() and type_signature[1].is_struct():
      # If both the first and second arguments are structs with the same
      # structure, simply re-use operand_2_value as. `tf.nest.map_structure`
      # below will map the binary operator pointwise to the leaves of the
      # structure.
      if structure.is_same_structure(type_signature[0], type_signature[1]):
        second_arg = operand_2_value
      else:
        raise TypeError('Cannot upcast one structure to a different structure. '
                        '{x} -> {y}'.format(
                            x=type_signature[1], y=type_signature[0]))
    elif type_signature[0].is_equivalent_to(type_signature[1]):
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
  return _tensorflow_comp(tensorflow, type_signature)


def create_indexing_operator(
    operand_type: computation_types.TensorType,
    index_type: computation_types.TensorType,
) -> ComputationProtoAndType:
  """Returns a tensorflow computation computing an indexing operation."""
  operand_type.check_tensor()
  index_type.check_tensor()
  if index_type.shape.rank != 0:
    raise TypeError(f'Expected index type to be a scalar, found {index_type}.')
  with tf.Graph().as_default() as graph:
    operand_value, operand_binding = tensorflow_utils.stamp_parameter_in_graph(
        'indexing_operand', operand_type, graph)
    index_value, index_binding = tensorflow_utils.stamp_parameter_in_graph(
        'index', index_type, graph)
    result_value = tf.gather(operand_value, index_value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph)
  type_signature = computation_types.FunctionType(
      computation_types.StructType((operand_type, index_type)), result_type)
  parameter_binding = pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(
          element=[operand_binding, index_binding]))
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return _tensorflow_comp(tensorflow, type_signature)


def create_empty_tuple() -> ComputationProtoAndType:
  """Returns a tensorflow computation returning an empty tuple.

  The returned computation has the type signature `( -> <>)`.
  """
  return create_computation_for_py_fn(lambda: structure.Struct([]), None)


def create_identity(
    type_signature: computation_types.Type) -> ComputationProtoAndType:
  """Returns a tensorflow computation representing an identity function.

  The returned computation has the type signature `(T -> T)`, where `T` is
  `type_signature`. NOTE: if `T` contains `computation_types.StructType`s
  without an associated container type, they will be given the container type
  `tuple` by this function.

  Args:
    type_signature: A `computation_types.Type` to use as the parameter type and
      result type of the identity function.

  Raises:
    TypeError: If `type_signature` contains any types which cannot appear in
      TensorFlow bindings.
  """
  type_analysis.check_tensorflow_compatible_type(type_signature)
  parameter_type = type_signature
  if parameter_type is None:
    raise TypeError('TensorFlow identity cannot be created for NoneType.')

  # TF relies on feeds not-identical to fetches in certain circumstances.
  if type_signature.is_tensor() or type_signature.is_sequence():
    identity_fn = tf.identity
  elif type_signature.is_struct():
    identity_fn = functools.partial(structure.map_structure, tf.identity)
  else:
    raise NotImplementedError(
        f'TensorFlow identity cannot be created for type {type_signature}')

  return create_computation_for_py_fn(identity_fn, parameter_type)


def create_replicate_input(type_signature: computation_types.Type,
                           count: int) -> ComputationProtoAndType:
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
  return create_computation_for_py_fn(lambda v: [v] * count, parameter_type)


def create_computation_for_py_fn(
    fn: types.FunctionType, parameter_type: Optional[computation_types.Type]
) -> ComputationProtoAndType:
  """Returns a tensorflow computation returning the result of `fn`.

  The returned computation has the type signature `(T -> U)`, where `T` is
  `parameter_type` and `U` is the type returned by `fn`.

  Args:
    fn: A Python function.
    parameter_type: A `computation_types.Type` or `None`.
  """
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
  return _tensorflow_comp(tensorflow, type_signature)
