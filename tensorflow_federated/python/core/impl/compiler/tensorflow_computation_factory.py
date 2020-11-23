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
from typing import Any, Callable, Optional, Tuple

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

ProtoAndType = Tuple[pb.Computation, computation_types.Type]


def _tensorflow_comp(
    tensorflow_proto: pb.TensorFlow,
    type_signature: computation_types.Type,
) -> ProtoAndType:
  serialized_type = type_serialization.serialize_type(type_signature)
  comp = pb.Computation(type=serialized_type, tensorflow=tensorflow_proto)
  return (comp, type_signature)


def create_constant(value, type_spec: computation_types.Type) -> ProtoAndType:
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


def create_binary_operator(
    operator, operand_type: computation_types.Type) -> ProtoAndType:
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
  return _tensorflow_comp(tensorflow, type_signature)


def create_binary_operator_with_upcast(
    type_signature: computation_types.StructType,
    operator: Callable[[Any, Any], Any]) -> ProtoAndType:
  """Creates TF computation upcasting its argument and applying `operator`.

  Args:
    type_signature: A `computation_types.StructType` with two elements, both
      only containing structs or tensors in their type tree. The first and
      second element must match in structure, or the second element may be a
      single tensor type that is broadcasted (upcast) to the leaves of the
      structure of the first type.
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


def create_empty_tuple() -> ProtoAndType:
  """Returns a tensorflow computation returning an empty tuple.

  The returned computation has the type signature `( -> <>)`.
  """
  return create_computation_for_py_fn(lambda: structure.Struct([]), None)


def create_identity(type_signature: computation_types.Type) -> ProtoAndType:
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
                           count: int) -> ProtoAndType:
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
    fn: types.FunctionType,
    parameter_type: Optional[computation_types.Type]) -> ProtoAndType:
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
