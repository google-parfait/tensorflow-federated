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
"""A library of construction functions for tensorflow computation structures."""

from collections.abc import Callable
import functools
from typing import Optional, TypeVar

import federated_language
from federated_language.proto import computation_pb2
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import serialization_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types


ComputationProtoAndType = tuple[
    computation_pb2.Computation, federated_language.Type
]
T = TypeVar('T', bound=federated_language.Type)


class TensorFlowComputationFactory:
  """An implementation of local computation factory for TF computations."""

  def __init__(self):
    pass

  def create_constant_from_scalar(
      self, value, type_spec: federated_language.Type
  ) -> ComputationProtoAndType:
    """Creates a TFF computation returning a constant based on a scalar value.

    The returned computation has the type signature `( -> T)`, where `T` may be
    either a scalar, or a nested structure made up of scalars.

    Args:
      value: A numpy scalar representing the value to return from the
        constructed computation (or to broadcast to all parts of a nested
        structure if `type_spec` is a structured type).
      type_spec: A `federated_language.Type` of the constructed constant. Must
        be either a tensor, or a nested structure of tensors.

    Returns:
      A tuple `(pb.Computation, federated_language.Type)` with the first element
      being a TFF computation with semantics as described above, and the second
      element representing the formal type of that computation.
    """
    return create_constant(value, type_spec)


def _tensorflow_comp(
    tensorflow_proto: computation_pb2.TensorFlow,
    type_signature: T,
) -> tuple[computation_pb2.Computation, T]:
  serialized_type = type_signature.to_proto()
  comp = computation_pb2.Computation(
      type=serialized_type, tensorflow=tensorflow_proto
  )
  return (comp, type_signature)


def create_constant(
    value, type_spec: federated_language.Type
) -> tuple[computation_pb2.Computation, federated_language.FunctionType]:
  """Returns a tensorflow computation returning a constant `value`.

  The returned computation has the type signature `( -> T)`, where `T` is
  `type_spec`.

  `value` must be a value convertible to a tensor or a structure of values, such
  that the dtype and shapes match `type_spec`. `type_spec` must contain only
  named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    value: A value to embed as a constant in the tensorflow graph.
    type_spec: A `federated_language.Type` to use as the argument to the
      constructed binary operator; must contain only named tuples and tensor
      types.

  Raises:
    TypeError: If the constraints of `type_spec` are violated.
  """
  if not federated_language.framework.is_generic_op_compatible_type(type_spec):
    raise TypeError(
        'Type spec {} cannot be constructed as a TensorFlow constant in TFF; '
        ' only nested tuples and tensors are permitted.'.format(type_spec)
    )
  inferred_value_type = type_conversions.tensorflow_infer_type(value)
  if isinstance(
      inferred_value_type, federated_language.StructType
  ) and not type_spec.is_assignable_from(inferred_value_type):
    raise TypeError(
        'Must pass a only tensor or structure of tensor values to '
        '`create_tensorflow_constant`; encountered a value {v} with inferred '
        'type {t!r}, but needed {s!r}'.format(
            v=value, t=inferred_value_type, s=type_spec
        )
    )
  if isinstance(inferred_value_type, federated_language.StructType):
    value = structure.from_container(value, recursive=True)
  tensor_dtypes_in_type_spec = []

  def _pack_dtypes(type_signature):
    """Appends dtype of `type_signature` to nonlocal variable."""
    if isinstance(type_signature, federated_language.TensorType):
      tensor_dtypes_in_type_spec.append(type_signature.dtype)
    return type_signature, False

  federated_language.framework.transform_type_postorder(type_spec, _pack_dtypes)

  if (
      any(np.issubdtype(x, np.integer) for x in tensor_dtypes_in_type_spec)
      and isinstance(inferred_value_type, federated_language.TensorType)
      and not np.issubdtype(inferred_value_type.dtype, np.integer)
  ):
    raise TypeError(
        'Only integers can be used as scalar values if our desired constant '
        'type spec contains any integer tensors; passed scalar {} of dtype {} '
        'for type spec {}.'.format(value, inferred_value_type.dtype, type_spec)
    )

  result_type = type_spec

  def _create_result_tensor(type_spec, value):
    """Packs `value` into `type_spec` recursively."""
    if isinstance(type_spec, federated_language.TensorType):
      if not federated_language.array_shape_is_fully_defined(type_spec.shape):
        raise ValueError(
            f'Expected the shape to be fully defined, found {type_spec.shape}.'
        )
      result = tf.constant(value, dtype=type_spec.dtype, shape=type_spec.shape)
    else:
      elements = []
      if isinstance(inferred_value_type, federated_language.StructType):
        # Copy the leaf values according to the type_spec structure.
        for (name, elem_type), value in zip(type_spec.items(), value):
          elements.append((name, _create_result_tensor(elem_type, value)))
      else:
        # "Broadcast" the value to each level of the type_spec structure.
        for _, elem_type in type_spec.items():  # pytype: disable=attribute-error
          elements.append((None, _create_result_tensor(elem_type, value)))
      result = structure.Struct(elements)
    return result

  with tf.Graph().as_default() as graph:
    result = _create_result_tensor(result_type, value)
    _, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph
    )

  type_signature = federated_language.FunctionType(None, result_type)
  tensorflow = computation_pb2.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding,
  )
  return _tensorflow_comp(tensorflow, type_signature)


def create_unary_operator(
    operator: Callable[..., object], operand_type: federated_language.Type
) -> ComputationProtoAndType:
  """Returns a tensorflow computation computing a unary operation.

  The returned computation has the type signature `(T -> U)`, where `T` is
  `operand_type` and `U` is the result of applying the `operator` to a value of
  type `T`

  Args:
    operator: A callable taking one argument representing the operation to
      encode For example: `tf.math.abs`.
    operand_type: A `federated_language.Type` to use as the argument to the
      constructed unary operator; must contain only named tuples and tensor
      types.

  Raises:
    TypeError: If the constraints of `operand_type` are violated or `operator`
      is not callable.
  """
  if (
      operand_type is None
      or not federated_language.framework.is_generic_op_compatible_type(
          operand_type
      )
  ):
    raise TypeError(
        '`operand_type` contains a type other than '
        '`federated_language.TensorType` and `federated_language.StructType`; '
        f'this is disallowed in the generic operators. Got: {operand_type} '
    )

  with tf.Graph().as_default() as graph:
    operand_value, operand_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', operand_type, graph
    )
    result_value = operator(operand_value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph
    )

  type_signature = federated_language.FunctionType(operand_type, result_type)
  parameter_binding = operand_binding
  tensorflow = computation_pb2.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding,
  )
  return _tensorflow_comp(tensorflow, type_signature)


def create_binary_operator(
    operator: Callable[..., object],
    operand_type: federated_language.Type,
    second_operand_type: Optional[federated_language.Type] = None,
) -> ComputationProtoAndType:
  """Returns a tensorflow computation computing a binary operation.

  The returned computation has the type signature `(<T,T> -> U)`, where `T` is
  `operand_type` and `U` is the result of applying the `operator` to a tuple of
  type `<T,T>`

  Note: If `operand_type` is a `federated_language.StructType`, then
  `operator` will be applied pointwise. This places the burden on callers of
  this function to construct the correct values to pass into the returned
  function. For example, to divide `[2, 2]` by `2`, first `2` must be packed
  into the data structure `[x, x]`, before the division operator of the
  appropriate type is called.

  Args:
    operator: A callable taking two arguments representing the operation to
      encode For example: `tf.math.add`, `tf.math.multiply`, and
      `tf.math.divide`.
    operand_type: A `federated_language.Type` to use as the argument to the
      constructed binary operator; must contain only named tuples and tensor
      types.
    second_operand_type: An optional `federated_language.Type` to use as the
      seocnd argument to the constructed binary operator. If `None`, operator
      uses `operand_type` for both arguments. Must contain only named tuples and
      tensor types.

  Raises:
    TypeError: If the constraints of `operand_type` are violated or `operator`
      is not callable.
  """
  if not federated_language.framework.is_generic_op_compatible_type(
      operand_type
  ):
    raise TypeError(
        '`operand_type` contains a type other than '
        '`federated_language.TensorType` and `federated_language.StructType`; '
        f'this is disallowed in the generic operators. Got: {operand_type} '
    )
  if second_operand_type is not None:
    if not federated_language.framework.is_generic_op_compatible_type(
        second_operand_type
    ):
      raise TypeError(
          '`second_operand_type` contains a type other than'
          ' `federated_language.TensorType` and'
          ' `federated_language.StructType`; this is disallowed in the generic'
          f' operators. Got: {second_operand_type} '
      )
  elif second_operand_type is None:
    second_operand_type = operand_type

  with tf.Graph().as_default() as graph:
    operand_1_value, operand_1_binding = (
        tensorflow_utils.stamp_parameter_in_graph('x', operand_type, graph)
    )
    operand_2_value, operand_2_binding = (
        tensorflow_utils.stamp_parameter_in_graph(
            'y', second_operand_type, graph
        )
    )
    result_value = operator(operand_1_value, operand_2_value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph
    )

  type_signature = federated_language.FunctionType(
      federated_language.StructType((operand_type, second_operand_type)),
      result_type,
  )
  parameter_binding = computation_pb2.TensorFlow.Binding(
      struct=computation_pb2.TensorFlow.StructBinding(
          element=[operand_1_binding, operand_2_binding]
      )  # pytype: disable=wrong-arg-types
  )
  tensorflow = computation_pb2.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding,
  )
  return _tensorflow_comp(tensorflow, type_signature)


def create_binary_operator_with_upcast(
    operator: Callable[[object, object], object],
    type_signature: federated_language.StructType,
) -> ComputationProtoAndType:
  """Creates TF computation upcasting its argument and applying `operator`.

  Args:
    operator: Callable defining the operator.
    type_signature: A `federated_language.StructType` with two elements, both
      only containing structs or tensors in their type tree. The first and
      second element must match in structure, or the second element may be a
      single tensor type that is broadcasted (upcast) to the leaves of the
      structure of the first type. This single tensor may be assignable to the
      tensor types at the leaves, or in the case that the leaves have fully
      defined shapes, this tensor may be `tf.broadcast`-ed to each of those
      shapes. In the case of non-assignability and non-fully defined shapes at
      the leaves of the structure, this function will raise.

  Returns:
    Same as `create_binary_operator()`.
  """
  py_typecheck.check_type(type_signature, federated_language.StructType)
  tensorflow_types.check_tensorflow_compatible_type(type_signature)
  if (
      not isinstance(type_signature, federated_language.StructType)
      or len(type_signature) != 2
  ):
    raise TypeError(
        'To apply a binary operator, we must by definition have an '
        'argument which is a `StructType` with 2 elements; '
        'asked to create a binary operator for type: {t}'.format(
            t=type_signature
        )
    )
  if federated_language.framework.type_contains(
      type_signature, lambda t: isinstance(t, federated_language.SequenceType)
  ):
    raise TypeError(
        'Applying binary operators in TensorFlow is only '
        'supported on Tensors and StructTypes; you '
        'passed {t} which contains a SequenceType.'.format(t=type_signature)
    )

  def _pack_into_type(to_pack: tf.Tensor, type_spec: federated_language.Type):
    """Pack Tensor value `to_pack` into the nested structure `type_spec`."""
    if isinstance(type_spec, federated_language.StructType):
      return structure.Struct([
          (elem_name, _pack_into_type(to_pack, elem_type))
          for elem_name, elem_type in type_spec.items()
      ])
    elif isinstance(type_spec, federated_language.TensorType):
      value_tensor_type = type_conversions.tensorflow_infer_type(to_pack)
      if type_spec.is_assignable_from(value_tensor_type):
        return to_pack
      elif not federated_language.array_shape_is_fully_defined(type_spec.shape):
        raise TypeError(
            'Cannot generate TensorFlow creating binary operator '
            'with first type not assignable from second, and '
            'first type without fully defined shapes. First '
            f'type contains an element of type: {type_spec}.\n'
            f'Packing value {to_pack} into this type is '
            'undefined.'
        )
      return tf.cast(tf.broadcast_to(to_pack, type_spec.shape), type_spec.dtype)  # pytype: disable=attribute-error

  with tf.Graph().as_default() as graph:
    first_arg, operand_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_signature[0], graph
    )
    operand_2_value, operand_2_binding = (
        tensorflow_utils.stamp_parameter_in_graph('y', type_signature[1], graph)
    )

    if isinstance(
        type_signature[0], federated_language.StructType
    ) and isinstance(type_signature[1], federated_language.StructType):
      # If both the first and second arguments are structs with the same
      # structure, simply re-use operand_2_value as. `tf.nest.map_structure`
      # below will map the binary operator pointwise to the leaves of the
      # structure.
      structure_1 = structure.to_odict_or_tuple(type_signature[0])
      structure_2 = structure.to_odict_or_tuple(type_signature[1])
      try:
        tree.assert_same_structure(structure_1, structure_2)
        second_arg = operand_2_value
      except (ValueError, TypeError) as e:
        raise TypeError(
            'Cannot upcast one structure to a different structure. '
            '{x} -> {y}'.format(x=type_signature[1], y=type_signature[0])
        ) from e
    elif type_signature[0].is_assignable_from(type_signature[1]):
      second_arg = operand_2_value
    else:
      second_arg = _pack_into_type(
          operand_2_value,
          type_signature[0],  # pytype: disable=wrong-arg-types
      )

    if isinstance(type_signature[0], federated_language.TensorType):
      result_value = operator(first_arg, second_arg)
    elif isinstance(type_signature[0], federated_language.StructType):
      result_value = structure._map_structure(  # pylint: disable=protected-access
          operator,
          first_arg,  # pytype: disable=wrong-arg-types
          second_arg,  # pytype: disable=wrong-arg-types
      )
    else:
      raise TypeError(
          'Encountered unexpected type {t}; can only handle Tensor '
          'and StructTypes.'.format(t=type_signature[0])
      )

  result_type, result_binding = tensorflow_utils.capture_result_from_graph(
      result_value, graph
  )

  type_signature = federated_language.FunctionType(type_signature, result_type)
  parameter_binding = computation_pb2.TensorFlow.Binding(
      struct=computation_pb2.TensorFlow.StructBinding(
          element=[operand_1_binding, operand_2_binding]
      )  # pytype: disable=wrong-arg-types
  )
  tensorflow = computation_pb2.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding,
  )
  return _tensorflow_comp(tensorflow, type_signature)


def create_empty_tuple() -> ComputationProtoAndType:
  """Returns a tensorflow computation returning an empty tuple.

  The returned computation has the type signature `( -> <>)`.
  """
  return create_computation_for_py_fn(lambda: structure.Struct([]), None)


def create_identity(
    type_signature: federated_language.Type,
) -> ComputationProtoAndType:
  """Returns a tensorflow computation representing an identity function.

  The returned computation has the type signature `(T -> T)`, where `T` is
  `type_signature`. NOTE: if `T` contains `federated_language.StructType`s
  without an associated container type, they will be given the container type
  `tuple` by this function.

  Args:
    type_signature: A `federated_language.Type` to use as the parameter type and
      result type of the identity function.

  Raises:
    TypeError: If `type_signature` contains any types which cannot appear in
      TensorFlow bindings.
  """
  tensorflow_types.check_tensorflow_compatible_type(type_signature)
  parameter_type = type_signature
  if parameter_type is None:
    raise TypeError('TensorFlow identity cannot be created for NoneType.')

  # TF relies on feeds not-identical to fetches in certain circumstances.
  if isinstance(
      type_signature,
      (
          federated_language.SequenceType,
          federated_language.TensorType,
      ),
  ):
    identity_fn = tf.identity
  elif isinstance(type_signature, federated_language.StructType):
    identity_fn = functools.partial(
        structure._map_structure,  # pylint: disable=protected-access
        tf.identity,
    )
  else:
    raise NotImplementedError(
        f'TensorFlow identity cannot be created for type {type_signature}'
    )

  return create_computation_for_py_fn(identity_fn, parameter_type)


def create_computation_for_py_fn(
    fn: Callable[..., object],
    parameter_type: Optional[federated_language.Type],
) -> ComputationProtoAndType:
  """Returns a tensorflow computation returning the result of `fn`.

  The returned computation has the type signature `(T -> U)`, where `T` is
  `parameter_type` and `U` is the type returned by `fn`.

  Args:
    fn: A Python function.
    parameter_type: A `federated_language.Type` or `None`.
  """
  if parameter_type is not None:
    py_typecheck.check_type(parameter_type, federated_language.Type)

  with tf.Graph().as_default() as graph:
    if parameter_type is not None:
      parameter_value, parameter_binding = (
          tensorflow_utils.stamp_parameter_in_graph('x', parameter_type, graph)
      )
      result = fn(parameter_value)
    else:
      parameter_binding = None
      result = fn()
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph
    )

  type_signature = federated_language.FunctionType(parameter_type, result_type)
  tensorflow = computation_pb2.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding,
  )
  return _tensorflow_comp(tensorflow, type_signature)
