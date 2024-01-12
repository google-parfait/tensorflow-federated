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
"""A library for constructing XLA computation."""

from typing import Union

from jax.lib import xla_client
import numpy as np

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.xla_backend import xla_serialization
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis


def _array_shape_from_tensor_type(
    tensor_type: computation_types.TensorType,
) -> xla_client.Shape:
  """Returns an `xla_client.Shape` from a `tff.TensorType`."""
  return xla_client.Shape.array_shape(
      xla_client.dtype_to_etype(tensor_type.dtype),
      tensor_type.shape,
  )


def _flattened_array_shapes_from_type(
    type_spec: Union[computation_types.TensorType, computation_types.StructType]
) -> list[xla_client.Shape]:
  """Returns a `list` of `xla_client.Shape` from a `tff.Type`.

  Args:
    type_spec: A `tff.TensorType` or a structure of `tff.TensorType`s.
  """
  if isinstance(type_spec, computation_types.TensorType):
    return [_array_shape_from_tensor_type(type_spec)]
  elif isinstance(type_spec, computation_types.StructType):
    result = []
    for flattened_type in structure.flatten(type_spec):
      if not isinstance(flattened_type, computation_types.TensorType):
        raise ValueError(
            'Expected all the types in the `tff.StructType` to be'
            f' `tff.TensorType`s, found {flattened_type}.'
        )
      result.append(_array_shape_from_tensor_type(flattened_type))
    return result
  else:
    raise NotImplementedError(f'Unexpected type found: {type(type_spec)}.')


def _create_binary_computation(
    binary_op, type_spec: computation_types.Type
) -> local_computation_factory_base.ComputationProtoAndType:
  """Helper for constructing computations for binary XLA operators.

  The constructed computation is of type `(<T,T> -> T)`, where `T` is the type
  of the operand (`type_spec`).

  Args:
    binary_op: A two-argument callable that constructs a binary xla op from
      tensor parameters (such as `xla_client.ops.Add` or similar).
    type_spec: The type of a single operand.

  Returns:
    An instance of `local_computation_factory_base.ComputationProtoAndType`.

  Raises:
    ValueError: if the arguments are invalid.
  """
  if not type_analysis.is_structure_of_tensors(type_spec):
    raise ValueError(
        'Expected `type_spec` to be a `tff.TensorType` or a structure of'
        f' `tff.TensorType`s, found {type_spec}.'
    )

  tensor_shapes = _flattened_array_shapes_from_type(type_spec)  # pytype: disable=wrong-arg-types
  num_tensors = len(tensor_shapes)
  builder = xla_client.XlaBuilder('comp')
  parameters = [
      xla_client.ops.Parameter(builder, i, shape)
      for i, shape in enumerate(tensor_shapes * 2)
  ]
  result_tensors = []
  for idx in range(num_tensors):
    result_tensors.append(
        binary_op(parameters[idx], parameters[idx + num_tensors])
    )
  xla_client.ops.Tuple(builder, result_tensors)
  xla_computation = builder.build()

  computation_type = computation_types.FunctionType(
      computation_types.StructType([type_spec, type_spec]), type_spec
  )
  computation_proto = xla_serialization.create_xla_tff_computation(
      xla_computation, list(range(2 * num_tensors)), computation_type
  )
  return (computation_proto, computation_type)


class XlaComputationFactory(
    local_computation_factory_base.LocalComputationFactory
):
  """A `LocalComputationFactory` for XLA computations."""

  def create_constant(
      self, value: object, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    if not type_analysis.is_structure_of_tensors(type_spec):
      raise ValueError(
          'Not a tensor or a structure of tensors: {}'.format(str(type_spec))
      )

    builder = xla_client.XlaBuilder('comp')

    def _constant_from_tensor(tensor_type: computation_types.TensorType):
      numpy_value = np.full(
          shape=tensor_type.shape,
          fill_value=value,
          dtype=tensor_type.dtype,
      )
      return xla_client.ops.Constant(builder, numpy_value)

    if isinstance(type_spec, computation_types.TensorType):
      tensors = [_constant_from_tensor(type_spec)]
    else:
      tensors = [_constant_from_tensor(x) for x in structure.flatten(type_spec)]

    xla_client.ops.Tuple(builder, tensors)
    xla_computation = builder.build()

    computation_type = computation_types.FunctionType(None, type_spec)
    computation_proto = xla_serialization.create_xla_tff_computation(
        xla_computation, [], computation_type
    )
    return (computation_proto, computation_type)

  def create_empty_tuple(
      self,
  ) -> local_computation_factory_base.ComputationProtoAndType:
    builder = xla_client.XlaBuilder('comp')
    result = []
    xla_client.ops.Tuple(builder, result)
    xla_computation = builder.build()

    computation_type = computation_types.FunctionType(
        None, computation_types.StructType([])
    )
    computation_proto = xla_serialization.create_xla_tff_computation(
        xla_computation, [], computation_type
    )
    return (computation_proto, computation_type)

  def create_random_uniform(
      self, low: object, high: object, type_spec: computation_types.TensorType
  ) -> local_computation_factory_base.ComputationProtoAndType:
    if not type_analysis.is_structure_of_tensors(type_spec):
      raise ValueError(
          'Not a tensor or a structure of tensors: {}'.format(str(type_spec))
      )

    builder = xla_client.XlaBuilder('comp')
    low = xla_client.ops.Constant(builder, np.array(low, dtype=type_spec.dtype))
    high = xla_client.ops.Constant(
        builder, np.array(high, dtype=type_spec.dtype)
    )
    shape = _array_shape_from_tensor_type(type_spec)
    result = [xla_client.ops.RngUniform(low, high, shape)]
    xla_client.ops.Tuple(builder, result)
    xla_computation = builder.build()

    computation_type = computation_types.FunctionType(None, type_spec)
    computation_proto = xla_serialization.create_xla_tff_computation(
        xla_computation, [], computation_type
    )
    return (computation_proto, computation_type)

  def create_identity(
      self, type_spec: computation_types.Type, **kwargs: object
  ) -> local_computation_factory_base.ComputationProtoAndType:
    if not type_analysis.is_structure_of_tensors(type_spec):
      raise ValueError(
          'Expected `type_spec` to be a `tff.TensorType` or a structure of'
          f' `tff.TensorType`s, found {type_spec}.'
      )

    builder = xla_client.XlaBuilder('comp')
    shapes = _flattened_array_shapes_from_type(type_spec)  # pytype: disable=wrong-arg-types
    parameters = [
        xla_client.ops.Parameter(builder, i, shape)
        for i, shape in enumerate(shapes)
    ]
    result = parameters
    xla_client.ops.Tuple(builder, result)
    xla_computation = builder.build()

    computation_type = computation_types.FunctionType(type_spec, type_spec)
    computation_proto = xla_serialization.create_xla_tff_computation(
        xla_computation, list(range(len(parameters))), computation_type
    )
    return (computation_proto, computation_type)

  def create_add(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_binary_computation(xla_client.ops.Add, type_spec)

  def create_subtract(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_binary_computation(xla_client.ops.Sub, type_spec)

  def create_multiply(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_binary_computation(xla_client.ops.Mul, type_spec)

  def create_divide(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_binary_computation(xla_client.ops.Div, type_spec)

  def create_min(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_binary_computation(xla_client.ops.Min, type_spec)

  def create_max(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_binary_computation(xla_client.ops.Max, type_spec)
