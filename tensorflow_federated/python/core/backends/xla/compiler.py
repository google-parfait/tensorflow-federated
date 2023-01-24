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
"""A compiler for the xla backend."""

from jax.lib import xla_client
import numpy as np

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.xla_context import xla_serialization


def _xla_tensor_shape_from_tff_tensor_type(tensor_type):
  py_typecheck.check_type(tensor_type, computation_types.TensorType)
  return xla_client.Shape.array_shape(
      xla_client.dtype_to_etype(tensor_type.dtype.as_numpy_dtype),
      tensor_type.shape.dims,
  )


def _xla_tensor_shape_list_from_from_tff_tensor_or_struct_type(type_spec):
  if isinstance(type_spec, computation_types.TensorType):
    return [_xla_tensor_shape_from_tff_tensor_type(type_spec)]
  py_typecheck.check_type(type_spec, computation_types.StructType)
  return [
      _xla_tensor_shape_from_tff_tensor_type(x)
      for x in structure.flatten(type_spec)
  ]


def _create_xla_binary_op_computation(type_spec, xla_binary_op_constructor):
  """Helper for constructing computations that implement binary operators.

  The constructed computation is of type `(<T,T> -> T)`, where `T` is the type
  of the operand (`type_spec`).

  Args:
    type_spec: The type of a single operand.
    xla_binary_op_constructor: A two-argument callable that constructs a binary
      xla op from tensor parameters (such as `xla_client.ops.Add` or similar).

  Returns:
    An instance of `local_computation_factory_base.ComputationProtoAndType`.

  Raises:
    ValueError: if the arguments are invalid.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if not type_analysis.is_structure_of_tensors(type_spec):
    raise ValueError(
        'Not a tensor or a structure of tensors: {}'.format(str(type_spec))
    )

  tensor_shapes = _xla_tensor_shape_list_from_from_tff_tensor_or_struct_type(
      type_spec
  )
  num_tensors = len(tensor_shapes)
  builder = xla_client.XlaBuilder('comp')
  params = [
      xla_client.ops.Parameter(builder, i, shape)
      for i, shape in enumerate(tensor_shapes * 2)
  ]
  result_tensors = []
  for idx in range(num_tensors):
    result_tensors.append(
        xla_binary_op_constructor(params[idx], params[idx + num_tensors])
    )
  xla_client.ops.Tuple(builder, result_tensors)
  xla_computation = builder.build()

  comp_type = computation_types.FunctionType(
      computation_types.StructType([(None, type_spec)] * 2), type_spec
  )
  comp_pb = xla_serialization.create_xla_tff_computation(
      xla_computation, list(range(2 * num_tensors)), comp_type
  )
  return (comp_pb, comp_type)


class XlaComputationFactory(
    local_computation_factory_base.LocalComputationFactory
):
  """An implementation of local computation factory for XLA computations."""

  def __init__(self):
    pass

  def create_constant_from_scalar(
      self, value, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    py_typecheck.check_type(type_spec, computation_types.Type)
    if not type_analysis.is_structure_of_tensors(type_spec):
      raise ValueError(
          'Not a tensor or a structure of tensors: {}'.format(str(type_spec))
      )

    builder = xla_client.XlaBuilder('comp')

    def _constant_from_tensor(tensor_type):
      py_typecheck.check_type(tensor_type, computation_types.TensorType)
      numpy_value = np.full(
          shape=tensor_type.shape.dims,
          fill_value=value,
          dtype=tensor_type.dtype.as_numpy_dtype,
      )
      return xla_client.ops.Constant(builder, numpy_value)

    if isinstance(type_spec, computation_types.TensorType):
      tensors = [_constant_from_tensor(type_spec)]
    else:
      tensors = [_constant_from_tensor(x) for x in structure.flatten(type_spec)]

    # Likewise, results are always returned as a single tuple with results.
    # This is always a flat tuple; the nested TFF structure is defined by the
    # binding.
    xla_client.ops.Tuple(builder, tensors)
    xla_computation = builder.build()

    comp_type = computation_types.FunctionType(None, type_spec)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_computation, [], comp_type
    )
    return (comp_pb, comp_type)

  def create_plus_operator(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_xla_binary_op_computation(type_spec, xla_client.ops.Add)

  def create_multiply_operator(
      self, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
    return _create_xla_binary_op_computation(type_spec, xla_client.ops.Mul)

  def create_scalar_multiply_operator(
      self,
      operand_type: computation_types.Type,
      scalar_type: computation_types.TensorType,
  ) -> local_computation_factory_base.ComputationProtoAndType:
    py_typecheck.check_type(operand_type, computation_types.Type)
    py_typecheck.check_type(scalar_type, computation_types.TensorType)
    if not type_analysis.is_structure_of_tensors(operand_type):
      raise ValueError(
          'Not a tensor or a structure of tensors: {}'.format(str(operand_type))
      )

    operand_shapes = _xla_tensor_shape_list_from_from_tff_tensor_or_struct_type(
        operand_type
    )
    scalar_shape = _xla_tensor_shape_from_tff_tensor_type(scalar_type)
    num_operand_tensors = len(operand_shapes)
    builder = xla_client.XlaBuilder('comp')
    params = [
        xla_client.ops.Parameter(builder, i, shape)
        for i, shape in enumerate(operand_shapes + [scalar_shape])
    ]
    scalar_ref = params[num_operand_tensors]
    result_tensors = []
    for idx in range(num_operand_tensors):
      result_tensors.append(xla_client.ops.Mul(params[idx], scalar_ref))
    xla_client.ops.Tuple(builder, result_tensors)
    xla_computation = builder.build()

    comp_type = computation_types.FunctionType(
        computation_types.StructType(
            [(None, operand_type), (None, scalar_type)]
        ),
        operand_type,
    )
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_computation, list(range(num_operand_tensors + 1)), comp_type
    )
    return (comp_pb, comp_type)
