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
"""Utility functions for writing executors."""

import asyncio
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization


# TODO(b/140752097): Factor out more commonalities between executorts to place
# in this helper file. The helpers that are currently here may not be the right
# ones. Exploit commonalities with transformations.
# TODO(b/134543154): Add a dedicated test for this library (not a priority since
# it's already transitively getting covered by existing executor tests).
async def delegate_entirely_to_executor(arg, arg_type, executor):
  """Delegates `arg` in its entirety to the target executor.

  The supported types of `arg` and the manner in which they are handled:

  * For instances of `pb.Computation`, calls `create_value()`.
  * For instances of `structure.Struct`, calls `create_struct()`.
  * Otherwise, must be `executor_value_base.ExecutorValue`, and assumed to
    already be owned by the target executor.

  Args:
    arg: The object to delegate to the target executor.
    arg_type: The type of this object.
    executor: The target executor to use.

  Returns:
    An instance of `executor_value_base.ExecutorValue` that represents the
    result of delegation.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(executor, executor_base.Executor)
  py_typecheck.check_type(arg_type, computation_types.Type)
  if isinstance(arg, pb.Computation):
    return await executor.create_value(arg, arg_type)
  elif isinstance(arg, structure.Struct):
    vals = await asyncio.gather(*[
        delegate_entirely_to_executor(value, type_spec, executor)
        for value, type_spec in zip(arg, arg_type)
    ])
    return await executor.create_struct(
        structure.Struct(
            zip((k for k, _ in structure.iter_elements(arg_type)), vals)))
  else:
    py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
    return arg


def parse_federated_aggregate_argument_types(type_spec):
  """Verifies and parses `type_spec` into constituents.

  Args:
    type_spec: An instance of `computation_types.StructType`.

  Returns:
    A tuple of (value_type, zero_type, accumulate_type, merge_type, report_type)
    for the 5 type constituents.
  """
  py_typecheck.check_type(type_spec, computation_types.StructType)
  py_typecheck.check_len(type_spec, 5)
  value_type = type_spec[0]
  py_typecheck.check_type(value_type, computation_types.FederatedType)
  item_type = value_type.member
  zero_type = type_spec[1]
  accumulate_type = type_spec[2]
  accumulate_type.check_equivalent_to(
      type_factory.reduction_op(zero_type, item_type))
  merge_type = type_spec[3]
  merge_type.check_equivalent_to(type_factory.binary_op(zero_type))
  report_type = type_spec[4]
  py_typecheck.check_type(report_type, computation_types.FunctionType)
  report_type.parameter.check_equivalent_to(zero_type)
  return value_type, zero_type, accumulate_type, merge_type, report_type


async def embed_tf_scalar_constant(executor, type_spec, value):
  """Embeds a constant `val` of TFF type `type_spec` in `executor`.

  Args:
    executor: An instance of `tff.framework.Executor`.
    type_spec: An instance of `tff.Type`.
    value: A scalar value.

  Returns:
    An instance of `tff.framework.ExecutorValue` containing an embedded value.
  """
  py_typecheck.check_type(executor, executor_base.Executor)
  proto, type_signature = tensorflow_computation_factory.create_constant(
      value, type_spec)
  result = await executor.create_value(proto, type_signature)
  return await executor.create_call(result)


async def embed_tf_binary_operator(executor, type_spec, op):
  """Embeds a binary operator `op` on `type_spec`-typed values in `executor`.

  Args:
    executor: An instance of `tff.framework.Executor`.
    type_spec: An instance of `tff.Type` of the type of values that the binary
      operator accepts as input and returns as output.
    op: An operator function (such as `tf.add` or `tf.multiply`) to apply to the
      tensor-level constituents of the values, pointwise.

  Returns:
    An instance of `tff.framework.ExecutorValue` representing the operator in
    a form embedded into the executor.
  """
  proto, type_signature = tensorflow_computation_factory.create_binary_operator(
      op, type_spec)
  return await executor.create_value(proto, type_signature)


def create_intrinsic_comp(intrinsic_def, type_spec):
  """Creates an intrinsic `pb.Computation`.

  Args:
    intrinsic_def: An instance of `intrinsic_defs.IntrinsicDef`.
    type_spec: The concrete type of the intrinsic (`computation_types.Type`).

  Returns:
    An instance of `pb.Computation` that represents the intrinsics.
  """
  py_typecheck.check_type(intrinsic_def, intrinsic_defs.IntrinsicDef)
  py_typecheck.check_type(type_spec, computation_types.Type)
  return pb.Computation(
      type=type_serialization.serialize_type(type_spec),
      intrinsic=pb.Intrinsic(uri=intrinsic_def.uri))


async def compute_intrinsic_federated_broadcast(
    executor: executor_base.Executor, arg: executor_value_base.ExecutorValue
) -> executor_value_base.ExecutorValue:
  """Computes a federated broadcast on the given `executor`.

  Args:
    executor: The executor to use.
    arg: The value to broadcast. Expected to be embedded in the `executor` and
      have federated type placed at `tff.SERVER` with all_equal of `True`.

  Returns:
    The result embedded in `executor`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(executor, executor_base.Executor)
  py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
  type_analysis.check_federated_type(
      arg.type_signature, placement=placement_literals.SERVER, all_equal=True)
  value = await arg.compute()
  type_signature = computation_types.FederatedType(
      arg.type_signature.member, placement_literals.CLIENTS, all_equal=True)
  return await executor.create_value(value, type_signature)


async def compute_intrinsic_federated_value(
    executor: executor_base.Executor, arg: executor_value_base.ExecutorValue,
    placement: placement_literals.PlacementLiteral
) -> executor_value_base.ExecutorValue:
  """Computes a federated value on the given `executor`.

  Args:
    executor: The executor to use.
    arg: The value to place.
    placement: The new placement of the value.

  Returns:
    The result embedded in `executor`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(executor, executor_base.Executor)
  py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
  py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
  value = await arg.compute()
  type_signature = computation_types.FederatedType(
      arg.type_signature, placement, all_equal=True)
  return await executor.create_value(value, type_signature)


async def compute_intrinsic_federated_weighted_mean(
    executor: executor_base.Executor, arg: executor_value_base.ExecutorValue
) -> executor_value_base.ExecutorValue:
  """Computes a federated weighted mean on the given `executor`.

  Args:
    executor: The executor to use.
    arg: The argument to embedded in `executor`.

  Returns:
    The result embedded in `executor`.
  """
  type_analysis.check_valid_federated_weighted_mean_argument_tuple_type(
      arg.type_signature)
  zip1_type = computation_types.FunctionType(
      computation_types.StructType([
          type_factory.at_clients(arg.type_signature[0].member),
          type_factory.at_clients(arg.type_signature[1].member)
      ]),
      type_factory.at_clients(
          computation_types.StructType(
              [arg.type_signature[0].member, arg.type_signature[1].member])))

  multiply_blk = building_block_factory.create_tensorflow_binary_operator_with_upcast(
      zip1_type.result.member, tf.multiply)

  map_type = computation_types.FunctionType(
      computation_types.StructType(
          [multiply_blk.type_signature, zip1_type.result]),
      type_factory.at_clients(multiply_blk.type_signature.result))

  sum1_type = computation_types.FunctionType(
      type_factory.at_clients(map_type.result.member),
      type_factory.at_server(map_type.result.member))

  sum2_type = computation_types.FunctionType(
      type_factory.at_clients(arg.type_signature[1].member),
      type_factory.at_server(arg.type_signature[1].member))

  zip2_type = computation_types.FunctionType(
      computation_types.StructType([sum1_type.result, sum2_type.result]),
      type_factory.at_server(
          computation_types.StructType(
              [sum1_type.result.member, sum2_type.result.member])))

  divide_blk = building_block_factory.create_tensorflow_binary_operator_with_upcast(
      zip2_type.result.member, tf.divide)

  async def _compute_multiply_fn():
    return await executor.create_value(multiply_blk.proto,
                                       multiply_blk.type_signature)

  async def _compute_multiply_arg():
    zip1_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS,
                                      zip1_type)
    zip_fn = await executor.create_value(zip1_comp, zip1_type)
    return await executor.create_call(zip_fn, arg)

  async def _compute_product_fn():
    map_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_MAP, map_type)
    return await executor.create_value(map_comp, map_type)

  async def _compute_product_arg():
    multiply_fn, multiply_arg = await asyncio.gather(_compute_multiply_fn(),
                                                     _compute_multiply_arg())
    return await executor.create_struct((multiply_fn, multiply_arg))

  async def _compute_products():
    product_fn, product_arg = await asyncio.gather(_compute_product_fn(),
                                                   _compute_product_arg())
    return await executor.create_call(product_fn, product_arg)

  async def _compute_total_weight():
    sum2_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_SUM, sum2_type)
    sum2_fn, sum2_arg = await asyncio.gather(
        executor.create_value(sum2_comp, sum2_type),
        executor.create_selection(arg, index=1))
    return await executor.create_call(sum2_fn, sum2_arg)

  async def _compute_sum_of_products():
    sum1_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_SUM, sum1_type)
    sum1_fn, products = await asyncio.gather(
        executor.create_value(sum1_comp, sum1_type), _compute_products())
    return await executor.create_call(sum1_fn, products)

  async def _compute_zip2_fn():
    zip2_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_ZIP_AT_SERVER,
                                      zip2_type)
    return await executor.create_value(zip2_comp, zip2_type)

  async def _compute_zip2_arg():
    sum_of_products, total_weight = await asyncio.gather(
        _compute_sum_of_products(), _compute_total_weight())
    return await executor.create_struct([sum_of_products, total_weight])

  async def _compute_divide_fn():
    return await executor.create_value(divide_blk.proto,
                                       divide_blk.type_signature)

  async def _compute_divide_arg():
    zip_fn, zip_arg = await asyncio.gather(_compute_zip2_fn(),
                                           _compute_zip2_arg())
    return await executor.create_call(zip_fn, zip_arg)

  async def _compute_apply_fn():
    apply_type = computation_types.FunctionType(
        computation_types.StructType(
            [divide_blk.type_signature, zip2_type.result]),
        type_factory.at_server(divide_blk.type_signature.result))
    apply_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_APPLY,
                                       apply_type)
    return await executor.create_value(apply_comp, apply_type)

  async def _compute_apply_arg():
    divide_fn, divide_arg = await asyncio.gather(_compute_divide_fn(),
                                                 _compute_divide_arg())
    return await executor.create_struct([divide_fn, divide_arg])

  async def _compute_divided():
    apply_fn, apply_arg = await asyncio.gather(_compute_apply_fn(),
                                               _compute_apply_arg())
    return await executor.create_call(apply_fn, apply_arg)

  return await _compute_divided()
