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
"""Utility functions for writing executors."""

import asyncio

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_value_base
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization


# TODO(b/140752097): Factor out more commonalities between executorts to place
# in this helper file. The helpers that are currently here may not be the right
# ones. Exploit commonalities with transformations.
# TODO(b/134543154): Add a dedicated test for this library (not a priority since
# it's already transitively getting covered by existing executor tests).
async def delegate_entirely_to_executor(arg, arg_type, executor):
  """Delegates `arg` in its entirety to the target executor.

  The supported types of `arg` and the manner in which they are handled:

  * For instances of `pb.Computation`, calls `create_value()`.
  * For instances of `anonymous_tuple.AnonymousTuple`, calls `create_tuple()`.
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
  elif isinstance(arg, anonymous_tuple.AnonymousTuple):
    v_elem = anonymous_tuple.to_elements(arg)
    t_elem = anonymous_tuple.to_elements(arg_type)
    vals = await asyncio.gather(*[
        delegate_entirely_to_executor(v, t, executor)
        for (_, v), (_, t) in zip(v_elem, t_elem)
    ])
    return await executor.create_tuple(
        anonymous_tuple.AnonymousTuple(list(zip([k for k, _ in t_elem], vals))))
  else:
    py_typecheck.check_type(arg, executor_value_base.ExecutorValue)
    return arg


def parse_federated_aggregate_argument_types(type_spec):
  """Verifies and parses `type_spec` into constituents.

  Args:
    type_spec: An instance of `computation_types.NamedTupleType`.

  Returns:
    A tuple of (value_type, zero_type, accumulate_type, merge_type, report_type)
    for the 5 type constituents.
  """
  py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
  py_typecheck.check_len(type_spec, 5)
  value_type = type_spec[0]
  py_typecheck.check_type(value_type, computation_types.FederatedType)
  item_type = value_type.member
  zero_type = type_spec[1]
  accumulate_type = type_spec[2]
  type_utils.check_equivalent_types(
      accumulate_type, type_factory.reduction_op(zero_type, item_type))
  merge_type = type_spec[3]
  type_utils.check_equivalent_types(merge_type,
                                    type_factory.binary_op(zero_type))
  report_type = type_spec[4]
  py_typecheck.check_type(report_type, computation_types.FunctionType)
  type_utils.check_equivalent_types(report_type.parameter, zero_type)
  return value_type, zero_type, accumulate_type, merge_type, report_type


async def embed_tf_scalar_constant(executor, type_spec, val):
  """Embeds a constant `val` of TFF type `type_spec` in `executor`.

  Args:
    executor: An instance of `tff.framework.Executor`.
    type_spec: An instance of `tff.Type`.
    val: A scalar value.

  Returns:
    An instance of `tff.framework.ExecutorValue` containing an embedded value.
  """
  py_typecheck.check_type(executor, executor_base.Executor)
  fn_building_block = (
      building_block_factory.create_tensorflow_constant(type_spec, val))
  embedded_val = await executor.create_call(await executor.create_value(
      fn_building_block.function.proto,
      fn_building_block.function.type_signature))
  type_utils.check_equivalent_types(embedded_val.type_signature, type_spec)
  return embedded_val


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
  # TODO(b/134543154): There is an opportunity here to import something more
  # in line with the usage (no building block wrapping, etc.)
  fn_building_block = (
      building_block_factory.create_tensorflow_binary_operator(type_spec, op))
  embedded_val = await executor.create_value(fn_building_block.proto,
                                             fn_building_block.type_signature)
  type_utils.check_equivalent_types(embedded_val.type_signature,
                                    type_factory.binary_op(type_spec))
  return embedded_val


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


async def compute_federated_weighted_mean(executor, arg):
  """Computes a federated weighted using simpler intrinsic coroutines.

  Args:
    executor: The executor to use.
    arg: The argument tuple value, which must be embedded in `executor`.

  Returns:
    The result embedded in `executor`.
  """
  type_utils.check_valid_federated_weighted_mean_argument_tuple_type(
      arg.type_signature)
  zip1_type = computation_types.FunctionType(
      computation_types.NamedTupleType([
          type_factory.at_clients(arg.type_signature[0].member),
          type_factory.at_clients(arg.type_signature[1].member)
      ]),
      type_factory.at_clients(
          computation_types.NamedTupleType(
              [arg.type_signature[0].member, arg.type_signature[1].member])))
  zip1_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS,
                                    zip1_type)
  zipped_arg = await executor.create_call(
      await executor.create_value(zip1_comp, zip1_type), arg)

  # TODO(b/134543154): Replace with something that produces a section of
  # plain TensorFlow code instead of constructing a lambda (so that this
  # can be executed directly on top of a plain TensorFlow-based executor).
  multiply_blk = building_block_factory.create_binary_operator_with_upcast(
      zipped_arg.type_signature.member, tf.multiply)

  map_type = computation_types.FunctionType(
      computation_types.NamedTupleType(
          [multiply_blk.type_signature, zipped_arg.type_signature]),
      type_factory.at_clients(multiply_blk.type_signature.result))
  map_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_MAP, map_type)
  products = await executor.create_call(
      await executor.create_value(map_comp, map_type), await
      executor.create_tuple([
          await executor.create_value(multiply_blk.proto,
                                      multiply_blk.type_signature), zipped_arg
      ]))
  sum1_type = computation_types.FunctionType(
      type_factory.at_clients(products.type_signature.member),
      type_factory.at_server(products.type_signature.member))
  sum1_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_SUM, sum1_type)
  sum_of_products = await executor.create_call(
      await executor.create_value(sum1_comp, sum1_type), products)
  sum2_type = computation_types.FunctionType(
      type_factory.at_clients(arg.type_signature[1].member),
      type_factory.at_server(arg.type_signature[1].member))
  sum2_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_SUM, sum2_type)
  total_weight = await executor.create_call(*(await asyncio.gather(
      executor.create_value(sum2_comp, sum2_type),
      executor.create_selection(arg, index=1))))
  zip2_type = computation_types.FunctionType(
      computation_types.NamedTupleType(
          [sum_of_products.type_signature, total_weight.type_signature]),
      type_factory.at_server(
          computation_types.NamedTupleType([
              sum_of_products.type_signature.member,
              total_weight.type_signature.member
          ])))
  zip2_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_ZIP_AT_SERVER,
                                    zip2_type)
  divide_arg = await executor.create_call(*(await asyncio.gather(
      executor.create_value(zip2_comp, zip2_type),
      executor.create_tuple([sum_of_products, total_weight]))))
  divide_blk = building_block_factory.create_binary_operator_with_upcast(
      divide_arg.type_signature.member, tf.divide)
  apply_type = computation_types.FunctionType(
      computation_types.NamedTupleType(
          [divide_blk.type_signature, divide_arg.type_signature]),
      type_factory.at_server(divide_blk.type_signature.result))
  apply_comp = create_intrinsic_comp(intrinsic_defs.FEDERATED_APPLY, apply_type)
  return await executor.create_call(*(await asyncio.gather(
      executor.create_value(apply_comp, apply_type),
      executor.create_tuple([
          await executor.create_value(divide_blk.proto,
                                      divide_blk.type_signature), divide_arg
      ]))))
