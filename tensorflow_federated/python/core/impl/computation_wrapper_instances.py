# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""Definitions of specific computation wrapper instances."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import computation_wrapper
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import function_utils
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import type_utils


def _tf_wrapper_fn(target_fn, parameter_type, unpack, name=None):
  """Wrapper function to plug Tensorflow logic in to TFF framework."""
  del name  # Unused.
  target_fn = function_utils.wrap_as_zero_or_one_arg_callable(
      target_fn, parameter_type, unpack)
  if not type_utils.is_tensorflow_compatible_type(parameter_type):
    raise TypeError('`tf_computation`s can accept only parameter types with '
                    'constituents `SequenceType`, `NamedTupleType` '
                    'and `TensorType`; you have attempted to create one '
                    'with the type {}.'.format(parameter_type))
  ctx_stack = context_stack_impl.context_stack
  comp_pb, extra_type_spec = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      target_fn, parameter_type, ctx_stack)
  return computation_impl.ComputationImpl(comp_pb, ctx_stack, extra_type_spec)


tensorflow_wrapper = computation_wrapper.ComputationWrapper(_tf_wrapper_fn)


def _tf2_wrapper_fn(target_fn, parameter_type, unpack, name=None):
  del name  # Unused.
  comp_pb, extra_type_spec = (
      tensorflow_serialization.serialize_tf2_as_tf_computation(
          target_fn, parameter_type, unpack=unpack))
  return computation_impl.ComputationImpl(comp_pb,
                                          context_stack_impl.context_stack,
                                          extra_type_spec)


tf2_wrapper = computation_wrapper.ComputationWrapper(_tf2_wrapper_fn)


def _federated_computation_wrapper_fn(target_fn,
                                      parameter_type,
                                      unpack,
                                      name=None):
  """Wrapper function to plug orchestration logic in to TFF framework."""
  target_fn = function_utils.wrap_as_zero_or_one_arg_callable(
      target_fn, parameter_type, unpack)
  ctx_stack = context_stack_impl.context_stack
  target_lambda = (
      federated_computation_utils.zero_or_one_arg_fn_to_building_block(
          target_fn,
          'arg' if parameter_type else None,
          parameter_type,
          ctx_stack,
          suggested_name=name))
  return computation_impl.ComputationImpl(target_lambda.proto, ctx_stack)


federated_computation_wrapper = computation_wrapper.ComputationWrapper(
    _federated_computation_wrapper_fn)


def to_computation_impl(building_block):
  """Converts a computation building block to a computation impl."""
  py_typecheck.check_type(building_block,
                          computation_building_blocks.ComputationBuildingBlock)
  return computation_impl.ComputationImpl(building_block.proto,
                                          context_stack_impl.context_stack)
