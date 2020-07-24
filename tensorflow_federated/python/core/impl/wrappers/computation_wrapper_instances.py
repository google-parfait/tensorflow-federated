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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.utils import function_utils
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper

# The documentation of the arguments and return values from the wrapper_fns
# is quite detailed and can be found in `computation_wrapper.py` along with
# the definitions of `_wrap` and `ComputationWrapper`. In order to avoid having
# to repeat those descriptions (and make any relevant changes in four separate
# places) the documentation here simply forwards readers over.
#
# pylint:disable=g-doc-args,g-doc-return-or-yield


def _tf_wrapper_fn(target_fn, parameter_type, unpack, name=None):
  """Wrapper function to plug Tensorflow logic into the TFF framework.

  This function is passed through `computation_wrapper.ComputationWrapper`.
  Documentation its arguments can be found inside the definition of that class.
  """
  del name  # Unused.
  target_fn = function_utils.wrap_as_zero_or_one_arg_callable(
      target_fn, parameter_type, unpack)
  if not type_analysis.is_tensorflow_compatible_type(parameter_type):
    raise TypeError('`tf_computation`s can accept only parameter types with '
                    'constituents `SequenceType`, `StructType` '
                    'and `TensorType`; you have attempted to create one '
                    'with the type {}.'.format(parameter_type))
  ctx_stack = context_stack_impl.context_stack
  comp_pb, extra_type_spec = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      target_fn, parameter_type, ctx_stack)
  return computation_impl.ComputationImpl(comp_pb, ctx_stack, extra_type_spec)


tensorflow_wrapper = computation_wrapper.ComputationWrapper(_tf_wrapper_fn)


def _tf2_wrapper_fn(target_fn, parameter_type, unpack, name=None):
  """Wrapper function to plug Tensorflow 2.0 logic into the TFF framework.

  This function is passed through `computation_wrapper.ComputationWrapper`.
  Documentation its arguments can be found inside the definition of that class.
  """
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
  """Wrapper function to plug orchestration logic into the TFF framework.

  This function is passed through `computation_wrapper.ComputationWrapper`.
  Documentation its arguments can be found inside the definition of that class.
  """
  target_fn = function_utils.wrap_as_zero_or_one_arg_callable(
      target_fn, parameter_type, unpack)
  ctx_stack = context_stack_impl.context_stack
  target_lambda, extra_type_spec = (
      federated_computation_utils.zero_or_one_arg_fn_to_building_block(
          target_fn,
          'arg' if parameter_type else None,
          parameter_type,
          ctx_stack,
          suggested_name=name))
  return computation_impl.ComputationImpl(target_lambda.proto, ctx_stack,
                                          extra_type_spec)


federated_computation_wrapper = computation_wrapper.ComputationWrapper(
    _federated_computation_wrapper_fn)

# pylint:enable=g-doc-args,g-doc-return-or-yield


def building_block_to_computation(building_block):
  """Converts a computation building block to a computation impl."""
  py_typecheck.check_type(building_block,
                          building_blocks.ComputationBuildingBlock)
  return computation_impl.ComputationImpl(building_block.proto,
                                          context_stack_impl.context_stack)
