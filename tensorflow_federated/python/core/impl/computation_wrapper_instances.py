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

from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import computation_wrapper
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import type_utils


def _tf_wrapper_fn(target_fn, parameter_type, name=None):
  """Wrapper function to plug Tensorflow logic in to TFF framework."""
  del name
  if not type_utils.check_tf_comp_whitelisted(parameter_type):
    raise TypeError('`tf_computation`s can accept only parameter types with '
                    'constituents `SequenceType`, `NamedTupleType` '
                    'and `TensorType`; you have attempted to create one '
                    'with the type {}.'.format(parameter_type))
  ctx_stack = context_stack_impl.context_stack
  comp_pb = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      target_fn, parameter_type, ctx_stack)
  return computation_impl.ComputationImpl(comp_pb, ctx_stack)


tensorflow_wrapper = computation_wrapper.ComputationWrapper(_tf_wrapper_fn)


def _federated_computation_wrapper_fn(target_fn, parameter_type, name=None):
  """Wrapper function to plug orchestration logic in to TFF framework."""
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
