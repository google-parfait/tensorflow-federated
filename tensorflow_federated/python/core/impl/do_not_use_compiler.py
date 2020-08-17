# Copyright 2020, The TensorFlow Federated Authors.
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
"""Library of compiler functions for usage in the native execution context."""

from absl import logging

from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances

# TODO(b/143772942): This module is temporary and can be removed once all the
# dependenices have been removed.


def _do_not_use_transform_to_native_form(comp):
  """Use `tff.backends.native.transform_to_native_form`."""
  proto = computation_impl.ComputationImpl.get_proto(comp)
  computation_building_block = building_blocks.ComputationBuildingBlock.from_proto(
      proto)
  try:
    logging.debug('Compiling TFF computation.')
    call_dominant_form, _ = transformations.transform_to_call_dominant(
        computation_building_block)
    logging.debug('Computation compiled to:')
    logging.debug(call_dominant_form.formatted_representation())
    return computation_wrapper_instances.building_block_to_computation(
        call_dominant_form)
  except ValueError as e:
    logging.debug('Compilation for native runtime failed with error %s', e)
    logging.debug('computation: %s',
                  computation_building_block.compact_representation())
    return comp


def _do_not_use_set_local_execution_context():
  factory = executor_stacks.local_executor_factory()
  context = execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=_do_not_use_transform_to_native_form)
  set_default_context.set_default_context(context)
