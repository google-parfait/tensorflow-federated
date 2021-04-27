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

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances


def transform_to_native_form(
    comp: computation_base.Computation,
    transform_math_to_tf: bool = False) -> computation_base.Computation:
  """Compiles a computation for execution in the TFF native runtime.

  This function transforms the proto underlying `comp` by transforming it
  to call-dominant form (see `tff.framework.transform_to_call_dominant` for
  definition).

  Args:
    comp: Instance of `computation_base.Computation` to compile.
    transform_math_to_tf: Whether to additional transform math to TensorFlow
      graphs. Necessary if running on a execution state without
      ReferenceResolvingExecutors underneath FederatingExecutors.

  Returns:
    A new `computation_base.Computation` representing the compiled version of
    `comp`.
  """
  proto = computation_impl.ComputationImpl.get_proto(comp)
  computation_building_block = building_blocks.ComputationBuildingBlock.from_proto(
      proto)
  try:
    logging.debug('Compiling TFF computation to CDF.')
    call_dominant_form, _ = transformations.transform_to_call_dominant(
        computation_building_block)
    logging.debug('Computation compiled to:')
    logging.debug(call_dominant_form.formatted_representation())
    if transform_math_to_tf:
      logging.debug('Compiling local computations to TensorFlow.')
      call_dominant_form, _ = transformations.compile_local_computation_to_tensorflow(
          call_dominant_form)
      logging.debug('Computation compiled to:')
      logging.debug(call_dominant_form.formatted_representation())
    call_dominant_form, _ = tree_transformations.transform_tf_call_ops_to_disable_grappler(
        call_dominant_form)
    return computation_wrapper_instances.building_block_to_computation(
        call_dominant_form)
  except ValueError as e:
    logging.debug('Compilation for native runtime failed with error %s', e)
    logging.debug('computation: %s',
                  computation_building_block.compact_representation())
    return comp
