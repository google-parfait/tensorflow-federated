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
"""A native form compiler for the native backend."""

from typing import Optional

from absl import logging
import federated_language
import tensorflow as tf

from tensorflow_federated.python.core.backends.mapreduce import compiler
from tensorflow_federated.python.core.environments.tensorflow_backend import compiled_computation_transformations
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_tree_transformations
from tensorflow_federated.python.core.impl.compiler import transformations


def transform_to_native_form(
    comp: federated_language.framework.ConcreteComputation,
    transform_math_to_tf: bool = False,
    grappler_config: Optional[tf.compat.v1.ConfigProto] = None,
) -> federated_language.framework.ConcreteComputation:
  """Compiles a computation for execution in the TFF native runtime.

  This function transforms the proto underlying `comp` by transforming it
  to call-dominant form (see `tff.framework.to_call_dominant` for
  definition).

  Args:
    comp: Instance of `federated_language.framework.ConcreteComputation` to
      compile.
    transform_math_to_tf: Whether to additional transform math to TensorFlow
      graphs. Necessary if running on a execution state without
      ReferenceResolvingExecutors underneath FederatingExecutors.
    grappler_config: Configuration for Grappler optimizations to perform on the
      TensorFlow computations. If `None`, Grappler will not be run and no
      optimizations wil be applied.

  Returns:
    A new `federated_language.framework.ConcreteComputation` representing the
    compiled
      version of `comp`.
  """
  proto = federated_language.framework.ConcreteComputation.get_proto(comp)
  computation_building_block = (
      federated_language.framework.ComputationBuildingBlock.from_proto(proto)
  )
  try:
    logging.debug('Compiling TFF computation to CDF.')
    with federated_language.framework.span(
        'transform_to_native_form', 'to_call_dominant', span=True
    ):
      call_dominant_form = transformations.to_call_dominant(
          computation_building_block
      )
    logging.debug('Computation compiled to:')
    logging.debug(call_dominant_form.formatted_representation())
    if transform_math_to_tf:
      logging.debug('Compiling local computations to TensorFlow.')
      with federated_language.framework.span(
          'transform_to_native_form',
          'compile_local_subcomputations_to_tensorflow',
          span=True,
      ):
        call_dominant_form = (
            compiler.compile_local_subcomputations_to_tensorflow(
                call_dominant_form
            )
        )
      logging.debug('Computation compiled to:')
      logging.debug(call_dominant_form.formatted_representation())
    if grappler_config is not None:
      with federated_language.framework.span(
          'transform_to_native_form', 'optimize_tf_graphs', span=True
      ):
        call_dominant_form, _ = (
            compiled_computation_transformations.optimize_tensorflow_graphs(
                call_dominant_form, grappler_config
            )
        )
    with federated_language.framework.span(
        'transform_to_native_form',
        'transform_tf_call_ops_disable_grappler',
        span=True,
    ):
      disabled_grapler_form, _ = (
          compiled_computation_transformations.transform_tf_call_ops_to_disable_grappler(
              call_dominant_form
          )
      )
    with federated_language.framework.span(
        'transform_to_native_form', 'transform_tf_add_ids', span=True
    ):
      form_with_ids, _ = (
          compiled_computation_transformations.transform_tf_add_ids(
              disabled_grapler_form
          )
      )
    return federated_language.framework.ConcreteComputation(
        computation_proto=form_with_ids.proto,
        context_stack=federated_language.framework.global_context_stack,
    )
  except ValueError as e:
    logging.debug('Compilation for native runtime failed with error %s', e)
    logging.debug(
        'computation: %s', computation_building_block.compact_representation()
    )
    return comp


def desugar_and_transform_to_native(comp):
  """Transform to native form and replace intrinsics with TensorFlow."""
  # Turn on static grappler. The function inlining is critical for GPU support,
  # otherwise variant placeholders that received datasets will be placed on GPUs
  # which don't have kernels for datastes, causing TF to error.
  grappler_config = tf.compat.v1.ConfigProto()
  aggressive = grappler_config.graph_options.rewrite_options.AGGRESSIVE
  rewrite_options = grappler_config.graph_options.rewrite_options
  rewrite_options.memory_optimization = aggressive
  rewrite_options.constant_folding = aggressive
  rewrite_options.arithmetic_optimization = aggressive
  rewrite_options.loop_optimization = aggressive
  rewrite_options.function_optimization = aggressive

  intrinsics_desugared_bb, _ = (
      tensorflow_tree_transformations.replace_intrinsics_with_bodies(
          comp.to_building_block()
      )
  )
  # Desugaring intrinsics injects TF computations; transforming to native form
  # adds TF cache IDs to them. It is crucial that these transformations execute
  # in this order.
  native_form = transform_to_native_form(
      federated_language.framework.ConcreteComputation(
          computation_proto=intrinsics_desugared_bb.proto,
          context_stack=federated_language.framework.global_context_stack,
      ),
      grappler_config=grappler_config,
  )
  return native_form
