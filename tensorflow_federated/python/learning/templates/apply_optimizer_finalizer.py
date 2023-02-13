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
"""Abstractions for finalization in learning algorithms."""

import collections
from collections.abc import Callable
from typing import Union

import tensorflow as tf

from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.tensorflow_libs import tensor_utils


def _build_tff_optimizer_initialize_and_next(
    model_weights_type: computation_types.Type,
    optimizer: optimizer_base.Optimizer,
):
  """Creates finalizer initialize and next functions for TFF optimizers."""

  @tensorflow_computation.tf_computation
  def init_fn():
    tensor_specs = type_conversions.type_to_tf_tensor_specs(
        model_weights_type.trainable
    )
    return optimizer.initialize(tensor_specs)

  optimizer_state_type = init_fn.type_signature.result

  @tensorflow_computation.tf_computation(
      optimizer_state_type,
      model_weights_type.trainable,
      model_weights_type.trainable,
  )
  @tf.function
  def next_fn(optimizer_state, trainable_weights, update):
    _, has_non_finite = tensor_utils.zero_all_if_any_non_finite(update)
    measurements = collections.OrderedDict(update_non_finite=has_non_finite)
    if tf.equal(has_non_finite, 1):
      # Do nothing if there are nans/infs in the update.
      return optimizer_state, trainable_weights, measurements
    new_state, new_weights = optimizer.next(
        optimizer_state, trainable_weights, update
    )
    return new_state, new_weights, measurements

  return init_fn, next_fn


def _build_keras_optimizer_initialize_and_next(
    model_weights_type: computation_types.Type,
    optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
):
  """Creates finalizer initialize and next functions for Keras optimizers."""

  @tensorflow_computation.tf_computation
  def init_fn():
    tensor_specs = type_conversions.type_to_tf_tensor_specs(
        model_weights_type.trainable
    )
    model_variables = tf.nest.map_structure(
        lambda s: tf.Variable(initial_value=tf.zeros(s.shape, s.dtype)),
        tensor_specs,
    )
    optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        optimizer_fn, model_variables, disjoint_init_and_next=True
    )
    return optimizer.initialize(tensor_specs)

  optimizer_state_type = init_fn.type_signature.result

  @tensorflow_computation.tf_computation(
      optimizer_state_type,
      model_weights_type.trainable,
      model_weights_type.trainable,
  )
  @tf.function
  def next_fn(optimizer_state, trainable_weights, update):
    _, has_non_finite = tensor_utils.zero_all_if_any_non_finite(update)
    measurements = collections.OrderedDict(update_non_finite=has_non_finite)
    if tf.equal(has_non_finite, 1):
      # Do nothing if there are nans/infs in the update.
      return optimizer_state, trainable_weights, measurements

    with tf.init_scope():
      # Create a structure of variables that the server optimizer can update.
      trainable_variables = tf.nest.map_structure(
          lambda t: tf.Variable(initial_value=tf.zeros(t.shape, t.dtype)),
          trainable_weights,
      )
      optimizer = keras_optimizer.build_or_verify_tff_optimizer(
          optimizer_fn, trainable_variables, disjoint_init_and_next=True
      )

    tf.nest.map_structure(
        lambda a, b: a.assign(b), trainable_variables, trainable_weights
    )
    optimizer_state, updated_weights = optimizer.next(
        optimizer_state, trainable_variables, update
    )
    # Keras optimizers mutate model variables in with the `next` step above, so
    # we skip calling the assignment for those optimizers.
    if not isinstance(optimizer, keras_optimizer.KerasOptimizer):
      tf.nest.map_structure(
          lambda a, b: a.assign(b), trainable_variables, updated_weights
      )
    return optimizer_state, trainable_variables, measurements

  return init_fn, next_fn


def build_apply_optimizer_finalizer(
    optimizer_fn: Union[
        optimizer_base.Optimizer, Callable[[], tf.keras.optimizers.Optimizer]
    ],
    model_weights_type: computation_types.StructType,
):
  """Builds finalizer that applies a step of an optimizer.

  The provided `model_weights_type` must be a non-federated `tff.Type` with the
  `tff.learning.models.ModelWeights` container.

  The 2nd input argument of the created `FinalizerProcess.next` expects a value
  matching `model_weights_type` and its 3rd argument expects value matching
  `model_weights_type.trainable`. The `optimizer` will be applied to the
  trainable model weights only, leaving non_trainable weights unmodified.

  The state of the process is the state of the `optimizer` and the process
  returns empty measurements.

  Args:
    optimizer_fn: A `tff.learning.optimizers.Optimizer` or a no-arg function
      that returns a `tf.keras.optimizers.Optimizer`. This optimizer is used to
      apply client updates to the server model.
    model_weights_type: A non-federated `tff.Type` of the model weights to be
      optimized, which must have a `tff.learning.models.ModelWeights` container.

  Returns:
    A `FinalizerProcess` that applies the `optimizer`.

  Raises:
    TypeError: If `value_type` does not have a
    `tff.learning.model.sModelWeights`
      Python container, or contains a `tff.types.FederatedType`.
  """
  if not isinstance(optimizer_fn, optimizer_base.Optimizer):
    if not callable(optimizer_fn) or not isinstance(
        optimizer_fn(),
        (
            tf.keras.optimizers.Optimizer,
            tf.keras.optimizers.legacy.Optimizer,
            tf.keras.optimizers.experimental.Optimizer,
        ),
    ):
      raise TypeError(
          'The optimizer_fn must be a `tff.learning.optimizers.Optimizer`, or '
          'a no-arg callable returning a `tf.keras.optimizers.Optimizer`.'
      )

  if (
      not model_weights_type.is_struct_with_python()
      or model_weights_type.python_container != model_weights.ModelWeights
      or type_analysis.contains_federated_types(model_weights_type)
  ):
    raise TypeError(
        'Provided value_type must be a tff.types.StructType with its python '
        'container being tff.learning.models.ModelWeights, not containing a '
        f'tff.types.FederatedType, but found: {model_weights_type}'
    )

  if isinstance(optimizer_fn, optimizer_base.Optimizer):
    init_tf, next_tf = _build_tff_optimizer_initialize_and_next(
        model_weights_type, optimizer_fn
    )
  else:
    init_tf, next_tf = _build_keras_optimizer_initialize_and_next(
        model_weights_type, optimizer_fn
    )

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_eval(init_tf, placements.SERVER)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_server(model_weights_type),
      computation_types.at_server(model_weights_type.trainable),
  )
  def next_fn(state, weights, update):
    optimizer_state, new_trainable_weights, measurements = (
        intrinsics.federated_map(next_tf, (state, weights.trainable, update))
    )
    new_weights = intrinsics.federated_zip(
        model_weights.ModelWeights(new_trainable_weights, weights.non_trainable)
    )
    return measured_process.MeasuredProcessOutput(
        optimizer_state, new_weights, measurements
    )

  if isinstance(optimizer_fn, optimizer_base.Optimizer):
    state_type = init_fn.type_signature.result.member

    @tensorflow_computation.tf_computation(state_type)
    def get_hparams_fn(state):
      return optimizer_fn.get_hparams(state)

    hparams_type = get_hparams_fn.type_signature.result

    @tensorflow_computation.tf_computation(state_type, hparams_type)
    def set_hparams_fn(state, hparams):
      return optimizer_fn.set_hparams(state, hparams)

  else:
    get_hparams_fn = None
    set_hparams_fn = None

  return finalizers.FinalizerProcess(
      init_fn,
      next_fn,
      get_hparams_fn=get_hparams_fn,
      set_hparams_fn=set_hparams_fn,
  )
