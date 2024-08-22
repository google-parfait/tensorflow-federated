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
from typing import Any, Optional, Union

import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import tensor_utils
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import finalizers

_MeasurementsType = collections.OrderedDict[str, tf.Tensor]


def reject_non_finite_update(
    state: Any, update: Any
) -> tuple[tf.Tensor, _MeasurementsType]:
  """Rejects the update if any non-finite value is in the update.

  This is the default `should_reject_update` function used in
  `build_apply_optimizer_finalizer`.

  Args:
    state: Unused optimzier state.
    update: The update to be applied to the model's weights by the optimizer.

  Returns:
    A tuple of:
      - should_reject (bool tensor): True if the update should be rejected,
      False otherwise.
      - measurements (OrderedDict): A dict with a single key
      (`update_non_finite`) an an integer tensor of whether the update was
      rejected.
  """
  del state
  _, has_non_finite = tensor_utils.zero_all_if_any_non_finite(update)
  measurements = collections.OrderedDict(update_non_finite=has_non_finite)
  return tf.equal(has_non_finite, 1), measurements


def _build_tff_optimizer_initialize_and_next(
    model_weights_type: computation_types.Type,
    optimizer: optimizer_base.Optimizer,
    should_reject_update: Callable[
        [Any, Any], tuple[Union[bool, tf.Tensor], Optional[_MeasurementsType]]
    ],
):
  """Creates finalizer initialize and next functions for TFF optimizers."""

  @tensorflow_computation.tf_computation
  def init_fn():
    tensor_specs = type_conversions.type_to_tf_tensor_specs(
        model_weights_type.trainable  # pytype: disable=attribute-error
    )
    return optimizer.initialize(tensor_specs)

  optimizer_state_type = init_fn.type_signature.result

  @tensorflow_computation.tf_computation(
      optimizer_state_type,
      model_weights_type.trainable,  # pytype: disable=attribute-error
      model_weights_type.trainable,  # pytype: disable=attribute-error
  )
  @tf.function
  def next_fn(optimizer_state, trainable_weights, update):
    new_state, new_weights = optimizer.next(
        optimizer_state, trainable_weights, update
    )
    should_reject, measurements = should_reject_update(new_state, update)
    if should_reject:
      # Do nothing if the update should be rejected.
      return optimizer_state, trainable_weights, measurements
    return new_state, new_weights, measurements

  return init_fn, next_fn


def build_apply_optimizer_finalizer(
    optimizer_fn: optimizer_base.Optimizer,
    model_weights_type: computation_types.StructType,
    should_reject_update: Callable[
        [Any, Any], tuple[Union[bool, tf.Tensor], Optional[_MeasurementsType]]
    ] = reject_non_finite_update,
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
    optimizer_fn: A `tff.learning.optimizers.Optimizer`. This optimizer is used
      to apply client updates to the server model.
    model_weights_type: A non-federated `tff.Type` of the model weights to be
      optimized, which must have a `tff.learning.models.ModelWeights` container.
    should_reject_update: A callable that takes the optimizer state and the
      model weights update, and returns a boolean or a bool tensor indicating if
      the model weights update should be rejected and an OrderedDict of
      measurements. If the model weights update is reject, we will fall back to
      the previous round's optimizer state and model weight, this is a no-op
      otherwise. The default function is `reject_non_finite_update` which checks
      if there is any non-finite value in the model update and returns the
      results.

  Returns:
    A `FinalizerProcess` that applies the `optimizer`.

  Raises:
    TypeError: If `value_type` does not have a
    `tff.learning.model.sModelWeights`
      Python container, or contains a `tff.types.FederatedType`.
  """
  if (
      not isinstance(model_weights_type, computation_types.StructWithPythonType)
      or model_weights_type.python_container != model_weights.ModelWeights
      or type_analysis.contains_federated_types(model_weights_type)
  ):
    raise TypeError(
        'Provided value_type must be a tff.types.StructType with its python '
        'container being tff.learning.models.ModelWeights, not containing a '
        f'tff.types.FederatedType, but found: {model_weights_type}'
    )

  init_tf, next_tf = _build_tff_optimizer_initialize_and_next(
      model_weights_type, optimizer_fn, should_reject_update
  )

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_eval(init_tf, placements.SERVER)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(model_weights_type, placements.SERVER),
      computation_types.FederatedType(
          model_weights_type.trainable, placements.SERVER
      ),
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
