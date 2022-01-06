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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Abstractions for finalization in learning algorithms."""

from typing import Callable, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base


class FinalizerResultTypeError(TypeError):
  """`TypeError` for finalizer not updating model weights as expected."""


class FinalizerProcess(measured_process.MeasuredProcess):
  """A stateful process for finalization of a round of training.

  A `FinalizerProcess` is a `tff.templates.MeasuredProcess` that formalizes the
  type signature of `initialize_fn` and `next_fn` for the work performed by
  server in a learning process after aggregating model updates from clients.

  The `initialize_fn` and `next_fn` must have the following type signatures:
  ```
    - initialize_fn: ( -> S@SERVER)
    - next_fn: (<S@SERVER,
                 A@SERVER,
                 B@SERVER>
                ->
                <state=S@SERVER,
                 result=A@SERVER,
                 measurements=M@SERVER>)
  ```

  `FinalizerProcess` requires `next_fn` with a second and a third input
  argument, which are both placed at `SERVER`. The second type `A` represents
  the current server parameters to be updated, while the third type `B`
  represents an update to the parameter `A`, and often matches the type of `B`.

  The `result` field of the returned `tff.templates.MeasuredProcessOutput` must
  be placed at `SERVER`, be of type matching that of second input argument (`B`)
  and represents the updated ("finalized") model parameters.
  """

  def __init__(self, initialize_fn, next_fn):
    super().__init__(initialize_fn, next_fn, next_is_multi_arg=True)

    if not initialize_fn.type_signature.result.is_federated():
      raise errors.TemplateNotFederatedError(
          f'Provided `initialize_fn` must return a federated type, but found '
          f'return type:\n{initialize_fn.type_signature.result}\nTip: If you '
          f'see a collection of federated types, try wrapping the returned '
          f'value in `tff.federated_zip` before returning.')
    next_types = (
        structure.flatten(next_fn.type_signature.parameter) +
        structure.flatten(next_fn.type_signature.result))
    if not all([t.is_federated() for t in next_types]):
      offending_types = '\n- '.join(
          [t for t in next_types if not t.is_federated()])
      raise errors.TemplateNotFederatedError(
          f'Provided `next_fn` must be a *federated* computation, that is, '
          f'operate on `tff.FederatedType`s, but found\n'
          f'next_fn with type signature:\n{next_fn.type_signature}\n'
          f'The non-federated types are:\n {offending_types}.')

    if initialize_fn.type_signature.result.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The state controlled by an `FinalizerProcess` must be placed at '
          f'the SERVER, but found type: {initialize_fn.type_signature.result}.')
    # Note that state of next_fn being placed at SERVER is now ensured by the
    # assertions in base class which would otherwise raise
    # TemplateStateNotAssignableError.

    next_fn_param = next_fn.type_signature.parameter
    if not next_fn_param.is_struct():
      raise errors.TemplateNextFnNumArgsError(
          f'The `next_fn` must have exactly two input arguments, but found '
          f'the following input type which is not a Struct: {next_fn_param}.')
    if len(next_fn_param) != 3:
      next_param_str = '\n- '.join([str(t) for t in next_fn_param])
      raise errors.TemplateNextFnNumArgsError(
          f'The `next_fn` must have exactly three input arguments, but found '
          f'{len(next_fn_param)} input arguments:\n{next_param_str}')
    model_weights_param = next_fn_param[1]
    update_from_clients_param = next_fn_param[2]
    if model_weights_param.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The second input argument of `next_fn` must be placed at SERVER '
          f'but found {model_weights_param}.')
    if update_from_clients_param.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The third input argument of `next_fn` must be placed at SERVER '
          f'but found {update_from_clients_param}.')

    next_fn_result = next_fn.type_signature.result
    if next_fn_result.result.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The "result" attribute of the return type of `next_fn` must be '
          f'placed at SERVER, but found {next_fn_result.result}.')
    if not model_weights_param.member.is_assignable_from(
        next_fn_result.result.member):
      raise FinalizerResultTypeError(
          f'The second input argument of `next_fn` must match the "result" '
          f'attribute of the return type of `next_fn`. Found:\n'
          f'Second input argument: {next_fn_param[1].member}\n'
          f'Result attribute: {next_fn_result.result.member}.')
    if next_fn_result.measurements.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The "measurements" attribute of return type of `next_fn` must be '
          f'placed at SERVER, but found {next_fn_result.measurements}.')


def _build_tff_optimizer_initialize_and_next(
    model_weights_type: computation_types.Type,
    optimizer: optimizer_base.Optimizer):
  """Creates finalizer initialize and next functions for TFF optimizers."""

  @computations.tf_computation
  def init_fn():
    tensor_specs = type_conversions.type_to_tf_tensor_specs(
        model_weights_type.trainable)
    return optimizer.initialize(tensor_specs)

  optimizer_state_type = init_fn.type_signature.result

  @computations.tf_computation(optimizer_state_type,
                               model_weights_type.trainable,
                               model_weights_type.trainable)
  def next_fn(optimizer_state, trainable_weights, update):
    return optimizer.next(optimizer_state, trainable_weights, update)

  return init_fn, next_fn


def _build_keras_optimizer_initialize_and_next(
    model_weights_type: computation_types.Type,
    optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer]):
  """Creates finalizer initialize and next functions for Keras optimizers."""

  @computations.tf_computation
  def init_fn():
    tensor_specs = type_conversions.type_to_tf_tensor_specs(
        model_weights_type.trainable)
    model_variables = tf.nest.map_structure(
        lambda s: tf.Variable(initial_value=tf.zeros(s.shape, s.dtype)),
        tensor_specs)
    optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        optimizer_fn, model_variables, disjoint_init_and_next=True)
    return optimizer.initialize(tensor_specs)

  optimizer_state_type = init_fn.type_signature.result

  @computations.tf_computation(optimizer_state_type,
                               model_weights_type.trainable,
                               model_weights_type.trainable)
  @tf.function
  def next_fn(optimizer_state, trainable_weights, update):
    with tf.init_scope():
      # Create a structure of variables that the server optimizer can update.
      trainable_variables = tf.nest.map_structure(
          lambda t: tf.Variable(initial_value=tf.zeros(t.shape, t.dtype)),
          trainable_weights)
      optimizer = keras_optimizer.build_or_verify_tff_optimizer(
          optimizer_fn, trainable_variables, disjoint_init_and_next=True)

    tf.nest.map_structure(lambda a, b: a.assign(b), trainable_variables,
                          trainable_weights)
    optimizer_state, updated_weights = optimizer.next(optimizer_state,
                                                      trainable_variables,
                                                      update)
    # Keras optimizers mutate model variables in with the `next` step above, so
    # we skip calling the assignment for those optimizers.
    if not isinstance(optimizer, keras_optimizer.KerasOptimizer):
      tf.nest.map_structure(lambda a, b: a.assign(b), trainable_variables,
                            updated_weights)
    return optimizer_state, trainable_variables

  return init_fn, next_fn


def build_apply_optimizer_finalizer(
    optimizer_fn: Union[optimizer_base.Optimizer,
                        Callable[[], tf.keras.optimizers.Optimizer]],
    model_weights_type: computation_types.StructType):
  """Builds finalizer that applies a step of an optimizer.

  The provided `model_weights_type` must be a non-federated `tff.Type` with the
  `tff.learning.ModelWeights` container.

  The 2nd input argument of the created `FinalizerProcess.next` expects a value
  matching `model_weights_type` and its 3rd argument expects value matching
  `model_weights_type.trainable`. The `optimizer` will be applied to the
  trainable model weights only, leaving non_trainable weights unmodified.

  The state of the process is the state of the `optimizer` and the process
  returns empty measurements.

  Args:
    optimizer_fn: A `tff.learning.optimizers.Optimizer` or a no-arg function
      that returns a `tf.keras.optimizers.Optimizer`.
      This optimizer is used to apply client updates to the server model.
    model_weights_type: A non-federated `tff.Type` of the model weights to be
      optimized, which must have a `tff.learning.ModelWeights` container.

  Returns:
    A `FinalizerProcess` that applies the `optimizer`.

  Raises:
    TypeError: If `value_type` does not have a `tff.learning.ModelWeights`
      Python container, or contains a `tff.types.FederatedType`.
  """
  if not isinstance(optimizer_fn, optimizer_base.Optimizer):
    if not callable(optimizer_fn) or not isinstance(
        optimizer_fn(), tf.keras.optimizers.Optimizer):
      raise TypeError(
          'The optimizer_fn must be a `tff.learning.optimizers.Optimizer`, or '
          'a no-arg callable returning a `tf.keras.optimizers.Optimizer`.')

  if (not model_weights_type.is_struct_with_python() or
      model_weights_type.python_container != model_utils.ModelWeights or
      type_analysis.contains_federated_types(model_weights_type)):
    raise TypeError(
        f'Provided value_type must be a tff.types.StructType with its python '
        f'container being tff.learning.ModelWeights, not containing a '
        f'tff.types.FederatedType, but found: {model_weights_type}')

  if isinstance(optimizer_fn, optimizer_base.Optimizer):
    init_tf, next_tf = _build_tff_optimizer_initialize_and_next(
        model_weights_type, optimizer_fn)
  else:
    init_tf, next_tf = _build_keras_optimizer_initialize_and_next(
        model_weights_type, optimizer_fn)

  @computations.federated_computation
  def init_fn():
    return intrinsics.federated_eval(init_tf, placements.SERVER)

  @computations.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_server(model_weights_type),
      computation_types.at_server(model_weights_type.trainable))
  def next_fn(state, weights, update):
    optimizer_state, new_trainable_weights = intrinsics.federated_map(
        next_tf, (state, weights.trainable, update))
    new_weights = intrinsics.federated_zip(
        model_utils.ModelWeights(new_trainable_weights, weights.non_trainable))
    empty_measurements = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(optimizer_state, new_weights,
                                                  empty_measurements)

  return FinalizerProcess(init_fn, next_fn)
