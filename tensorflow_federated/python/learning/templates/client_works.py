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
"""Abstractions for client work in learning algorithms."""

from typing import Callable
import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base


@attr.s(frozen=True)
class ClientResult():
  """A structure containing the result of `ClientWorkProcess.next` computation.

  Attributes:
    update: The local update to model weights produced by clients.
    update_weight: A weight for weighted aggregation of the `update`.
  """
  update = attr.ib()
  update_weight = attr.ib()


class ModelWeightsTypeError(TypeError):
  """`TypeError` for incorrect container of model weights."""


class ClientDataTypeError(TypeError):
  """`TypeError` for incorrect type of client data."""


class ClientResultTypeError(TypeError):
  """`TypeError` for incorrect structure of result of client work."""


class ClientWorkProcess(measured_process.MeasuredProcess):
  """A stateful process capturing work at clients during learning.

  Client work encapsulates the main work performed by clinets as part of a
  federated learning algorithm, such as several steps of gradient descent based
  on the client data, and returning a update to the initial model weights.

  A `ClientWorkProcess` is a `tff.templates.MeasuredProcess` that formalizes the
  type signature of `initialize_fn` and `next_fn` for the core work performed by
  clients in a learning process.

  The `initialize_fn` and `next_fn` must have the following type signatures:
  ```
    - initialize_fn: ( -> S@SERVER)
    - next_fn: (<S@SERVER,
                 ModelWeights(TRAINABLE, NON_TRAINABLE)@CLIENTS,
                 DATA@CLIENTS>
                ->
                <state=S@SERVER,
                 result=ClientResult(TRAINABLE, W)@CLIENTS,
                 measurements=M@SERVER>)
  ```
  with `W` and `M` being arbitrary types not dependent on other types here.

  `ClientWorkProcess` requires `next_fn` with a second and a third input
  argument, which are both values placed at `CLIENTS`. The second argument is
  initial model weights to be used for the work to be performed by clients. It
  must be of a type matching `tff.learning.ModelWeights`, for these to be
  assignable to the weights of a `tff.learning.Model`. The third argument must
  be a `tff.SequenceType` representing the data available at clients.

  The `result` field of the returned `tff.templates.MeasuredProcessOutput` must
  be placed at `CLIENTS`, and be of type matching `ClientResult`, of which the
  `update` field represents the update to the trainable model weights, and
  `update_weight` represents the weight to be used for weighted aggregation of
  the updates.

  The `measurements` field of the returned `tff.templates.MeasuredProcessOutput`
  must be placed at `SERVER`. Thus, implementation of this process must include
  aggregation of any metrics computed during training. TODO(b/190334722):
  Confirm this aspect, or change it.
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
          f'The state controlled by a `ClientWorkProcess` must be placed at '
          f'the SERVER, but found type: {initialize_fn.type_signature.result}.')
    # Note that state of next_fn being placed at SERVER is now ensured by the
    # assertions in base class which would otherwise raise
    # TemplateStateNotAssignableError.

    next_fn_param = next_fn.type_signature.parameter
    if not next_fn_param.is_struct():
      raise errors.TemplateNextFnNumArgsError(
          f'The `next_fn` must have exactly three input arguments, but found '
          f'the following input type which is not a Struct: {next_fn_param}.')
    if len(next_fn_param) != 3:
      next_param_str = '\n- '.join([str(t) for t in next_fn_param])
      raise errors.TemplateNextFnNumArgsError(
          f'The `next_fn` must have exactly three input arguments, but found '
          f'{len(next_fn_param)} input arguments:\n{next_param_str}')
    model_weights_param = next_fn_param[1]
    client_data_param = next_fn_param[2]
    if model_weights_param.placement != placements.CLIENTS:
      raise errors.TemplatePlacementError(
          f'The second input argument of `next_fn` must be placed at CLIENTS '
          f'but found {model_weights_param}.')
    if (not model_weights_param.member.is_struct_with_python() or
        model_weights_param.member.python_container
        is not model_utils.ModelWeights):
      raise ModelWeightsTypeError(
          f'The second input argument of `next_fn` must have the '
          f'`tff.learning.ModelWeights` container but found '
          f'{model_weights_param}')
    if client_data_param.placement != placements.CLIENTS:
      raise errors.TemplatePlacementError(
          f'The third input argument of `next_fn` must be placed at CLIENTS '
          f'but found {client_data_param}.')
    if not client_data_param.member.is_sequence():
      raise ClientDataTypeError(
          f'The third input argument of `next_fn` must be a sequence but found '
          f'{client_data_param}.')

    next_fn_result = next_fn.type_signature.result
    if (not next_fn_result.result.is_federated() or
        next_fn_result.result.placement != placements.CLIENTS):
      raise errors.TemplatePlacementError(
          f'The "result" attribute of the return type of `next_fn` must be '
          f'placed at CLIENTS, but found {next_fn_result.result}.')
    if (not next_fn_result.result.member.is_struct_with_python() or
        next_fn_result.result.member.python_container is not ClientResult):
      raise ClientResultTypeError(
          f'The "result" attribute of the return type of `next_fn` must have '
          f'the `ClientResult` container, but found {next_fn_result.result}.')
    if not model_weights_param.member.trainable.is_assignable_from(
        next_fn_result.result.member.update):
      raise ClientResultTypeError(
          f'The "update" attribute of returned `ClientResult` must match '
          f'the "trainable" attribute of the `tff.learning.ModelWeights` '
          f'expected as second input argument of the `next_fn`. Found:\n'
          f'Second input argument: {model_weights_param.member.trainable}\n'
          f'Update attribute of result: {next_fn_result.result.member.update}.')
    if next_fn_result.measurements.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The "measurements" attribute of return type of `next_fn` must be '
          f'placed at SERVER, but found {next_fn_result.measurements}.')


# TODO(b/190334722): Add model metric handling and aggregation and report it in
# the measurement field of the output.
def build_model_delta_client_work(model_fn: Callable[[], model_lib.Model],
                                  optimizer: optimizer_base.Optimizer):
  """Builds `ClientWorkProcess` returning change to the trained model weights.

  The created `ClientWorkProcess` expects model weights that can be assigned to
  the model created by `model_fn`, and will apply `optimizer` to optimize the
  model using the client data. The returned `ClientResult` will contain the
  difference between the trained and initial trainable model weights (aka
  "model delta") as update, and the update_weight will be the number of examples
  used in training. The type signature for client data is derived from the input
  spec of the model.

  This method is the recommended starting point for forking a custom
  implementation of the `ClientWorkProcess`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    optimizer: A `tff.learning.optimizers.Optimizer`.

  Returns:
    A `ClientWorkProcess`.
  """
  py_typecheck.check_callable(model_fn)
  # TODO(b/190334722): Include support for Keras optimizers via
  # tff.learning.optimizers.KerasOptimizer when ready.
  py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
  weights_type, data_type = _weights_and_data_type_from_model_fn(model_fn)
  # TODO(b/161529310): We flatten and convert the trainable specs to tuple, as
  # "for batch in data:" pattern would try to stack the tensors in a list.
  optimizer_tensor_specs = _flat_tuple(
      type_conversions.type_to_tf_tensor_specs(weights_type.trainable))

  @computations.tf_computation(weights_type, data_type)
  @tf.function
  def local_update(initial_weights, data):
    # TODO(b/190334722): Restructure so that model_fn only needs to be invoked
    # once.
    with tf.init_scope():
      model = model_fn()
    model_weights = model_utils.ModelWeights.from_model(model)

    tf.nest.map_structure(lambda weight, value: weight.assign(value),
                          model_weights, initial_weights)
    num_examples = tf.constant(0, tf.int32)
    optimizer_state = optimizer.initialize(optimizer_tensor_specs)

    # TODO(b/161529310): Different from creating an iterator using iter(data).
    for batch in data:
      with tf.GradientTape() as tape:
        outputs = model.forward_pass(batch)
      gradients = tape.gradient(outputs.loss, model_weights.trainable)
      num_examples += tf.shape(outputs.predictions)[0]

      optimizer_state, updated_weights = optimizer.next(
          optimizer_state, _flat_tuple(model_weights.trainable),
          _flat_tuple(gradients))
      updated_weights = tf.nest.pack_sequence_as(model_weights.trainable,
                                                 updated_weights)
      tf.nest.map_structure(lambda weight, value: weight.assign(value),
                            model_weights.trainable, updated_weights)

    model_delta = tf.nest.map_structure(lambda x, y: x - y,
                                        initial_weights.trainable,
                                        model_weights.trainable)
    return ClientResult(
        update=model_delta, update_weight=tf.cast(num_examples, tf.float32))

  @computations.federated_computation
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  @computations.federated_computation(
      init_fn.type_signature.result, computation_types.at_clients(weights_type),
      computation_types.at_clients(data_type))
  def next_fn(state, weights, client_data):
    client_result = intrinsics.federated_map(local_update,
                                             (weights, client_data))
    empty_measurements = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(state, client_result,
                                                  empty_measurements)

  return ClientWorkProcess(init_fn, next_fn)


def _weights_and_data_type_from_model_fn(model_fn):
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
  data_type = computation_types.SequenceType(model.input_spec)
  model_weights_type = model_utils.weights_type_from_model(model)
  return model_weights_type, data_type


def _flat_tuple(struct):
  return tuple(tf.nest.flatten(struct))
