# Copyright 2019, The TensorFlow Federated Authors.
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
"""An implementation of the FedAvg algorithm with learning rate schedules.

This is intended to be a somewhat minimal implementation of Federated
Averaging that allows for client and server learning rate scheduling.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

# TODO(b/147626125): Merge with fed_avg.py to allow for learning rate schedules
# in the reparameterized federated averaging framework.

# TODO(b/149402127): Implement a check to zero out client updates if any value
# is non-finite.

import collections

import attr
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.utils import adapters
from tensorflow_federated.python.tensorflow_libs import tensor_utils

ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')


def _initialize_optimizer_vars(model, optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  assert optimizer.variables()


def _get_weights(model):
  return ModelWeights(
      trainable=tuple(model.trainable_variables),
      non_trainable=tuple(model.non_trainable_variables))


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
  -   `optimizer_state`: The server optimizer variables.
  -   `round_num`: The current training round, as a float.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  # This is a float to avoid type incompatibility when calculating learning rate
  # schedules.

  @classmethod
  def from_tff_result(cls, anon_tuple):
    """Constructs a `ServerState` from any compatible anonymous tuple."""
    model = ModelWeights(
        trainable=tuple(anon_tuple.model.trainable),
        non_trainable=tuple(anon_tuple.model.non_trainable))

    return cls(
        model=model,
        optimizer_state=list(anon_tuple.optimizer_state),
        round_num=anon_tuple.round_num)

  @classmethod
  def assign_weights_to_keras_model(cls, reference_model, keras_model):
    """Assign the model weights to the weights of a `tf.keras.Model`.

    Args:
      reference_model: the `ModelWeights` object to assign weights from.
      keras_model: the `tf.keras.Model` object to assign weights to.
    """
    if not isinstance(reference_model, ModelWeights):
      raise TypeError('The reference model must be an instance of '
                      'fed_avg_schedule.ModelWeights.')

    def assign_weights(keras_weights, tff_weights):
      for k, w in zip(keras_weights, tff_weights):
        k.assign(w)

    assign_weights(keras_model.trainable_weights, reference_model.trainable)
    assign_weights(keras_model.non_trainable_weights,
                   reference_model.non_trainable)


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`, increase the round number.

  Args:
    model: A `tff.learning.TrainableModel`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tff.utils.assign(model_weights, server_state.model)
  # Server optimizer variables must be initialized prior to invoking this
  tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  if has_non_finite_weight > 0:
    return server_state

  # Apply the update to the model. We must multiply weights_delta by -1.0 to
  # view it as a gradient that should be applied to the server_optimizer.
  grads_and_vars = [
      (-1.0 * x, v) for x, v in zip(weights_delta, model_weights.trainable)
  ]

  server_optimizer.apply_gradients(grads_and_vars)

  # Create a new state based on the updated model.
  return tff.utils.update_state(
      server_state,
      model=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1.0)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


def create_client_update_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """
  @tf.function
  def client_update(model,
                    dataset,
                    initial_weights,
                    client_optimizer,
                    client_weight_fn=None):
    """Updates client model.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.Model.weights` from server.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.

    Returns:
      A 'ClientOutput`.
    """

    model_weights = _get_weights(model)
    tff.utils.assign(model_weights, initial_weights)

    num_examples = tf.constant(0, dtype=tf.int32)
    for batch in dataset:
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights.trainable)
      grads_and_vars = zip(grads, model_weights.trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      num_examples += tf.shape(output.predictions)[0]

    aggregated_outputs = model.report_local_outputs()
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    elif client_weight_fn is None:
      client_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_weight = client_weight_fn(aggregated_outputs)

    return ClientOutput(
        weights_delta, client_weight, aggregated_outputs,
        collections.OrderedDict([('num_examples', num_examples)]))

  return client_update


def build_server_init_fn(model_fn, server_optimizer_fn):
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model` and `ServerState.optimizer_state` are
  initialized via their constructor functions. The attribute
  `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    server_optimizer = server_optimizer_fn()
    model = model_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model=_get_weights(model),
        optimizer_state=server_optimizer.variables(),
        round_num=0.0)

  return server_init_tf


def build_fed_avg_process(model_fn,
                          client_optimizer_fn,
                          client_lr=0.1,
                          server_optimizer_fn=tf.keras.optimizers.SGD,
                          server_lr=1.0,
                          client_weight_fn=None):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    client_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    server_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  dummy_model = model_fn()

  server_init_tf = build_server_init_fn(model_fn, server_optimizer_fn)
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  round_num_type = server_state_type.round_num

  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

  @tff.tf_computation(tf_dataset_type, model_weights_type, round_num_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(learning_rate=client_lr)
    client_update = create_client_update_fn()
    return client_update(model_fn(), tf_dataset, initial_model_weights,
                         client_optimizer, client_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(learning_rate=server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  @tff.federated_computation(
      tff.FederatedType(server_state_type, tff.SERVER),
      tff.FederatedType(tf_dataset_type, tff.CLIENTS))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """
    client_model = tff.federated_broadcast(server_state.model)
    client_round_num = tff.federated_broadcast(server_state.round_num)
    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_model, client_round_num))

    client_weight = client_outputs.client_weight
    model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=client_weight)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, model_delta))

    aggregated_outputs = dummy_model.federated_output_computation(
        client_outputs.model_output)
    aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  @tff.federated_computation
  def initialize_fn():
    return tff.federated_value(server_init_tf(), tff.SERVER)

  tff_iterative_process = tff.templates.IterativeProcess(
      initialize_fn=initialize_fn, next_fn=run_one_round)

  return FederatedAveragingProcessAdapter(tff_iterative_process)


class FederatedAveragingProcessAdapter(adapters.IterativeProcessPythonAdapter):
  """Converts iterative process results from anonymous tuples.

  Converts to ServerState and unpacks metrics. This simplifies tasks such as
  recording metrics.
  """

  def __init__(self, iterative_process):
    self._iterative_process = iterative_process

  def initialize(self):
    initial_state = self._iterative_process.initialize()
    return ServerState.from_tff_result(initial_state)

  def next(self, state, data):
    state, metrics = self._iterative_process.next(state, data)
    state = ServerState.from_tff_result(state)
    metrics = metrics._asdict(recursive=True)
    outputs = None
    return adapters.IterationResult(state, metrics, outputs)
