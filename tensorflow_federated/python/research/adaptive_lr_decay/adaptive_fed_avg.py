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
"""Interpolation between FedAvg and FedSGD with adaptive learning rate decay.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections

import attr
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.utils import adapters


def _initialize_optimizer_vars(model, optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')


def _get_weights(model):
  return tff.learning.ModelWeights(
      trainable=tuple(model.trainable_variables),
      non_trainable=tuple(model.non_trainable_variables))


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Attributes:
    model: A dictionary of the model's trainable and non-trainable weights.
    optimizer_state: The server optimizer variables.
    client_lr_callback: A `callback.LROnPlateau` instance.
    server_lr_callback: A `callback.LROnPlateau` instance.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  client_lr_callback = attr.ib()
  server_lr_callback = attr.ib()

  @classmethod
  def from_tff_result(cls, anon_tuple, from_anon_client_callback,
                      from_anon_server_callback):
    """Constructs a `ServerState` from any compatible anonymous tuple."""
    model = tff.learning.ModelWeights(
        trainable=tuple(anon_tuple.model.trainable),
        non_trainable=tuple(anon_tuple.model.non_trainable))

    return cls(
        model=model,
        optimizer_state=list(anon_tuple.optimizer_state),
        client_lr_callback=from_anon_client_callback(
            anon_tuple.client_lr_callback),
        server_lr_callback=from_anon_server_callback(
            anon_tuple.server_lr_callback))

  @classmethod
  def assign_weights_to_keras_model(cls, reference_weights, keras_model):
    """Assign the model weights to the weights of a `tf.keras.Model`.

    Args:
      reference_weights: the `tff.learning.ModelWeights` object to assign
        weights from.
      keras_model: the `tf.keras.Model` object to assign weights to.
    """
    if not isinstance(reference_weights, tff.learning.ModelWeights):
      raise TypeError('The reference weights must be an instance of '
                      'tff.learning.ModelWeights.')
    def assign_weights(keras_weights, tff_weights):
      for k, w in zip(keras_weights, tff_weights):
        k.assign(w)

    assign_weights(keras_model.trainable_weights, reference_weights.trainable)
    assign_weights(keras_model.non_trainable_weights,
                   reference_weights.non_trainable)


@tf.function
def server_update(model, server_optimizer, server_state, aggregated_gradients,
                  client_monitor_value, client_callback_update_fn,
                  server_monitor_value, server_callback_update_fn):
  """Updates `server_state` according to `weights_delta` and output metrics.

  The `model_weights` attribute of `server_state` is updated according to the
  ``weights_delta`. The `client_lr_callback` and `server_lr_callback` attributes
  are updated according to the `client_monitor_value` and `server_monitor_value`
  arguments, respectively.

  Args:
    model: A `tff.learning.TrainableModel`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    aggregated_gradients: A weighted average over clients of the per-client
      gradient sums.
    client_monitor_value: The updated round metric used to update the client
      callback.
    client_callback_update_fn: A function that accepts a client callback and
      client monitor value, and returns an updated client callback.
    server_monitor_value: The updated round metric used to update the server
      callback.
    server_callback_update_fn: A function that accepts a server callback and
      server monitor value, and returns an updated server callback.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tff.utils.assign(model_weights, server_state.model)
  # Server optimizer variables must be initialized prior to invoking this
  tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

  # Apply the update to the model. Note that we do not multiply by -1.0, since
  # we actually accumulate the client gradients.
  grads_and_vars = [
      (x, v) for x, v in zip(aggregated_gradients, model_weights.trainable)
  ]

  server_optimizer.apply_gradients(grads_and_vars)

  updated_client_lr_callback = client_callback_update_fn(
      server_state.client_lr_callback, client_monitor_value)
  updated_server_lr_callback = server_callback_update_fn(
      server_state.server_lr_callback, server_monitor_value)

  # Create a new state based on the updated model.
  return tff.utils.update_state(
      server_state,
      model=model_weights,
      optimizer_state=server_optimizer.variables(),
      client_lr_callback=updated_client_lr_callback,
      server_lr_callback=updated_server_lr_callback)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Attributes:
    accumulated_gradients: A list of accumulated gradients for the model's
      trainable variables. Note: This is a sum of gradients, not the difference
      between the initial brodcast model, and the trained model (as in
      `tff.learning.build_federated_averaging_process`).
    client_weight: Weight to be used in a weighted mean when aggregating
      the `weights_delta`.
    initial_model_output: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      evaluating on the input dataset (before training).
    model_output: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
    optimizer_output: Additional metrics or other outputs defined by the
      optimizer.
  """
  accumulated_gradients = attr.ib()
  client_weight = attr.ib()
  initial_model_output = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


@tf.function
def get_client_output(model, dataset, weights):
  """Evaluates the metrics of a client model."""
  model_weights = _get_weights(model)
  tff.utils.assign(model_weights, weights)
  for batch in dataset:
    model.forward_pass(batch)
  return model.report_local_outputs()


@tf.function
def client_update(model,
                  dataset,
                  initial_weights,
                  client_optimizer,
                  client_weight_fn=None):
  """Updates the client model with local training.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    initial_weights: A `tff.learning.Model.weights` from server.
    client_optimizer: A `tf.keras.optimizer.Optimizer` object.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A 'ClientOutput`.
  """

  model_weights = _get_weights(model)
  tff.utils.assign(model_weights, initial_weights)

  num_examples = tf.constant(0, dtype=tf.int32)
  grad_sums = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  for batch in dataset:
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch)
    grads = tape.gradient(output.loss, model_weights.trainable)
    grads_and_vars = zip(grads, model_weights.trainable)
    client_optimizer.apply_gradients(grads_and_vars)
    num_examples += tf.shape(output.predictions)[0]
    grad_sums = tf.nest.map_structure(tf.add, grad_sums, grads)

  aggregated_outputs = model.report_local_outputs()

  if client_weight_fn is None:
    client_weight = tf.cast(num_examples, dtype=tf.float32)
  else:
    client_weight = client_weight_fn(aggregated_outputs)

  return ClientOutput(
      accumulated_gradients=grad_sums,
      client_weight=client_weight,
      initial_model_output=aggregated_outputs,
      model_output=aggregated_outputs,
      optimizer_output=collections.OrderedDict([('num_examples', num_examples)
                                               ]))


def build_server_init_fn(model_fn, server_optimizer_fn, client_lr_callback,
                         server_lr_callback):
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model` and `ServerState.optimizer_state` are
  initialized via their constructor functions.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    client_lr_callback: A `ReduceLROnPlateau` callback.
    server_lr_callback: A `ReduceLROnPlateau` callback.

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
        client_lr_callback=client_lr_callback,
        server_lr_callback=server_lr_callback)

  return server_init_tf


def build_fed_avg_process(model_fn,
                          client_lr_callback,
                          client_callback_update_fn,
                          server_lr_callback,
                          server_callback_update_fn,
                          client_optimizer_fn=tf.keras.optimizers.SGD,
                          server_optimizer_fn=tf.keras.optimizers.SGD,
                          client_weight_fn=None):
  """Builds the TFF computations for FedAvg with learning rate decay.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    client_lr_callback: A `ReduceLROnPlateau` callback.
    client_callback_update_fn: A function that updates the client callback.
    server_lr_callback: A `ReduceLROnPlateau` callback.
    server_callback_update_fn: A function that updates the server callback.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  dummy_model = model_fn()
  client_monitor = client_lr_callback.monitor
  server_monitor = server_lr_callback.monitor

  server_init_tf = build_server_init_fn(model_fn, server_optimizer_fn,
                                        client_lr_callback, server_lr_callback)

  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

  client_lr_type = server_state_type.client_lr_callback.learning_rate
  client_monitor_value_type = server_state_type.client_lr_callback.best
  server_monitor_value_type = server_state_type.server_lr_callback.best

  @tff.tf_computation(tf_dataset_type, model_weights_type, client_lr_type)
  def client_update_fn(tf_dataset, initial_model_weights, client_lr):
    client_optimizer = client_optimizer_fn(learning_rate=client_lr)
    initial_model_output = get_client_output(model_fn(), tf_dataset,
                                             initial_model_weights)
    client_state = client_update(model_fn(), tf_dataset, initial_model_weights,
                                 client_optimizer, client_weight_fn)
    return tff.utils.update_state(
        client_state, initial_model_output=initial_model_output)

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      client_monitor_value_type, server_monitor_value_type)
  def server_update_fn(server_state, model_delta, client_monitor_value,
                       server_monitor_value):
    model = model_fn()
    server_lr = server_state.server_lr_callback.learning_rate
    server_optimizer = server_optimizer_fn(learning_rate=server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta,
                         client_monitor_value, client_callback_update_fn,
                         server_monitor_value, server_callback_update_fn)

  @tff.federated_computation(
      tff.FederatedType(server_state_type, tff.SERVER),
      tff.FederatedType(tf_dataset_type, tff.CLIENTS))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Note that in addition to updating the server weights according to the client
    model weight deltas, we extract metrics (governed by the `monitor` attribute
    of the `client_lr_callback` and `server_lr_callback` attributes of the
    `server_state`) and use these to update the client learning rate callbacks.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation` before and during local
      client training.
    """
    client_model = tff.federated_broadcast(server_state.model)
    client_lr = tff.federated_broadcast(
        server_state.client_lr_callback.learning_rate)
    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, client_model, client_lr))

    client_weight = client_outputs.client_weight
    aggregated_gradients = tff.federated_mean(
        client_outputs.accumulated_gradients, weight=client_weight)

    initial_aggregated_outputs = dummy_model.federated_output_computation(
        client_outputs.initial_model_output)
    if isinstance(initial_aggregated_outputs.type_signature,
                  tff.NamedTupleType):
      initial_aggregated_outputs = tff.federated_zip(initial_aggregated_outputs)

    aggregated_outputs = dummy_model.federated_output_computation(
        client_outputs.model_output)
    if isinstance(aggregated_outputs.type_signature, tff.NamedTupleType):
      aggregated_outputs = tff.federated_zip(aggregated_outputs)
    client_monitor_value = initial_aggregated_outputs[client_monitor]
    server_monitor_value = initial_aggregated_outputs[server_monitor]

    server_state = tff.federated_map(
        server_update_fn, (server_state, aggregated_gradients,
                           client_monitor_value, server_monitor_value))

    return server_state, initial_aggregated_outputs, aggregated_outputs

  @tff.federated_computation
  def initialize_fn():
    return tff.federated_value(server_init_tf(), tff.SERVER)

  tff_iterative_process = tff.templates.IterativeProcess(
      initialize_fn=initialize_fn, next_fn=run_one_round)

  return FedAvgDecayAdapter(tff_iterative_process,
                            client_lr_callback.from_anonymous_tuple,
                            server_lr_callback.from_anonymous_tuple)


class FedAvgDecayAdapter(adapters.IterativeProcessPythonAdapter):
  """Converts iterative process results from anonymous tuples.

  Converts to ServerState and unpacks metrics. This simplifies tasks such as
  recording metrics.
  """

  def __init__(self, iterative_process, from_anon_client_callback,
               from_anon_server_callback):
    self._iterative_process = iterative_process
    self._from_anon_client_callback = from_anon_client_callback
    self._from_anon_server_callback = from_anon_server_callback

  def initialize(self):
    initial_state = self._iterative_process.initialize()
    return ServerState.from_tff_result(initial_state,
                                       self._from_anon_client_callback,
                                       self._from_anon_server_callback)

  def next(self, state, data):
    state, initial_metrics, metrics = self._iterative_process.next(state, data)
    state = ServerState.from_tff_result(state, self._from_anon_client_callback,
                                        self._from_anon_server_callback)
    initial_metrics = initial_metrics._asdict(recursive=True)
    metrics = metrics._asdict(recursive=True)
    total_metrics = {
        'before_training': initial_metrics,
        'during_training': metrics
    }
    outputs = None
    return adapters.IterationResult(state, total_metrics, outputs)
