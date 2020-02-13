# Lint as: python3
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
"""An implementation of the Targeted Attacks in Federated Learning.

This is intended to implement and simulate existing targeted attacks in
federated learning systems. Most of the implementations are based on
'tff.ressearch.basedline_fedavg'. Similar simulation scripts can be used by
replacing relevant functions and plugging-in correct parameters.

Based on the following papers:

Analyzing Federated Learning through an Adversarial Lens
    Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal,
    Seraphin Calo ICML 2019.
    https://arxiv.org/abs/1811.12470

How To Back door Federated Learning
    Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin,
    Vitaly Shmatikov
    https://arxiv.org/abs/1807.00459
"""

import collections

import attr
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.targeted_attack.aggregate_fn import build_stateless_mean
from tensorflow_federated.python.tensorflow_libs import tensor_utils


@attr.s(eq=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `weights_delta_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  weights_delta_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


@attr.s(eq=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Variables of optimizer.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  delta_aggregate_state = attr.ib()


def _create_optimizer_vars(model, optimizer):
  model_weights = _get_weights(model)
  delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta),
      tf.nest.flatten(model_weights.trainable))
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  return optimizer.variables()


def _get_weights(model):
  model_weights = collections.namedtuple('ModelWeights',
                                         'trainable non_trainable')
  return model_weights(
      trainable=tensor_utils.to_var_dict(model.trainable_variables),
      non_trainable=tensor_utils.to_var_dict(model.non_trainable_variables))


def _get_norm(weights):
  """Compute the norm of a weight matrix.

  Args:
    weights: a OrderedDict specifying weight matrices at different layers.
  Returns:
    The norm of all layer weight matrices.
  """
  return tf.linalg.global_norm(tf.nest.flatten(weights))


@tf.function
def server_update(model, server_optimizer, server_optimizer_vars, server_state,
                  weights_delta, new_delta_aggregate_state):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `tff.learning.TrainableModel`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_optimizer_vars: A list of previous variables of server_optimzer.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.
    new_delta_aggregate_state: An update to the server state.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tf.nest.map_structure(lambda a, b: a.assign(b),
                        (model_weights, server_optimizer_vars),
                        (server_state.model, server_state.optimizer_state))

  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
      tf.nest.flatten(model_weights.trainable))
  server_optimizer.apply_gradients(grads_and_vars, name='server_update')

  return tff.utils.update_state(
      server_state,
      model=model_weights,
      optimizer_state=server_optimizer_vars,
      delta_aggregate_state=new_delta_aggregate_state)


class ClientExplicitBoosting:
  """Client tensorflow logic for explicit boosting."""

  def __init__(self, boost_factor):
    """Specify the boosting parameter.

    Args:
      boost_factor: A 'tf.float32' specifying how malicious update is boosted.
    """
    self.boost_factor = boost_factor

  @tf.function
  def __call__(self, model, benign_dataset, malicious_dataset, client_type,
               initial_weights):
    """Updates client model with client potentially being malicious.

    Args:
      model: A `tff.learning.Model`.
      benign_dataset: A 'tf.data.Dataset' consisting of benign dataset.
      malicious_dataset: A 'tf.data.Dataset' consisting of malicious dataset.
      client_type: A 'tf.bool' indicating whether the client is malicious; iff
        `True` the client will construct its update using `malicious_dataset`,
        otherwise will construct the update using `benign_dataset`.
      initial_weights: A `tff.learning.Model.weights` from server.

    Returns:
      A 'ClientOutput`.
    """
    model_weights = _get_weights(model)

    @tf.function
    def reduce_fn(num_examples_sum, batch):
      """Runs `tff.learning.Model.train_on_batch` on local client batch."""
      output = model.train_on_batch(batch)
      return num_examples_sum + tf.shape(output.predictions)[0]

    @tf.function
    def compute_benign_update():
      """compute benign update sent back to the server."""
      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                            initial_weights)

      num_examples_sum = benign_dataset.reduce(
          initial_state=tf.constant(0), reduce_func=reduce_fn)

      weights_delta_benign = tf.nest.map_structure(lambda a, b: a - b,
                                                   model_weights.trainable,
                                                   initial_weights.trainable)

      aggregated_outputs = model.report_local_outputs()

      return weights_delta_benign, aggregated_outputs, num_examples_sum

    @tf.function
    def compute_malicious_update():
      """compute malicious update sent back to the server."""

      weights_delta_benign, aggregated_outputs, num_examples_sum \
          = compute_benign_update()

      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                            initial_weights)

      malicious_dataset.reduce(
          initial_state=tf.constant(0), reduce_func=reduce_fn)

      weights_delta_malicious = tf.nest.map_structure(lambda a, b: a - b,
                                                      model_weights.trainable,
                                                      initial_weights.trainable)

      nested_boost_factor = collections.OrderedDict([
          (key, self.boost_factor) for key in weights_delta_malicious.keys()
      ])

      weights_delta = tf.nest.map_structure(
          tf.add, weights_delta_benign,
          tf.nest.map_structure(tf.multiply, nested_boost_factor,
                                weights_delta_malicious))

      return weights_delta, aggregated_outputs, num_examples_sum

    result = tf.cond(
        tf.equal(client_type, True), compute_malicious_update,
        compute_benign_update)
    weights_delta, aggregated_outputs, num_examples_sum = result

    weights_delta_weight = tf.cast(num_examples_sum, tf.float32)

    weight_norm = _get_norm(weights_delta)

    return ClientOutput(
        weights_delta, weights_delta_weight, aggregated_outputs,
        tensor_utils.to_odict({
            'num_examples': num_examples_sum,
            'weight_norm': weight_norm,
        }))


def build_server_init_fn(model_fn, server_optimizer_fn, delta_aggregate_state):
  """Builds a `tff.tf_computation` that returns initial `ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    delta_aggregate_state: The initial state of the delta_aggregator.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = _create_optimizer_vars(model, server_optimizer)
    return ServerState(
        model=_get_weights(model),
        optimizer_state=server_optimizer_vars,
        delta_aggregate_state=delta_aggregate_state)

  return server_init_tf


def build_server_update_fn(model_fn, server_optimizer_fn, server_state_type,
                           model_weights_type):
  """Builds a `tff.tf_computation` that updates `ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    server_state_type: type_signature of server state.
    model_weights_type: type_signature of model weights.

  Returns:
    A `tff.tf_computation` that updates `ServerState`.
  """

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      server_state_type.delta_aggregate_state)
  def server_update_tf(server_state, model_delta, new_delta_aggregate_state):
    """Updates the `server_state`.

    Args:
      server_state: The `ServerState`.
      model_delta: The model difference from clients.
      new_delta_aggregate_state: An update to the server state.

    Returns:
      The updated `ServerState`.
    """
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = _create_optimizer_vars(model, server_optimizer)

    return server_update(model, server_optimizer, server_optimizer_vars,
                         server_state, model_delta, new_delta_aggregate_state)

  return server_update_tf


def build_client_update_fn(model_fn, client_update_tf, tf_dataset_type,
                           model_weights_type):
  """Builds a `tff.tf_computation` in the presense of malicious clients.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    client_update_tf: A 'tf.function' that computes the ClientOutput
    tf_dataset_type: type_signature of dataset.
    model_weights_type: type_signature of model weights.

  Returns:
    A `tff.tf_computation` for local model optimization with type signature:
    '@tff.tf_computation(tf_dataset_type, tf_dataset_type,
                      tf.bool, model_weights_type)'
  """

  @tff.tf_computation(tf_dataset_type, tf_dataset_type, tf.bool,
                      model_weights_type)
  def client_delta_tf(benign_dataset, malicious_dataset, client_type,
                      initial_model_weights):
    """Performs client local model optimization.

    Args:
      benign_dataset: A 'tf.data.Dataset' consisting of benign dataset
      malicious_dataset: A 'tf.data.Dataset' consisting of malicious dataset
      client_type: A 'tf.bool' indicating whether the client is malicious
      initial_model_weights: A `tff.learning.Model.weights` from server.

    Returns:
      A 'ClientOutput`.
    """
    return client_update_tf(model_fn(), benign_dataset, malicious_dataset,
                            client_type, initial_model_weights)

  return client_delta_tf


def build_run_one_round_fn_attacked(server_update_fn, client_update_fn,
                                    stateful_delta_aggregate_fn,
                                    dummy_model_for_metadata,
                                    federated_server_state_type,
                                    federated_dataset_type):
  """Builds a `tff.federated_computation` for a round of training.

  Args:
    server_update_fn: A function for updates in the server.
    client_update_fn: A function for updates in the clients.
    stateful_delta_aggregate_fn: A 'tff.computation'that takes in model deltas
      placed@CLIENTS to an aggregated model delta placed@SERVER.
    dummy_model_for_metadata: A dummy `tff.learning.TrainableModel`.
    federated_server_state_type: type_signature of federated server state.
    federated_dataset_type: type_signature of federated dataset.

  Returns:
    A `tff.federated_computation` for a round of training.
  """

  federated_bool_type = tff.FederatedType(tf.bool, tff.CLIENTS)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type, federated_dataset_type,
                             federated_bool_type)
  def run_one_round(server_state, federated_dataset, malicious_dataset,
                    malicious_clients):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.
      malicious_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.
        consisting of malicious datasets.
      malicious_clients: A federated `tf.bool` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """

    client_model = tff.federated_broadcast(server_state.model)

    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, malicious_dataset, malicious_clients, client_model))

    weight_denom = client_outputs.weights_delta_weight

    new_delta_aggregate_state, round_model_delta = stateful_delta_aggregate_fn(
        server_state.delta_aggregate_state,
        client_outputs.weights_delta,
        weight=weight_denom)

    server_state = tff.federated_map(
        server_update_fn,
        (server_state, round_model_delta, new_delta_aggregate_state))

    aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
        client_outputs.model_output)
    aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  return run_one_round


def build_federated_averaging_process_attacked(
    model_fn,
    stateful_delta_aggregate_fn=build_stateless_mean(),
    client_update_tf=ClientExplicitBoosting(boost_factor=1.0),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)):
  """Builds the TFF computations for optimization using federated averaging with potentially malicious clients.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    stateful_delta_aggregate_fn: A 'tff.computation' that aggregates model
      deltas placed@CLIENTS to an aggregated model delta placed@SERVER.
    client_update_tf: a 'tf.function' computes the ClientOutput.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.utils.IterativeProcess`.
  """

  dummy_model_for_metadata = model_fn()

  server_init_tf = build_server_init_fn(
      model_fn, server_optimizer_fn, stateful_delta_aggregate_fn.initialize())
  server_state_type = server_init_tf.type_signature.result
  server_update_fn = build_server_update_fn(model_fn, server_optimizer_fn,
                                            server_state_type,
                                            server_state_type.model)
  tf_dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec)

  client_update_fn = build_client_update_fn(model_fn, client_update_tf,
                                            tf_dataset_type,
                                            server_state_type.model)

  federated_server_state_type = tff.FederatedType(server_state_type, tff.SERVER)

  federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

  run_one_round_tff = build_run_one_round_fn_attacked(
      server_update_fn, client_update_fn, stateful_delta_aggregate_fn,
      dummy_model_for_metadata, federated_server_state_type,
      federated_dataset_type)

  return tff.utils.IterativeProcess(
      initialize_fn=tff.federated_computation(
          lambda: tff.federated_value(server_init_tf(), tff.SERVER)),
      next_fn=run_one_round_tff)


class ClientProjectBoost:
  """Client tensorflow logic for norm bounded attack."""

  def __init__(self, boost_factor, norm_bound, round_num):
    """Specify the attacking parameter.

    Args:
      boost_factor: A 'tf.float32' specifying how malicious update is boosted.
      norm_bound: A 'tf.float32' specifying the norm bound before boosting.
      round_num: A 'tf.int32' specifying the number of iterative rounds.
    """
    self.boost_factor = boost_factor
    self.norm_bound = norm_bound
    self.round_num = round_num

  @tf.function
  def __call__(self, model, benign_dataset, malicious_dataset,
               client_is_malicious, initial_weights):
    """Updates client model with client potentially being malicious.

    Args:
      model: A `tff.learning.Model`.
      benign_dataset: A 'tf.data.Dataset' consisting of benign dataset.
      malicious_dataset: A 'tf.data.Dataset' consisting of malicious dataset.
      client_is_malicious: A 'tf.bool' showing whether the client is malicious.
      initial_weights: A `tff.learning.Model.weights` from server.

    Returns:
      A 'ClientOutput`.
    """
    model_weights = _get_weights(model)

    @tf.function
    def clip_by_norm(gradient, norm):
      """Clip the gradient by its l2 norm."""

      delta_norm = _get_norm(gradient)

      if delta_norm < tf.cast(norm, tf.float32):
        return gradient
      else:
        delta_mul_factor = tf.math.divide_no_nan(
            tf.cast(norm, tf.float32), delta_norm)
        nested_mul_factor = collections.OrderedDict([
            (key, delta_mul_factor) for key in gradient.keys()
        ])
        return tf.nest.map_structure(tf.multiply, nested_mul_factor, gradient)

    @tf.function
    def project_weights(weights, initial_weights, norm):
      """Project the weight onto l2 ball around initial_weights with radius norm."""
      weights_delta = tf.nest.map_structure(lambda a, b: a - b, weights,
                                            initial_weights)

      return tf.nest.map_structure(tf.add, clip_by_norm(weights_delta, norm),
                                   initial_weights)

    @tf.function
    def reduce_fn(num_examples_sum, batch):
      """Runs `tff.learning.Model.train_on_batch` on local client batch."""
      output = model.train_on_batch(batch)
      return num_examples_sum + tf.shape(output.predictions)[0]

    @tf.function
    def compute_benign_update():
      """compute benign update sent back to the server."""
      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                            initial_weights)

      num_examples_sum = benign_dataset.reduce(
          initial_state=tf.constant(0), reduce_func=reduce_fn)

      weights_delta_benign = tf.nest.map_structure(lambda a, b: a - b,
                                                   model_weights.trainable,
                                                   initial_weights.trainable)

      aggregated_outputs = model.report_local_outputs()

      return weights_delta_benign, aggregated_outputs, num_examples_sum

    @tf.function
    def compute_malicious_update():
      """compute malicious update sent back to the server."""

      _, aggregated_outputs, num_examples_sum = compute_benign_update()

      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                            initial_weights)

      for _ in range(self.round_num):
        benign_dataset.reduce(
            initial_state=tf.constant(0), reduce_func=reduce_fn)
        malicious_dataset.reduce(
            initial_state=tf.constant(0), reduce_func=reduce_fn)

        tf.nest.map_structure(
            lambda a, b: a.assign(b), model_weights.trainable,
            project_weights(model_weights.trainable, initial_weights.trainable,
                            tf.cast(self.norm_bound, tf.float32)))

      weights_delta_malicious = tf.nest.map_structure(lambda a, b: a - b,
                                                      model_weights.trainable,
                                                      initial_weights.trainable)

      nested_boost_factor = collections.OrderedDict([
          (key, self.boost_factor) for key in weights_delta_malicious.keys()
      ])

      weights_delta = tf.nest.map_structure(tf.multiply, nested_boost_factor,
                                            weights_delta_malicious)

      return weights_delta, aggregated_outputs, num_examples_sum

    if client_is_malicious:
      result = compute_malicious_update()
    else:
      result = compute_benign_update()
    weights_delta, aggregated_outputs, num_examples_sum = result

    weights_delta_weight = tf.cast(num_examples_sum, tf.float32)
    weight_norm = _get_norm(weights_delta)

    return ClientOutput(
        weights_delta, weights_delta_weight, aggregated_outputs,
        tensor_utils.to_odict({
            'num_examples': num_examples_sum,
            'weight_norm': weight_norm,
        }))
