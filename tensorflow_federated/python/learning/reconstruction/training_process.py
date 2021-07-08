# Copyright 2020, Google LLC.
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
"""An implementation of Federated Reconstruction (FedRecon).

This is a federated learning algorithm designed for
`tff.learning.reconstruction.Model`s. `tff.learning.reconstruction.Model`s
introduce a partition of variables into global variables and local variables.

At a high level, local variables are reconstructed (via training) on client
devices at the beginning of each round and never sent to the server. Each
client's local variables are then used to update global variables. Global
variable deltas are aggregated normally on the server as in Federated Averaging
and sent to new clients at the beginning of the next round.

During each round:
1. A random subset of clients is selected.
2. Each client receives the latest global variables from the server.
3. Each client locally reconstructs its local variables.
4. Each client computes an update for the global variables.
5. The server aggregates global variables across users and updates them for the
   next round.

Note that clients are stateless since the local variables are not stored across
rounds.

Based on the paper:
Federated Reconstruction: Partially Local Federated Learning
    Karan Singhal, Hakim Sidahmed, Zachary Garrett, Shanshan Wu, Keith Rush,
    Sushant Prakash. https://arxiv.org/abs/2102.03448
"""

import collections
import functools
from typing import Callable, List, Optional, Union

import attr
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process as aggregation_process_lib
from tensorflow_federated.python.core.templates import iterative_process as iterative_process_lib
from tensorflow_federated.python.core.templates import measured_process as measured_process_lib
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.learning.reconstruction import keras_utils
from tensorflow_federated.python.learning.reconstruction import model as model_lib
from tensorflow_federated.python.learning.reconstruction import reconstruction_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils

# Type aliases for readability.
AggregationFactory = Union[factory.WeightedAggregationFactory,
                           factory.UnweightedAggregationFactory]
LossFn = Callable[[], tf.keras.losses.Loss]
MetricsFn = Callable[[], List[tf.keras.metrics.Metric]]
ModelFn = Callable[[], model_lib.Model]
OptimizerFn = Callable[[], tf.keras.optimizers.Optimizer]


@attr.s(eq=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during training.

  Attributes:
    weights_delta: A dictionary of updates to the model's global trainable
      variables.
    client_weight: Weight to be used in a weighted mean when aggregating
      weights_delta.
    model_output: A structure reflecting the losses and metrics produced during
      training on the input dataset.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()


def _build_server_init_fn(
    model_fn: ModelFn,
    server_optimizer_fn: OptimizerFn,
    aggregation_init_computation: computation_base.Computation,
    broadcast_init_computation: computation_base.Computation,
) -> computation_base.Computation:
  """Builds a `tff.Computation` that returns initial server state.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.reconstruction.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    aggregation_init_computation: A `tff.Computation` which initializes state
      for the aggregation process which will perform aggregation from clients to
      server.
    broadcast_init_computation: A `tff.Computation` which initializes state for
      the process that broadcasts the model weights on the server to the
      clients.

  Returns:
    A `tff.Computation` that returns an initial
      `tff.learning.framework.ServerState`.
  """

  @computations.tf_computation
  def server_init_tf():
    """Initialize the TensorFlow-only portions of the server state."""
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = reconstruction_utils.create_optimizer_vars(
        model, server_optimizer)
    return reconstruction_utils.get_global_variables(
        model), server_optimizer_vars

  @computations.federated_computation()
  def server_init_tff():
    """Returns a state placed at `tff.SERVER`."""
    tf_init_tuple = intrinsics.federated_eval(server_init_tf, placements.SERVER)
    return intrinsics.federated_zip(
        optimizer_utils.ServerState(
            model=tf_init_tuple[0],
            optimizer_state=tf_init_tuple[1],
            delta_aggregate_state=aggregation_init_computation(),
            model_broadcast_state=broadcast_init_computation()))

  return server_init_tff


def _build_server_update_fn(
    model_fn: ModelFn,
    server_optimizer_fn: OptimizerFn,
    server_state_type: computation_types.Type,
    model_weights_type: computation_types.Type,
    aggregator_state_type: computation_types.Type,
    broadcaster_state_type: computation_types.Type,
) -> computation_base.Computation:
  """Builds a `tff.Computation` that updates `ServerState`.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.reconstruction.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    server_state_type: Type of server state.
    model_weights_type: Type of model weights.
    aggregator_state_type: Type of the `state` element of the
      `tff.templates.AggregationProcess` used to perform aggregation.
    broadcaster_state_type: Type of the `state` element of the
      `tff.templates.MeasuredProcess` used to perform broadcasting.

  Returns:
    A `tff.Computation` that updates `ServerState`.
  """

  @computations.tf_computation(server_state_type, model_weights_type.trainable,
                               aggregator_state_type, broadcaster_state_type)
  @tf.function
  def server_update(server_state, weights_delta, aggregator_state,
                    broadcaster_state):
    """Updates the `server_state` based on `weights_delta`.

    Args:
      server_state: A `tff.learning.framework.ServerState`, the state to be
        updated.
      weights_delta: The model delta in global trainable variables from clients.
      aggregator_state: The state of the aggregator after performing
        aggregation.
      broadcaster_state: The state of the broadcaster after broadcasting.

    Returns:
      The updated `tff.learning.framework.ServerState`.
    """
    with tf.init_scope():
      model = model_fn()
      server_optimizer = server_optimizer_fn()
      # Create optimizer variables so we have a place to assign the optimizer's
      # state.
      server_optimizer_vars = reconstruction_utils.create_optimizer_vars(
          model, server_optimizer)

    global_model_weights = reconstruction_utils.get_global_variables(model)
    # Initialize the model with the current state.
    tf.nest.map_structure(lambda a, b: a.assign(b),
                          (global_model_weights, server_optimizer_vars),
                          (server_state.model, server_state.optimizer_state))

    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    # We ignore the update if the weights_delta is non finite.
    if tf.equal(has_non_finite_weight, 0):
      grads_and_vars = tf.nest.map_structure(
          lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
          tf.nest.flatten(global_model_weights.trainable))
      server_optimizer.apply_gradients(grads_and_vars, name='server_update')

    # Create a new state based on the updated model.
    return structure.update_struct(
        server_state,
        model=global_model_weights,
        optimizer_state=server_optimizer_vars,
        model_broadcast_state=broadcaster_state,
        delta_aggregate_state=aggregator_state,
    )

  return server_update


def _build_client_update_fn(
    model_fn: ModelFn,
    *,  # Callers should use keyword args for below.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn],
    dataset_type: computation_types.SequenceType,
    model_weights_type: computation_types.Type,
    client_optimizer_fn: OptimizerFn,
    reconstruction_optimizer_fn: OptimizerFn,
    dataset_split_fn: reconstruction_utils.DatasetSplitFn,
    client_weighting: client_weight_lib.ClientWeightType,
) -> computation_base.Computation:
  """Builds a `tff.Computation` for local reconstruction and update.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.reconstruction.Model`.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      compute local model updates during reconstruction and post-reconstruction
      and evaluate the model during training. The final loss metric is the
      example-weighted mean loss across batches and across clients. The loss
      metric does not include reconstruction batches in the loss.
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. Metrics results are computed locally as described
      by the metric, and are aggregated across clients as in
      `federated_aggregate_keras_metric`. If None, no metrics are applied.
      Metrics are not computed on reconstruction batches.
    dataset_type: Type of TF dataset.
    model_weights_type: Type of model weights.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for training the model weights on the
      client post-reconstruction.
    reconstruction_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for reconstructing the local variables
      with global variables frozen. This optimizer is used before the one given
      by client_optimizer_fn.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a client
      dataset and producing two TF datasets. The first is iterated over during
      reconstruction, and the second is iterated over post-reconstruction. This
      can be used to preprocess datasets to e.g. iterate over them for multiple
      epochs or use disjoint data for reconstruction and post-reconstruction.
    client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
      built-in weighting method, or a callable that takes the local metrics of
      the model and returns a tensor that provides the weight in the federated
      average of model deltas.

  Returns:
    A `tff.Computation` for the local client update.
  """

  @computations.tf_computation(dataset_type, model_weights_type)
  @tf.function
  def client_update(dataset, initial_model_weights):
    """Performs client local model optimization.

    Args:
      dataset: A `tf.data.Dataset` that provides training examples.
      initial_model_weights: A `tff.learning.ModelWeights` containing the
        starting global trainable and non-trainable weights.

    Returns:
      A `ClientOutput`.
    """
    with tf.init_scope():
      model = model_fn()
      client_optimizer = client_optimizer_fn()
      reconstruction_optimizer = reconstruction_optimizer_fn()

      metrics = []
      if metrics_fn is not None:
        metrics.extend(metrics_fn())
      # To be used to calculate example-weighted mean across batches and
      # clients.
      metrics.append(keras_utils.MeanLossMetric(loss_fn()))
      # To be used to calculate batch loss for model updates.
      client_loss = loss_fn()

    global_model_weights = reconstruction_utils.get_global_variables(model)
    local_model_weights = reconstruction_utils.get_local_variables(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), global_model_weights,
                          initial_model_weights)

    @tf.function
    def reconstruction_reduce_fn(num_examples_sum, batch):
      """Runs reconstruction training on local client batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)
        batch_loss = client_loss(
            y_true=output.labels, y_pred=output.predictions)

      gradients = tape.gradient(batch_loss, local_model_weights.trainable)
      reconstruction_optimizer.apply_gradients(
          zip(gradients, local_model_weights.trainable))

      return num_examples_sum + output.num_examples

    @tf.function
    def train_reduce_fn(num_examples_sum, batch):
      """Runs one step of client optimizer on local client batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)
        batch_loss = client_loss(
            y_true=output.labels, y_pred=output.predictions)

      gradients = tape.gradient(batch_loss, global_model_weights.trainable)
      client_optimizer.apply_gradients(
          zip(gradients, global_model_weights.trainable))

      # Update each metric.
      for metric in metrics:
        metric.update_state(y_true=output.labels, y_pred=output.predictions)

      return num_examples_sum + output.num_examples

    recon_dataset, post_recon_dataset = dataset_split_fn(dataset)

    # If needed, do reconstruction, training the local variables while keeping
    # the global ones frozen.
    if local_model_weights.trainable:
      # Ignore output number of examples used in reconstruction, since this
      # isn't included in `client_weight`.
      recon_dataset.reduce(
          initial_state=tf.constant(0), reduce_func=reconstruction_reduce_fn)

    # Train the global variables, keeping local variables frozen.
    num_examples_sum = post_recon_dataset.reduce(
        initial_state=tf.constant(0), reduce_func=train_reduce_fn)

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          global_model_weights.trainable,
                                          initial_model_weights.trainable)

    # We ignore the update if the weights_delta is non finite.
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    model_local_outputs = keras_utils.read_metric_variables(metrics)

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0.0, dtype=tf.float32)
    elif client_weighting is client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      client_weight = tf.cast(num_examples_sum, dtype=tf.float32)
    elif client_weighting is client_weight_lib.ClientWeighting.UNIFORM:
      client_weight = tf.constant(1.0, dtype=tf.float32)
    else:
      client_weight = client_weighting(model_local_outputs)

    return ClientOutput(weights_delta, client_weight, model_local_outputs)

  return client_update


def _build_run_one_round_fn(
    server_update_fn: computation_base.Computation,
    client_update_fn: computation_base.Computation,
    federated_output_computation: computation_base.Computation,
    federated_server_state_type: computation_types.FederatedType,
    federated_dataset_type: computation_types.FederatedType,
    aggregation_process: aggregation_process_lib.AggregationProcess,
    broadcast_process: measured_process_lib.MeasuredProcess,
) -> computation_base.Computation:
  """Builds a `tff.Computation` for a round of training.

  Args:
    server_update_fn: A function for updates in the server.
    client_update_fn: A function for updates in the clients.
    federated_output_computation: A `tff.Computation` for aggregating local
      model outputs across clients.
    federated_server_state_type: A `tff.FederatedType` whose `member` attribute
      is a `tff.Type` for the server state.
    federated_dataset_type: A `tff.FederatedType` whose `member` attribute is a
      `tff.SequenceType` for the federated dataset.
    aggregation_process: Instance of `tff.templates.AggregationProcess` to
      perform aggregation during the round.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients.

  Returns:
    A `tff.Computation` for a round of training.
  """

  @computations.federated_computation(federated_server_state_type,
                                      federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `tff.learning.framework.ServerState` with placement
        `tff.SERVER`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `tff.learning.framework.ServerState` and aggregated
        metrics.
    """
    broadcast_output = broadcast_process.next(
        server_state.model_broadcast_state, server_state.model)

    client_outputs = intrinsics.federated_map(
        client_update_fn, (federated_dataset, broadcast_output.result))

    if aggregation_process.is_weighted:
      aggregation_output = aggregation_process.next(
          server_state.delta_aggregate_state,
          client_outputs.weights_delta,
          weight=client_outputs.client_weight)
    else:
      aggregation_output = aggregation_process.next(
          server_state.delta_aggregate_state, client_outputs.weights_delta)

    round_model_delta = aggregation_output.result

    server_state = intrinsics.federated_map(
        server_update_fn, (server_state, round_model_delta,
                           aggregation_output.state, broadcast_output.state))

    aggregated_model_outputs = federated_output_computation(
        client_outputs.model_output)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            broadcast=broadcast_output.measurements,
            aggregation=aggregation_output.measurements,
            train=aggregated_model_outputs))
    return server_state, measurements

  return run_one_round


def _is_valid_broadcast_process(
    process: measured_process_lib.MeasuredProcess) -> bool:
  """Validates a `MeasuredProcess` adheres to the broadcast signature.

  A valid broadcast process is one whose argument is placed at `SERVER` and
  whose output is placed at `CLIENTS`.

  Args:
    process: A measured process to validate.

  Returns:
    `True` iff the process is a validate broadcast process, otherwise `False`.
  """
  init_type = process.initialize.type_signature
  next_type = process.next.type_signature
  is_valid_stateful_process = (
      init_type.result.placement is placements.SERVER and
      next_type.parameter[0].placement is placements.SERVER and
      next_type.result.state.placement is placements.SERVER and
      next_type.result.measurements.placement is placements.SERVER)
  return (isinstance(process, measured_process_lib.MeasuredProcess) and
          is_valid_stateful_process and
          next_type.parameter[1].placement is placements.SERVER and
          next_type.result.result.placement is placements.CLIENTS)


def _instantiate_aggregation_process(
    aggregation_factory,
    model_weights_type) -> aggregation_process_lib.AggregationProcess:
  """Constructs aggregation process given factory, checking compatibilty."""
  if aggregation_factory is None:
    aggregation_factory = mean.MeanFactory()
  py_typecheck.check_type(aggregation_factory,
                          factory.AggregationFactory.__args__)

  # We give precedence to unweighted aggregation.
  if isinstance(aggregation_factory, factory.UnweightedAggregationFactory):
    aggregation_process = aggregation_factory.create(
        model_weights_type.trainable)
  elif isinstance(aggregation_factory, factory.WeightedAggregationFactory):
    aggregation_process = aggregation_factory.create(
        model_weights_type.trainable, computation_types.TensorType(tf.float32))
  else:
    raise ValueError('Unknown type of aggregation factory: {}'.format(
        type(aggregation_factory)))

  process_signature = aggregation_process.next.type_signature
  input_client_value_type = process_signature.parameter[1]
  result_server_value_type = process_signature.result[1]
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError('`aggregation_factory` does not produce a '
                    'compatible `AggregationProcess`. The processes must '
                    'retain the type structure of the inputs on the '
                    f'server, but got {input_client_value_type.member} != '
                    f'{result_server_value_type.member}.')

  return aggregation_process


# TODO(b/192094313): refactor to accept tff.learning.Optimizer arguments
def build_training_process(
    model_fn: ModelFn,
    *,  # Callers pass below args by name.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn] = None,
    server_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 1.0),
    client_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 0.1),
    reconstruction_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 0.1),
    dataset_split_fn: Optional[reconstruction_utils.DatasetSplitFn] = None,
    client_weighting: Optional[client_weight_lib.ClientWeightType] = None,
    broadcast_process: Optional[measured_process_lib.MeasuredProcess] = None,
    aggregation_factory: Optional[AggregationFactory] = None,
) -> iterative_process_lib.IterativeProcess:
  """Builds the IterativeProcess for optimization using FedRecon.

  Returns a `tff.templates.IterativeProcess` for Federated Reconstruction. On
  the client, computation can be divided into two stages: (1) reconstruction of
  local variables and (2) training of global variables.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.reconstruction.Model`. This method must *not* capture
      Tensorflow tensors or variables and use them. must be constructed entirely
      from scratch on each invocation, returning the same pre-constructed model
      each call will result in an error.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      compute local model updates during reconstruction and post-reconstruction
      and evaluate the model during training. The final loss metric is the
      example-weighted mean loss across batches and across clients. The loss
      metric does not include reconstruction batches in the loss.
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. Metrics results are computed locally as described
      by the metric, and are aggregated across clients as in
      `federated_aggregate_keras_metric`. If None, no metrics are applied.
      Metrics are not computed on reconstruction batches.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for applying updates to the global model
      on the server.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for local client training after
      reconstruction.
    reconstruction_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` used to reconstruct the local variables,
      with the global ones frozen, or the first stage described above.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a single
      TF dataset and producing two TF datasets. The first is iterated over
      during reconstruction, and the second is iterated over
      post-reconstruction. This can be used to preprocess datasets to e.g.
      iterate over them for multiple epochs or use disjoint data for
      reconstruction and post-reconstruction. If None, split client data in half
      for each user, using one half for reconstruction and the other for
      evaluation. See `reconstruction_utils.build_dataset_split_fn` for options.
    client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
      built-in weighting method, or a callable that takes the local metrics of
      the model and returns a tensor that provides the weight in the federated
      average of model deltas. If None, defaults to weighting by number of
      examples.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`. If set to default None,
      the server model is broadcast to the clients using the default
      `tff.federated_broadcast`.
    aggregation_factory: An optional instance of
      `tff.aggregators.WeightedAggregationFactory` or
      `tff.aggregators.UnweightedAggregationFactory` determining the method of
      aggregation to perform. If unspecified, uses a default
      `tff.aggregators.MeanFactory` which computes a stateless mean across
      clients (weighted depending on `client_weighting`).

  Raises:
    TypeError: If `broadcast_process` does not have the expected signature.
    TypeError: If `aggregation_factory` does not have the expected signature.
    ValueError: If  `aggregation_factory` is not a
      `tff.aggregators.WeightedAggregationFactory` or a
      `tff.aggregators.UnweightedAggregationFactory`.
    ValueError: If `aggregation_factory` is a
      `tff.aggregators.UnweightedAggregationFactory` but `client_weighting` is
      not `tff.learning.ClientWeighting.UNIFORM`.

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  with tf.Graph().as_default():
    throwaway_model_for_metadata = model_fn()

  model_weights_type = type_conversions.type_from_tensors(
      reconstruction_utils.get_global_variables(throwaway_model_for_metadata))

  if client_weighting is None:
    client_weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES
  if (isinstance(aggregation_factory, factory.UnweightedAggregationFactory) and
      client_weighting is not client_weight_lib.ClientWeighting.UNIFORM):
    raise ValueError(f'Expected `tff.learning.ClientWeighting.UNIFORM` client '
                     f'weighting with unweighted aggregator, instead got '
                     f'{client_weighting}')

  if broadcast_process is None:
    broadcast_process = optimizer_utils.build_stateless_broadcaster(
        model_weights_type=model_weights_type)
  if not _is_valid_broadcast_process(broadcast_process):
    raise TypeError(
        'broadcast_process type signature does not conform to expected '
        'signature (<state@S, input@S> -> <state@S, result@C, measurements@S>).'
        ' Got: {t}'.format(t=broadcast_process.next.type_signature))
  broadcaster_state_type = (
      broadcast_process.initialize.type_signature.result.member)

  aggregation_process = _instantiate_aggregation_process(
      aggregation_factory, model_weights_type)
  aggregator_state_type = (
      aggregation_process.initialize.type_signature.result.member)

  server_init_tff = _build_server_init_fn(model_fn, server_optimizer_fn,
                                          aggregation_process.initialize,
                                          broadcast_process.initialize)
  server_state_type = server_init_tff.type_signature.result.member

  server_update_fn = _build_server_update_fn(
      model_fn,
      server_optimizer_fn,
      server_state_type,
      server_state_type.model,
      aggregator_state_type=aggregator_state_type,
      broadcaster_state_type=broadcaster_state_type)

  dataset_type = computation_types.SequenceType(
      throwaway_model_for_metadata.input_spec)
  if dataset_split_fn is None:
    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        split_dataset=True)
  client_update_fn = _build_client_update_fn(
      model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      dataset_type=dataset_type,
      model_weights_type=server_state_type.model,
      client_optimizer_fn=client_optimizer_fn,
      reconstruction_optimizer_fn=reconstruction_optimizer_fn,
      dataset_split_fn=dataset_split_fn,
      client_weighting=client_weighting)

  federated_server_state_type = computation_types.at_server(server_state_type)
  federated_dataset_type = computation_types.at_clients(dataset_type)
  # Create placeholder metrics to produce a corresponding federated output
  # computation.
  metrics = []
  if metrics_fn is not None:
    metrics.extend(metrics_fn())
  metrics.append(keras_utils.MeanLossMetric(loss_fn()))
  federated_output_computation = (
      keras_utils.federated_output_computation_from_metrics(metrics))

  run_one_round_tff = _build_run_one_round_fn(
      server_update_fn,
      client_update_fn,
      federated_output_computation,
      federated_server_state_type,
      federated_dataset_type,
      aggregation_process=aggregation_process,
      broadcast_process=broadcast_process,
  )

  process = iterative_process_lib.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round_tff)

  @computations.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  process.get_model_weights = get_model_weights
  return process
