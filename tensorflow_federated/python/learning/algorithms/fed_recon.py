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
`tff.learning.models.ReconstructionModel`s.
`tff.learning.models.ReconstructionModel`s introduce a partition of variables
into global variables and local variables.

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
    Sushant Prakash. NeurIPS 2021. https://arxiv.org/abs/2102.03448
"""

import collections
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process as measured_process_lib
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import tensor_utils
from tensorflow_federated.python.learning.metrics import aggregator as metrics_aggregators
from tensorflow_federated.python.learning.metrics import keras_finalizer as metrics_finalizers_lib
from tensorflow_federated.python.learning.models import reconstruction_model
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import apply_optimizer_finalizer
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import learning_process

# Type aliases for readability.
ReconstructionModel = reconstruction_model.ReconstructionModel
AggregationFactory = Union[
    factory.WeightedAggregationFactory, factory.UnweightedAggregationFactory
]
LossFn = Callable[[], tf.keras.losses.Loss]
MetricsFn = Callable[[], list[tf.keras.metrics.Metric]]
MetricFinalizersType = collections.OrderedDict[str, Callable[[Any], Any]]
ModelFn = Callable[[], ReconstructionModel]


# TODO: b/230109170 - re-enable pylint after fixing bug.
# pylint: disable=g-bare-generic
def _build_reconstruction_client_work(
    model_fn: ModelFn,
    *,  # Callers should use keyword args for below.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn],
    client_optimizer_fn: optimizer_base.Optimizer,
    reconstruction_optimizer_fn: optimizer_base.Optimizer,
    dataset_split_fn: reconstruction_model.ReconstructionDatasetSplitFn,
    client_weighting: client_weight_lib.ClientWeightType,
    metrics_aggregator: Callable[
        [MetricFinalizersType, computation_types.StructWithPythonType],
        computation_base.Computation,
    ],
) -> client_works.ClientWorkProcess:
  # pylint: enable=g-bare-generic
  """Builds a `tff.Computation` for local reconstruction and update.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.ReconstructionModel`.
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
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer` for training the
      model weights on the client post-reconstruction.
    reconstruction_optimizer_fn: A `tff.learning.optimizers.Optimizer` for
      reconstructing the local variables with global variables frozen. This
      optimizer is used before the one given by `client_optimizer_fn`.
    dataset_split_fn: A `tff.learning.models.ReconstructionDatasetSplitFn`
      taking in aclient dataset and producing two TF datasets. The first is
      iterated overduring reconstruction, and the second is iterated over
      post-reconstruction. This can be used to preprocess datasets to e.g.
      iterate over them for multiple epochs or use disjoint data for
      reconstruction and post-reconstruction.
    client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
      built-in weighting method, or a callable that takes the local metrics of
      the model and returns a tensor that provides the weight in the federated
      average of model deltas.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics.

  Returns:
    A `tff.learning.templates.ClientWorkProcess` for the local client update.
  """
  with tf.Graph().as_default():
    model_for_metadata = model_fn()
  model_weights_type = reconstruction_model.global_weights_type_from_model(
      model_for_metadata
  )
  element_type = computation_types.tensorflow_to_type(
      model_for_metadata.input_spec
  )
  dataset_type = computation_types.SequenceType(element_type)

  @federated_computation.federated_computation
  def initialize():
    # FedRecon client work is stateless (empty tuple).
    return intrinsics.federated_value((), placements.SERVER)

  # Metric finalizer functions that will be populated while tracing
  # `client_update` and used later in the federated computation.
  metric_finalizers: collections.OrderedDict[
      str, metrics_finalizers_lib.KerasMetricFinalizer
  ] = collections.OrderedDict()

  @tensorflow_computation.tf_computation(model_weights_type, dataset_type)
  @tf.function
  def client_update(initial_model_weights, dataset):
    """Performs client local model optimization.

    Args:
      initial_model_weights: A `tff.learning.models.ModelWeights` containing the
        starting global trainable and non-trainable weights.
      dataset: A `tf.data.Dataset` that provides training examples.

    Returns:
      A `ClientOutput`.
    """
    with tf.init_scope():
      model = model_fn()
      loss_metric = tf.keras.metrics.MeanMetricWrapper(loss_fn(), name='loss')
      if metrics_fn is None:
        metrics = [loss_metric]
      else:
        metrics = metrics_fn() + [loss_metric]
      nonlocal metric_finalizers
      for metric in metrics:
        if metric.name in metric_finalizers:
          raise ValueError(
              f'Duplicate metric name detected: {metric.name}. '
              f'Already saw metrics {list(metric_finalizers.keys())}'
          )
        metric_finalizers[metric.name] = (
            metrics_finalizers_lib.create_keras_metric_finalizer(metric)
        )
      # To be used to calculate batch loss for model updates.
      client_loss = loss_fn()

    global_model_weights = ReconstructionModel.get_global_variables(model)
    local_model_weights = ReconstructionModel.get_local_variables(model)
    tf.nest.map_structure(
        lambda a, b: a.assign(b), global_model_weights, initial_model_weights
    )
    client_optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        client_optimizer_fn,
        global_model_weights.trainable,
        disjoint_init_and_next=False,
    )
    reconstruction_optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        reconstruction_optimizer_fn,
        local_model_weights.trainable,
        disjoint_init_and_next=False,
    )

    @tf.function
    def reconstruction_reduce_fn(state, batch):
      """Runs reconstruction training on local client batch."""
      num_examples_sum, optimizer_state = state
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)
        batch_loss = client_loss(
            y_true=output.labels, y_pred=output.predictions
        )

      gradients = tape.gradient(batch_loss, local_model_weights.trainable)
      updated_optimizer_state, updated_weights = reconstruction_optimizer.next(
          optimizer_state,
          tuple(local_model_weights.trainable),
          tuple(gradients),
      )
      if not isinstance(
          reconstruction_optimizer, keras_optimizer.KerasOptimizer
      ):
        # TFF optimizers require assigning the updated tensors back into the
        # model variables. (With Keras optimizers we don't need to do this,
        # because Keras optimizers mutate the model variables within the `next`
        # step.)
        tf.nest.map_structure(
            lambda a, b: a.assign(b),
            local_model_weights.trainable,
            list(updated_weights),
        )

      return num_examples_sum + output.num_examples, updated_optimizer_state

    @tf.function
    def train_reduce_fn(state, batch):
      """Runs one step of client optimizer on local client batch."""
      num_examples_sum, optimizer_state = state
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)
        batch_loss = client_loss(
            y_true=output.labels, y_pred=output.predictions
        )
      gradients = tape.gradient(batch_loss, global_model_weights.trainable)
      updated_optimizer_state, updated_weights = client_optimizer.next(
          optimizer_state,
          tuple(global_model_weights.trainable),
          tuple(gradients),
      )
      if not isinstance(client_optimizer, keras_optimizer.KerasOptimizer):
        # Keras optimizer mutates model variables within the `next` step.
        tf.nest.map_structure(
            lambda a, b: a.assign(b),
            global_model_weights.trainable,
            list(updated_weights),
        )
      # Update each metric.
      for metric in metrics:
        metric.update_state(y_true=output.labels, y_pred=output.predictions)
      return num_examples_sum + output.num_examples, updated_optimizer_state

    recon_dataset, post_recon_dataset = dataset_split_fn(dataset)

    # If needed, do reconstruction, training the local variables while keeping
    # the global ones frozen.
    if local_model_weights.trainable:
      # Ignore output number of examples used in reconstruction, since this
      # isn't included in `client_weight`.
      def initial_state_reconstruction_reduce():
        trainable_tensor_specs = tf.nest.map_structure(
            lambda v: tf.TensorSpec(v.shape, v.dtype),
            local_model_weights.trainable,
        )
        # We convert the trainable specs to tuple, as the data iteration pattern
        # might try to stack the tensors in a list.
        initial_num_examples = tf.constant(0)
        return initial_num_examples, reconstruction_optimizer.initialize(
            tuple(trainable_tensor_specs)
        )

      recon_dataset.reduce(
          initial_state=initial_state_reconstruction_reduce(),
          reduce_func=reconstruction_reduce_fn,
      )

    # Train the global variables, keeping local variables frozen.
    def initial_state_train_reduce():
      trainable_tensor_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype),
          global_model_weights.trainable,
      )
      # We convert the trainable specs to tuple, as the data iteration pattern
      # might try to stack the tensors in a list.
      initial_num_examples = tf.constant(0)
      return initial_num_examples, client_optimizer.initialize(
          tuple(trainable_tensor_specs)
      )

    num_examples_sum, _ = post_recon_dataset.reduce(
        initial_state=initial_state_train_reduce(), reduce_func=train_reduce_fn
    )

    # The finalizer component expects model updates computed as `initial -
    # final`.
    weights_delta = tf.nest.map_structure(
        lambda a, b: a - b,
        initial_model_weights.trainable,
        global_model_weights.trainable,
    )

    # We ignore the update if the weights_delta is non finite.
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta)
    )

    unfinalized_metrics = ReconstructionModel.read_metric_variables(metrics)

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0.0, dtype=tf.float32)
    elif client_weighting is client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      client_weight = tf.cast(num_examples_sum, dtype=tf.float32)
    elif client_weighting is client_weight_lib.ClientWeighting.UNIFORM:
      client_weight = tf.constant(1.0, dtype=tf.float32)
    else:
      client_weight = client_weighting(unfinalized_metrics)

    return (
        client_works.ClientResult(weights_delta, client_weight),
        unfinalized_metrics,
    )

  @federated_computation.federated_computation(
      computation_types.FederatedType((), placements.SERVER),
      computation_types.FederatedType(model_weights_type, placements.CLIENTS),
      computation_types.FederatedType(dataset_type, placements.CLIENTS),
  )
  def next_fn(state, incoming_model_weights, client_datasets):
    del state  # Unused.
    client_result, unfinalized_metrics = intrinsics.federated_map(
        client_update, (incoming_model_weights, client_datasets)
    )
    metrics_aggregation_computation = metrics_aggregator(
        metric_finalizers, unfinalized_metrics.type_signature.member
    )
    finalized_metrics = intrinsics.federated_zip(
        collections.OrderedDict(
            train=metrics_aggregation_computation(unfinalized_metrics)
        )
    )
    return measured_process_lib.MeasuredProcessOutput(
        state=intrinsics.federated_value((), placements.SERVER),
        result=client_result,
        measurements=finalized_metrics,
    )

  return client_works.ClientWorkProcess(
      initialize_fn=initialize,
      next_fn=next_fn,
      # TODO: b/257966299 - expose get/set_hparams functions to that FedRecon
      # parameters can be explored in black-box hparam tuners.
      get_hparams_fn=None,
      set_hparams_fn=None,
  )


# TODO: b/230109170 - re-enable pylint after fixing bug.
# pylint: disable=g-bare-generic
def build_fed_recon(
    model_fn: Callable[[], ReconstructionModel],
    *,  # Callers pass below args by name.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn] = None,
    server_optimizer_fn: optimizer_base.Optimizer = sgdm.build_sgdm(
        learning_rate=1.0
    ),
    client_optimizer_fn: optimizer_base.Optimizer = sgdm.build_sgdm(
        learning_rate=0.1
    ),
    reconstruction_optimizer_fn: optimizer_base.Optimizer = sgdm.build_sgdm(
        learning_rate=0.1
    ),
    dataset_split_fn: Optional[
        reconstruction_model.ReconstructionDatasetSplitFn
    ] = None,
    client_weighting: Optional[client_weight_lib.ClientWeightType] = None,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator_factory: Optional[AggregationFactory] = None,
    metrics_aggregator: Optional[
        Callable[
            [MetricFinalizersType, computation_types.StructWithPythonType],
            computation_base.Computation,
        ]
    ] = metrics_aggregators.sum_then_finalize,
) -> learning_process.LearningProcess:
  # pylint: enable=g-bare-generic
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
    server_optimizer_fn:  A `tff.learning.optimizers.Optimizer`for applying
      updates to the global model on the server.
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer` for local client
      training after reconstruction.
    reconstruction_optimizer_fn: A `tff.learning.optimizers.Optimizer` used to
      reconstruct the local variables, with the global ones frozen, or the first
      stage described above.
    dataset_split_fn: A `tff.learning.models.ReconstructionDatasetSplitFn`
      taking in a single TF dataset and producing two TF datasets. The first is
      iterated over during reconstruction, and the second is iterated over
      post-reconstruction. This can be used to preprocess datasets to e.g.
      iterate over them for multiple epochs or use disjoint data for
      reconstruction and post-reconstruction. If None, split client data in half
      for each user, using one half for reconstruction and the other for
      evaluation. See
      `tff.learning.models.ReconstructionModel.build_dataset_split_fn` for
      options.
    client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
      built-in weighting method, or a callable that takes the local metrics of
      the model and returns a tensor that provides the weight in the federated
      average of model deltas. If None, defaults to weighting by number of
      examples.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via
      `tff.learning.templates.build_broadcast_process`.
    model_aggregator_factory: An optional instance of
      `tff.aggregators.WeightedAggregationFactory` or
      `tff.aggregators.UnweightedAggregationFactory` determining the method of
      aggregation to perform. If unspecified, uses a default
      `tff.aggregators.MeanFactory` which computes a stateless mean across
      clients (weighted depending on `client_weighting`).
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics. If
      `None`, this is set to `tff.learning.metrics.sum_then_finalize`.

  Returns:
    A `tff.learning.templates.LearningProcess`.

  Raises:
    TypeError: If `model_fn` does not return instances of
      `tff.learning.models.ReconstructionModel`.
    ValueError: If `model_aggregator_factory` is a
      `tff.aggregators.UnweightedAggregationFactory` and `client_weighting` is
      any value other than `tff.learning.ClientWeighting.UNIFORM`.
  """

  @tensorflow_computation.tf_computation
  def build_initial_model_weights():
    model = model_fn()
    if not isinstance(model, ReconstructionModel):
      raise TypeError(
          '`model_fn` must return an instance of '
          f'`tff.learning.models.ReconstructionModel`. Got a: {type(model)}'
      )
    return ReconstructionModel.get_global_variables(model)

  model_weights_type = build_initial_model_weights.type_signature.result

  if client_weighting is None:
    client_weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES
  if (
      isinstance(model_aggregator_factory, factory.UnweightedAggregationFactory)
      and client_weighting is not client_weight_lib.ClientWeighting.UNIFORM
  ):
    raise ValueError(
        'Expected `tff.learning.ClientWeighting.UNIFORM` client '
        'weighting with unweighted aggregator, instead got '
        f'{client_weighting}'
    )

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)

  if model_aggregator_factory is None:
    model_aggregator_factory = mean.MeanFactory()
  if isinstance(model_aggregator_factory, factory.UnweightedAggregationFactory):
    model_aggregator_factory = factory_utils.as_weighted_aggregator(
        model_aggregator_factory
    )
  py_typecheck.check_type(
      model_aggregator_factory, factory.WeightedAggregationFactory
  )
  model_aggregator = model_aggregator_factory.create(
      model_weights_type.trainable, computation_types.TensorType(np.float32)
  )

  if dataset_split_fn is None:
    dataset_split_fn = ReconstructionModel.build_dataset_split_fn(
        split_dataset=True
    )

  client_work = _build_reconstruction_client_work(
      model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      client_optimizer_fn=client_optimizer_fn,
      reconstruction_optimizer_fn=reconstruction_optimizer_fn,
      dataset_split_fn=dataset_split_fn,
      client_weighting=client_weighting,
      metrics_aggregator=metrics_aggregator,
  )

  finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
      optimizer_fn=server_optimizer_fn, model_weights_type=model_weights_type
  )

  return composers.compose_learning_process(
      initial_model_weights_fn=build_initial_model_weights,
      model_weights_distributor=model_distributor,
      client_work=client_work,
      model_update_aggregator=model_aggregator,
      model_finalizer=finalizer,
  )
