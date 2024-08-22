# Copyright 2018, The TensorFlow Federated Authors.
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
"""An implementation of model delta `ClientWork` template.

The term 'model delta' refers to difference between the model weights at the
start and the end of local training. This is for used for example in the
implementation of the generalized FedAvg algorithm implemented in
`tff.learning.algorithms.build_weighted_fed_avg`.
"""

import collections
from collections.abc import Callable, Mapping
from typing import Any, Optional

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning import tensor_utils
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import client_works


# TODO: b/213433744 - Make this method private.
def build_model_delta_update_with_tff_optimizer(
    model_fn: Callable[[], variable.VariableModel],
    *,
    weighting: client_weight_lib.ClientWeighting,
    loop_implementation: loop_builder.LoopImplementation,
):
  """Creates client update logic in FedAvg using a TFF optimizer.

  Args:
    model_fn: A no-arg callable returning a `tff.learning.models.VariableModel`.
    weighting: A `tff.learning.ClientWeighting` value.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tf.function`.
  """
  model = model_fn()
  dataset_reduce_fn = loop_builder.build_training_loop(
      loop_implementation=loop_implementation
  )

  @tf.function
  def client_update(optimizer, initial_weights, data, optimizer_hparams=None):
    model_weights = model_weights_lib.ModelWeights.from_model(model)
    tf.nest.map_structure(
        lambda a, b: a.assign(b), model_weights, initial_weights
    )

    def reduce_fn(state, batch):
      """Trains a `tff.learning.models.VariableModel` on a batch of data."""
      num_examples_sum, optimizer_state = state
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model_weights.trainable)
      optimizer_state, updated_weights = optimizer.next(
          optimizer_state,
          tuple(tf.nest.flatten(model_weights.trainable)),
          tuple(tf.nest.flatten(gradients)),
      )
      updated_weights = tf.nest.pack_sequence_as(
          model_weights.trainable, updated_weights
      )
      tf.nest.map_structure(
          lambda a, b: a.assign(b), model_weights.trainable, updated_weights
      )

      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)

      return num_examples_sum, optimizer_state

    def initial_state_for_reduce_fn():
      initial_num_examples = tf.zeros(shape=[], dtype=tf.int64)
      # TODO: b/161529310 - We flatten and convert the trainable specs to tuple,
      # as "for batch in data:" pattern would try to stack the tensors in list.
      trainable_tensor_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype),
          tuple(tf.nest.flatten(model_weights.trainable)),
      )
      # TODO: b/245968233 - Reduce to a single `initialize` call once TFF
      # optimizers can inject hyperparameters upon initialization.
      optimizer_state = optimizer.initialize(trainable_tensor_specs)
      if optimizer_hparams is not None:
        optimizer_state = optimizer.set_hparams(
            optimizer_state, optimizer_hparams
        )
      return (initial_num_examples, optimizer_state)

    num_examples, _ = dataset_reduce_fn(
        reduce_fn, data, initial_state_for_reduce_fn
    )
    client_update = tf.nest.map_structure(
        tf.subtract, initial_weights.trainable, model_weights.trainable
    )
    # TODO: b/122071074 - Consider moving this functionality into
    # tff.federated_mean?
    client_update, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(client_update)
    )
    # If there are any non-finite model weights values, we set the
    # `client_weight` to zero so that this client does not contribute to the
    # aggregated model deltas at the server. We also reset the client local
    # metrics via `model.reset_metrics()`, so that these non-finite training
    # metrics are excluded from the aggregated metrics at the server.
    # TODO: b/327051011 - Add a metric to count the number of clients with
    # non-finite model deltas.
    client_weight = _choose_client_weight(
        weighting, has_non_finite_delta, num_examples
    )
    if has_non_finite_delta > 0:
      model.reset_metrics()
    model_output = model.report_local_unfinalized_metrics()
    return (
        client_works.ClientResult(
            update=client_update, update_weight=client_weight
        ),
        model_output,
    )

  return client_update


def _choose_client_weight(weighting, has_non_finite_delta, num_examples):
  if has_non_finite_delta > 0:
    return tf.constant(0.0, tf.float32)
  else:
    if weighting == client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      return tf.cast(num_examples, tf.float32)
    elif weighting == client_weight_lib.ClientWeighting.UNIFORM:
      return tf.constant(1.0, tf.float32)
    else:
      raise ValueError(f'Unexpected weighting argument: {weighting}')


def build_model_delta_client_work(
    model_fn: Callable[[], variable.VariableModel],
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    *,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for federated averaging.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`. This method must *not* capture
      TensorFlow tensors or variables and use them. The model must be
      constructed entirely from scratch on each invocation, returning the same
      pre-constructed model each call will result in an error.
    optimizer: A `tff.learning.optimizers.Optimizer`.
    client_weighting:  A `tff.learning.ClientWeighting` value.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.models.VariableModel.metric_finalizers()`) returns a
      `tff.Computation` for aggregating the unfinalized metrics.  If `None`,
      this is set to `tff.learning.metrics.sum_then_finalize`.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `ClientWorkProcess`.
  """
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)

  if metrics_aggregator is None:
    metrics_aggregator = aggregator.sum_then_finalize

  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    metrics_aggregation_fn = metrics_aggregator(model.metric_finalizers())
  element_type = computation_types.tensorflow_to_type(model.input_spec)
  data_type = computation_types.SequenceType(element_type)
  weights_type = model_weights_lib.weights_type_from_model(model)

  # We initialize the optimizer for the purposes of extracting its
  # hyperparameters, using a "whimsy" spec.
  whimsy_specs = tf.TensorSpec(shape=(), dtype=tf.float32)
  whimsy_opt_state = optimizer.initialize(whimsy_specs)
  initial_hparams = optimizer.get_hparams(whimsy_opt_state)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value(initial_hparams, placements.SERVER)

  state_type = init_fn.type_signature.result.member
  # In this case, the state is exactly equal to the hyperparameters being
  # used by the underlying optimizer, so their type is the same.
  hparams_type = state_type

  @tensorflow_computation.tf_computation(state_type)
  def get_hparams_fn(state):
    return state

  @tensorflow_computation.tf_computation(state_type, hparams_type)
  def set_hparams_fn(state, hparams):
    del state
    return hparams

  @tensorflow_computation.tf_computation(state_type, weights_type, data_type)
  def client_update_computation(state, initial_model_weights, dataset):
    optimizer_hparams = state
    client_update = build_model_delta_update_with_tff_optimizer(
        model_fn=model_fn,
        weighting=client_weighting,
        loop_implementation=loop_implementation,
    )
    return client_update(
        optimizer, initial_model_weights, dataset, optimizer_hparams
    )

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(weights_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, weights, client_data):
    state_at_clients = intrinsics.federated_broadcast(state)
    client_result, model_outputs = intrinsics.federated_map(
        client_update_computation, (state_at_clients, weights, client_data)
    )
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        state, client_result, measurements
    )

  return client_works.ClientWorkProcess(
      init_fn,
      next_fn,
      get_hparams_fn=get_hparams_fn,
      set_hparams_fn=set_hparams_fn,
  )


def build_functional_model_delta_update(
    model: functional.FunctionalModel,
    *,
    weighting: client_weight_lib.ClientWeighting,
    loop_implementation: loop_builder.LoopImplementation,
):
  """Creates client update logic in FedAvg.

  Args:
    model: A `tff.learning.models.FunctionalModel`.
    weighting: A `tff.learning.ClientWeighting` value.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tf.function`.
  """

  @tf.function
  def client_update_fn(
      optimizer: optimizer_base.Optimizer,
      initial_weights: Any,
      dataset: tf.data.Dataset,
  ):
    initial_trainable_weights = initial_weights[0]
    trainable_tensor_specs = tf.nest.map_structure(
        tf.TensorSpec.from_tensor,
        tuple(tf.nest.flatten(initial_trainable_weights)),
    )

    def reduce_func(training_state, batch):
      model_weights, optimizer_state, metrics_state, num_examples = (
          training_state
      )
      trainable_weights, non_trainable_weights = model_weights
      if isinstance(batch, Mapping):
        x = batch['x']
        y = batch['y']
      else:
        x, y = batch

      with tf.GradientTape() as tape:
        # Must explicitly watch non-tf.Variable tensors.
        tape.watch(trainable_weights)
        batch_output = model.predict_on_batch(model_weights, x, training=True)
        batch_loss = model.loss(output=batch_output, label=y)

      gradients = tape.gradient(batch_loss, trainable_weights)
      optimizer_state, trainable_weights = optimizer.next(
          optimizer_state, trainable_weights, gradients
      )
      predictions = tf.nest.flatten(batch_output)[0]
      batch_num_examples = tf.shape(predictions)[0]
      num_examples += tf.cast(batch_num_examples, tf.int64)
      model_weights = (trainable_weights, non_trainable_weights)

      # TODO: b/272099796 - Update `update_metrics_state` of FunctionalModel
      metrics_state = model.update_metrics_state(
          metrics_state,
          batch_output=variable.BatchOutput(
              loss=batch_loss,
              predictions=batch_output,
              num_examples=batch_num_examples,
          ),
          labels=y,
      )
      return model_weights, optimizer_state, metrics_state, num_examples

    initial_training_state = tf.function(
        lambda: (
            initial_weights,
            optimizer.initialize(trainable_tensor_specs),
            model.initialize_metrics_state(),
            tf.constant(0, tf.int64),  # num_examples
        )
    )

    ds_reduce_fn = loop_builder.build_training_loop(
        loop_implementation=loop_implementation
    )
    final_training_state = ds_reduce_fn(
        reduce_func, dataset, initial_training_state
    )
    model_weights, _, metrics_state, num_examples = final_training_state
    trainable_weights, _ = model_weights
    # After all local batches, compute the delta between the trained model
    # and the initial incoming model weights.
    client_model_update = tf.nest.map_structure(
        tf.subtract, initial_trainable_weights, trainable_weights
    )
    client_model_update, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(client_model_update)
    )
    # If there are any non-finite model weights values, we set the
    # `client_weight` to zero so that this client does not contribute to the
    # aggregated model deltas at the server. We also reset the client local
    # metrics by `initialize_metrics_state`, so that these non-finite training
    # metrics are excluded from the aggregated metrics at the server.
    client_weight = _choose_client_weight(
        weighting, has_non_finite_delta, num_examples
    )
    unfinalized_metrics = (
        model.initialize_metrics_state()
        if has_non_finite_delta > 0
        else metrics_state
    )
    return (
        client_works.ClientResult(
            update=client_model_update, update_weight=client_weight
        ),
        unfinalized_metrics,
    )

  return client_update_fn


def build_functional_model_delta_client_work(
    *,
    model: functional.FunctionalModel,
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for federated averaging.

  This differs from `tff.learning.templates.build_model_delta_client_work` in
  that it only accepts `tff.learning.models.FunctionalModel` and
  `tff.learning.optimizers.Optimizer` type arguments, resulting in TensorFlow
  graphs that do not contain `tf.Variable` operations.

  Args:
    model: A `tff.learning.models.FunctionalModel` to train.
    optimizer: A `tff.learning.optimizers.Optimizer` to use for local, on-client
      optimization.
    client_weighting:  A `tff.learning.ClientWeighting` value.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.models.VariableModel.metric_finalizers()`) and returns a
      `tff.Computation` for aggregating the unfinalized metrics.  If `None`,
      this is set to `tff.learning.metrics.sum_then_finalize`.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `ClientWorkProcess`.
  """
  py_typecheck.check_type(model, functional.FunctionalModel)
  py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  element_type = computation_types.tensorflow_to_type(model.input_spec)
  data_type = computation_types.SequenceType(element_type)

  def ndarray_to_tensorspec(ndarray):
    return tf.TensorSpec(
        shape=ndarray.shape, dtype=tf.dtypes.as_dtype(ndarray.dtype)
    )

  # Wrap in a `ModelWeights` structure that is required by the `finalizer.`
  weights_spec = model_weights_lib.ModelWeights(
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[0]),
      tuple(ndarray_to_tensorspec(w) for w in model.initial_weights[1]),
  )
  weights_type = computation_types.tensorflow_to_type(weights_spec)

  @tensorflow_computation.tf_computation(weights_type, data_type)
  def client_update_computation(initial_model_weights, dataset):
    # Switch to the tuple expected by FunctionalModel.
    initial_model_weights = (
        initial_model_weights.trainable,
        initial_model_weights.non_trainable,
    )
    client_update = build_functional_model_delta_update(
        model=model,
        weighting=client_weighting,
        loop_implementation=loop_implementation,
    )
    return client_update(optimizer, initial_model_weights, dataset)

  @federated_computation.federated_computation
  def init_fn():
    # Empty tuple means "no state" / stateless.
    return intrinsics.federated_value((), placements.SERVER)

  if metrics_aggregator is None:
    metrics_aggregator = aggregator.sum_then_finalize

  @federated_computation.federated_computation(
      computation_types.FederatedType((), placements.SERVER),
      computation_types.FederatedType(weights_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, weights, client_data):
    client_result, unfinalized_metrics = intrinsics.federated_map(
        client_update_computation, (weights, client_data)
    )
    metrics_aggregation_fn = metrics_aggregator(model.finalize_metrics)
    finalized_training_metrics = metrics_aggregation_fn(unfinalized_metrics)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=finalized_training_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)
