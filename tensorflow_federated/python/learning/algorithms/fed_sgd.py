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
"""An implementation of the Federated SGD algorithm.

This is the baseline algorithm from:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from collections.abc import Callable, Mapping
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning import tensor_utils
from tensorflow_federated.python.learning.metrics import aggregator as metric_aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import apply_optimizer_finalizer
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import learning_process


def _build_client_update(
    model: variable.VariableModel,
    loop_implementation: loop_builder.LoopImplementation,
):
  """Creates client update logic for FedSGD.

  Args:
    model: A `tff.learning.models.VariableModel` used to compute gradients.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tf.function`.
  """
  dataset_reduce_fn = loop_builder.build_training_loop(
      loop_implementation=loop_implementation
  )

  @tf.function
  def client_update(initial_weights, dataset):
    model_weights = model_weights_lib.ModelWeights.from_model(model)
    tf.nest.map_structure(
        lambda a, b: a.assign(b), model_weights, initial_weights
    )

    def reduce_fn(state, batch):
      """Runs forward_pass on batch and sums the weighted gradients."""
      accumulated_gradients, num_examples_sum = state

      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      gradients = tape.gradient(
          output.loss,
          model_weights.trainable,
          unconnected_gradients=tf.UnconnectedGradients.ZERO,
      )
      num_examples = tf.cast(output.num_examples, tf.float32)
      accumulated_gradients = tuple(
          accumulator + num_examples * gradient
          for accumulator, gradient in zip(accumulated_gradients, gradients)
      )

      # We may be able to optimize the reduce function to avoid doubling the
      # number of required variables here (e.g. keeping two copies of all
      # gradients). If you're looking to optimize memory usage this might be a
      # place to look.
      return (accumulated_gradients, num_examples_sum + num_examples)

    def _zero_initial_state():
      """Create a tuple of (gradient accumulators, num examples)."""
      return tuple(
          tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
      ), tf.constant(0, dtype=tf.float32)

    gradient_sums, num_examples_sum = dataset_reduce_fn(
        reduce_fn, dataset, _zero_initial_state
    )

    # We now normalize to compute the average gradient over all examples.
    average_gradient = tf.nest.map_structure(
        lambda gradient: gradient / num_examples_sum, gradient_sums
    )

    model_output = model.report_local_unfinalized_metrics()
    average_gradient, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(average_gradient)
    )
    if has_non_finite_delta > 0:
      client_weight = tf.constant(0.0)
    else:
      client_weight = num_examples_sum

    return (
        client_works.ClientResult(
            update=average_gradient, update_weight=client_weight
        ),
        model_output,
    )

  return client_update


def _build_fed_sgd_client_work(
    model_fn: Callable[[], variable.VariableModel],
    metrics_aggregator: types.MetricsAggregatorType,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> client_works.ClientWorkProcess:
  """Creates a `tff.learning.templates.ClientWorkProcess` for federated SGD.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`. This method must *not* capture
      TensorFlow tensors or variables and use them. The model must be
      constructed entirely from scratch on each invocation, returning the same
      pre-constructed model each call will result in an error.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.models.VariableModel.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`),
      and returns a `tff.Computation` for aggregating the unfinalized metrics.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tff.learning.templates.ClientWorkProcess`.
  """
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    metrics_aggregation_fn = metrics_aggregator(
        model.metric_finalizers(),
    )
  element_type = computation_types.tensorflow_to_type(model.input_spec)
  data_type = computation_types.SequenceType(element_type)
  weights_type = model_weights_lib.weights_type_from_model(model)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  @tensorflow_computation.tf_computation(weights_type, data_type)
  def client_update_computation(initial_model_weights, dataset):
    client_update = _build_client_update(
        model_fn(), loop_implementation=loop_implementation
    )
    return client_update(initial_model_weights, dataset)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(weights_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, model_weights, client_data):
    client_result, model_outputs = intrinsics.federated_map(
        client_update_computation, (model_weights, client_data)
    )
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def _build_functional_client_update(
    model: functional.FunctionalModel,
    loop_implementation: loop_builder.LoopImplementation,
) -> Callable[[Any, Any], client_works.ClientResult]:
  """Creates client update logic for FedSGD.

  Args:
    model: A `tff.learning.models.FunctionalModel` used to compute gradients.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tf.function`.
  """
  dataset_reduce_fn = loop_builder.build_training_loop(
      loop_implementation=loop_implementation
  )

  @tf.function
  def client_update(initial_weights, dataset):
    trainable_weights, non_trainable_weights = (
        initial_weights.trainable,
        initial_weights.non_trainable,
    )

    def reduce_fn(state, batch):
      """Runs prediction and loss on batch and sums the weighted gradients."""
      accumulated_gradients, metrics_state, num_examples_sum = state
      if isinstance(batch, Mapping):
        x = batch['x']
        y = batch['y']
      else:
        x, y = batch

      with tf.GradientTape() as tape:
        tape.watch(trainable_weights)
        batch_output = model.predict_on_batch(
            model_weights=(trainable_weights, non_trainable_weights),
            x=x,
            training=True,
        )
        batch_loss = model.loss(output=batch_output, label=y)
        predictions = tf.nest.flatten(batch_output)[0]
        batch_num_examples = tf.shape(predictions)[0]

        gradients = tape.gradient(
            batch_loss,
            trainable_weights,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        num_examples = tf.cast(batch_num_examples, tf.float32)
        accumulated_gradients = tuple(
            accumulator + num_examples * gradient
            for accumulator, gradient in zip(accumulated_gradients, gradients)
        )

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
      return (
          accumulated_gradients,
          metrics_state,
          num_examples_sum + num_examples,
      )

    def _zero_initial_state():
      """Create a tuple of (gradient accumulators, metrics, num examples)."""
      return (
          tuple(tf.nest.map_structure(tf.zeros_like, trainable_weights)),
          model.initialize_metrics_state(),
          tf.constant(0, dtype=tf.float32),
      )

    gradient_sums, metrics_state, num_examples_sum = dataset_reduce_fn(
        reduce_fn, dataset, _zero_initial_state
    )
    # We now normalize to compute the average gradient over all examples.
    average_gradient = tf.nest.map_structure(
        lambda gradient: gradient / num_examples_sum, gradient_sums
    )
    average_gradient, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(average_gradient)
    )
    if has_non_finite_delta > 0:
      client_weight = tf.constant(0.0)
    else:
      client_weight = num_examples_sum
    return (
        client_works.ClientResult(
            update=average_gradient, update_weight=client_weight
        ),
        metrics_state,
    )

  return client_update


def _build_functional_fed_sgd_client_work(
    model: functional.FunctionalModel,
    metrics_aggregator: types.MetricsAggregatorType,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> client_works.ClientWorkProcess:
  """Creates a `tff.learning.templates.ClientWorkProcess` for federated SGD.

  This differs from `_build_fed_sgd_client_work` in that it only accepts
  `tff.learning.models.FunctionalModel`, resulting in TensorFlow
  graphs that do not contain `tf.Variable` operations.

  Args:
    model: A `tff.learning.models.FunctionalModel` to train.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.models.VariableModel.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`),
      and returns a `tff.Computation` for aggregating the unfinalized metrics.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tff.learning.templates.ClientWorkProcess`.
  """
  py_typecheck.check_type(model, functional.FunctionalModel)
  element_type = computation_types.tensorflow_to_type(model.input_spec)
  data_type = computation_types.SequenceType(element_type)

  def ndarray_to_tensorspec(ndarray):
    return tf.TensorSpec(shape=ndarray.shape, dtype=ndarray.dtype)

  # Wrap in a `ModelWeights` structure that is required by the `finalizer.`
  trainable_weights, non_trainable_weights = model.initial_weights
  weights_spec = model_weights_lib.ModelWeights(
      tuple(ndarray_to_tensorspec(w) for w in trainable_weights),
      tuple(ndarray_to_tensorspec(w) for w in non_trainable_weights),
  )
  weights_type = computation_types.tensorflow_to_type(weights_spec)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  @tensorflow_computation.tf_computation(weights_type, data_type)
  def client_update_computation(initial_model_weights, dataset):
    client_update = _build_functional_client_update(
        model, loop_implementation=loop_implementation
    )
    return client_update(initial_model_weights, dataset)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(weights_type, placements.CLIENTS),
      computation_types.FederatedType(data_type, placements.CLIENTS),
  )
  def next_fn(state, model_weights, client_data):
    client_result, unfinalized_metrics = intrinsics.federated_map(
        client_update_computation, (model_weights, client_data)
    )
    metrics_aggregation_fn = metrics_aggregator(
        model.finalize_metrics, unfinalized_metrics.type_signature.member
    )
    train_metrics = metrics_aggregation_fn(unfinalized_metrics)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return measured_process.MeasuredProcessOutput(
        state, client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


DEFAULT_SERVER_OPTIMIZER_FN = sgdm.build_sgdm(learning_rate=0.1)


def build_fed_sgd(
    model_fn: Union[
        Callable[[], variable.VariableModel], functional.FunctionalModel
    ],
    server_optimizer_fn: optimizer_base.Optimizer = DEFAULT_SERVER_OPTIMIZER_FN,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    metrics_aggregator: Optional[types.MetricsAggregatorType] = None,
    loop_implementation: loop_builder.LoopImplementation = loop_builder.LoopImplementation.DATASET_REDUCE,
) -> learning_process.LearningProcess:
  """Builds a learning process that performs federated SGD.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  federated SGD on client models. The learning process has the following methods
  inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with type signature `( -> S@SERVER)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState`
      representing the initial state of the server.
  *   `next`: A `tff.Computation` with type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>)` where `S` is a
      `LearningAlgorithmState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. This computation returns a
      `LearningAlgorithmState` representing the updated server state and the
      metrics during client training and any other metrics from broadcast and
      aggregation processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `next`, and `M` represents the type
      of the model weights used during training.
  *   `set_model_weights`: A `tff.Computation` with type signature
      `(<S, M> -> S)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `M` represents the type of the model weights
      used during training.

  Each time `next` is called, the server model is broadcast to each client using
  a distributor. Each client sums the gradients for each batch in its local
  dataset (without updating its model) to calculate, and averages the gradients
  based on their number of examples. These average gradients are then aggregated
  at the server, and are applied at the server using an optimizer.

  This implements the original FedSGD algorithm in [McMahan et al.,
  2017](https://arxiv.org/abs/1602.05629).

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`, or an instance of a
      `tff.learning.models.FunctionalModel`. When passing a callable, the
      callable must *not* capture TensorFlow tensors or variables and use them.
      The model must be constructed entirely from scratch on each invocation,
      returning the same pre-constructed model each call will result in an
      error.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer` used to apply
      client updates to the server model.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.models.VariableModel.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`),
      and returns a `tff.Computation` for aggregating the unfinalized metrics.
    loop_implementation: Changes the implementation of the training loop
      generated. See `tff.learning.LoopImplementation` for more details.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  if not isinstance(model_fn, Callable):
    if not isinstance(model_fn, functional.FunctionalModel):
      raise TypeError(
          'If `model_fn` is not a callable, it must be an instance of '
          f'tff.learning.models.FunctionalModel. Got {type(model_fn)}'
      )

    @tensorflow_computation.tf_computation()
    def initial_model_weights_fn():
      trainable_weights, non_trainable_weights = model_fn.initial_weights
      return model_weights_lib.ModelWeights(
          tuple(tf.convert_to_tensor(w) for w in trainable_weights),
          tuple(tf.convert_to_tensor(w) for w in non_trainable_weights),
      )

  else:

    @tensorflow_computation.tf_computation()
    def initial_model_weights_fn():
      model = model_fn()  # pytype: disable=not-callable
      if not isinstance(model, variable.VariableModel):
        raise TypeError(
            'When `model_fn` is a callable, it returns instances of'
            ' tff.learning.models.VariableModel. Instead callable returned'
            f' type: {type(model)}'
        )
      return model_weights_lib.ModelWeights.from_model(model)

  model_weights_type = initial_model_weights_fn.type_signature.result

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)

  model_update_type = model_weights_type.trainable
  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  aggregator = model_aggregator.create(
      model_update_type, computation_types.TensorType(np.float32)
  )

  if metrics_aggregator is None:
    metrics_aggregator = metric_aggregator.sum_then_finalize
  if not callable(model_fn):
    client_work = _build_functional_fed_sgd_client_work(
        model_fn, metrics_aggregator, loop_implementation=loop_implementation
    )
  else:
    client_work = _build_fed_sgd_client_work(
        model_fn, metrics_aggregator, loop_implementation=loop_implementation
    )
  finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type
  )
  return composers.compose_learning_process(
      initial_model_weights_fn,
      model_distributor,
      client_work,
      aggregator,
      finalizer,
  )
