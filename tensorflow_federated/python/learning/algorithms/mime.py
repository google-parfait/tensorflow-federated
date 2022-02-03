# Copyright 2022, The TensorFlow Federated Authors.
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
"""An implementation of Mime Lite algorithm.

The algorithm is proposed by the paper:

Breaking the centralized barrier for cross-device federated learning.
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank
    Reddi, Sebastian U. Stich, and Ananda Theertha Suresh.
    Advances in Neural Information Processing Systems 34 (2021).
    https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf
"""

import collections
from typing import Callable, Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.algorithms import aggregation
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.metrics import aggregator as metric_aggregator
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.tensorflow_libs import tensor_utils


# TODO(b/152633983): Move to the tff.learning.Model class.
def _reset_local_model_variables(model):
  for var in model.local_variables:
    if var.initial_value is not None:
      var.assign(var.initial_value)
    else:
      var.assign(tf.zeros_like(var))


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


def _build_client_update_fn_for_mime_lite(
    model_fn: Callable[[], model_lib.Model],
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    use_experimental_simulation_loop: bool = False):
  """Builds the `tf_computation` for Mime Lite client training."""

  @computations.tf_computation
  def client_update_fn(global_optimizer_state, initial_weights, data):
    model = model_fn()
    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)
    weight_tensor_specs = type_conversions.type_to_tf_tensor_specs(
        model_utils.weights_type_from_model(model))

    @tf.function
    def client_update(global_optimizer_state, initial_weights, data):
      model_weights = model_utils.ModelWeights.from_model(model)
      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                            initial_weights)

      def full_gradient_reduce_fn(state, batch):
        """Sums individual gradients, to be later divided by num_examples."""
        gradient_sum, num_examples_sum = state
        with tf.GradientTape() as tape:
          output = model.forward_pass(batch, training=True)
        if output.num_examples is None:
          num_examples = tf.shape(output.predictions, out_type=tf.int64)[0]
        else:
          num_examples = tf.cast(output.num_examples, tf.int64)
        # TODO(b/161529310): We flatten and convert to tuple, as tf.data
        # iterators would try to stack the tensors in list into a single tensor.
        gradients = tuple(
            tf.nest.flatten(
                tape.gradient(output.loss, model_weights.trainable)))
        gradient_sum = tf.nest.map_structure(
            lambda g_sum, g: g_sum + g * tf.cast(num_examples, g.dtype),
            gradient_sum, gradients)
        num_examples_sum += num_examples
        return gradient_sum, num_examples_sum

      def initial_state_for_full_gradient_reduce_fn():
        initial_gradient_sum = tf.nest.map_structure(
            lambda spec: tf.zeros(spec.shape, spec.dtype),
            tuple(tf.nest.flatten(weight_tensor_specs.trainable)))
        initial_num_examples_sum = tf.constant(0, tf.int64)
        return initial_gradient_sum, initial_num_examples_sum

      full_gradient, num_examples = dataset_reduce_fn(
          full_gradient_reduce_fn, data,
          initial_state_for_full_gradient_reduce_fn)
      # Compute the average gradient.
      full_gradient = tf.nest.map_structure(
          lambda g: tf.math.divide_no_nan(g, tf.cast(num_examples, g.dtype)),
          full_gradient)

      # Resets the local model variables, including metrics states, as we are
      # not interested in metrics based on the full gradient evaluation, only
      # from the subsequent training.
      _reset_local_model_variables(model)

      def train_reduce_fn(state, batch):
        with tf.GradientTape() as tape:
          output = model.forward_pass(batch, training=True)
        gradients = tape.gradient(output.loss, model_weights.trainable)
        # Mime Lite keeps optimizer state unchanged during local training.
        _, updated_weights = optimizer.next(global_optimizer_state,
                                            model_weights.trainable, gradients)
        tf.nest.map_structure(lambda a, b: a.assign(b), model_weights.trainable,
                              updated_weights)
        return state

      # Performs local training, updating `tf.Variable`s in `model_weights`.
      dataset_reduce_fn(
          train_reduce_fn, data, initial_state_fn=lambda: tf.zeros(shape=[0]))

      client_weights_delta = tf.nest.map_structure(tf.subtract,
                                                   initial_weights.trainable,
                                                   model_weights.trainable)
      model_output = model.report_local_unfinalized_metrics()

      # TODO(b/122071074): Consider moving this functionality into aggregation.
      client_weights_delta, has_non_finite_delta = (
          tensor_utils.zero_all_if_any_non_finite(client_weights_delta))
      client_weight = _choose_client_weight(client_weighting,
                                            has_non_finite_delta, num_examples)
      return client_works.ClientResult(
          update=client_weights_delta,
          update_weight=client_weight), model_output, full_gradient

    return client_update(global_optimizer_state, initial_weights, data)

  return client_update_fn


def _build_mime_lite_client_work(
    model_fn: Callable[[], model_lib.Model],
    optimizer: optimizer_base.Optimizer,
    client_weighting: client_weight_lib.ClientWeighting,
    full_gradient_aggregator: Optional[
        factory.WeightedAggregationFactory] = None,
    metrics_aggregator: Optional[Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation]] = None,
    use_experimental_simulation_loop: bool = False
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for Mime Lite.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    optimizer: A `tff.learning.optimizers.Optimizer` which will be used for both
      creating and updating a global optimizer state, as well as optimization at
      clients given the global state, which is fixed during the optimization.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method.
    full_gradient_aggregator: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics. If
      `None`, this is set to `tff.learning.metrics.sum_then_finalize`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `ClientWorkProcess`.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.MeanFactory()
  py_typecheck.check_type(full_gradient_aggregator,
                          factory.WeightedAggregationFactory)
  if metrics_aggregator is None:
    metrics_aggregator = metric_aggregator.sum_then_finalize

  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        model.report_local_unfinalized_metrics())
    metrics_aggregation_fn = metrics_aggregator(model.metric_finalizers(),
                                                unfinalized_metrics_type)
  data_type = computation_types.SequenceType(model.input_spec)
  weights_type = model_utils.weights_type_from_model(model)
  weight_tensor_specs = type_conversions.type_to_tf_tensor_specs(weights_type)

  full_gradient_aggregator = full_gradient_aggregator.create(
      weights_type.trainable, computation_types.TensorType(tf.float32))

  @computations.federated_computation
  def init_fn():
    specs = weight_tensor_specs.trainable
    optimizer_state = intrinsics.federated_eval(
        computations.tf_computation(lambda: optimizer.initialize(specs)),
        placements.SERVER)
    aggregator_state = full_gradient_aggregator.initialize()
    return intrinsics.federated_zip((optimizer_state, aggregator_state))

  client_update_fn = _build_client_update_fn_for_mime_lite(
      model_fn, optimizer, client_weighting, use_experimental_simulation_loop)

  @computations.tf_computation(init_fn.type_signature.result.member[0],
                               weights_type.trainable)
  def update_optimizer_state(state, aggregate_gradient):
    whimsy_weights = tf.nest.map_structure(lambda g: tf.zeros(g.shape, g.dtype),
                                           aggregate_gradient)
    updated_state, _ = optimizer.next(state, whimsy_weights, aggregate_gradient)
    return updated_state

  @computations.federated_computation(
      init_fn.type_signature.result, computation_types.at_clients(weights_type),
      computation_types.at_clients(data_type))
  def next_fn(state, weights, client_data):
    optimizer_state, aggregator_state = state
    optimizer_state_at_clients = intrinsics.federated_broadcast(optimizer_state)
    client_result, model_outputs, full_gradient = (
        intrinsics.federated_map(
            client_update_fn,
            (optimizer_state_at_clients, weights, client_data)))
    full_gradient_agg_output = full_gradient_aggregator.next(
        aggregator_state, full_gradient, client_result.update_weight)
    updated_optimizer_state = intrinsics.federated_map(
        update_optimizer_state,
        (optimizer_state, full_gradient_agg_output.result))

    new_state = intrinsics.federated_zip(
        (updated_optimizer_state, full_gradient_agg_output.state))
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics))
    return measured_process.MeasuredProcessOutput(new_state, client_result,
                                                  measurements)

  return client_works.ClientWorkProcess(init_fn, next_fn)


def build_weighted_mime_lite(
    model_fn: Callable[[], model_lib.Model],
    optimizer: optimizer_base.Optimizer,
    client_weighting: Optional[
        client_weight_lib
        .ClientWeighting] = client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    full_gradient_aggregator: Optional[
        factory.WeightedAggregationFactory] = None,
    metrics_aggregator: Optional[Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation]] = None,
    use_experimental_simulation_loop: bool = False
) -> learning_process.LearningProcess:
  """Builds a learning process that performs Mime Lite.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  Mime Lite algorithm on client models. The iterative process has the following
  methods inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` representing the initial
      state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `{B*}@CLIENTS` represents the client datasets.
      The output `L` contains the updated server state, as well as aggregated
      metrics at the server, including client training metrics and any other
      metrics from distribution and aggregation processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `next` and `M` represents the type
      of the model weights used during training.

  Each time the `next` method is called, the server model is communicated to
  each client using the provided `model_distributor`. For each client, local
  training is performed using `optimizer`, where its state is communicated by
  the server, and kept intact during local training. The state is updated only
  at the server based on the full gradient evaluated by the clients based on the
  current server model state. The client full gradients are aggregated by
  weighted `full_gradient_aggregator`. Each client computes the difference
  between the client model after training and its initial model. These model
  deltas are then aggregated by weighted `model_aggregator`. Both of the
  aggregations are weighted, according to `client_weighting`. The aggregate
  model delta is added to the existing server model state.

  The Mime Lite algorithm is based on the paper
  "Breaking the centralized barrier for cross-device federated learning."
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank
    Reddi, Sebastian U. Stich, and Ananda Theertha Suresh.
    Advances in Neural Information Processing Systems 34 (2021).
    https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf

  Note that Keras optimizers are not supported. This is due to the Mime Lite
  algorithm applying the optimizer without changing it state at clients
  (optimizer's `tf.Variable`s in the case of Keras), which is not possible with
  Keras optimizers without reaching into private implementation details and
  incurring additional computation and memory cost at clients.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    optimizer: A `tff.learning.optimizers.Optimizer` which will be used for both
      creating and updating a global optimizer state, as well as optimization at
      clients given the global state, which is fixed during the optimization.
    client_weighting: A member of `tff.learning.ClientWeighting` that specifies
      a built-in weighting method. By default, weighting by number of examples
      is used.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    full_gradient_aggregator: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics. If
      `None`, this is set to `tff.learning.metrics.sum_then_finalize`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_type(optimizer, optimizer_base.Optimizer)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)

  @computations.tf_computation()
  def initial_model_weights_fn():
    return model_utils.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result
  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)
  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  py_typecheck.check_type(model_aggregator, factory.WeightedAggregationFactory)
  model_aggregator = model_aggregator.create(
      model_weights_type.trainable, computation_types.TensorType(tf.float32))
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.MeanFactory()
  py_typecheck.check_type(full_gradient_aggregator,
                          factory.WeightedAggregationFactory)

  client_work = _build_mime_lite_client_work(
      model_fn=model_fn,
      optimizer=optimizer,
      client_weighting=client_weighting,
      full_gradient_aggregator=full_gradient_aggregator,
      metrics_aggregator=metrics_aggregator,
      use_experimental_simulation_loop=use_experimental_simulation_loop)
  finalizer = finalizers.build_apply_optimizer_finalizer(
      sgdm.build_sgdm(1.0), model_weights_type)
  return composers.compose_learning_process(initial_model_weights_fn,
                                            model_distributor, client_work,
                                            model_aggregator, finalizer)


def build_unweighted_mime_lite(
    model_fn: Callable[[], model_lib.Model],
    optimizer: optimizer_base.Optimizer,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.UnweightedAggregationFactory] = None,
    full_gradient_aggregator: Optional[
        factory.UnweightedAggregationFactory] = None,
    metrics_aggregator: Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation] = metric_aggregator.sum_then_finalize,
    use_experimental_simulation_loop: bool = False
) -> learning_process.LearningProcess:
  """Builds a learning process that performs Mime Lite.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  Mime Lite algorithm on client models. The iterative process has the following
  methods inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` representing the initial
      state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `{B*}@CLIENTS` represents the client datasets.
      The output `L` contains the updated server state, as well as aggregated
      metrics at the server, including client training metrics and any other
      metrics from distribution and aggregation processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `next` and `M` represents the type
      of the model weights used during training.

  Each time the `next` method is called, the server model is communicated to
  each client using the provided `model_distributor`. For each client, local
  training is performed using `optimizer`, where its state is communicated by
  the server, and kept intact during local training. The state is updated only
  at the server based on the full gradient evaluated by the clients based on the
  current server model state. The client full gradients are aggregated by
  unweighted `full_gradient_aggregator`. Each client computes the difference
  between the client model after training and its initial model. These model
  deltas are then aggregated by unweighted `model_aggregator`. The aggregate
  model delta is added to the existing server model state.

  The Mime Lite algorithm is based on the paper
  "Breaking the centralized barrier for cross-device federated learning."
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank
    Reddi, Sebastian U. Stich, and Ananda Theertha Suresh.
    Advances in Neural Information Processing Systems 34 (2021).
    https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf

  Note that Keras optimizers are not supported. This is due to the Mime Lite
  algorithm applying the optimizer without changing it state at clients
  (optimizer's `tf.Variable`s in the case of Keras), which is not possible with
  Keras optimizers without reaching into private implementation details and
  incurring additional computation and memory cost at clients.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    optimizer: A `tff.learning.optimizers.Optimizer` which will be used for both
      creating and updating a global optimizer state, as well as optimization at
      clients given the global state, which is fixed during the optimization.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.UnweightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.UnweightedMeanFactory`.
    full_gradient_aggregator: An optional
      `tff.aggregators.UnweightedAggregationFactory` used to aggregate the full
      gradients on client datasets. If `None`, this is set to
      `tff.aggregators.UnweightedMeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `tff.learning.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `tff.learning.Model.report_local_unfinalized_metrics()`), and
      returns a `tff.Computation` for aggregating the unfinalized metrics. If
      `None`, this is set to `tff.learning.metrics.sum_then_finalize`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  if model_aggregator is None:
    model_aggregator = mean.UnweightedMeanFactory()
  py_typecheck.check_type(model_aggregator,
                          factory.UnweightedAggregationFactory)
  if full_gradient_aggregator is None:
    full_gradient_aggregator = mean.UnweightedMeanFactory()
  py_typecheck.check_type(full_gradient_aggregator,
                          factory.UnweightedAggregationFactory)

  return build_weighted_mime_lite(
      model_fn=model_fn,
      optimizer=optimizer,
      client_weighting=client_weight_lib.ClientWeighting.UNIFORM,
      model_distributor=model_distributor,
      model_aggregator=aggregation.as_weighted_aggregator(model_aggregator),
      full_gradient_aggregator=aggregation.as_weighted_aggregator(
          full_gradient_aggregator),
      metrics_aggregator=metrics_aggregator,
      use_experimental_simulation_loop=use_experimental_simulation_loop)
