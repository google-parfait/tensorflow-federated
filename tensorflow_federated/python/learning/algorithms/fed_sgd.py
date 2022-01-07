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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""An implementation of the Federated SGD algorithm.

This is the baseline algorithm from:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from typing import Callable, Optional, Union

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.tensorflow_libs import tensor_utils


def _build_client_update(model: model_lib.Model,
                         use_experimental_simulation_loop: bool = False):
  """Creates client update logic for FedSGD.

  Args:
    model: A `tff.learning.Model` used to compute gradients.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tf.function`.
  """
  dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
      use_experimental_simulation_loop)

  @tf.function
  def client_update(initial_weights, dataset):
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                          initial_weights)

    def reduce_fn(state, batch):
      """Runs forward_pass on batch and sums the weighted gradients."""
      accumulated_gradients, num_examples_sum = state

      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      gradients = tape.gradient(output.loss, model_weights.trainable)
      num_examples = tf.cast(output.num_examples, tf.float32)
      accumulated_gradients = tuple(
          accumulator + num_examples * gradient
          for accumulator, gradient in zip(accumulated_gradients, gradients))

      # We may be able to optimize the reduce function to avoid doubling the
      # number of required variables here (e.g. keeping two copies of all
      # gradients). If you're looking to optimize memory usage this might be a
      # place to look.
      return (accumulated_gradients, num_examples_sum + num_examples)

    def _zero_initial_state():
      """Create a tuple of (gradient accumulators, num examples)."""
      return tuple(
          tf.nest.map_structure(tf.zeros_like,
                                model_weights.trainable)), tf.constant(
                                    0, dtype=tf.float32)

    gradient_sums, num_examples_sum = dataset_reduce_fn(
        reduce_fn=reduce_fn,
        dataset=dataset,
        initial_state_fn=_zero_initial_state)

    # We now normalize to compute the average gradient over all examples.
    average_gradient = tf.nest.map_structure(
        lambda gradient: gradient / num_examples_sum, gradient_sums)

    model_output = model.report_local_outputs()
    stat_output = collections.OrderedDict(num_examples=num_examples_sum)

    average_gradient, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(average_gradient))
    if has_non_finite_delta > 0:
      client_weight = tf.constant(0.0)
    else:
      client_weight = num_examples_sum

    return client_works.ClientResult(
        update=average_gradient,
        update_weight=client_weight), model_output, stat_output

  return client_update


def _build_fed_sgd_client_work(
    model_fn: Callable[[], model_lib.Model],
    use_experimental_simulation_loop: bool = False
) -> client_works.ClientWorkProcess:
  """Creates a `tff.learning.templates.ClientWorkProcess` for federated SGD.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `tff.learning.templates.ClientWorkProcess`.
  """
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
  data_type = computation_types.SequenceType(model.input_spec)
  weights_type = model_utils.weights_type_from_model(model)

  @computations.federated_computation
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  @computations.tf_computation(weights_type, data_type)
  def client_update_computation(initial_model_weights, dataset):
    client_update = _build_client_update(model_fn(),
                                         use_experimental_simulation_loop)
    return client_update(initial_model_weights, dataset)

  @computations.federated_computation(
      init_fn.type_signature.result, computation_types.at_clients(weights_type),
      computation_types.at_clients(data_type))
  def next_fn(state, model_weights, client_data):
    client_result, model_outputs, stat_output = intrinsics.federated_map(
        client_update_computation, (model_weights, client_data))
    train_metrics = model.federated_output_computation(model_outputs)
    stat_metrics = intrinsics.federated_sum(stat_output)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics, stat=stat_metrics))
    return measured_process.MeasuredProcessOutput(state, client_result,
                                                  measurements)

  return client_works.ClientWorkProcess(init_fn, next_fn)


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)


def build_fed_sgd(
    model_fn: Callable[[], model_lib.Model],
    server_optimizer_fn: Union[optimizer_base.Optimizer, Callable[
        [], tf.keras.optimizers.Optimizer]] = DEFAULT_SERVER_OPTIMIZER_FN,
    distributor: Optional[distributors.DistributionProcess] = None,
    model_update_aggregation_factory: Optional[
        factory.WeightedAggregationFactory] = None,
    use_experimental_simulation_loop: bool = False,
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
      `LearningAlgorithmState` representing the updated server state and metrics
      that are the result of `tff.learning.Model.federated_output_computation`
      during client training and any other metrics from broadcast and
      aggregation processes.
  *   get_model_weights: A `tff.Computation` with the type signature `(S -> W)`
      where `S` is a `LearningAlgorithmState` matching
      the output of `initialize`, and `W` is a `tff.learning.ModelWeights`
      representing the current model weights of the state.

  Each time `next` is called, the server model is broadcast to each client using
  a distributor. Each client sums the gradients for each batch in its local
  dataset (without updating its model) to calculate, and averages the gradients
  based on their number of examples. These average gradients are then aggregated
  at the server, and are applied at the server using a
  `tf.keras.optimizers.Optimizer`.

  This implements the original FedSGD algorithm in [McMahan et al.,
  2017](https://arxiv.org/abs/1602.05629).

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`. The optimizer is used to
      apply client updates to the server model.
    distributor: An optional `DistributionProcess` that broadcasts the model
      weights on the server to the clients. If set to `None`, the distributor is
      constructed via `distributors.build_broadcast_process`.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` that constructs
      `tff.templates.AggregationProcess` for aggregating the client model
      updates on the server. If `None`, uses a default constructed
      `tff.aggregators.MeanFactory`, creating a stateless mean aggregation.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  py_typecheck.check_callable(model_fn)

  @computations.tf_computation()
  def initial_model_weights_fn():
    return model_utils.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  if distributor is None:
    distributor = distributors.build_broadcast_process(model_weights_type)

  if model_update_aggregation_factory is None:
    model_update_aggregation_factory = mean.MeanFactory()
  aggregator = model_update_aggregation_factory.create(
      model_weights_type.trainable, computation_types.TensorType(tf.float32))

  client_work = _build_fed_sgd_client_work(
      model_fn,
      use_experimental_simulation_loop=use_experimental_simulation_loop)
  finalizer = finalizers.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type)
  return composers.compose_learning_process(initial_model_weights_fn,
                                            distributor, client_work,
                                            aggregator, finalizer)
