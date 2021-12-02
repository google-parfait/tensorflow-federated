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
"""An implementation of the FedProx algorithm.

Based on the paper:

"Federated Optimization in Heterogeneous Networks" by Tian Li, Anit Kumar Sahu,
Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. MLSys 2020.
See https://arxiv.org/abs/1812.06127 for the full paper.
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


def build_proximal_client_update_with_tff_optimizer(
    model_fn,
    proximal_strength: float,
    use_experimental_simulation_loop: bool = False):
  """Creates client update logic in FedProx using a TFF optimizer.

  In contrast to using a `tf.keras.optimizers.Optimizer`, we avoid creating
  `tf.Variable`s associated with the optimizer state within the scope of the
  client work, as they are not necessary. This also means that the client's
  model weights are updated by computing `optimizer.next` and then assigning
  the result to the model weights (while a `tf.keras.optimizers.Optimizer` will
  modify the model weight in place using `optimizer.apply_gradients`).

  Args:
    model_fn: A no-arg callable returning a `tff.learning.Model`.
    proximal_strength: A nonnegative float representing the parameter of
      FedProx's regularization term. When set to `0`, the client update reduces
      to that of FedAvg. Higher values prevent clients from moving too far from
      the server model during local training.
    use_experimental_simulation_loop: Controls the reduce loop function for the
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tf.function`.
  """
  model = model_fn()
  dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
      use_experimental_simulation_loop)

  @tf.function
  def client_update(optimizer, initial_weights, data):
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                          initial_weights)

    def reduce_fn(state, batch):
      """Trains a `tff.learning.Model` on a batch of data."""
      num_examples_sum, optimizer_state = state
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model_weights.trainable)
      proximal_delta = tf.nest.map_structure(tf.subtract,
                                             model_weights.trainable,
                                             initial_weights.trainable)
      proximal_term = tf.nest.map_structure(lambda x: proximal_strength * x,
                                            proximal_delta)
      gradients = tf.nest.map_structure(tf.add, gradients, proximal_term)

      optimizer_state, updated_weights = optimizer.next(optimizer_state,
                                                        model_weights.trainable,
                                                        gradients)
      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights.trainable,
                            updated_weights)

      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)

      return num_examples_sum, optimizer_state

    def initial_state_for_reduce_fn():
      trainable_tensor_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights.trainable)
      return tf.zeros(
          shape=[],
          dtype=tf.int64), optimizer.initialize(trainable_tensor_specs)

    num_examples, _ = dataset_reduce_fn(
        reduce_fn, data, initial_state_fn=initial_state_for_reduce_fn)
    client_update = tf.nest.map_structure(tf.subtract,
                                          initial_weights.trainable,
                                          model_weights.trainable)
    model_output = model.report_local_outputs()
    stat_output = collections.OrderedDict(num_examples=num_examples)

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    client_update, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(client_update))
    # Zero out the weight if there are any non-finite values.
    if has_non_finite_delta > 0:
      client_weight = tf.constant(0.0)
    else:
      client_weight = tf.cast(num_examples, tf.float32)

    return client_works.ClientResult(
        update=client_update,
        update_weight=client_weight), model_output, stat_output

  return client_update


def build_proximal_client_update_with_keras_optimizer(
    model_fn,
    proximal_strength: float,
    use_experimental_simulation_loop: bool = False):
  """Creates client update logic in FedProx using a `tf.keras` optimizer.

  In contrast to using a `tff.learning.optimizers.Optimizer`, we have to
  maintain `tf.Variable`s associated with the optimizer state within the scope
  of the client work. Additionally, the client model weights are modified in
  place by using `optimizer.apply_gradients`).

  Args:
    model_fn: A no-arg callable returning a `tff.learning.Model`.
    proximal_strength: A nonnegative float representing the parameter of
      FedProx's regularization term. When set to `0`, the client update reduces
      to that of FedAvg. Higher values prevent clients from moving too far from
      the server model during local training.
    use_experimental_simulation_loop: Controls the reduce loop function for the
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tf.function`.
  """
  model = model_fn()
  dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
      use_experimental_simulation_loop)

  @tf.function
  def client_update(optimizer, initial_weights, data):
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                          initial_weights)

    def reduce_fn(num_examples_sum, batch):
      """Trains a `tff.learning.Model` on a batch of data."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model_weights.trainable)
      proximal_delta = tf.nest.map_structure(tf.subtract,
                                             model_weights.trainable,
                                             initial_weights.trainable)
      proximal_term = tf.nest.map_structure(lambda x: proximal_strength * x,
                                            proximal_delta)
      gradients = tf.nest.map_structure(tf.add, gradients, proximal_term)
      grads_and_vars = zip(gradients, model_weights.trainable)
      optimizer.apply_gradients(grads_and_vars)

      # TODO(b/199782787): Add a unit test for a model that does not compute
      # `num_examples` in its forward pass.
      if output.num_examples is None:
        num_examples_sum += tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples_sum += tf.cast(output.num_examples, tf.int64)

      return num_examples_sum

    def initial_state_for_reduce_fn():
      return tf.zeros(shape=[], dtype=tf.int64)

    num_examples = dataset_reduce_fn(
        reduce_fn, data, initial_state_fn=initial_state_for_reduce_fn)
    client_update = tf.nest.map_structure(tf.subtract,
                                          initial_weights.trainable,
                                          model_weights.trainable)
    model_output = model.report_local_outputs()
    stat_output = collections.OrderedDict(num_examples=num_examples)

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    client_update, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(client_update))
    # Zero out the weight if there are any non-finite values.
    if has_non_finite_delta > 0:
      client_weight = tf.constant(0.0)
    else:
      client_weight = tf.cast(num_examples, tf.float32)

    return client_works.ClientResult(
        update=client_update,
        update_weight=client_weight), model_output, stat_output

  return client_update


def build_fed_prox_client_work(
    model_fn: Callable[[], model_lib.Model],
    proximal_strength: float,
    optimizer_fn: Union[optimizer_base.Optimizer,
                        Callable[[], tf.keras.optimizers.Optimizer]],
    use_experimental_simulation_loop: bool = False
) -> client_works.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for federated averaging.

  This client work is constructed in slightly different manners depending on
  whether `optimizer_fn` is a `tff.learning.optimizers.Optimizer`, or a no-arg
  callable returning a `tf.keras.optimizers.Optimizer`.

  If it is a `tff.learning.optimizers.Optimizer`, we avoid creating
  `tf.Variable`s associated with the optimizer state within the scope of the
  client work, as they are not necessary. This also means that the client's
  model weights are updated by computing `optimizer.next` and then assigning
  the result to the model weights (while a `tf.keras.optimizers.Optimizer` will
  modify the model weight in place using `optimizer.apply_gradients`).

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    proximal_strength: A nonnegative float representing the parameter of
      FedProx's regularization term. When set to `0`, the algorithm reduces to
      FedAvg. Higher values prevent clients from moving too far from the server
      model during local training.
    optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg callable
      that returns a `tf.keras.Optimizer`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `ClientWorkProcess`.
  """
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
  data_type = computation_types.SequenceType(model.input_spec)
  weights_type = model_utils.weights_type_from_model(model)

  if isinstance(optimizer_fn, optimizer_base.Optimizer):

    @computations.tf_computation(weights_type, data_type)
    def client_update_computation(initial_model_weights, dataset):
      client_update = build_proximal_client_update_with_tff_optimizer(
          model_fn, proximal_strength, use_experimental_simulation_loop)
      return client_update(optimizer_fn, initial_model_weights, dataset)

  else:

    @computations.tf_computation(weights_type, data_type)
    def client_update_computation(initial_model_weights, dataset):
      optimizer = optimizer_fn()
      client_update = build_proximal_client_update_with_keras_optimizer(
          model_fn, proximal_strength, use_experimental_simulation_loop)
      return client_update(optimizer, initial_model_weights, dataset)

  @computations.federated_computation
  def init_fn():
    return intrinsics.federated_value((), placements.SERVER)

  @computations.federated_computation(
      init_fn.type_signature.result, computation_types.at_clients(weights_type),
      computation_types.at_clients(data_type))
  def next_fn(state, weights, client_data):
    client_result, model_outputs, stat_output = intrinsics.federated_map(
        client_update_computation, (weights, client_data))
    train_metrics = model.federated_output_computation(model_outputs)
    stat_metrics = intrinsics.federated_sum(stat_output)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(train=train_metrics, stat=stat_metrics))
    return measured_process.MeasuredProcessOutput(state, client_result,
                                                  measurements)

  return client_works.ClientWorkProcess(init_fn, next_fn)


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)


def build_example_weighted_fed_prox_process(
    model_fn: Callable[[], model_lib.Model],
    proximal_strength: float,
    client_optimizer_fn: Union[optimizer_base.Optimizer,
                               Callable[[], tf.keras.optimizers.Optimizer]],
    server_optimizer_fn: Union[optimizer_base.Optimizer, Callable[
        [], tf.keras.optimizers.Optimizer]] = DEFAULT_SERVER_OPTIMIZER_FN,
    distributor: Optional[distributors.DistributionProcess] = None,
    model_update_aggregation_factory: Optional[
        factory.WeightedAggregationFactory] = None,
    use_experimental_simulation_loop: bool = False
) -> learning_process.LearningProcess:
  """Builds a learning process that performs the FedProx algorithm.

  This function creates a `LearningProcess` that performs example-weighted
  FedProx on client models. This algorithm behaves the same as federated
  averaging, except that it uses a proximal regularization term that encourages
  clients to not drift too far from the server model.

  The iterative process has the following methods inherited from
  `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `LearningAlgorithmState` representing the
      initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `LearningAlgorithmState` whose type matches the output of `initialize`
      and `{B*}@CLIENTS` represents the client datasets. The output `L`
      contains the updated server state, as well as metrics that are the result
      of `tff.learning.Model.federated_output_computation` during client
      training, and any other metrics from broadcast and aggregation processes.
  *   `report`: A `tff.Computation` with type signature `( -> M@SERVER)`, where
      `M` represents the type of the model weights used during training.

  Each time the `next` method is called, the server model is broadcast to each
  client using a broadcast function. For each client, local training is
  performed using `client_optimizer_fn`. Each client computes the difference
  between the client model after training and the initial broadcast model.
  These model deltas are then aggregated at the server using a weighted
  aggregation function. Clients weighted by the number of examples they see
  thoughout local training. The aggregate model delta is applied at the server
  using a server optimizer, as in the FedOpt framework proposed in
  [Reddi et al., 2021](https://arxiv.org/abs/2003.00295).

  Note: The default server optimizer function is `tf.keras.optimizers.SGD`
  with a learning rate of 1.0, which corresponds to adding the model delta to
  the current server model. This recovers the original FedProx algorithm in
  [Li et al., 2020](https://arxiv.org/abs/1812.06127). More
  sophisticated federated averaging procedures may use different learning rates
  or server optimizers.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    proximal_strength: A nonnegative float representing the parameter of
      FedProx's regularization term. When set to `0`, the algorithm reduces to
      FedAvg. Higher values prevent clients from moving too far from the server
      model during local training.
    client_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`. By default, this uses
      `tf.keras.optimizers.SGD` with a learning rate of 1.0.
    distributor: An optional `DistributionProcess` that broadcasts the model
      weights on the server to the clients. If set to `None`, the distributor is
      constructed via `distributors.build_broadcast_process`.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` used to aggregate client
      updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `LearningProcess`.

  Raises:
    ValueError: If `proximal_parameter` is not a nonnegative float.
  """
  if not isinstance(proximal_strength, float) or proximal_strength < 0.0:
    raise ValueError(
        'proximal_strength must be a nonnegative float, found {}'.format(
            proximal_strength))

  py_typecheck.check_callable(model_fn)

  @computations.tf_computation()
  def initial_model_weights_fn():
    return model_utils.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  if distributor is None:
    distributor = distributors.build_broadcast_process(model_weights_type)

  if model_update_aggregation_factory is None:
    model_update_aggregation_factory = mean.MeanFactory()
  py_typecheck.check_type(model_update_aggregation_factory,
                          factory.WeightedAggregationFactory)
  aggregator = model_update_aggregation_factory.create(
      model_weights_type.trainable, computation_types.TensorType(tf.float32))
  process_signature = aggregator.next.type_signature
  input_client_value_type = process_signature.parameter[1]
  result_server_value_type = process_signature.result[1]
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError('`model_update_aggregation_factory` does not produce a '
                    'compatible `AggregationProcess`. The processes must '
                    'retain the type structure of the inputs on the '
                    f'server, but got {input_client_value_type.member} != '
                    f'{result_server_value_type.member}.')

  client_work = build_fed_prox_client_work(model_fn, proximal_strength,
                                           client_optimizer_fn,
                                           use_experimental_simulation_loop)
  finalizer = finalizers.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type)
  return composers.compose_learning_process(initial_model_weights_fn,
                                            distributor, client_work,
                                            aggregator, finalizer)
