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

from typing import Callable, Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientSgd(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated SGD."""

  def __init__(
      self,
      model: model_lib.Model,
      client_weighting: client_weight_lib.ClientWeightType = client_weight_lib
      .ClientWeighting.NUM_EXAMPLES,
      use_experimental_simulation_loop: bool = False):
    """Constructs the client computation for Federated SGD.

    Note: All variable creation required for the client computation (e.g. model
    variable construction) must occur in during construction, and not during
    `__call__`.

    Args:
      model: A `learning.Model` for which gradients are computed.
      client_weighting: A value of `tff.learning.ClientWeighting` that
        specifies a built-in weighting method, or a callable that takes the
        output of `model.report_local_outputs` and returns a tensor that
        provides the weight in the federated average of model deltas.
      use_experimental_simulation_loop: Controls the reduce loop function for
        input dataset. An experimental reduce loop is used for simulation.
    """
    client_weight_lib.check_is_client_weighting_or_callable(client_weighting)
    self._client_weighting = client_weighting

    self._model = model_utils.enhance(model)
    py_typecheck.check_type(self._model, model_utils.EnhancedModel)

    self._dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)

  @property
  def variables(self):
    return []

  @tf.function
  def __call__(self, dataset, initial_weights):
    model = self._model

    # TODO(b/113112108): Remove this temporary workaround and restore check for
    # `tf.data.Dataset` after subclassing the currently used custom data set
    # representation from it.
    if 'Dataset' not in str(type(dataset)):
      raise TypeError('Expected a data set, found {}.'.format(
          py_typecheck.type_string(type(dataset))))

    tf.nest.map_structure(lambda a, b: a.assign(b), model.weights,
                          initial_weights)
    flat_trainable_weights = tuple(tf.nest.flatten(model.weights.trainable))

    def reduce_fn(state, batch):
      """Runs forward_pass on batch and sums the weighted gradients."""
      flat_accumulated_grads, num_examples_sum = state

      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      flat_grads = tape.gradient(output.loss, flat_trainable_weights)
      if output.num_examples is None:
        num_examples = tf.shape(output.predictions, out_type=tf.int64)[0]
      else:
        num_examples = tf.cast(output.num_examples, tf.int64)

      batch_weight = tf.cast(num_examples, tf.float32)

      flat_accumulated_grads = tuple(
          accumulator + batch_weight * grad
          for accumulator, grad in zip(flat_accumulated_grads, flat_grads))

      # The TF team is aware of an optimization in the reduce state to avoid
      # doubling the number of required variables here (e.g. keeping two copies
      # of all gradients). If you're looking to optimize memory usage this might
      # be a place to look.
      return (flat_accumulated_grads, num_examples_sum + num_examples)

    def _zero_initial_state():
      """Create a tuple of (gradient accumulators, batch weight, num examples)."""
      return (tuple(tf.zeros_like(w) for w in flat_trainable_weights),
              tf.constant(0, dtype=tf.int64))

    flat_grad_sums, num_examples_sum = self._dataset_reduce_fn(
        reduce_fn=reduce_fn,
        dataset=dataset,
        initial_state_fn=_zero_initial_state)
    grad_sums = tf.nest.pack_sequence_as(model.weights.trainable,
                                         flat_grad_sums)
    num_examples_as_float = tf.cast(num_examples_sum, tf.float32)

    # For SGD, the delta is just the negative of the average gradient:
    weights_delta = tf.nest.map_structure(
        lambda gradient: -1.0 * gradient / num_examples_as_float, grad_sums)

    model_output = model.report_local_outputs()

    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    if has_non_finite_delta > 0:
      weights_delta_weight = tf.constant(0.0)
    elif self._client_weighting is client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      weights_delta_weight = num_examples_as_float
    elif self._client_weighting is client_weight_lib.ClientWeighting.UNIFORM:
      weights_delta_weight = tf.constant(1.0)
    else:
      weights_delta_weight = self._client_weighting(model_output)

    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, model.report_local_outputs(),
        tensor_utils.to_odict({
            'client_weight': weights_delta_weight,
            'has_non_finite_delta': has_non_finite_delta,
        }))


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)


# TODO(b/192094313): refactor to accept tff.learning.Optimizer arguments
def build_federated_sgd_process(
    model_fn: Callable[[], model_lib.Model],
    server_optimizer_fn: Callable[
        [], tf.keras.optimizers.Optimizer] = DEFAULT_SERVER_OPTIMIZER_FN,
    *,  # Require named (non-positional) parameters for the following kwargs:
    client_weighting: Optional[client_weight_lib.ClientWeightType] = None,
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    model_update_aggregation_factory: Optional[
        factory.WeightedAggregationFactory] = None,
    use_experimental_simulation_loop: bool = False,
) -> iterative_process.IterativeProcess:
  """Builds the TFF computations for optimization using federated SGD.

  This function creates a `tff.templates.IterativeProcess` that performs
  federated SGD on client models. The iterative process has the following
  methods inherited from `tff.templates.IterativeProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `tff.learning.framework.ServerState`
      representing the initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>)` where `S` is a
      `tff.learning.framework.ServerState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. This computation returns a
      `tff.learning.framework.ServerState` representing the updated server state
      and metrics that are the result of
      `tff.learning.Model.federated_output_computation` during client training
      and any other metrics from broadcast and aggregation processes.

  The iterative process also has the following method not inherited from
  `tff.templates.IterativeProcess`:

  *   `get_model_weights`: A `tff.Computation` that takes as input the
      a `tff.learning.framework.ServerState`, and returns a
      `tff.learning.ModelWeights` containing the state's model weights.

  Each time the `next` method is called, the server model is broadcast to each
  client using a broadcast function. Each client sums the gradients at each
  batch in the client's local dataset without updating the model to calculate
  the batch gradient on client, and averages the gradient based on its number of
  examples. These averaged batch gradients are then aggregated at the server
  using an aggregation function. The aggregate gradients are applied at the
  server by using the
  `tf.keras.optimizers.Optimizer.apply_gradients` method of the server
  optimizer.

  This implements the original FedSGD algorithm in [McMahan et al.,
  2017](https://arxiv.org/abs/1602.05629).

  Note: the default server optimizer function is `tf.keras.optimizers.SGD`
  with a learning rate of 0.1. More sophisticated federated SGD procedures may
  use different learning rates or server optimizers.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    server_optimizer_fn: A no-arg function that returns a `tf.Optimizer`. The
      `apply_gradients` method of this optimizer is used to apply client updates
      to the server model.
    client_weighting: A value of `tff.learning.ClientWeighting` that specifies a
      built-in weighting method, or a callable that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If `None`, defaults to weighting
      by number of examples.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` that constructs
      `tff.templates.AggregationProcess` for aggregating the client model
      updates on the server. If `None`, uses a default constructed
      `tff.aggregators.MeanFactory`, creating a stateless mean aggregation.
    use_experimental_simulation_loop: Controls the reduce loop function for
        input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  if isinstance(model_update_aggregation_factory,
                factory.UnweightedAggregationFactory):
    if client_weighting is None:
      client_weighting = client_weight_lib.ClientWeighting.UNIFORM
    elif client_weighting is not client_weight_lib.ClientWeighting.UNIFORM:
      raise ValueError('Cannot use non-uniform client weighting with '
                       'unweighted aggregation.')
  elif client_weighting is None:
    client_weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES

  def client_sgd_avg(model_fn: Callable[[], model_lib.Model]) -> ClientSgd:
    return ClientSgd(
        model_fn(),
        client_weighting=client_weighting,
        use_experimental_simulation_loop=use_experimental_simulation_loop)

  iter_proc = optimizer_utils.build_model_delta_optimizer_process(
      model_fn,
      model_to_client_delta_fn=client_sgd_avg,
      server_optimizer_fn=server_optimizer_fn,
      broadcast_process=broadcast_process,
      model_update_aggregation_factory=model_update_aggregation_factory)

  server_state_type = iter_proc.state_type.member

  @computations.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  iter_proc.get_model_weights = get_model_weights
  return iter_proc
