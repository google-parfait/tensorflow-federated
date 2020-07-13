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
"""An implementation of the Federated Averaging algorithm.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from typing import Any, Callable, Optional

import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientFedAvg(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated Averaging."""

  def __init__(self,
               model: model_lib.Model,
               optimizer: tf.keras.optimizers.Optimizer,
               client_weight_fn: Optional[Callable[[Any], tf.Tensor]] = None):
    """Creates the client computation for Federated Averaging.

    Args:
      model: A `tff.learning.Model` instance.
      optimizer: A `tf.keras.Optimizer` instance.
      client_weight_fn: an optional callable that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.
    """
    py_typecheck.check_type(model, model_lib.Model)
    self._model = model_utils.enhance(model)
    self._optimizer = optimizer
    py_typecheck.check_type(self._model, model_utils.EnhancedModel)

    if client_weight_fn is not None:
      py_typecheck.check_callable(client_weight_fn)
      self._client_weight_fn = client_weight_fn
    else:
      self._client_weight_fn = None

  @property
  def variables(self):
    return []

  @tf.function
  def __call__(self, dataset, initial_weights):
    model = self._model
    optimizer = self._optimizer
    tf.nest.map_structure(lambda a, b: a.assign(b), model.weights,
                          initial_weights)

    @tf.function
    def reduce_fn(num_examples_sum, batch):
      """Train `tff.learning.Model` on local client batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, model.weights.trainable)
      optimizer.apply_gradients(zip(gradients, model.weights.trainable))

      if output.num_examples is None:
        return num_examples_sum + tf.shape(output.predictions)[0]
      else:
        return num_examples_sum + output.num_examples

    num_examples_sum = dataset.reduce(
        initial_state=tf.constant(0), reduce_func=reduce_fn)

    weights_delta = tf.nest.map_structure(tf.subtract, model.weights.trainable,
                                          initial_weights.trainable)
    aggregated_outputs = model.report_local_outputs()

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    # Zero out the weight if there are any non-finite values.
    if has_non_finite_delta > 0:
      weights_delta_weight = tf.constant(0.0)
    elif self._client_weight_fn is None:
      weights_delta_weight = tf.cast(num_examples_sum, tf.float32)
    else:
      weights_delta_weight = self._client_weight_fn(aggregated_outputs)

    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, aggregated_outputs,
        collections.OrderedDict(
            num_examples=num_examples_sum,
            has_non_finite_delta=has_non_finite_delta,
        ))


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)


def build_federated_averaging_process(
    model_fn: Callable[[], model_lib.Model],
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    server_optimizer_fn: Callable[
        [], tf.keras.optimizers.Optimizer] = DEFAULT_SERVER_OPTIMIZER_FN,
    client_weight_fn: Callable[[Any], tf.Tensor] = None,
    stateful_delta_aggregate_fn: Optional[tff.utils.StatefulAggregateFn] = None,
    stateful_model_broadcast_fn: Optional[tff.utils.StatefulBroadcastFn] = None,
    *,
    broadcast_process: Optional[tff.templates.MeasuredProcess] = None,
    aggregation_process: Optional[tff.templates.MeasuredProcess] = None,
) -> tff.templates.IterativeProcess:
  """Builds an iterative process that performs federated averaging.

  This function creates a `tff.templates.IterativeProcess` that performs
  federated averaging on client models. The iterative process has the following
  methods:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `tff.learning.framework.ServerState`
      representing the initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>)` where `S` is a
      `tff.learning.framework.ServerState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. This computation returns a
      `tff.learning.framework.ServerState` representing the updated server state
      and training metrics that are the result of
      `tff.learning.Model.federated_output_computation` during client training.

  Each time the `next` method is called, the server model is broadcast to each
  client using a broadcast function. For each client, one epoch of local
  training is performed via the `tf.keras.optimizers.Optimizer.apply_gradients`
  method of the client optimizer. Each client computes the difference between
  the client model after training and the initial broadcast model. These model
  deltas are then aggregated at the server using some aggregation function. The
  aggregate model delta is applied at the server by using the
  `tf.keras.optimizers.Optimizer.apply_gradients` method of the server
  optimizer.

  Note: the default server optimizer function is `tf.keras.optimizers.SGD`
  with a learning rate of 1.0, which corresponds to adding the model delta to
  the current server model. This recovers the original FedAvg algorithm in
  [McMahan et al., 2017](https://arxiv.org/abs/1602.05629). More
  sophisticated federated averaging procedures may use different learning rates
  or server optimizers.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    client_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
    server_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
      By default, this uses `tf.keras.optimizers.SGD` with a learning rate of
      1.0.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor providing the weight in
      the federated average of model deltas. If not provided, the default is the
      total number of examples processed on device.
    stateful_delta_aggregate_fn: A `tff.utils.StatefulAggregateFn` where the
      `next_fn` performs a federated aggregation and upates state. It must have
      TFF type `(<state@SERVER, value@CLIENTS, weights@CLIENTS> ->
      <state@SERVER, aggregate@SERVER>)`, where the `value` type is
      `tff.learning.framework.ModelWeights.trainable` corresponding to the
      object returned by `model_fn`. By default performs arithmetic mean
      aggregation, weighted by `client_weight_fn`.  Must be `None` if
      `aggregation_process` is not `None`.
    stateful_model_broadcast_fn: A `tff.utils.StatefulBroadcastFn` where the
      `next_fn` performs a federated broadcast and upates state. It must have
      TFF type `(<state@SERVER, value@SERVER> -> <state@SERVER,
      value@CLIENTS>)`, where the `value` type is
      `tff.learning.framework.ModelWeights` corresponding to the object returned
      by `model_fn`. The default is the identity broadcast.  Must be `None` if
      `broadcast_process` is not `None`.
    broadcast_process: a `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`.
      Must be `None` if
      `stateful_model_broadcast_fn` is not `None`.
    aggregation_process: a `tff.templates.MeasuredProcess` that aggregates the
      model updates on the clients back to the server. It must support the
      signature `({input_values}@CLIENTS-> output_values@SERVER)`.
      Must be
      `None` if `stateful_delta_aggregate_fn` is not `None`.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  def client_fed_avg(model_fn):
    return ClientFedAvg(model_fn(), client_optimizer_fn(), client_weight_fn)

  return optimizer_utils.build_model_delta_optimizer_process(
      model_fn,
      model_to_client_delta_fn=client_fed_avg,
      server_optimizer_fn=server_optimizer_fn,
      stateful_delta_aggregate_fn=stateful_delta_aggregate_fn,
      stateful_model_broadcast_fn=stateful_model_broadcast_fn,
      broadcast_process=broadcast_process,
      aggregation_process=aggregation_process)
