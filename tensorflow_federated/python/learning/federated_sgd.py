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

from typing import Any, Callable, Optional

import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientSgd(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated SGD."""

  def __init__(self, model, batch_weight_fn=None):
    """Constructs the client computation for Federated SGD.

    Args:
      model: A `learning.Model` for which gradients are computed.
      batch_weight_fn: A function that takes a batch (as passed to forward_pass)
        and returns a float32 weight. If not provided, the default uses the size
        of the batch (as measured by the batch dimension of the predictions
        returned by forward_pass).
    """
    if batch_weight_fn is not None:
      py_typecheck.check_callable(batch_weight_fn)
    self._batch_weight_fn = batch_weight_fn

    self._model = model_utils.enhance(model)
    py_typecheck.check_type(self._model, model_utils.EnhancedModel)

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

    @tf.function
    def reduce_fn(state, batch):
      """Runs forward_pass on batch and sums the weighted gradients."""
      flat_accumulated_grads, batch_weight_sum = state

      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      flat_grads = tape.gradient(output.loss, flat_trainable_weights)

      if self._batch_weight_fn is not None:
        batch_weight = self._batch_weight_fn(batch)
      else:
        batch_weight = tf.cast(tf.shape(output.predictions)[0], tf.float32)

      flat_accumulated_grads = tuple(
          accumulator + batch_weight * grad
          for accumulator, grad in zip(flat_accumulated_grads, flat_grads))

      # The TF team is aware of an optimization in the reduce state to avoid
      # doubling the number of required variables here (e.g. keeping two copies
      # of all gradients). If you're looking to optimize memory usage this might
      # be a place to look.
      return (flat_accumulated_grads, batch_weight_sum + batch_weight)

    def _zero_initial_state():
      """Create a tuple of (tuple of gradient accumulators, batch weight sum)."""
      return (tuple(tf.zeros_like(w) for w in flat_trainable_weights),
              tf.constant(0.0))

    flat_grad_sums, batch_weight_sum = dataset.reduce(
        initial_state=_zero_initial_state(), reduce_func=reduce_fn)
    grad_sums = tf.nest.pack_sequence_as(model.weights.trainable,
                                         flat_grad_sums)

    # For SGD, the delta is just the negative of the average gradient:
    weights_delta = tf.nest.map_structure(
        lambda gradient: -1.0 * gradient / batch_weight_sum, grad_sums)
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    if has_non_finite_delta > 0:
      weights_delta_weight = tf.constant(0.0)
    else:
      weights_delta_weight = batch_weight_sum
    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, model.report_local_outputs(),
        tensor_utils.to_odict({
            'client_weight': weights_delta_weight,
            'has_non_finite_delta': has_non_finite_delta,
        }))


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)


def build_federated_sgd_process(
    model_fn: Callable[[], model_lib.Model],
    server_optimizer_fn: Callable[
        [], tf.keras.optimizers.Optimizer] = DEFAULT_SERVER_OPTIMIZER_FN,
    client_weight_fn: Callable[[Any], tf.Tensor] = None,
    stateful_delta_aggregate_fn: Optional[tff.utils.StatefulAggregateFn] = None,
    stateful_model_broadcast_fn: Optional[tff.utils.StatefulBroadcastFn] = None,
    *,
    broadcast_process: Optional[tff.templates.MeasuredProcess] = None,
    aggregation_process: Optional[tff.templates.MeasuredProcess] = None,
) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for optimization using federated SGD.

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
  client using a broadcast function. Each client sums the gradients at each
  batch in the client's local dataset. These gradient sums are then aggregated
  at the server using an aggregation function. The aggregate gradients are
  applied at the server by using the
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
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of the aggregated gradients. If not provided, the
      default is the total number of examples processed on device.
    stateful_delta_aggregate_fn: A `tff.utils.StatefulAggregateFn` where the
      `next_fn` performs a federated aggregation and upates state. It must have
      TFF type `(<state@SERVER, value@CLIENTS, weights@CLIENTS> ->
      <state@SERVER, aggregate@SERVER>)`, where the `value` type is
      `tff.learning.framework.ModelWeights.trainable` corresponding to the
      object returned by `model_fn`. By default performs arithmetic mean
      aggregation, weighted by `client_weight_fn`. Must be `None` if
      `aggregation_process` is not `None`.
    stateful_model_broadcast_fn: A `tff.utils.StatefulBroadcastFn` where the
      `next_fn` performs a federated broadcast and upates state. It must have
      TFF type `(<state@SERVER, value@SERVER> -> <state@SERVER,
      value@CLIENTS>)`, where the `value` type is
      `tff.learning.framework.ModelWeights` corresponding to the object returned
      by `model_fn`. The default is the identity broadcast. Must be `None` if
      `broadcast_process` is not `None`.
    broadcast_process: a `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`. Must be `None` if
      `stateful_model_broadcast_fn` is not `None`.
    aggregation_process: a `tff.templates.MeasuredProcess` that aggregates the
      model updates on the clients back to the server. It must support the
      signature `({input_values}@CLIENTS-> output_values@SERVER)`. Must be
      `None` if `stateful_delta_aggregate_fn` is not `None`.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  def client_sgd_avg(model_fn):
    return ClientSgd(model_fn(), client_weight_fn)

  return optimizer_utils.build_model_delta_optimizer_process(
      model_fn,
      model_to_client_delta_fn=client_sgd_avg,
      server_optimizer_fn=server_optimizer_fn,
      stateful_delta_aggregate_fn=stateful_delta_aggregate_fn,
      stateful_model_broadcast_fn=stateful_model_broadcast_fn,
      broadcast_process=broadcast_process,
      aggregation_process=aggregation_process)
