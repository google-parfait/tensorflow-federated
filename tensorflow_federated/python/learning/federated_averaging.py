# Lint as: python3
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

import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientFedAvg(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated Averaging."""

  def __init__(self, model, client_weight_fn=None):
    """Creates the client computation for Federated Averaging.

    Args:
      model: A `tff.learning.TrainableModel`.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.
    """
    self._model = model_utils.enhance(model)
    py_typecheck.check_type(self._model, model_utils.EnhancedTrainableModel)

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
    # TODO(b/113112108): Remove this temporary workaround and restore check for
    # `tf.data.Dataset` after subclassing the currently used custom data set
    # representation from it.
    if 'Dataset' not in str(type(dataset)):
      raise TypeError('Expected a data set, found {}.'.format(
          py_typecheck.type_string(type(dataset))))

    model = self._model
    tf.nest.map_structure(lambda a, b: a.assign(b), model.weights,
                          initial_weights)

    @tf.function
    def reduce_fn(num_examples_sum, batch):
      """Runs `tff.learning.Model.train_on_batch` on local client batch."""
      output = model.train_on_batch(batch)
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
    if self._client_weight_fn is None:
      weights_delta_weight = tf.cast(num_examples_sum, tf.float32)
    else:
      weights_delta_weight = self._client_weight_fn(aggregated_outputs)
    # Zero out the weight if there are any non-finite values.
    if has_non_finite_delta > 0:
      weights_delta_weight = tf.constant(0.0)

    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, aggregated_outputs,
        tensor_utils.to_odict({
            'num_examples': num_examples_sum,
            'has_non_finite_delta': has_non_finite_delta,
        }))


def build_federated_averaging_process(
    model_fn,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_weight_fn=None,
    stateful_delta_aggregate_fn=None,
    stateful_model_broadcast_fn=None):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a `tf.Optimizer`. The
      `apply_gradients` method of this optimizer is used to apply client updates
      to the server model. The default creates a `tf.keras.optimizers.SGD` with
      a learning rate of 1.0, which simply adds the average client delta to the
      server's model.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.
    stateful_delta_aggregate_fn: A `tff.utils.StatefulAggregateFn` where the
      `next_fn` performs a federated aggregation and upates state. That is, it
      has TFF type `(state@SERVER, value@CLIENTS, weights@CLIENTS) ->
      (state@SERVER, aggregate@SERVER)`, where the `value` type is
      `tff.learning.framework.ModelWeights.trainable` corresponding to the
      object returned by `model_fn`. By default performs arithmetic mean
      aggregation, weighted by `client_weight_fn`.
    stateful_model_broadcast_fn: A `tff.utils.StatefulBroadcastFn` where the
      `next_fn` performs a federated broadcast and upates state. That is, it has
      TFF type `(state@SERVER, value@SERVER) -> (state@SERVER, value@CLIENTS)`,
      where the `value` type is `tff.learning.framework.ModelWeights`
      corresponding to the object returned by `model_fn`. By default performs
      identity broadcast.

  Returns:
    A `tff.utils.IterativeProcess`.
  """

  def client_fed_avg(model_fn):
    return ClientFedAvg(model_fn(), client_weight_fn)

  if stateful_delta_aggregate_fn is None:
    stateful_delta_aggregate_fn = optimizer_utils.build_stateless_mean()
  else:
    py_typecheck.check_type(stateful_delta_aggregate_fn,
                            tff.utils.StatefulAggregateFn)

  if stateful_model_broadcast_fn is None:
    stateful_model_broadcast_fn = optimizer_utils.build_stateless_broadcaster()
  else:
    py_typecheck.check_type(stateful_model_broadcast_fn,
                            tff.utils.StatefulBroadcastFn)

  return optimizer_utils.build_model_delta_optimizer_process(
      model_fn, client_fed_avg, server_optimizer_fn,
      stateful_delta_aggregate_fn, stateful_model_broadcast_fn)
