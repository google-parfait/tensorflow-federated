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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils

nest = tf.contrib.framework.nest


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
    if isinstance(self._model, model_lib.TrainableModel):
      raise ValueError(
          'Do not pass a TrainableModel to ClientSgd, as the '
          'built-in local training algorithm would be ignored. '
          'This failure could be made into a warning if this is inconvenient.')

  @property
  def variables(self):
    return []

  # TODO(b/124777499): Remove `autograph=False` when possible.
  @tf.contrib.eager.function(autograph=False)
  def __call__(self, dataset, initial_weights):
    model = self._model

    # TODO(b/113112108): Remove this temporary workaround and restore check for
    # `tf.data.Dataset` after subclassing the currently used custom data set
    # representation from it.
    if 'Dataset' not in str(type(dataset)):
      raise TypeError('Expected a data set, found {}.'.format(
          py_typecheck.type_string(type(dataset))))

    nest.map_structure(tf.assign, model.weights, initial_weights)
    flat_trainable_weights = tuple(nest.flatten(model.weights.trainable))

    # TODO(b/124777499): Remove `autograph=False` when possible.
    @tf.contrib.eager.function(autograph=False)
    def reduce_fn(state, batch):
      """Runs forward_pass on batch and sums the weighted gradients."""
      flat_accumulated_grads, batch_weight_sum = state

      with tf.contrib.eager.GradientTape() as tape:
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
    grad_sums = nest.pack_sequence_as(model.weights.trainable, flat_grad_sums)

    # For SGD, the delta is just the negative of the average gradient:
    weights_delta = nest.map_structure(
        lambda gradient: -1.0 * gradient / batch_weight_sum, grad_sums)
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    weights_delta_weight = tf.cond(
        tf.equal(has_non_finite_delta,
                 0), lambda: batch_weight_sum, lambda: tf.constant(0.0))

    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, model.report_local_outputs(),
        tensor_utils.to_odict({
            'client_weight': weights_delta_weight,
            'has_non_finite_delta': has_non_finite_delta,
        }))


def build_federated_sgd_process(
    model_fn,
    server_optimizer_fn=lambda: gradient_descent.SGD(learning_rate=0.1),
    client_weight_fn=None):
  """Builds the TFF computations for optimization using federated SGD.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a `tf.Optimizer`. The
      `apply_gradients` method of this optimizer is used to apply client updates
      to the server model.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.utils.IterativeProcess`.
  """

  def client_sgd_avg(model_fn):
    return ClientSgd(model_fn(), client_weight_fn)

  return optimizer_utils.build_model_delta_optimizer_process(
      model_fn, client_sgd_avg, server_optimizer_fn)
