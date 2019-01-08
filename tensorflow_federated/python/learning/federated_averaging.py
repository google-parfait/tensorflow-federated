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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class ClientFedAvg(optimizer_utils.ClientDeltaFn):
  """Client TensorFlow logic for Federated Averaging."""

  def __init__(self, model, client_weight_fn=None):
    """Creates the client computation for Federated Averaging.

    Args:
      model: A `learning.TrainableModel`.
      client_weight_fn: Optional function that takes the output of
        model.report_local_outputs() and returns a tensor that provides
        the weight in the federated average of model deltas. If not provided,
        the default is the total number of examples processed on device.
    """
    self._model = model_utils.enhance(model)
    py_typecheck.check_type(self._model, model_utils.EnhancedTrainableModel)

    self._num_examples = tf.Variable(0, name='num_examples')
    if client_weight_fn is not None:
      py_typecheck.check_callable(client_weight_fn)
      self._client_weight_fn = client_weight_fn
    else:
      self._client_weight_fn = lambda _: tf.cast(self._num_examples, tf.float32)

  @property
  def variables(self):
    return [self._num_examples]

  @tf.contrib.eager.function(autograph=False)
  def __call__(self, dataset, initial_weights):
    # N.B. When not in eager mode, this code must be wrapped as a defun
    # as it uses program-order semantics to avoid adding many explicit
    # control dependencies.
    model = self._model
    py_typecheck.check_type(dataset, tf.data.Dataset)

    # TODO(b/120801384): We should initialize model.local_variables here.
    # Or, we may just need a convention that TFF initializes all variables
    # before invoking the TF function.

    tf.contrib.framework.nest.map_structure(tf.assign, model.weights,
                                            initial_weights)

    @tf.contrib.eager.function(autograph=False)
    def reduce_fn(dummy_state, batch):
      """Runs train_on_batch on batch."""
      output = model.train_on_batch(batch)
      tf.assign_add(self._num_examples, tf.shape(output.predictions)[0])
      return dummy_state

    # TODO(b/121400757): Remove dummy_output when bug fixed.
    dummy_output = dataset.reduce(
        initial_state=tf.constant(0.0), reduce_func=reduce_fn)

    weights_delta = tf.contrib.framework.nest.map_structure(
        tf.subtract, model.weights.trainable, initial_weights.trainable)

    aggregated_outputs = model.report_local_outputs()
    weights_delta_weight = self._client_weight_fn(aggregated_outputs)  # pylint:disable=not-callable

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_average?
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    weights_delta_weight = tf.cond(
        tf.equal(has_non_finite_delta,
                 0), lambda: weights_delta_weight, lambda: tf.constant(0.0))

    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, aggregated_outputs,
        tensor_utils.to_odict({
            'num_examples': self._num_examples.value(),
            'has_non_finite_delta': has_non_finite_delta,
            'workaround for b/121400757': dummy_output,
        }))


def build_federated_averaging_process(model_fn,
                                      server_optimizer_fn=None,
                                      client_weight_fn=None):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a `tf.Optimizer`. The
      apply_gradients method of this optimizer is used to apply client updates
      to the server model. The default returns a `tf.train.GradientDescent` with
      a learning_rate of 1.0, which simply adds the average client delta to the
      server's model.
      client_weight_fn: Optional function that takes the output of
        model.report_local_outputs() and returns a tensor that provides
        the weight in the federated average of model deltas. If not provided,
        the default is the total number of examples processed on device.

  Returns:
    A `tff.utils.IterativeProcess`.
  """

  def client_fed_avg(model_fn):
    return ClientFedAvg(model_fn(), client_weight_fn)

  return optimizer_utils.build_model_delta_optimizer_tff(
      model_fn, client_fed_avg, server_optimizer_fn)
