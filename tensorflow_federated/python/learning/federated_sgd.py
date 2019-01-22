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

import tensorflow as tf

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
    del model
    py_typecheck.check_type(self._model, model_utils.EnhancedModel)
    if isinstance(self._model, model_lib.TrainableModel):
      raise ValueError(
          'Do not pass a TrainableModel to ClientSgd, as the '
          'built-in local training algorithm would be ignored. '
          'This failure could be made into a warning if this is inconvenient.')

    def _get_grad_var(name, tensor):
      return tf.Variable(
          lambda: tf.zeros_like(tensor), name='{}_grad'.format(name))

    self._grad_sum_vars = nest.map_structure_with_paths(
        _get_grad_var, self._model.weights.trainable)
    self._batch_weight_sum = tf.Variable(0.0, name='batch_weight_sum')

  @property
  def variables(self):
    return [self._batch_weight_sum] + nest.flatten(self._grad_sum_vars)

  @tf.contrib.eager.function(autograph=False)
  def __call__(self, dataset, initial_weights):
    # N.B. When not in eager mode, this code must be wrapped as a defun
    # as it uses program-order semantics to avoid adding many explicit
    # control dependencies.
    model = self._model
    py_typecheck.check_type(dataset, tf.data.Dataset)

    nest.map_structure(tf.assign, model.weights, initial_weights)

    @tf.contrib.eager.function(autograph=False)
    def reduce_fn(dummy_state, batch):
      """Runs forward_pass on batch."""
      with tf.contrib.eager.GradientTape() as tape:
        output = model.forward_pass(batch)

      flat_vars = nest.flatten(model.weights.trainable)
      grads = nest.pack_sequence_as(self._grad_sum_vars,
                                    tape.gradient(output.loss, flat_vars))

      if self._batch_weight_fn is not None:
        batch_weight = self._batch_weight_fn(batch)
      else:
        batch_weight = tf.cast(tf.shape(output.predictions)[0], tf.float32)

      tf.assign_add(self._batch_weight_sum, batch_weight)
      nest.map_structure(
          lambda v, g:  # pylint:disable=g-long-lambda
          tf.assign_add(v, batch_weight * g),
          self._grad_sum_vars,
          grads)

      return dummy_state

    # TODO(b/121400757): Remove dummy_output when bug fixed.
    dummy_output = dataset.reduce(
        initial_state=tf.constant(0.0), reduce_func=reduce_fn)

    # For SGD, the delta is just the negative of the average gradient:
    # TODO(b/109733734): Might be better to send the weighted grad sums
    # and the denominator separately?
    weights_delta = nest.map_structure(
        lambda g: -1.0 * g / self._batch_weight_sum, self._grad_sum_vars)
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    weights_delta_weight = tf.cond(
        tf.equal(has_non_finite_delta,
                 0), lambda: self._batch_weight_sum, lambda: tf.constant(0.0))

    return optimizer_utils.ClientOutput(
        weights_delta, weights_delta_weight, model.report_local_outputs(),
        tensor_utils.to_odict({
            'client_weight': weights_delta_weight,
            'has_non_finite_delta': has_non_finite_delta,
            'workaround for b/121400757': dummy_output,
        }))
