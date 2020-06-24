# Copyright 2019, The TensorFlow Federated Authors.
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
"""A simple implementation of federated evaluation."""

import collections

import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.learning import model_utils


def build_federated_evaluation(model_fn):
  """Builds the TFF computation for federated evaluation of the given model.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and federated data, and returns the evaluation metrics
    as aggregated by `tff.learning.Model.federated_output_computation`.
  """
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  # TODO(b/124477628): Ideally replace the need for stamping throwaway models
  # with some other mechanism.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_utils.weights_type_from_model(model)
    batch_type = tff.to_type(model.input_spec)

  @tff.tf_computation(model_weights_type, tff.SequenceType(batch_type))
  def client_eval(incoming_model_weights, dataset):
    """Returns local outputs after evaluting `model_weights` on `dataset`."""

    model = model_utils.enhance(model_fn())

    @tf.function
    def _tf_client_eval(incoming_model_weights, dataset):
      """Evaluation TF work."""

      tff.utils.assign(model.weights, incoming_model_weights)

      def reduce_fn(prev_loss, batch):
        model_output = model.forward_pass(batch, training=False)
        return prev_loss + tf.cast(model_output.loss, tf.float64)

      dataset.reduce(tf.constant(0.0, dtype=tf.float64), reduce_fn)

      return collections.OrderedDict([('local_outputs',
                                       model.report_local_outputs())])

    return _tf_client_eval(incoming_model_weights, dataset)

  @tff.federated_computation(
      tff.FederatedType(model_weights_type, tff.SERVER),
      tff.FederatedType(tff.SequenceType(batch_type), tff.CLIENTS))
  def server_eval(server_model_weights, federated_dataset):
    client_outputs = tff.federated_map(
        client_eval,
        [tff.federated_broadcast(server_model_weights), federated_dataset])
    return model.federated_output_computation(client_outputs.local_outputs)

  return server_eval
