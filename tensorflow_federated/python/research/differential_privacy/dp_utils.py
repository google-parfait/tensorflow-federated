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
"""Utilities supporting DP-FedAvg experiments."""

import tensorflow_federated as tff

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.research.utils import adapters


class DPFedAvgProcessAdapter(adapters.IterativeProcessPythonAdapter):
  """Converts iterative process results from anonymous tuples.

  Converts to ServerState and unpacks metrics, including adding the vector
  clips as metrics.
  """

  def __init__(self, iterative_process, per_vector_clipping, adaptive_clipping):
    self._iterative_process = iterative_process
    self._per_vector_clipping = per_vector_clipping
    self._adaptive_clipping = adaptive_clipping

  def _get_clip(self, state):
    return state.numerator_state.sum_state.l2_norm_clip

  def _server_state_from_tff_result(self, result):
    if self._per_vector_clipping:
      per_vector_aggregate_states = [
          anonymous_tuple.to_odict(elt, recursive=True) for _, elt in
          anonymous_tuple.iter_elements(result.delta_aggregate_state)
      ]
    else:
      per_vector_aggregate_states = anonymous_tuple.to_odict(
          result.delta_aggregate_state, recursive=True)
    return tff.learning.framework.ServerState(
        tff.learning.ModelWeights(
            tuple(result.model.trainable), tuple(result.model.non_trainable)),
        list(result.optimizer_state), per_vector_aggregate_states,
        tuple(result.model_broadcast_state))

  def initialize(self):
    initial_state = self._iterative_process.initialize()
    return self._server_state_from_tff_result(initial_state)

  def next(self, state, data):
    state, metrics = self._iterative_process.next(state, data)
    python_state = self._server_state_from_tff_result(state)
    metrics = metrics._asdict(recursive=True)
    if self._adaptive_clipping:
      if self._per_vector_clipping:
        metrics.update({
            ('clip_' + str(i)): self._get_clip(vector_state)
            for i, vector_state in enumerate(state.delta_aggregate_state)
        })
      else:
        metrics.update({'clip': self._get_clip(state.delta_aggregate_state)})

    outputs = None
    return adapters.IterationResult(python_state, metrics, outputs)


def assign_weights_to_keras_model(reference_model, keras_model):
  """Assign the model weights to the weights of a `tf.keras.Model`.

  Args:
    reference_model: The model to assign weights from. Must be an instance of
      `tff.learning.ModelWeights`.
    keras_model: the `tf.keras.Model` object to assign weights to.
  """
  if not isinstance(reference_model, tff.learning.ModelWeights):
    raise TypeError('The reference model must be an instance of '
                    'tff.learning.ModelWeights.')

  def assign_weights(keras_weights, tff_weights):
    for k, w in zip(keras_weights, tff_weights):
      k.assign(w)

  assign_weights(keras_model.trainable_weights, reference_model.trainable)
  assign_weights(keras_model.non_trainable_weights,
                 reference_model.non_trainable)
