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
