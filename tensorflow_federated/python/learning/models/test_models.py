# Copyright 2022, The TensorFlow Federated Authors.
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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Module of `FunctionalModel` implementations for useful for tests."""

from typing import Any

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning.models import functional


def build_functional_linear_regression(
    feature_dim: int = 2) -> functional.FunctionalModel:
  """Build a linear regression FunctionalModel for testing."""
  input_spec = (
      tf.TensorSpec([None, feature_dim], tf.float32),
      tf.TensorSpec([None, 1], tf.float32),
  )

  initial_trainable_weights = (np.reshape(
      np.zeros([feature_dim]),
      [feature_dim, 1]).astype(np.float32), np.zeros([1]).astype(np.float32))
  initial_non_trainable_weights = ()
  initial_weights = (initial_trainable_weights, initial_non_trainable_weights)

  @tf.function
  def predict_on_batch(weights: functional.ModelWeights,
                       x: Any,
                       training: bool = True) -> Any:
    trainable_weights, _ = weights
    kernel, bias = trainable_weights
    del training  # Unused.
    return x @ kernel + bias

  @tf.function
  def forward_pass(weights: functional.ModelWeights,
                   batch_input: Any,
                   training: bool = True) -> model_lib.BatchOutput:
    x, y = batch_input
    if not input_spec[1].is_compatible_with(y):
      raise ValueError("Expected batch_input[1] to be compatible with "
                       f"{input_spec[1]} but found {y}")
    if not input_spec[0].is_compatible_with(x):
      raise ValueError("Expected batch_input[0] to be compatible with "
                       "{input_spec[0]} but found {x}")
    predictions = predict_on_batch(weights, x=x, training=training)
    residuals = predictions - y
    num_examples = tf.gather(tf.shape(predictions), 0)
    total_loss = tf.reduce_sum(tf.pow(residuals, 2))
    average_loss = total_loss / tf.cast(num_examples, tf.float32)
    return model_lib.BatchOutput(
        loss=average_loss, predictions=predictions, num_examples=num_examples)

  return functional.FunctionalModel(
      initial_weights=initial_weights,
      forward_pass_fn=forward_pass,
      predict_on_batch_fn=predict_on_batch,
      input_spec=input_spec)
