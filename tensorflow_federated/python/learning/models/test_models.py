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
"""Module of `FunctionalModel` implementations for useful for tests."""

import collections
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import variable


def build_functional_linear_regression(
    feature_dim: int = 2,
    has_unconnected: bool = False,
) -> functional.FunctionalModel:
  """Build a linear regression FunctionalModel for testing."""
  input_spec = (
      tf.TensorSpec([None, feature_dim], tf.float32),
      tf.TensorSpec([None, 1], tf.float32),
  )

  if has_unconnected:
    initial_trainable_weights = (
        np.reshape(np.zeros([feature_dim]), [feature_dim, 1]).astype(
            np.float32
        ),
        np.zeros([1]).astype(np.float32),
        np.zeros([1]).astype(np.float32),
    )
  else:
    initial_trainable_weights = (
        np.reshape(np.zeros([feature_dim]), [feature_dim, 1]).astype(
            np.float32
        ),
        np.zeros([1]).astype(np.float32),
    )
  initial_non_trainable_weights = ()
  initial_weights = (initial_trainable_weights, initial_non_trainable_weights)

  @tf.function
  def predict_on_batch(
      weights: functional.ModelWeights, x: Any, training: bool = True
  ) -> Any:
    trainable_weights, _ = weights
    if has_unconnected:
      kernel, bias, unconnected = trainable_weights
      del unconnected  # Unconnected weights
    else:
      kernel, bias = trainable_weights
    del training  # Unused.
    return x @ kernel + bias

  def loss(output: Any, label: Any, sample_weight: Any) -> float:
    del sample_weight
    return tf.math.reduce_mean(tf.math.pow(output - label, 2.0))

  @tf.function
  def initialize_metrics() -> types.MetricsState:
    return collections.OrderedDict(
        loss=tf.constant(0.0, tf.float32), num_examples=tf.constant(0, tf.int32)
    )

  @tf.function
  def update_metrics_state(
      state: types.MetricsState,
      labels: Any,
      batch_output: variable.BatchOutput,
      sample_weight: Optional[Any] = None,
  ) -> types.MetricsState:
    del sample_weight  # Unused.
    batch_size = tf.shape(labels)[0]
    predictions = batch_output.predictions
    loss = tf.math.reduce_sum(tf.math.pow(predictions - labels, 2.0))
    return collections.OrderedDict(
        loss=state["loss"] + loss,
        num_examples=state["num_examples"] + batch_size,
    )

  @tf.function
  def finalize_metrics(state: types.MetricsState):
    return collections.OrderedDict(
        loss=tf.math.divide_no_nan(
            state["loss"], tf.cast(state["num_examples"], tf.float32)
        ),
        num_examples=state["num_examples"],
    )

  return functional.FunctionalModel(
      initial_weights=initial_weights,
      predict_on_batch_fn=predict_on_batch,
      loss_fn=loss,
      metrics_fns=(initialize_metrics, update_metrics_state, finalize_metrics),
      input_spec=input_spec,
  )
