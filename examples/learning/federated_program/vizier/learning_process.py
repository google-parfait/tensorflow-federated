# Copyright 2023, The TensorFlow Federated Authors.
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
"""The `tff.learning.templates.LearningProcess`s for this federated program."""

import tensorflow as tf
import tensorflow_federated as tff


def create_learning_processes(
    input_spec,
) -> tuple[
    tff.learning.templates.LearningProcess,
    tff.learning.templates.LearningProcess,
]:
  """Creates the `tff.learning.templates.LearningProcess`s for this program.

  Args:
    input_spec: The `input_spec` to use when creating the model.

  Returns:
    A `tuple` containing the train learning process and evaluation learning
    process.
  """

  def _model_fn() -> tff.learning.models.VariableModel:
    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=5),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(10),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

  client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
  server_optimizer = tff.learning.optimizers.build_adam(learning_rate=0.001)
  model_aggregator = tff.learning.robust_aggregator()
  train_prcess = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=_model_fn,
      client_optimizer_fn=client_optimizer,
      server_optimizer_fn=server_optimizer,
      model_aggregator=model_aggregator,
  )
  evaluation_process = tff.learning.algorithms.build_fed_eval(_model_fn)

  return train_prcess, evaluation_process
