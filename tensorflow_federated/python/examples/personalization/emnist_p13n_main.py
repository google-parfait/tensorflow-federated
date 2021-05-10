# Copyright 2020, The TensorFlow Federated Authors.
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
"""A simple personalization experiment on EMNIST-62 using local fine-tuning."""

import collections
import functools
from typing import Dict, List, Tuple

from absl import app
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.examples.personalization.p13n_utils import build_personalize_fn
from tensorflow_federated.python.examples.personalization.p13n_utils import evaluate_fn


def _get_emnist_datasets(
) -> Tuple[List[tf.data.Dataset], List[Dict[str, tf.data.Dataset]]]:
  """Pre-process EMNIST-62 dataset for FedAvg and personalization."""

  def element_fn(element):
    return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

  def preprocess_train_data(dataset):
    """Pre-process the dataset for training the global model."""
    num_epochs_per_round = 10
    batch_size = 20
    buffer_size = 1000
    return dataset.repeat(num_epochs_per_round).map(element_fn).shuffle(
        buffer_size).batch(batch_size)

  def preprocess_p13n_data(dataset):
    """Pre-process the dataset for training/evaluating a personalized model."""
    # Note: the clients are expected to provide *unbatched* datasets here. Any
    # pre-processing of the dataset (such as batching) can be done inside the
    # `personalize_fn`s. This allows users to evaluate different personalization
    # strategies with different pre-processing methods.
    return dataset.map(element_fn)

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=False)  # EMNIST has 3400 clients.

  # Shuffle the client ids before splitting into training and personalization.
  client_ids = list(
      np.random.RandomState(seed=42).permutation(emnist_train.client_ids))

  # The first 2500 clients are used for training a global model.
  federated_train_data = [
      preprocess_train_data(emnist_train.create_tf_dataset_for_client(c))
      for c in client_ids[:2500]
  ]

  # The remaining 900 clients are used for training and evaluating personalized
  # models. Each client's input is an `OrdereDict` with at least two keys:
  # `train_data` and `test_data`, and each key is mapped to a `tf.data.Dataset`.
  federated_p13n_data = []
  for c in client_ids[2500:]:
    federated_p13n_data.append(
        collections.OrderedDict([
            ('train_data',
             preprocess_p13n_data(
                 emnist_train.create_tf_dataset_for_client(c))),
            ('test_data',
             preprocess_p13n_data(emnist_test.create_tf_dataset_for_client(c)))
        ]))

  return federated_train_data, federated_p13n_data


def _create_conv_dropout_model(only_digits: bool = True) -> tf.keras.Model:
  """A convolutional model to use for EMNIST experiments.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(
          64, kernel_size=(3, 3), activation='relu', data_format=data_format),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  return model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Create datasets for training global and personalized models.
  federated_train_data, federated_p13n_data = _get_emnist_datasets()

  def model_fn() -> tff.learning.Model:
    """Build a `tff.learning.Model` for training EMNIST."""
    keras_model = _create_conv_dropout_model(only_digits=False)
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=federated_train_data[0].element_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  # Build a standard federated averaging process for training the global model.
  client_opt = lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
  server_opt = lambda: tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.9)
  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=client_opt,
      server_optimizer_fn=server_opt)

  # Initialize the server state of the FedAvg process.
  server_state = iterative_process.initialize()

  # Create a dictionary of two personalization strategies: one uses SGD while
  # the other one uses Adam optimizer to train a personalized model.
  #
  # Here `personalize_fn_dict` is an `OrderedDict` that maps a strategy name to
  # a no-argument function which, when being called, returns a `tf.function`
  # that represents a personalization strategy. Customers can define arbitrary
  # personalization strategis (i.e., `tf.function`s) that take a
  # `tff.learning.Model`, an unbatched `tf.data.Dataset` for train, an unbatched
  # `tf.data.Dataset` for test, (and an extra `context` object), and returns the
  # personalization metrics (see `build_personalize_fn` for an example).
  personalize_fn_dict = collections.OrderedDict()
  sgd_opt = lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
  personalize_fn_dict['sgd'] = functools.partial(
      build_personalize_fn,
      optimizer_fn=sgd_opt,
      batch_size=20,
      num_epochs=3,
      num_epochs_per_eval=1)
  adam_opt = lambda: tf.keras.optimizers.Adam(learning_rate=0.002)
  personalize_fn_dict['adam'] = functools.partial(
      build_personalize_fn,
      optimizer_fn=adam_opt,
      batch_size=20,
      num_epochs=3,
      num_epochs_per_eval=1)

  # Build the `tff.Computation` for evaluating the personalization strategies.
  # Here `p13n_eval` is a `tff.Computation` with the following type signature:
  # <model_weights@SERVER, datasets@CLIENTS> -> personalization_metrics@SERVER.
  p13n_eval = tff.learning.build_personalization_eval(
      model_fn=model_fn,
      personalize_fn_dict=personalize_fn_dict,
      baseline_evaluate_fn=evaluate_fn,
      max_num_clients=100)  # Metrics from at most 100 clients will be returned.

  # Train a global model using the standard FedAvg algorithm.
  num_total_rounds = 5
  num_clients_per_round = 10
  print(f'Running FedAvg for {num_total_rounds} training rounds...')
  for _ in range(1, num_total_rounds + 1):
    sampled_train_data = list(
        np.random.choice(
            federated_train_data, num_clients_per_round, replace=False))
    server_state, _ = iterative_process.next(server_state, sampled_train_data)

  print('Evaluating the personalization strategies on the global model...')
  # Run `p13n_eval` on a federated dataset of 50 randomly sampled clients.
  # The returned `p13n_metrics` is a nested dictionary that stores the
  # evaluation metrics from 50 clients.
  num_clients_do_p13n_eval = 50
  sampled_p13n_data = list(
      np.random.choice(
          federated_p13n_data, num_clients_do_p13n_eval, replace=False))
  p13n_metrics = p13n_eval(server_state.model, sampled_p13n_data)
  # Specifically, `p13n_metrics` is an `OrderedDict` that maps
  # key 'baseline_metrics' to the evaluation metrics of the initial global
  # model (computed by `baseline_evaluate_fn` argument in `p13n_eval`), and
  # maps keys (strategy names) in `personalize_fn_dict` to the evaluation
  # metrics of the corresponding personalization strategies.
  #
  # Only metrics from at most `max_num_clients` participating clients are
  # collected (clients are sampled without replacement). Each metric is
  # mapped to a list of scalars (each scalar comes from one client). Metric
  # values at the same position, e.g., `metric_1[i]`, `metric_2[i]`, ...,
  # come from the same client.
  #
  # Users can save `p13n_metrics` to file for further analysis. For
  # simplcity, we extract and print three values here:
  # 1. mean accuracy of the initial global model;
  # 2. mean accuracy of SGD-trained personalized models obtained at Epoch 1.
  # 3. mean accuracy of Adam-trained personalized models obtained at Epoch 1.
  global_model_accuracies = np.array(
      p13n_metrics['baseline_metrics']['sparse_categorical_accuracy'])
  mean_global_acc = np.mean(global_model_accuracies).item()
  print(f'Mean accuracy of the global model: {mean_global_acc}.')

  print('Mean accuracy of the personalized models at Epoch 1:')
  personalized_models_accuracies_sgd = np.array(
      p13n_metrics['sgd']['epoch_1']['sparse_categorical_accuracy'])
  mean_p13n_acc_sgd = np.mean(personalized_models_accuracies_sgd).item()
  print(f'SGD-trained personalized models: {mean_p13n_acc_sgd}.')

  personalized_models_accuracies_adam = np.array(
      p13n_metrics['adam']['epoch_1']['sparse_categorical_accuracy'])
  mean_p13n_acc_adam = np.mean(personalized_models_accuracies_adam).item()
  print(f'Adam-trained personalized models: {mean_p13n_acc_adam}.')


if __name__ == '__main__':
  app.run(main)
