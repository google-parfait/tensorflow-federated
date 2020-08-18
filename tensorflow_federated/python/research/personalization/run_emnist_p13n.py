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

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.personalization.p13n_utils import build_personalize_fn
from tensorflow_federated.python.research.personalization.p13n_utils import evaluate_fn
from tensorflow_federated.python.research.utils.models import emnist_models

# All parameters (except the last one) defined here are for FedAvg training.
# For simplicity, parameters used in personalization (or `p13n` for short) are
# explicitly set in the `main` function below.
flags.DEFINE_integer('num_clients_per_round', 20,
                     'Number of clients per FedAvg training round.')
flags.DEFINE_integer('num_epochs_per_round', 10,
                     'Number of training epochs in each FedAvg round.')
flags.DEFINE_integer('batch_size', 20, 'Batch size used in FedAvg.')
flags.DEFINE_integer('num_total_rounds', 10,
                     'Number of total FedAvg training rounds.')
flags.DEFINE_integer(
    'num_rounds_per_p13n_eval', 5,
    'Number of FedAvg training rounds between two p13n evals.')
flags.DEFINE_bool(
    'shuffle_clients_before_split', True,
    'Whether to shuffle the clients before splitting into training and p13n.')

FLAGS = flags.FLAGS

SEED = 42  # Seed used when splitting the clients into training and p13n.


def _get_emnist_datasets():
  """Pre-process EMNIST-62 dataset for FedAvg and personalization."""

  def element_fn(element):
    return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

  def preprocess_train_data(dataset):
    """Pre-process the dataset for training the global model."""
    return dataset.repeat(
        FLAGS.num_epochs_per_round).map(element_fn).shuffle(1000).batch(
            FLAGS.batch_size)

  def preprocess_p13n_data(dataset):
    """Pre-process the dataset for training/evaluating a personalized model."""
    # Note: the clients are expected to provide *unbatched* datasets here. Any
    # pre-processing of the dataset (such as batching) can be done inside the
    # `personalize_fn`s. This allows users to evaluate different personalization
    # strategies with different pre-processing methods.
    return dataset.map(element_fn)

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=False)  # EMNIST has 3400 clients.

  if FLAGS.shuffle_clients_before_split:
    # Shuffle the client ids before splitting into training and personalization.
    client_ids = list(
        np.random.RandomState(SEED).permutation(emnist_train.client_ids))
  else:
    client_ids = emnist_train.client_ids

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Create datasets for training global and personalized models.
  federated_train_data, federated_p13n_data = _get_emnist_datasets()

  def model_fn():
    """Build a `tff.learning.Model` for training EMNIST."""
    keras_model = emnist_models.create_conv_dropout_model(only_digits=False)
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
      train_batch_size=20,
      max_num_epochs=10,
      num_epochs_per_eval=1,
      test_batch_size=20)
  adam_opt = lambda: tf.keras.optimizers.Adam(learning_rate=0.02)
  personalize_fn_dict['adam'] = functools.partial(
      build_personalize_fn,
      optimizer_fn=adam_opt,
      train_batch_size=20,
      max_num_epochs=10,
      num_epochs_per_eval=1,
      test_batch_size=20)

  # Build the `tff.Computation` for evaluating the personalization strategies.
  # Here `p13n_eval` is a `tff.Computation` with the following type signature:
  # <model_weights@SERVER, datasets@CLIENTS> -> personalization_metrics@SERVER.
  p13n_eval = tff.learning.build_personalization_eval(
      model_fn=model_fn,
      personalize_fn_dict=personalize_fn_dict,
      baseline_evaluate_fn=functools.partial(evaluate_fn, batch_size=10),
      max_num_samples=900)  # Metrics from all p13n clients will be returned.

  # Start the training loop.
  for round_idx in range(1, FLAGS.num_total_rounds + 1):
    sampled_train_data = list(
        np.random.choice(
            federated_train_data, FLAGS.num_clients_per_round, replace=False))
    server_state, _ = iterative_process.next(server_state, sampled_train_data)

    if round_idx % FLAGS.num_rounds_per_p13n_eval == 0:
      # Invoke the constructed `tff.Computation`. Below we run `p13n_eval` for
      # 18 rounds with 50 clients per round. This will take some time to finish.
      # The returned `p13n_metrics` is a nested dictionary that stores all the
      # personalization metrics from the clients.
      p13n_metrics = p13n_eval(server_state.model, federated_p13n_data[:50])
      p13n_metrics = p13n_metrics._asdict(recursive=True)  # Convert to a dict.
      for i in range(1, 18):
        current_p13n_metrics = p13n_eval(
            server_state.model,
            federated_p13n_data[i * 50:(i + 1) * 50])._asdict(recursive=True)

        p13n_metrics = tf.nest.map_structure(
            lambda a, b: tf.concat([a, b], axis=0), p13n_metrics,
            current_p13n_metrics)
      # Specifically, `p13n_metrics` is an `OrderedDict` that maps
      # key 'baseline_metrics' to the evaluation metrics of the initial global
      # model (computed by `baseline_evaluate_fn` argument in `p13n_eval`), and
      # maps keys (strategy names) in `personalize_fn_dict` to the evaluation
      # metrics of the corresponding personalization strategies.
      #
      # Only metrics from at most `max_num_samples` participating clients are
      # collected (clients are sampled without replacement). Each metric is
      # mapped to a list of scalars (each scalar comes from one client). Metric
      # values at the same position, e.g., metric_1[i], metric_2[i]..., come
      # from the same client.
      #
      # Users can save `p13n_metrics` to file for further analysis. For
      # simplcity, we extract and print two values here:
      # 1. mean accuracy of the initial global model;
      # 2. mean accuracy of the personalized models obtained at Epoch 1.
      print('Current Round {}'.format(round_idx))

      global_model_accuracies = np.array(
          p13n_metrics['baseline_metrics']['sparse_categorical_accuracy'])
      print('Mean accuracy of the global model: {}'.format(
          np.mean(global_model_accuracies).item()))

      personalized_models_accuracies = np.array(
          p13n_metrics['sgd']['epoch_1']['sparse_categorical_accuracy'])
      print('Mean accuracy of the personalized models at Epoch 1: {}'.format(
          np.mean(personalized_models_accuracies).item()))


if __name__ == '__main__':
  app.run(main)
