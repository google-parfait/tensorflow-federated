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
"""Federated Shakespeare next character prediction library using TFF."""

import functools
from typing import Any, Callable, Optional

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils.datasets import shakespeare_dataset
from tensorflow_federated.python.research.utils.models import shakespeare_models


# Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
VOCAB_SIZE = len(shakespeare_dataset.CHAR_VOCAB) + 4


def create_shakespeare_model(sequence_length):
  """Constructs a `tf.keras.Model` to train."""
  return shakespeare_models.create_recurrent_model(
      vocab_size=VOCAB_SIZE, sequence_length=sequence_length)


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  pad_token, _, _, _ = shakespeare_dataset.get_special_tokens()

  return [
      keras_metrics.NumBatchesCounter(),
      keras_metrics.NumExamplesCounter(),
      keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token]),
  ]


def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    assign_weights_fn: Callable[[Any, tf.keras.Model], None],
    client_epochs_per_round: int,
    client_batch_size: int,
    clients_per_round: int,
    max_batches_per_client: Optional[int] = -1,
    client_datasets_random_seed: Optional[int] = None,
    sequence_length: Optional[int] = 80,
    total_rounds: Optional[int] = 1500,
    experiment_name: Optional[str] = 'federated_shakespeare',
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    max_eval_batches: Optional[int] = None,
    **kwargs):
  """Runs an iterative process on a Shakespeare next character prediction task.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `tensorflow_federated.python.research.utils.training_loop`.

   We assume that the iterative process has the following functional type
   signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  Args:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, and
      a `client_weight_fn`, and returns a `tff.templates.IterativeProcess`. The
      `model_fn` must return a `tff.learning.Model`.
    assign_weights_fn: A function that accepts the server state `S` and a
      `tf.keras.Model`, and updates the weights in the Keras model. This is used
      to do evaluation using Keras.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    max_batches_per_client: An optional int specifying the number of batches
      taken by each client at each round. If `-1`, the entire client dataset is
      used.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    sequence_length: An int specifying the length of the character sequences
      used for prediction.
    total_rounds: The number of federated training rounds.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    max_eval_batches: If set to a positive integer, evaluation datasets are
      capped to at most that many batches. If set to None or a nonpositive
      integer, the full evaluation datasets are used.
    **kwargs: Additional arguments configuring the training loop. For details
      on supported arguments, see
      `tensorflow_federated/python/research/utils/training_utils.py`.
  """

  train_clientdata = shakespeare_dataset.construct_character_level_datasets(
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      sequence_length=sequence_length,
      max_batches_per_client=max_batches_per_client)

  _, test_dataset = shakespeare_dataset.get_centralized_datasets(
      train_batch_size=client_batch_size,
      max_test_batches=max_eval_batches,
      sequence_length=sequence_length)

  model_builder = functools.partial(
      create_shakespeare_model, sequence_length=sequence_length)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  input_spec = train_clientdata.create_tf_dataset_for_client(
      train_clientdata.client_ids[0]).element_spec

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  training_process = iterative_process_builder(
      tff_model_fn, client_weight_fn=client_weight_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_dataset=train_clientdata,
      train_clients_per_round=clients_per_round,
      random_seed=client_datasets_random_seed)

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=test_dataset,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      validation_fn=evaluate_fn,
      test_fn=evaluate_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
