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
"""Federated Stack Overflow tag prediction (via logistic regression) using TFF."""

import functools
from typing import Any, Callable, Optional

from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils.datasets import stackoverflow_lr_dataset
from tensorflow_federated.python.research.utils.models import stackoverflow_lr_models


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  return [
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(top_k=5, name='recall_at_5'),
  ]


def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    assign_weights_fn: Callable[[Any, tf.keras.Model], None],
    client_epochs_per_round: int,
    client_batch_size: int,
    clients_per_round: int,
    client_datasets_random_seed: Optional[int] = None,
    vocab_tokens_size: Optional[int] = 10000,
    vocab_tags_size: Optional[int] = 500,
    max_elements_per_user: Optional[int] = 1000,
    num_validation_examples: Optional[int] = 10000):
  """Runs an iterative process on the Stack Overflow logistic regression task.

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
      returns a `tff.templates.IterativeProcess`. The `model_fn` must return a
      `tff.learning.Model`.
    assign_weights_fn: A function that accepts the server state `S` and a
      `tf.keras.Model`, and updates the weights in the Keras model. This is used
      to do evaluation using Keras.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    vocab_tokens_size: Integer dictating the number of most frequent words to
      use in the vocabulary.
    vocab_tags_size: Integer dictating the number of most frequent tags to use
      in the label creation.
    max_elements_per_user: The maximum number of elements processed for each
      client's dataset.
    num_validation_examples: The number of test examples to use for validation.
  """

  stackoverflow_train, stackoverflow_validation, stackoverflow_test = stackoverflow_lr_dataset.get_stackoverflow_datasets(
      vocab_tokens_size=vocab_tokens_size,
      vocab_tags_size=vocab_tags_size,
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      max_training_elements_per_user=max_elements_per_user,
      num_validation_examples=num_validation_examples)

  input_spec = stackoverflow_train.create_tf_dataset_for_client(
      stackoverflow_train.client_ids[0]).element_spec

  model_builder = functools.partial(
      stackoverflow_lr_models.create_logistic_model,
      vocab_tokens_size=vocab_tokens_size,
      vocab_tags_size=vocab_tags_size)

  loss_builder = functools.partial(
      tf.keras.losses.BinaryCrossentropy,
      from_logits=False,
      reduction=tf.keras.losses.Reduction.SUM)

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  training_process = iterative_process_builder(tff_model_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_dataset=stackoverflow_train,
      train_clients_per_round=clients_per_round,
      random_seed=client_datasets_random_seed)

  evaluate_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=stackoverflow_validation,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  test_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      eval_dataset=stackoverflow_validation.concatenate(stackoverflow_test),
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      validation_fn=evaluate_fn,
      test_fn=test_fn)
