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
"""Baseline experiment on centralized Stack Overflow LR data."""

from typing import Any, Mapping, Optional

import tensorflow as tf

from tensorflow_federated.python.research.utils import centralized_training_loop
from tensorflow_federated.python.research.utils.datasets import stackoverflow_lr_dataset
from tensorflow_federated.python.research.utils.models import stackoverflow_lr_models


def run_centralized(optimizer: tf.keras.optimizers.Optimizer,
                    experiment_name: str,
                    root_output_dir: str,
                    num_epochs: int,
                    batch_size: int,
                    decay_epochs: Optional[int] = None,
                    lr_decay: Optional[float] = None,
                    hparams_dict: Optional[Mapping[str, Any]] = None,
                    vocab_tokens_size: Optional[int] = 10000,
                    vocab_tags_size: Optional[int] = 500,
                    num_validation_examples: Optional[int] = 10000,
                    max_batches: Optional[int] = None):
  """Trains an RNN on the Stack Overflow next word prediction task.

  Args:
    optimizer: A `tf.keras.optimizers.Optimizer` used to perform training.
    experiment_name: The name of the experiment. Part of the output directory.
    root_output_dir: The top-level output directory for experiment runs. The
      `experiment_name` argument will be appended, and the directory will
      contain tensorboard logs, metrics written as CSVs, and a CSV of
      hyperparameter choices (if `hparams_dict` is used).
    num_epochs: The number of training epochs.
    batch_size: The batch size, used for train, validation, and test.
    decay_epochs: The number of epochs of training before decaying the learning
      rate. If None, no decay occurs.
    lr_decay: The amount to decay the learning rate by after `decay_epochs`
      training epochs have occurred.
    hparams_dict: A mapping with string keys representing the hyperparameters
      and their values. If not None, this is written to CSV.
    vocab_tokens_size: Integer dictating the number of most frequent words to
      use in the vocabulary.
    vocab_tags_size: Integer dictating the number of most frequent tags to use
      in the label creation.
    num_validation_examples: The number of test examples to use for validation.
    max_batches: If set to a positive integer, datasets are capped to at most
      that many batches. If set to None or a nonpositive integer, the full
      datasets are used.
  """

  train_dataset, validation_dataset, test_dataset = stackoverflow_lr_dataset.get_centralized_datasets(
      train_batch_size=batch_size,
      max_train_batches=max_batches,
      max_validation_batches=max_batches,
      max_test_batches=max_batches,
      vocab_tokens_size=vocab_tokens_size,
      vocab_tags_size=vocab_tags_size,
      num_validation_examples=num_validation_examples)

  model = stackoverflow_lr_models.create_logistic_model(
      vocab_tokens_size=vocab_tokens_size, vocab_tags_size=vocab_tags_size)

  model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(
          from_logits=False, reduction=tf.keras.losses.Reduction.SUM),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall(top_k=5)])

  centralized_training_loop.run(
      keras_model=model,
      train_dataset=train_dataset,
      validation_dataset=validation_dataset,
      test_dataset=test_dataset,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      num_epochs=num_epochs,
      hparams_dict=hparams_dict,
      decay_epochs=decay_epochs,
      lr_decay=lr_decay)
