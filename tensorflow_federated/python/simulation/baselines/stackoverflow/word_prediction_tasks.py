# Copyright 2021, The TensorFlow Federated Authors.
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
"""Library for creating word prediction tasks on Stack Overflow."""

from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import keras_metrics
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.stackoverflow import constants
from tensorflow_federated.python.simulation.baselines.stackoverflow import word_prediction_models
from tensorflow_federated.python.simulation.baselines.stackoverflow import word_prediction_preprocessing
from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import stackoverflow


def create_word_prediction_task_from_datasets(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec],
    sequence_length: int,
    vocab_size: int,
    num_out_of_vocab_buckets: int,
    train_data: client_data.ClientData,
    test_data: client_data.ClientData,
    validation_data: client_data.ClientData,
) -> baseline_task.BaselineTask:
  """Creates a baseline task for next-word prediction on Stack Overflow.

  The goal of the task is to take `sequence_length` words from a post and
  predict the next word. Here, all posts are drawn from the Stack Overflow
  forum, and a client corresponds to a user.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    sequence_length: A positive integer dictating the length of each word
      sequence in a client's dataset. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_SEQUENCE_LENGTH`.
    vocab_size: Integer dictating the number of most frequent words in the
      entire corpus to use for the task's vocabulary. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_WORD_VOCAB_SIZE`.
    num_out_of_vocab_buckets: The number of out-of-vocabulary buckets to use.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.
    validation_data: A `tff.simulation.datasets.ClientData` used for validation.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer')
  if vocab_size < 1:
    raise ValueError('vocab_size must be a positive integer')
  if num_out_of_vocab_buckets < 1:
    raise ValueError('num_out_of_vocab_buckets must be a positive integer')

  vocab = list(stackoverflow.load_word_counts(vocab_size=vocab_size).keys())

  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)

  train_preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
      train_client_spec,
      vocab,
      sequence_length=sequence_length,
      num_out_of_vocab_buckets=num_out_of_vocab_buckets)
  eval_preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
      eval_client_spec,
      vocab,
      sequence_length=sequence_length,
      num_out_of_vocab_buckets=num_out_of_vocab_buckets)

  task_datasets = task_data.BaselineTaskDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data=validation_data,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  special_tokens = word_prediction_preprocessing.get_special_tokens(
      vocab_size, num_out_of_vocab_buckets=num_out_of_vocab_buckets)
  pad_token = special_tokens.padding
  oov_tokens = special_tokens.out_of_vocab
  eos_token = special_tokens.end_of_sentence

  def metrics_builder():
    return [
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy', masked_tokens=[pad_token]),
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_without_out_of_vocab',
            masked_tokens=[pad_token] + oov_tokens),
        # Notice that the beginning of sentence token never appears in the
        # ground truth label.
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_without_out_of_vocab_or_end_of_sentence',
            masked_tokens=[pad_token, eos_token] + oov_tokens),
    ]

  # The total vocabulary size is the number of words in the vocabulary, plus
  # the number of out-of-vocabulary tokens, plus three tokens used for
  # padding, beginning of sentence and end of sentence.
  extended_vocab_size = (
      vocab_size + special_tokens.get_number_of_special_tokens())

  def model_fn() -> model.Model:
    return keras_utils.from_keras_model(
        keras_model=word_prediction_models.create_recurrent_model(
            vocab_size=extended_vocab_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=task_datasets.element_type_structure,
        metrics=metrics_builder())

  return baseline_task.BaselineTask(task_datasets, model_fn)


def create_word_prediction_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    sequence_length: int = constants.DEFAULT_SEQUENCE_LENGTH,
    vocab_size: int = constants.DEFAULT_WORD_VOCAB_SIZE,
    num_out_of_vocab_buckets: int = 1,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False) -> baseline_task.BaselineTask:
  """Creates a baseline task for next-word prediction on Stack Overflow.

  The goal of the task is to take `sequence_length` words from a post and
  predict the next word. Here, all posts are drawn from the Stack Overflow
  forum, and a client corresponds to a user.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    sequence_length: A positive integer dictating the length of each word
      sequence in a client's dataset. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_SEQUENCE_LENGTH`.
    vocab_size: Integer dictating the number of most frequent words in the
      entire corpus to use for the task's vocabulary. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_WORD_VOCAB_SIZE`.
    num_out_of_vocab_buckets: The number of out-of-vocabulary buckets to use.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
    use_synthetic_data: A boolean indicating whether to use synthetic Stack
      Overflow data. This option should only be used for testing purposes, in
      order to avoid downloading the entire Stack Overflow dataset.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if use_synthetic_data:
    synthetic_data = stackoverflow.get_synthetic()
    stackoverflow_train = synthetic_data
    stackoverflow_validation = synthetic_data
    stackoverflow_test = synthetic_data
  else:
    stackoverflow_train, stackoverflow_validation, stackoverflow_test = (
        stackoverflow.load_data(cache_dir=cache_dir))

  return create_word_prediction_task_from_datasets(
      train_client_spec, eval_client_spec, sequence_length, vocab_size,
      num_out_of_vocab_buckets, stackoverflow_train, stackoverflow_test,
      stackoverflow_validation)
