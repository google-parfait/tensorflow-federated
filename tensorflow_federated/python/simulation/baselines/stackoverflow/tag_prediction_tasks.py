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
"""Library for creating tag prediction tasks on Stack Overflow."""

from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.stackoverflow import constants
from tensorflow_federated.python.simulation.baselines.stackoverflow import tag_prediction_preprocessing
from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import stackoverflow


def _build_logistic_regression_model(input_size: int, output_size: int):
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          output_size, activation='sigmoid', input_shape=(input_size,))
  ])


def create_tag_prediction_task_from_datasets(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec],
    word_vocab_size: int,
    tag_vocab_size: int,
    train_data: client_data.ClientData,
    test_data: client_data.ClientData,
    validation_data: client_data.ClientData,
) -> baseline_task.BaselineTask:
  """Creates a baseline task for tag prediction on Stack Overflow.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    word_vocab_size: Integer dictating the number of most frequent words in the
      entire corpus to use for the task's vocabulary. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_WORD_VOCAB_SIZE`.
    tag_vocab_size: Integer dictating the number of most frequent tags in the
      entire corpus to use for the task's labels. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_TAG_VOCAB_SIZE`.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.
    validation_data: A `tff.simulation.datasets.ClientData` used for validation.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if word_vocab_size < 1:
    raise ValueError('word_vocab_size must be a positive integer')
  if tag_vocab_size < 1:
    raise ValueError('tag_vocab_size must be a positive integer')

  word_vocab = list(stackoverflow.load_word_counts(vocab_size=word_vocab_size))
  tag_vocab = list(stackoverflow.load_tag_counts().keys())[:tag_vocab_size]

  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=100, shuffle_buffer_size=1)

  train_preprocess_fn = tag_prediction_preprocessing.create_preprocess_fn(
      train_client_spec, word_vocab, tag_vocab)
  eval_preprocess_fn = tag_prediction_preprocessing.create_preprocess_fn(
      eval_client_spec, word_vocab, tag_vocab)

  task_datasets = task_data.BaselineTaskDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data=validation_data,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  def model_fn() -> model.Model:
    return keras_utils.from_keras_model(
        keras_model=_build_logistic_regression_model(
            input_size=word_vocab_size, output_size=tag_vocab_size),
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM),
        input_spec=task_datasets.element_type_structure,
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(top_k=5, name='recall_at_5'),
        ])

  return baseline_task.BaselineTask(task_datasets, model_fn)


def create_tag_prediction_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    word_vocab_size: int = constants.DEFAULT_WORD_VOCAB_SIZE,
    tag_vocab_size: int = constants.DEFAULT_TAG_VOCAB_SIZE,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False,
) -> baseline_task.BaselineTask:
  """Creates a baseline task for tag prediction on Stack Overflow.

  The goal of the task is to predict the tags associated to a post based on a
  bag-of-words representation of the post.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    word_vocab_size: Integer dictating the number of most frequent words in the
      entire corpus to use for the task's vocabulary. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_WORD_VOCAB_SIZE`.
    tag_vocab_size: Integer dictating the number of most frequent tags in the
      entire corpus to use for the task's labels. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_TAG_VOCAB_SIZE`.
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

  return create_tag_prediction_task_from_datasets(
      train_client_spec, eval_client_spec, word_vocab_size, tag_vocab_size,
      stackoverflow_train, stackoverflow_test, stackoverflow_validation)
