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
"""Library for creating baseline tasks on Shakespeare."""

from typing import Callable, Optional, Tuple

import tensorflow as tf

from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import keras_metrics
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.shakespeare import shakespeare_models
from tensorflow_federated.python.simulation.baselines.shakespeare import shakespeare_preprocessing
from tensorflow_federated.python.simulation.datasets import shakespeare

# Vocabulary with out-of-vocabulary, padding, beginning-of-sentence, and
# end-of-sentence tokens.
VOCAB_LENGTH = len(shakespeare_preprocessing.CHAR_VOCAB) + 4
DEFAULT_SEQUENCE_LENGTH = 20
_PreprocessFn = Callable[[tf.data.Dataset], tf.data.Dataset]
_ModelFn = Callable[[], model.Model]


def _get_preprocessing_functions(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: client_spec.ClientSpec,
    sequence_length) -> Tuple[_PreprocessFn, _PreprocessFn]:
  """Creates train and eval preprocessing functions for a CIFAR-100 task."""
  train_preprocess_fn = shakespeare_preprocessing.create_preprocess_fn(
      num_epochs=train_client_spec.num_epochs,
      batch_size=train_client_spec.batch_size,
      max_elements=train_client_spec.max_elements,
      shuffle_buffer_size=train_client_spec.shuffle_buffer_size,
      sequence_length=sequence_length)
  eval_preprocess_fn = shakespeare_preprocessing.create_preprocess_fn(
      num_epochs=eval_client_spec.num_epochs,
      batch_size=eval_client_spec.batch_size,
      max_elements=eval_client_spec.max_elements,
      shuffle_buffer_size=eval_client_spec.shuffle_buffer_size,
      sequence_length=sequence_length)
  return train_preprocess_fn, eval_preprocess_fn


def create_character_prediction_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    use_synthetic_data: bool = False
) -> Tuple[task_data.BaselineTaskDatasets, _ModelFn]:
  """Creates a baseline task for next-character prediction on Shakespeare.

  The goal of the task is to take `sequence_length` characters (eg. alpha-
  numeric characters and puctuation characters) and predict the next character.
  Here, all sentences are drawn from the collected works of William Shakespeare,
  and a client corresponds to role in a play.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    sequence_length: A positive integer dictating the length of each example in
      a client's dataset.
    use_synthetic_data: A boolean indicating whether to use synthetic
      Shakespeare data. This option should only be used for testing purposes, in
      order to avoid downloading the entire Shakespeare dataset.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer')

  if use_synthetic_data:
    synthetic_data = shakespeare.get_synthetic()
    shakespeare_train = synthetic_data
    shakespeare_test = synthetic_data
  else:
    shakespeare_train, shakespeare_test = shakespeare.load_data()

  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=32, shuffle_buffer_size=1)
  train_preprocess_fn, eval_preprocess_fn = _get_preprocessing_functions(
      train_client_spec, eval_client_spec, sequence_length)
  task_datasets = task_data.BaselineTaskDatasets(
      train_data=shakespeare_train,
      test_data=shakespeare_test,
      validation_data=None,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  keras_model = shakespeare_models.create_recurrent_model(
      vocab_size=VOCAB_LENGTH, sequence_length=sequence_length)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  pad_token, _, _, _ = shakespeare_preprocessing.get_special_tokens()
  metrics = [
      keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token])
  ]

  def model_fn() -> model.Model:
    return keras_utils.from_keras_model(
        keras_model=keras_model,
        loss=loss,
        input_spec=task_datasets.element_type_structure,
        metrics=metrics)

  return baseline_task.BaselineTask(task_datasets, model_fn)
