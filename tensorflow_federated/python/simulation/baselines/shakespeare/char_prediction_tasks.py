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

from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import keras_metrics
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.shakespeare import char_prediction_models
from tensorflow_federated.python.simulation.baselines.shakespeare import char_prediction_preprocessing
from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import shakespeare

# Vocabulary with out-of-vocabulary, padding, beginning-of-sentence, and
# end-of-sentence tokens.
VOCAB_LENGTH = len(char_prediction_preprocessing.CHAR_VOCAB) + 4
DEFAULT_SEQUENCE_LENGTH = 80


def create_character_prediction_task_from_datasets(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec],
    sequence_length: int,
    train_data: client_data.ClientData,
    test_data: client_data.ClientData,
) -> baseline_task.BaselineTask:
  """Creates a baseline task for next-character prediction on Shakespeare.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    sequence_length: A positive integer dictating the length of each example in
      a client's dataset. By default, this is set to
      `tff.simulation.baselines.shakespeare.DEFAULT_SEQUENCE_LENGTH`.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """

  if sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer')

  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=32, shuffle_buffer_size=1
    )

  train_preprocess_fn = char_prediction_preprocessing.create_preprocess_fn(
      train_client_spec, sequence_length
  )
  eval_preprocess_fn = char_prediction_preprocessing.create_preprocess_fn(
      eval_client_spec, sequence_length
  )

  task_datasets = task_data.BaselineTaskDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data=None,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn,
  )

  pad_token, _, _, _ = char_prediction_preprocessing.get_special_tokens()

  def model_fn() -> variable.VariableModel:
    return keras_utils.from_keras_model(
        keras_model=char_prediction_models.create_recurrent_model(
            vocab_size=VOCAB_LENGTH, sequence_length=sequence_length
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=task_datasets.element_type_structure,
        metrics=[
            keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
            keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token]),
        ],
    )

  return baseline_task.BaselineTask(task_datasets, model_fn)


def create_character_prediction_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False,
) -> baseline_task.BaselineTask:
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
      a client's dataset. By default, this is set to
      `tff.simulation.baselines.shakespeare.DEFAULT_SEQUENCE_LENGTH`.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
    use_synthetic_data: A boolean indicating whether to use synthetic
      Shakespeare data. This option should only be used for testing purposes, in
      order to avoid downloading the entire Shakespeare dataset.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if use_synthetic_data:
    synthetic_data = shakespeare.get_synthetic()
    train_data = synthetic_data
    test_data = synthetic_data
  else:
    train_data, test_data = shakespeare.load_data(cache_dir=cache_dir)

  return create_character_prediction_task_from_datasets(
      train_client_spec,
      eval_client_spec,
      sequence_length,
      train_data,
      test_data,
  )
