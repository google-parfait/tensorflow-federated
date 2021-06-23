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
"""Library for creating character recognition tasks on EMNIST."""

import enum
from typing import Optional, Union

import tensorflow as tf

from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.emnist import emnist_models
from tensorflow_federated.python.simulation.baselines.emnist import emnist_preprocessing
from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import emnist


class CharacterRecognitionModel(enum.Enum):
  """Enum for EMNIST character recognition models."""
  CNN_DROPOUT = 'cnn_dropout'
  CNN = 'cnn'
  TWO_LAYER_DNN = '2nn'


_CHARACTER_RECOGNITION_MODELS = [e.value for e in CharacterRecognitionModel]


def _get_character_recognition_model(model_id: Union[str,
                                                     CharacterRecognitionModel],
                                     only_digits: bool) -> tf.keras.Model:
  """Constructs a `tf.keras.Model` for character recognition."""
  try:
    model_enum = CharacterRecognitionModel(model_id)
  except ValueError:
    raise ValueError('The model argument must be one of {}, found {}'.format(
        _CHARACTER_RECOGNITION_MODELS, model_id))

  if model_enum == CharacterRecognitionModel.CNN_DROPOUT:
    keras_model = emnist_models.create_conv_dropout_model(
        only_digits=only_digits)
  elif model_enum == CharacterRecognitionModel.CNN:
    keras_model = emnist_models.create_original_fedavg_cnn_model(
        only_digits=only_digits)
  elif model_enum == CharacterRecognitionModel.TWO_LAYER_DNN:
    keras_model = emnist_models.create_two_hidden_layer_model(
        only_digits=only_digits)
  else:
    raise ValueError('The model id must be one of {}, found {}'.format(
        _CHARACTER_RECOGNITION_MODELS, model_id))
  return keras_model


def create_character_recognition_task_from_datasets(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec],
    model_id: Union[str, CharacterRecognitionModel], only_digits: bool,
    train_data: client_data.ClientData,
    test_data: client_data.ClientData) -> baseline_task.BaselineTask:
  """Creates a baseline task for character recognition on EMNIST.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    model_id: A string identifier for a character recognition model. Must be one
      of 'cnn_dropout', 'cnn', or '2nn'. These correspond respectively to a CNN
      model with dropout, a CNN model with no dropout, and a densely connected
      network with two hidden layers of width 200.
    only_digits: A boolean indicating whether to use the full EMNIST-62 dataset
      containing 62 alphanumeric classes (`True`) or the smaller EMNIST-10
      dataset with only 10 numeric classes (`False`).
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  emnist_task = 'character_recognition'

  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)

  train_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
      train_client_spec, emnist_task=emnist_task)
  eval_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
      eval_client_spec, emnist_task=emnist_task)

  task_datasets = task_data.BaselineTaskDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data=None,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  def model_fn() -> model.Model:
    return keras_utils.from_keras_model(
        keras_model=_get_character_recognition_model(model_id, only_digits),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=task_datasets.element_type_structure,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return baseline_task.BaselineTask(task_datasets, model_fn)


def create_character_recognition_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    model_id: Union[str, CharacterRecognitionModel] = 'cnn_dropout',
    only_digits: bool = False,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False) -> baseline_task.BaselineTask:
  """Creates a baseline task for character recognition on EMNIST.

  The goal of the task is to minimize the sparse categorical crossentropy
  between the output labels of the model and the true label of the image. When
  `only_digits = True`, there are 10 possible labels (the digits 0-9), while
  when `only_digits = False`, there are 62 possible labels (both numbers and
  letters).

  This classification can be done using a number of different models, specified
  using the `model_id` argument. Below we give a list of the different models
  that can be used:

  *   `model_id = cnn_dropout`: A moderately sized convolutional network. Uses
  two convolutional layers, a max pooling layer, and dropout, followed by two
  dense layers.
  *   `model_id = cnn`: A moderately sized convolutional network, without any
  dropout layers. Matches the architecture of the convolutional network used
  by (McMahan et al., 2017) for the purposes of testing the FedAvg algorithm.
  *   `model_id = 2nn`: A densely connected network with 2 hidden layers, each
  with 200 hidden units and ReLU activations.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    model_id: A string identifier for a character recognition model. Must be one
      of 'cnn_dropout', 'cnn', or '2nn'. These correspond respectively to a CNN
      model with dropout, a CNN model with no dropout, and a densely connected
      network with two hidden layers of width 200.
    only_digits: A boolean indicating whether to use the full EMNIST-62 dataset
      containing 62 alphanumeric classes (`True`) or the smaller EMNIST-10
      dataset with only 10 numeric classes (`False`).
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
    use_synthetic_data: A boolean indicating whether to use synthetic EMNIST
      data. This option should only be used for testing purposes, in order to
      avoid downloading the entire EMNIST dataset.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if use_synthetic_data:
    synthetic_data = emnist.get_synthetic()
    emnist_train = synthetic_data
    emnist_test = synthetic_data
  else:
    emnist_train, emnist_test = emnist.load_data(
        only_digits=only_digits, cache_dir=cache_dir)

  return create_character_recognition_task_from_datasets(
      train_client_spec, eval_client_spec, model_id, only_digits, emnist_train,
      emnist_test)
