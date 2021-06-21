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
"""Library for creating baseline tasks on CIFAR-100."""

import enum
from typing import Callable, Optional, Tuple, Union

import tensorflow as tf

from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.cifar import cifar_preprocessing
from tensorflow_federated.python.simulation.baselines.cifar import resnet_models
from tensorflow_federated.python.simulation.datasets import cifar100


class ResnetModel(enum.Enum):
  """Enum for ResNet classification models."""
  RESNET18 = 'resnet18'
  RESNET34 = 'resnet34'
  RESNET50 = 'resnet50'
  RESNET101 = 'resnet101'
  RESNET152 = 'resnet152'


_NUM_CLASSES = 100
_RESNET_MODELS = [e.value for e in ResnetModel]
_PreprocessFn = Callable[[tf.data.Dataset], tf.data.Dataset]
_ModelFn = Callable[[], model.Model]


def _get_resnet_model(model_id: Union[str, ResnetModel],
                      input_shape: Tuple[int, int, int]) -> tf.keras.Model:
  """Constructs a `tf.keras.Model` for digit recognition."""
  try:
    model_enum = ResnetModel(model_id)
  except ValueError:
    raise ValueError('The model argument must be one of {}, found {}'.format(
        model, ResnetModel))

  if model_enum == ResnetModel.RESNET18:
    keras_model_fn = resnet_models.create_resnet18
  elif model_enum == ResnetModel.RESNET34:
    keras_model_fn = resnet_models.create_resnet34
  elif model_enum == ResnetModel.RESNET50:
    keras_model_fn = resnet_models.create_resnet50
  elif model_enum == ResnetModel.RESNET101:
    keras_model_fn = resnet_models.create_resnet101
  elif model_enum == ResnetModel.RESNET152:
    keras_model_fn = resnet_models.create_resnet152
  else:
    raise ValueError('The model id must be one of {}, found {}'.format(
        _RESNET_MODELS, model_enum))
  return keras_model_fn(input_shape=input_shape, num_classes=_NUM_CLASSES)


def _get_preprocessing_functions(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: client_spec.ClientSpec,
    crop_shape) -> Tuple[_PreprocessFn, _PreprocessFn]:
  """Creates train and eval preprocessing functions for a CIFAR-100 task."""
  train_preprocess_fn = cifar_preprocessing.create_preprocess_fn(
      num_epochs=train_client_spec.num_epochs,
      batch_size=train_client_spec.batch_size,
      max_elements=train_client_spec.max_elements,
      shuffle_buffer_size=train_client_spec.shuffle_buffer_size,
      crop_shape=crop_shape)
  eval_preprocess_fn = cifar_preprocessing.create_preprocess_fn(
      num_epochs=eval_client_spec.num_epochs,
      batch_size=eval_client_spec.batch_size,
      max_elements=eval_client_spec.max_elements,
      shuffle_buffer_size=eval_client_spec.shuffle_buffer_size,
      crop_shape=crop_shape)
  return train_preprocess_fn, eval_preprocess_fn


def create_image_classification_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    model_id: Union[str, ResnetModel] = 'resnet18',
    crop_height: int = 24,
    crop_width: int = 24,
    use_synthetic_data: bool = False
) -> Tuple[task_data.BaselineTaskDatasets, _ModelFn]:
  """Creates a baseline task for image classification on CIFAR-100.

  The goal of the task is to minimize the sparse categorical crossentropy
  between the output labels of the model and the true label of the image.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    model_id: A string identifier for a digit recognition model. Must be one of
      `resnet18`, `resnet34`, `resnet50`, `resnet101` and `resnet152. These
      correspond to various ResNet architectures. Unlike standard ResNet
      architectures though, the batch normalization layers are replaced with
      group normalization.
    crop_height: An integer specifying the desired height for cropping images.
      Must be between 1 and 32 (the height of uncropped CIFAR-100 images).
    crop_width: An integer specifying the desired width for cropping images.
      Must be between 1 and 32 (the width of uncropped CIFAR-100 images).
    use_synthetic_data: A boolean indicating whether to use synthetic CIFAR-100
      data. This option should only be used for testing purposes, in order to
      avoid downloading the entire CIFAR-100 dataset.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if crop_height < 1 or crop_width < 1 or crop_height > 32 or crop_width > 32:
    raise ValueError('The crop_height and crop_width must be between 1 and 32.')
  crop_shape = (crop_height, crop_width, 3)

  if use_synthetic_data:
    synthetic_data = cifar100.get_synthetic()
    cifar_train = synthetic_data
    cifar_test = synthetic_data
  else:
    cifar_train, cifar_test = cifar100.load_data()

  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)
  train_preprocess_fn, eval_preprocess_fn = _get_preprocessing_functions(
      train_client_spec, eval_client_spec, crop_shape)
  task_datasets = task_data.BaselineTaskDatasets(
      train_data=cifar_train,
      test_data=cifar_test,
      validation_data=None,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  keras_model = _get_resnet_model(model_id, crop_shape)
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

  def model_fn() -> model.Model:
    return keras_utils.from_keras_model(
        keras_model=keras_model,
        loss=loss,
        input_spec=task_datasets.element_type_structure,
        metrics=metrics)

  return baseline_task.BaselineTask(task_datasets, model_fn)
