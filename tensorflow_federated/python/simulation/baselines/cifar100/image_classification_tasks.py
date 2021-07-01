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
from typing import Optional, Tuple, Union

import tensorflow as tf

from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.cifar100 import image_classification_preprocessing
from tensorflow_federated.python.simulation.baselines.cifar100 import resnet_models
from tensorflow_federated.python.simulation.datasets import cifar100
from tensorflow_federated.python.simulation.datasets import client_data


class ResnetModel(enum.Enum):
  """Enum for ResNet classification models."""
  RESNET18 = 'resnet18'
  RESNET34 = 'resnet34'
  RESNET50 = 'resnet50'
  RESNET101 = 'resnet101'
  RESNET152 = 'resnet152'


_NUM_CLASSES = 100
_RESNET_MODELS = [e.value for e in ResnetModel]
DEFAULT_CROP_HEIGHT = 24
DEFAULT_CROP_WIDTH = 24


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


def create_image_classification_task_with_datasets(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec],
    model_id: Union[str, ResnetModel],
    crop_height: int,
    crop_width: int,
    train_data: client_data.ClientData,
    test_data: client_data.ClientData,
) -> baseline_task.BaselineTask:
  """Creates a baseline task for image classification on CIFAR-100.

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
      Must be between 1 and 32 (the height of uncropped CIFAR-100 images). By
      default, this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_HEIGHT`.
    crop_width: An integer specifying the desired width for cropping images.
      Must be between 1 and 32 (the width of uncropped CIFAR-100 images). By
      default this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_WIDTH`.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if crop_height < 1 or crop_width < 1 or crop_height > 32 or crop_width > 32:
    raise ValueError('The crop_height and crop_width must be between 1 and 32.')
  crop_shape = (crop_height, crop_width, 3)

  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)

  train_preprocess_fn = image_classification_preprocessing.create_preprocess_fn(
      train_client_spec, crop_shape=crop_shape)
  eval_preprocess_fn = image_classification_preprocessing.create_preprocess_fn(
      eval_client_spec, crop_shape=crop_shape)

  task_datasets = task_data.BaselineTaskDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data=None,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  def model_fn() -> model.Model:
    return keras_utils.from_keras_model(
        keras_model=_get_resnet_model(model_id, crop_shape),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=task_datasets.element_type_structure,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return baseline_task.BaselineTask(task_datasets, model_fn)


def create_image_classification_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    model_id: Union[str, ResnetModel] = 'resnet18',
    crop_height: int = DEFAULT_CROP_HEIGHT,
    crop_width: int = DEFAULT_CROP_WIDTH,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False) -> baseline_task.BaselineTask:
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
      Must be between 1 and 32 (the height of uncropped CIFAR-100 images). By
      default, this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_HEIGHT`.
    crop_width: An integer specifying the desired width for cropping images.
      Must be between 1 and 32 (the width of uncropped CIFAR-100 images). By
      default this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_WIDTH`.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
    use_synthetic_data: A boolean indicating whether to use synthetic CIFAR-100
      data. This option should only be used for testing purposes, in order to
      avoid downloading the entire CIFAR-100 dataset.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if use_synthetic_data:
    synthetic_data = cifar100.get_synthetic()
    cifar_train = synthetic_data
    cifar_test = synthetic_data
  else:
    cifar_train, cifar_test = cifar100.load_data(cache_dir=cache_dir)

  return create_image_classification_task_with_datasets(train_client_spec,
                                                        eval_client_spec,
                                                        model_id, crop_height,
                                                        crop_width, cifar_train,
                                                        cifar_test)
