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
"""Library for creating autoencoder tasks on EMNIST."""

from typing import Optional

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


def create_autoencoder_task_from_datasets(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec],
    train_data: client_data.ClientData,
    test_data: client_data.ClientData) -> baseline_task.BaselineTask:
  """Creates a baseline task for autoencoding on EMNIST.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  emnist_task = 'autoencoder'

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
        keras_model=emnist_models.create_autoencoder_model(),
        loss=tf.keras.losses.MeanSquaredError(),
        input_spec=task_datasets.element_type_structure,
        metrics=[
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError()
        ])

  return baseline_task.BaselineTask(task_datasets, model_fn)


def create_autoencoder_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    only_digits: bool = False,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False) -> baseline_task.BaselineTask:
  """Creates a baseline task for autoencoding on EMNIST.

  This task involves performing autoencoding on the EMNIST dataset using a
  densely connected bottleneck network. The model uses 8 layers of widths
  `[1000, 500, 250, 30, 250, 500, 1000, 784]`, with the final layer being the
  output layer. Each layer uses a sigmoid activation function, except the
  smallest layer, which uses a linear activation function.

  The goal of the task is to minimize the mean squared error between the input
  to the network and the output of the network.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
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

  return create_autoencoder_task_from_datasets(train_client_spec,
                                               eval_client_spec, emnist_train,
                                               emnist_test)
