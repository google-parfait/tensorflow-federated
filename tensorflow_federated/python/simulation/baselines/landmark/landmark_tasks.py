# Copyright 2022, Google LLC.
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
"""Library for creating Baseline Task on GLDv2."""

from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.baselines.landmark import landmark_preprocessing
from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import gldv2
from tensorflow_federated.python.simulation.models import mobilenet_v2


_IMAGE_SIZE = landmark_preprocessing.IMAGE_SIZE
_NUM_GROUPS = 8
_NUM_CLASSES = 2028


def create_landmark_classification_task_from_datasets(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec],
    train_data: client_data.ClientData,
    test_data: tf.data.Dataset,
    debug_seed: Optional[int] = None,
) -> baseline_task.BaselineTask:
  """Creates a baseline task of image classification on GLDv2.

  The goal of the task is to minimize the sparse categorical cross entropy
  between the output labels of the model and the true label of the image. A
  MobilenetV2 model is created that expects input image data with a shape of
  [_IMAGE_SIZE, _IMAGE_SIZE, 3] and group normalization layers with a group
  number of _NUM_GROUPS.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a `batch_size` of 64.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tf.data.Dataset` used for testing.
    debug_seed: An optional integer seed to force deterministic model
      initialization and dataset shuffle buffers. This is intended for
      unittesting.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if eval_client_spec is None:
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1
    )

  train_preprocess_fn = landmark_preprocessing.create_preprocess_fn(
      train_client_spec,
      is_training=True,
      debug_seed=debug_seed,
  )
  eval_preprocess_fn = landmark_preprocessing.create_preprocess_fn(
      eval_client_spec,
      is_training=False,
      debug_seed=debug_seed,
  )

  task_datasets = task_data.BaselineTaskDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data=None,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn,
  )

  def model_fn() -> variable.VariableModel:
    return keras_utils.from_keras_model(
        keras_model=mobilenet_v2.create_mobilenet_v2(
            input_shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3),
            num_groups=_NUM_GROUPS,
            num_classes=_NUM_CLASSES,
        ),
        input_spec=task_datasets.element_type_structure,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

  return baseline_task.BaselineTask(task_datasets, model_fn)


def create_landmark_classification_task(
    train_client_spec: client_spec.ClientSpec,
    eval_client_spec: Optional[client_spec.ClientSpec] = None,
    use_gld23k: bool = False,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False,
    debug_seed: Optional[int] = None,
) -> baseline_task.BaselineTask:
  """Creates a baseline task of image classification on GLDv2.

  The goal of the task is to minimize the sparse categorical cross entropy
  between the output labels of the model and the true label of the image. A
  MobilenetV2 model is created that expects input image data with a shape of
  [128, 128, 3] and group normalization layers with a group number of 8.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a `batch_size` of 64.
    use_gld23k: An optional boolean. When true, a smaller version of the GLDv2
      landmark dataset will be loaded. This gld23k dataset is used for faster
      prototyping.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      non-specified, they will be cached to the default cache directory `cache`.
    use_synthetic_data: An optional boolean indicating whether to use synthetic
      GLDv2 data. This option should only be used for testing purposes, in order
      to avoid downloading the entire GLDv2 dataset.
    debug_seed: An optional integer seed to force deterministic model
      initialization. This is intended for unittesting.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if use_synthetic_data:
    landmark_train = gldv2.get_synthetic()
    landmark_test = landmark_train.create_tf_dataset_for_client(
        landmark_train.client_ids[0]
    )
  else:
    landmark_train, landmark_test = gldv2.load_data(
        gld23k=use_gld23k, cache_dir=cache_dir if cache_dir else 'cache'
    )

  return create_landmark_classification_task_from_datasets(
      train_client_spec,
      eval_client_spec,
      landmark_train,
      landmark_test,
      debug_seed,
  )
