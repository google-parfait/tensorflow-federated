# Copyright 2023, The TensorFlow Federated Authors.
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
"""The `tff.program.FederatedDataSource`s for this federated program."""

import tensorflow as tf
import tensorflow_federated as tff


def create_data_sources() -> (
    tuple[
        tff.program.FederatedDataSource,
        tff.program.FederatedDataSource,
        tf.TensorSpec,
    ]
):
  """Creates the `tff.program.FederatedDataSource`s for this program.

  Returns:
    A `tuple` containing the train data source, evaluation data source, and the
    `element_type_structure` of the train data which is used when constructing
    the model.
  """
  train_data, evaluation_data = tff.simulation.datasets.emnist.load_data(
      only_digits=True
  )

  def _preprocess_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    def _create_tuple(element):
      return (element['pixels'], element['label'])

    dataset = dataset.batch(batch_size=32)
    dataset = dataset.map(_create_tuple)
    return dataset

  train_data = train_data.preprocess(_preprocess_fn)
  datasets = []
  for client_id in train_data.client_ids:
    dataset = train_data.create_tf_dataset_for_client(client_id)
    datasets.append(dataset)
  train_data_source = tff.program.DatasetDataSource(datasets)
  evaluation_data = evaluation_data.preprocess(_preprocess_fn)
  evaluation_data_source = tff.program.DatasetDataSource(datasets)

  return (
      train_data_source,
      evaluation_data_source,
      train_data.element_type_structure,
  )
