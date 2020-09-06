# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for shared sequentiality scheduling utilities."""

import collections
import functools

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated_research.optimization.shared import schedule_utils


def model_builder():
  # Create a simple linear regression model, single output.
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(1,))
  ])
  return model


@tf.function
def get_sample_batch():
  return next(iter(create_tf_dataset_for_client(0)))


def create_tf_dataset_for_client(client_id, batch_data=True):
  # Create client data for y = 2*x+3
  np.random.seed(client_id)
  x = np.random.rand(6, 1).astype(np.float32)
  y = 2 * x + 3
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict([('x', x), ('y', y)]))
  if batch_data:
    dataset = dataset.batch(2)
  return dataset


class TrainingUtilsTest(tf.test.TestCase):

  def test_build_scheduled_client_datasets_fn(self):
    single_client_dataset_fn = functools.partial(
        create_tf_dataset_for_client, batch_data=False)
    tff_dataset = tff.simulation.client_data.ConcreteClientData(
        [2], single_client_dataset_fn)
    client_datasets_fn = schedule_utils.build_scheduled_client_datasets_fn(
        tff_dataset,
        clients_per_round=1,
        client_batch_size=2,
        client_epochs_per_round=3,
        total_rounds=2,
        num_stages=2,
        batch_growth_factor=2,
        epochs_decrease_amount=1)
    # This should be a dataset with 3 batches of size 4, with client ids [2]
    client_datasets, client_ids = client_datasets_fn(round_num=2)
    self.assertEqual(client_ids, [2])

    client_dataset = client_datasets[0]
    sample_batch = next(iter(client_dataset))
    self.assertEqual(sample_batch['x'].shape[0], 4)

    num_batches = 0
    num_examples = 0
    for batch in client_dataset:
      num_batches += 1
      num_examples += batch['x'].shape[0]
    self.assertEqual(num_batches, 3)
    self.assertEqual(num_examples, 12)


if __name__ == '__main__':
  tf.test.main()
