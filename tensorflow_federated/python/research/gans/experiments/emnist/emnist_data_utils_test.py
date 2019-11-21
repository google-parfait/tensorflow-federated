# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""Test Federated EMNIST dataset utilities."""

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.gans.experiments.emnist import emnist_data_utils

BATCH_SIZE = 7


def _summarize_model(model):
  model.summary()
  print('\n\n\n')


def _get_example_client_dataset():
  client_data = tff.simulation.datasets.emnist.get_synthetic(num_clients=1)
  return client_data.create_tf_dataset_for_client(client_data.client_ids[0])


def _get_example_client_dataset_containing_lowercase():
  _, client_data = tff.simulation.datasets.emnist.load_data(only_digits=False)
  return client_data.create_tf_dataset_for_client(client_data.client_ids[0])


class EmnistTest(tf.test.TestCase):

  def test_preprocessed_img_inversion(self):
    raw_images_ds = _get_example_client_dataset()

    # Inversion turned off, average pixel is dark.
    standard_images_ds = emnist_data_utils.preprocess_img_dataset(
        raw_images_ds, invert_imagery=False, batch_size=BATCH_SIZE)
    for batch in iter(standard_images_ds):
      for image in batch:
        self.assertLessEqual(np.average(image), -0.7)

    # Inversion turned on, average pixel is light.
    inverted_images_ds = emnist_data_utils.preprocess_img_dataset(
        raw_images_ds, invert_imagery=True, batch_size=BATCH_SIZE)
    for batch in iter(inverted_images_ds):
      for image in batch:
        self.assertGreaterEqual(np.average(image), 0.7)

  def test_preprocessed_img_labels_are_case_agnostic(self):
    raw_images_ds = _get_example_client_dataset_containing_lowercase()

    raw_ds_iterator = iter(raw_images_ds)
    # The first element in the raw dataset is an uppercase 'I' (label is 18).
    self.assertEqual(next(raw_ds_iterator)['label'].numpy(), 18)
    # The second element in the raw dataset is an uppercase 'C' (label is 12).
    self.assertEqual(next(raw_ds_iterator)['label'].numpy(), 12)
    # The third element in the raw dataset is a lowercase 'd' (label is 39).
    self.assertEqual(next(raw_ds_iterator)['label'].numpy(), 47)

    processed_ds = emnist_data_utils.preprocess_img_dataset(
        raw_images_ds, include_label=True, batch_size=BATCH_SIZE, shuffle=False)
    _, label_batch = next(iter(processed_ds))
    processed_label_iterator = iter(label_batch)
    # The first element (in first batch) in the processed dataset has a case
    # agnostic label of 18 (i.e., assert that value remains unchanged).
    self.assertEqual(next(processed_label_iterator).numpy(), 18)
    # The second element (in first batch) in the processed dataset has a case
    # agnostic label of 12 (i.e., assert that value remains unchanged).
    self.assertEqual(next(processed_label_iterator).numpy(), 12)
    # The third element (in first batch) in the processed dataset should now
    # have a case agnostic label of 47 - 26 = 21.
    self.assertEqual(next(processed_label_iterator).numpy(), 47 - 26)

    for _, label_batch in iter(processed_ds):
      for label in label_batch:
        self.assertGreaterEqual(label, 0)
        self.assertLessEqual(label, 36)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
