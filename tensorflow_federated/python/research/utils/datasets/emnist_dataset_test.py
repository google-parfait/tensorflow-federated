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

import tensorflow as tf

from tensorflow_federated.python.research.utils.datasets import emnist_dataset

TEST_BATCH_SIZE = emnist_dataset.TEST_BATCH_SIZE


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class DatasetTest(tf.test.TestCase):

  def test_emnist_dataset_structure(self):
    emnist_train, emnist_test = emnist_dataset.get_emnist_datasets(
        client_batch_size=10, client_epochs_per_round=1, only_digits=True)
    self.assertEqual(len(emnist_train.client_ids), 3383)
    sample_train_ds = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])

    train_batch = next(iter(sample_train_ds))
    train_batch_shape = train_batch[0].shape
    test_batch = next(iter(emnist_test))
    test_batch_shape = test_batch[0].shape
    self.assertEqual(train_batch_shape.as_list(), [10, 28, 28, 1])
    self.assertEqual(test_batch_shape.as_list(), [TEST_BATCH_SIZE, 28, 28, 1])

  def test_take_without_repeat(self):
    emnist_train, _ = emnist_dataset.get_emnist_datasets(
        client_batch_size=10,
        client_epochs_per_round=1,
        max_batches_per_client=10,
        only_digits=True)
    self.assertEqual(len(emnist_train.client_ids), 3383)
    for i in range(10):
      client_ds = emnist_train.create_tf_dataset_for_client(
          emnist_train.client_ids[i])
      self.assertLessEqual(_compute_length_of_dataset(client_ds), 10)

  def test_take_with_repeat(self):
    emnist_train, _ = emnist_dataset.get_emnist_datasets(
        client_batch_size=10,
        client_epochs_per_round=-1,
        max_batches_per_client=10,
        only_digits=True)
    self.assertEqual(len(emnist_train.client_ids), 3383)
    for i in range(10):
      client_ds = emnist_train.create_tf_dataset_for_client(
          emnist_train.client_ids[i])
      self.assertEqual(_compute_length_of_dataset(client_ds), 10)

  def test_raises_no_repeat_and_no_take(self):
    with self.assertRaisesRegex(
        ValueError, 'Argument client_epochs_per_round is set to -1'):
      emnist_dataset.get_emnist_datasets(
          client_batch_size=10,
          client_epochs_per_round=-1,
          max_batches_per_client=-1)

  def test_global_emnist_dataset_structure(self):
    global_train, global_test = emnist_dataset.get_centralized_datasets(
        train_batch_size=32, test_batch_size=100, only_digits=False)

    train_batch = next(iter(global_train))
    train_batch_shape = train_batch[0].shape
    test_batch = next(iter(global_test))
    test_batch_shape = test_batch[0].shape
    self.assertEqual(train_batch_shape.as_list(), [32, 28, 28, 1])
    self.assertEqual(test_batch_shape.as_list(), [100, 28, 28, 1])


if __name__ == '__main__':
  tf.test.main()
