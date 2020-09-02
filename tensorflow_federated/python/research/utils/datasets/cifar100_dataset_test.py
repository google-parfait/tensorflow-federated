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

import collections

import tensorflow as tf

from tensorflow_federated.python.research.utils.datasets import cifar100_dataset

TEST_BATCH_SIZE = cifar100_dataset.TEST_BATCH_SIZE


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class DatasetTest(tf.test.TestCase):

  def test_centralized_cifar_structure(self):
    crop_shape = (24, 24, 3)
    cifar_train, cifar_test = cifar100_dataset.get_centralized_datasets(
        train_batch_size=20, test_batch_size=100, crop_shape=crop_shape)
    train_batch = next(iter(cifar_train))
    train_batch_shape = tuple(train_batch[0].shape)
    self.assertEqual(train_batch_shape, (20, 24, 24, 3))
    test_batch = next(iter(cifar_test))
    test_batch_shape = tuple(test_batch[0].shape)
    self.assertEqual(test_batch_shape, (100, 24, 24, 3))

  def test_federated_cifar_structure(self):
    crop_shape = (32, 32, 3)
    cifar_train, _ = cifar100_dataset.get_federated_cifar100(
        client_epochs_per_round=1, train_batch_size=10, crop_shape=crop_shape)
    client_id = cifar_train.client_ids[0]
    client_dataset = cifar_train.create_tf_dataset_for_client(client_id)
    train_batch = next(iter(client_dataset))
    train_batch_shape = tuple(train_batch[0].shape)
    self.assertEqual(train_batch_shape, (10, 32, 32, 3))

  def test_no_op_crop_process_cifar_example(self):
    crop_shape = (1, 1, 1, 3)
    x = tf.constant([[[[1.0, -1.0, 0.0]]]])  # Has shape (1, 1, 1, 3), mean 0
    x = x / tf.math.reduce_std(x)  # x now has variance 1
    dummy_example = collections.OrderedDict(image=x, label=0)
    processed_dummy_example = cifar100_dataset.preprocess_cifar_example(
        dummy_example, crop_shape=crop_shape, distort=False)
    self.assertEqual(processed_dummy_example[0].shape, crop_shape)
    self.assertAllClose(x, processed_dummy_example[0], rtol=1e-03)
    self.assertEqual(processed_dummy_example[1], 0)

  def test_raises_length_2_crop(self):
    with self.assertRaises(ValueError):
      cifar100_dataset.get_federated_cifar100(
          client_epochs_per_round=1, train_batch_size=10, crop_shape=(32, 32))
    with self.assertRaises(ValueError):
      cifar100_dataset.get_centralized_datasets(
          train_batch_size=10, crop_shape=(32, 32))

  def test_raises_no_repeat_and_no_take(self):
    with self.assertRaisesRegex(
        ValueError, 'Argument client_epochs_per_round is set to -1'):
      cifar100_dataset.get_federated_cifar100(
          client_epochs_per_round=-1,
          train_batch_size=10,
          max_batches_per_client=-1)

  def test_take_with_repeat(self):
    cifar_train, _ = cifar100_dataset.get_federated_cifar100(
        client_epochs_per_round=-1,
        train_batch_size=10,
        max_batches_per_client=10)
    self.assertEqual(len(cifar_train.client_ids), 500)
    for i in range(10):
      client_ds = cifar_train.create_tf_dataset_for_client(
          cifar_train.client_ids[i])
      self.assertEqual(_compute_length_of_dataset(client_ds), 10)


if __name__ == '__main__':
  tf.test.main()
