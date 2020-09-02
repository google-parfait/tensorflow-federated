# Copyright 2020, The TensorFlow Federated Authors.
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
"""Tests for Stackoverflow data loader."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.research.utils.datasets import stackoverflow_lr_dataset


TEST_BATCH_SIZE = stackoverflow_lr_dataset.TEST_BATCH_SIZE


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class DatasetTest(tf.test.TestCase):

  def test_stackoverflow_dataset_structure(self):
    stackoverflow_train, stackoverflow_validation, stackoverflow_test = stackoverflow_lr_dataset.get_stackoverflow_datasets(
        vocab_tokens_size=100,
        vocab_tags_size=5,
        max_training_elements_per_user=10,
        client_batch_size=10,
        client_epochs_per_round=1,
        num_validation_examples=10000)
    self.assertEqual(len(stackoverflow_train.client_ids), 342477)
    sample_train_ds = stackoverflow_train.create_tf_dataset_for_client(
        stackoverflow_train.client_ids[0])

    train_batch = next(iter(sample_train_ds))
    valid_batch = next(iter(stackoverflow_validation))
    test_batch = next(iter(stackoverflow_test))
    self.assertEqual(train_batch[0].shape.as_list(), [10, 100])
    self.assertEqual(train_batch[1].shape.as_list(), [10, 5])
    self.assertEqual(valid_batch[0].shape.as_list(), [TEST_BATCH_SIZE, 100])
    self.assertEqual(valid_batch[1].shape.as_list(), [TEST_BATCH_SIZE, 5])
    self.assertEqual(test_batch[0].shape.as_list(), [TEST_BATCH_SIZE, 100])
    self.assertEqual(test_batch[1].shape.as_list(), [TEST_BATCH_SIZE, 5])

  def test_global_stackoverflow_dataset_structure(self):
    global_train, global_val, global_test = stackoverflow_lr_dataset.get_centralized_datasets(
        train_batch_size=32,
        validation_batch_size=100,
        test_batch_size=500,
        vocab_tokens_size=100,
        vocab_tags_size=5,
        num_validation_examples=10000)

    train_batch = next(iter(global_train))
    val_batch = next(iter(global_val))
    test_batch = next(iter(global_test))
    self.assertEqual(train_batch[0].shape.as_list(), [32, 100])
    self.assertEqual(train_batch[1].shape.as_list(), [32, 5])
    self.assertEqual(val_batch[0].shape.as_list(), [100, 100])
    self.assertEqual(val_batch[1].shape.as_list(), [100, 5])
    self.assertEqual(test_batch[0].shape.as_list(), [500, 100])
    self.assertEqual(test_batch[1].shape.as_list(), [500, 5])

  @test.skip_test_for_gpu
  def test_take_with_repeat(self):
    so_train, _, _ = stackoverflow_lr_dataset.get_stackoverflow_datasets(
        vocab_tokens_size=1000,
        vocab_tags_size=500,
        max_training_elements_per_user=128,
        client_batch_size=10,
        client_epochs_per_round=-1,
        max_batches_per_user=8,
        num_validation_examples=500)
    for i in range(10):
      client_ds = so_train.create_tf_dataset_for_client(so_train.client_ids[i])
      self.assertEqual(_compute_length_of_dataset(client_ds), 8)

  @test.skip_test_for_gpu
  def test_raises_no_repeat_and_no_take(self):
    with self.assertRaisesRegex(
        ValueError, 'Argument client_epochs_per_round is set to -1'):
      stackoverflow_lr_dataset.get_stackoverflow_datasets(
          vocab_tokens_size=1000,
          vocab_tags_size=500,
          max_training_elements_per_user=128,
          client_batch_size=10,
          client_epochs_per_round=-1,
          max_batches_per_user=-1,
          num_validation_examples=500)


if __name__ == '__main__':
  tf.test.main()
