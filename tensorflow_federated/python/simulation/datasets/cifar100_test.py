# Lint as: python3
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

from tensorflow_federated.python.simulation.datasets import cifar100

tf.compat.v1.enable_v2_behavior()


class LoadDataTest(tf.test.TestCase):

  def test_load_train_data(self):
    cifar_train, _ = cifar100.load_data()
    self.assertLen(cifar_train.client_ids, 500)
    self.assertCountEqual(cifar_train.client_ids, [str(i) for i in range(500)])

    self.assertEqual(
        cifar_train.element_type_structure,
        collections.OrderedDict([
            ('coarse_label', tf.TensorSpec(shape=(), dtype=tf.int64)),
            ('image', tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8)),
            ('label', tf.TensorSpec(shape=(), dtype=tf.int64)),
        ]))

    expected_coarse_labels = []
    expected_labels = []
    for i in range(20):
      expected_coarse_labels += [i] * 2500
    for i in range(100):
      expected_labels += [i] * 500

    coarse_labels = []
    labels = []
    for client_id in cifar_train.client_ids:
      client_data = self.evaluate(
          list(cifar_train.create_tf_dataset_for_client(client_id)))
      self.assertLen(client_data, 100)
      for x in client_data:
        coarse_labels.append(x['coarse_label'])
        labels.append(x['label'])
    self.assertLen(coarse_labels, 50000)
    self.assertLen(labels, 50000)
    self.assertCountEqual(coarse_labels, expected_coarse_labels)
    self.assertCountEqual(labels, expected_labels)

  def test_load_test_data(self):
    _, cifar_test = cifar100.load_data()
    self.assertLen(cifar_test.client_ids, 100)
    self.assertCountEqual(cifar_test.client_ids, [str(i) for i in range(100)])

    self.assertEqual(
        cifar_test.element_type_structure,
        collections.OrderedDict([
            ('coarse_label', tf.TensorSpec(shape=(), dtype=tf.int64)),
            ('image', tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8)),
            ('label', tf.TensorSpec(shape=(), dtype=tf.int64)),
        ]))

    expected_coarse_labels = []
    expected_labels = []
    for i in range(20):
      expected_coarse_labels += [i] * 500
    for i in range(100):
      expected_labels += [i] * 100

    coarse_labels = []
    labels = []
    for client_id in cifar_test.client_ids:
      client_data = self.evaluate(
          list(cifar_test.create_tf_dataset_for_client(client_id)))
      self.assertLen(client_data, 100)
      for x in client_data:
        coarse_labels.append(x['coarse_label'])
        labels.append(x['label'])
    self.assertLen(coarse_labels, 10000)
    self.assertLen(labels, 10000)
    self.assertCountEqual(coarse_labels, expected_coarse_labels)
    self.assertCountEqual(labels, expected_labels)


if __name__ == '__main__':
  tf.test.main()
