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

from absl import flags
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import cifar100

FLAGS = flags.FLAGS
EXPECTED_ELEMENT_TYPE = collections.OrderedDict(
    coarse_label=tf.TensorSpec(shape=(), dtype=tf.int64),
    image=tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
    label=tf.TensorSpec(shape=(), dtype=tf.int64))


class CIFAR100Test(tf.test.TestCase):

  def test_get_synthetic(self):
    client_data = cifar100.get_synthetic()
    self.assertLen(client_data.client_ids, 1)
    self.assertEqual(client_data.element_type_structure, EXPECTED_ELEMENT_TYPE)

    data = self.evaluate(
        list(
            client_data.create_tf_dataset_for_client(
                client_data.client_ids[0])))
    coarse_labels = [x['coarse_label'] for x in data]
    images = [x['image'] for x in data]
    labels = [x['label'] for x in data]
    self.assertLen(coarse_labels, 5)
    self.assertCountEqual(coarse_labels, [4, 4, 4, 8, 10])

    self.assertLen(labels, 5)
    self.assertCountEqual(labels, [0, 51, 51, 88, 71])

    self.assertLen(images, 5)
    self.assertEqual(images[0].shape, (32, 32, 3))
    self.assertEqual(images[-1].shape, (32, 32, 3))

  def test_load_from_gcs(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')
    cifar_test = cifar100.load_data(FLAGS.test_tmpdir)[1]
    self.assertLen(cifar_test.client_ids, 100)
    self.assertCountEqual(cifar_test.client_ids, [str(i) for i in range(100)])
    self.assertEqual(cifar_test.element_type_structure, EXPECTED_ELEMENT_TYPE)
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
