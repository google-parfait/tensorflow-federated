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
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import emnist

FLAGS = flags.FLAGS
EXPECTED_ELEMENT_TYPE = collections.OrderedDict(
    label=tf.TensorSpec(shape=(), dtype=tf.int32),
    pixels=tf.TensorSpec(shape=(28, 28), dtype=tf.float32))


class EMNISTTest(tf.test.TestCase, absltest.TestCase):

  def test_get_synthetic(self):
    client_data = emnist.get_synthetic()
    self.assertLen(client_data.client_ids, 1)
    self.assertEqual(client_data.element_type_structure, EXPECTED_ELEMENT_TYPE)
    for client_id in client_data.client_ids:
      data = self.evaluate(
          list(client_data.create_tf_dataset_for_client(client_id)))
      images = [x['pixels'] for x in data]
      labels = [x['label'] for x in data]
      self.assertLen(labels, 10)
      self.assertCountEqual(labels, list(range(10)))
      self.assertLen(images, 10)
      self.assertEqual(images[0].shape, (28, 28))
      self.assertEqual(images[-1].shape, (28, 28))

  def test_load_from_gcs(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')

    def run_test(only_digits: bool, expect_num_clients):
      train, test = emnist.load_data(only_digits, cache_dir=FLAGS.test_tmpdir)
      self.assertLen(train.client_ids, expect_num_clients)
      self.assertLen(test.client_ids, expect_num_clients)
      self.assertEqual(train.element_type_structure, EXPECTED_ELEMENT_TYPE)
      self.assertEqual(test.element_type_structure, EXPECTED_ELEMENT_TYPE)

    with self.subTest('only_digits'):
      run_test(True, 3383)
    with self.subTest('all'):
      run_test(False, 3400)


if __name__ == '__main__':
  tf.test.main()
