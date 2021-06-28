# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.simulation.datasets import celeba

FLAGS = flags.FLAGS
_EXPECTED_TYPE = collections.OrderedDict(
    sorted([(celeba.IMAGE_NAME,
             tf.TensorSpec(shape=(84, 84, 3), dtype=tf.int64))] +
           [(field_name, tf.TensorSpec(shape=(), dtype=tf.bool))
            for field_name in celeba.ATTRIBUTE_NAMES]))


class CelebATest(tf.test.TestCase):

  def _check_num_examples(self, client_data, expected_min_num_examples):
    # Check that clients have at least expected_min_num_examples. To do this
    # check for every client takes way too long and makes the unit test run time
    # painful, so just check the first ten clients.
    for client_id in client_data.client_ids[:10]:
      dataset = self.evaluate(
          list(client_data.create_tf_dataset_for_client(client_id)))
      self.assertGreaterEqual(len(dataset), expected_min_num_examples)

  def test_load_from_gcs(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')

    def run_test(split_by_clients: bool, expected_num_train_clients: int,
                 expected_num_test_clients: int,
                 expected_min_num_train_examples: int,
                 expected_min_num_test_examples: int):
      train, test = celeba.load_data(
          split_by_clients, cache_dir=FLAGS.test_tmpdir)
      self.assertLen(train.client_ids, expected_num_train_clients)
      self.assertLen(test.client_ids, expected_num_test_clients)
      self.assertIsInstance(train.element_type_structure,
                            collections.OrderedDict)
      self.assertIsInstance(test.element_type_structure,
                            collections.OrderedDict)
      self.assertEqual(_EXPECTED_TYPE, train.element_type_structure)
      self.assertEqual(_EXPECTED_TYPE, test.element_type_structure)
      self._check_num_examples(train, expected_min_num_train_examples)
      self._check_num_examples(test, expected_min_num_test_examples)

    with self.subTest('split_by_clients'):
      run_test(
          split_by_clients=True,
          expected_num_train_clients=8408,
          expected_num_test_clients=935,
          expected_min_num_train_examples=5,
          expected_min_num_test_examples=5)
    with self.subTest('split_by_examples'):
      run_test(
          split_by_clients=False,
          expected_num_train_clients=9343,
          expected_num_test_clients=9343,
          expected_min_num_train_examples=4,
          expected_min_num_test_examples=1)


if __name__ == '__main__':
  tf.test.main()
