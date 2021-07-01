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

import collections

from absl import flags
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import stackoverflow

FLAGS = flags.FLAGS
EXPECTED_ELEMENT_TYPE = collections.OrderedDict(
    creation_date=tf.TensorSpec(shape=[], dtype=tf.string),
    score=tf.TensorSpec(shape=[], dtype=tf.int64),
    tags=tf.TensorSpec(shape=[], dtype=tf.string),
    title=tf.TensorSpec(shape=[], dtype=tf.string),
    tokens=tf.TensorSpec(shape=[], dtype=tf.string),
    type=tf.TensorSpec(shape=[], dtype=tf.string),
)


class StackoverflowTest(absltest.TestCase):

  def test_get_synthetic(self):
    client_data = stackoverflow.get_synthetic()
    synthetic_data_dictionary = stackoverflow.create_synthetic_data_dictionary()
    self.assertCountEqual(client_data.client_ids,
                          synthetic_data_dictionary.keys())
    self.assertEqual(client_data.element_type_structure, EXPECTED_ELEMENT_TYPE)
    dataset = client_data.create_tf_dataset_for_client(
        next(iter(synthetic_data_dictionary.keys())))
    self.assertEqual(dataset.element_spec, EXPECTED_ELEMENT_TYPE)

  def test_load_word_counts(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')
    word_counts = stackoverflow.load_word_counts()
    self.assertLen(word_counts, 6005329)
    self.assertEqual(word_counts['.'], 342309)
    self.assertEqual(word_counts['the'], 341937)

  def test_load_word_counts_small_vocab(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')
    num_words = 100
    word_counts = stackoverflow.load_word_counts(vocab_size=num_words)
    self.assertLen(word_counts, num_words)
    self.assertEqual(word_counts['.'], 342309)
    self.assertEqual(word_counts['the'], 341937)

  def test_load_tag_counts(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')
    tag_counts = stackoverflow.load_tag_counts()
    self.assertLen(tag_counts, 50454)
    self.assertEqual(tag_counts['html'], 167815)
    self.assertEqual(tag_counts['galois-field'], 58)

  def test_load_from_gcs(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')
    train, heldout, test = stackoverflow.load_data(cache_dir=FLAGS.test_tmpdir)
    self.assertLen(train.client_ids, 342_477)
    self.assertLen(heldout.client_ids, 38_758)
    self.assertLen(test.client_ids, 204_088)
    self.assertEqual(train.element_type_structure, EXPECTED_ELEMENT_TYPE)
    self.assertEqual(heldout.element_type_structure, EXPECTED_ELEMENT_TYPE)
    self.assertEqual(test.element_type_structure, EXPECTED_ELEMENT_TYPE)


if __name__ == '__main__':
  absltest.main()
