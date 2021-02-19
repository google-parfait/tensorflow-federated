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
"""Tests for tensorflow_federated.python.simulation.datasets.shakespeare."""

import collections

from absl import flags
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import shakespeare

FLAGS = flags.FLAGS
EXPECTED_ELEMENT_TYPE = collections.OrderedDict(
    snippets=tf.TensorSpec(shape=[], dtype=tf.string))


class ShakespeareTest(absltest.TestCase):

  def test_get_synthetic(self):
    client_data = shakespeare.get_synthetic()
    self.assertCountEqual(client_data.client_ids,
                          shakespeare._SYNTHETIC_SHAKESPEARE_DATA.keys())
    self.assertEqual(client_data.element_type_structure, EXPECTED_ELEMENT_TYPE)
    dataset = client_data.create_tf_dataset_for_client(
        next(iter(shakespeare._SYNTHETIC_SHAKESPEARE_DATA.keys())))
    self.assertEqual(dataset.element_spec, EXPECTED_ELEMENT_TYPE)

  def test_load_from_gcs(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        "skipTest to run test locally.")
    train, test = shakespeare.load_data(cache_dir=FLAGS.test_tmpdir)
    self.assertLen(train.client_ids, 715)
    self.assertLen(test.client_ids, 715)
    self.assertEqual(train.element_type_structure, EXPECTED_ELEMENT_TYPE)
    self.assertEqual(test.element_type_structure, EXPECTED_ELEMENT_TYPE)


if __name__ == '__main__':
  absltest.main()
