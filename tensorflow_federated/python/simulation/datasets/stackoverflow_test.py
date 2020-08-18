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
"""Tests for tensorflow_federated.python.simulation.datasets.stackoverflow."""

import collections

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import stackoverflow


class StackoverflowTest(absltest.TestCase):

  def test_get_synthetic(self):
    client_data = stackoverflow.get_synthetic()
    self.assertCountEqual(client_data.client_ids,
                          stackoverflow._SYNTHETIC_STACKOVERFLOW_DATA.keys())

    expected_type = collections.OrderedDict(
        creation_date=tf.TensorSpec(shape=[], dtype=tf.string),
        title=tf.TensorSpec(shape=[], dtype=tf.string),
        score=tf.TensorSpec(shape=[], dtype=tf.int64),
        tags=tf.TensorSpec(shape=[], dtype=tf.string),
        tokens=tf.TensorSpec(shape=[], dtype=tf.string),
        type=tf.TensorSpec(shape=[], dtype=tf.string),
    )

    self.assertEqual(client_data.element_type_structure, expected_type)

    dataset = client_data.create_tf_dataset_for_client(
        next(iter(stackoverflow._SYNTHETIC_STACKOVERFLOW_DATA.keys())))
    self.assertEqual(dataset.element_spec, expected_type)


if __name__ == '__main__':
  absltest.main()
