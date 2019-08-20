# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for ConcreteClientData."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.simulation import client_data as cd


class FilePerUserClientDataTest(tf.test.TestCase, absltest.TestCase):

  def test_concrete_client_data(self):
    client_ids = [1, 2, 3]

    def create_dataset_fn(client_id):
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = cd.ConcreteClientData(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    self.assertEqual(client_data.output_types, tf.int64)
    self.assertEqual(client_data.output_shapes, ())

    def length(ds):
      return tf.data.experimental.cardinality(ds).numpy()

    for i in client_ids:
      self.assertEqual(length(client_data.create_tf_dataset_for_client(i)), i)

    # Preprocess to only take the first example from each client
    client_data = client_data.preprocess(lambda d: d.take(1))
    for i in client_ids:
      self.assertEqual(length(client_data.create_tf_dataset_for_client(i)), 1)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
