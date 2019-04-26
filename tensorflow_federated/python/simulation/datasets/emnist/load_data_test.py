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
"""Tests for load_data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets.emnist import load_data


class LoadDataTest(tf.test.TestCase, absltest.TestCase):

  def test_synthetic(self):
    client_data = load_data.get_synthetic(num_clients=4)
    self.assertLen(client_data.client_ids, 4)

    self.assertEqual(
        client_data.output_types,
        collections.OrderedDict([('pixels', tf.float32), ('label', tf.int32)]))
    self.assertEqual(
        client_data.output_shapes,
        collections.OrderedDict([('pixels', (28, 28)), ('label', ())]))

    for client_id in client_data.client_ids:
      data = list(client_data.create_tf_dataset_for_client(client_id))
      images = [x['pixels'].numpy() for x in data]
      labels = [x['label'].numpy() for x in data]
      self.assertLen(labels, 10)
      self.assertCountEqual(labels, range(10))
      self.assertLen(images, 10)
      self.assertEqual(images[0].shape, (28, 28))
      self.assertEqual(images[-1].shape, (28, 28))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
