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
"""Tests for FromTensorSlicesClientData."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.simulation import from_tensor_slices_client_data


class FromTensorSlicesClientDataTest(tf.test.TestCase):

  def test_basic(self):
    tensor_slices_dict = {'a': [1, 2, 3], 'b': [4, 5]}
    client_data = from_tensor_slices_client_data.FromTensorSlicesClientData(
        tensor_slices_dict)
    self.assertCountEqual(client_data.client_ids, ['a', 'b'])
    self.assertEqual(client_data.output_types, tf.int32)
    self.assertEqual(client_data.output_shapes, ())

    def as_list(dataset):
      return [self.evaluate(x) for x in dataset]

    self.assertEqual(
        as_list(client_data.create_tf_dataset_for_client('a')), [1, 2, 3])
    self.assertEqual(
        as_list(client_data.create_tf_dataset_for_client('b')), [4, 5])

  def test_empty(self):
    with self.assertRaises(ValueError):
      from_tensor_slices_client_data.FromTensorSlicesClientData({'a': []})

  def test_shuffle_client_ids(self):
    tensor_slices_dict = {'a': [1, 1], 'b': [2, 2, 2], 'c': [3], 'd': [4, 4]}
    all_examples = [1, 1, 2, 2, 2, 3, 4, 4]
    client_data = from_tensor_slices_client_data.FromTensorSlicesClientData(
        tensor_slices_dict)

    def get_flat_dataset(seed):
      ds = client_data.create_tf_dataset_from_all_clients(seed=seed)
      return [x.numpy() for x in ds]

    d1 = get_flat_dataset(123)
    d2 = get_flat_dataset(456)
    self.assertNotEqual(d1, d2)  # Different random seeds, different order.
    self.assertCountEqual(d1, all_examples)
    self.assertCountEqual(d2, all_examples)

    # Test that the default behavior is to use a fresh random seed.
    # We could get unlucky, but we are very unlikely to get unlucky
    # 100 times in a row.
    found_not_equal = False
    for _ in range(100):
      if get_flat_dataset(seed=None) != get_flat_dataset(seed=None):
        found_not_equal = True
        break
    self.assertTrue(found_not_equal)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
