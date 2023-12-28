# Copyright 2022, The TensorFlow Federated Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import client_id_data_source


class ClientIdDataSourceIteratorTest(parameterized.TestCase):

  def test_init_sets_federated_type(self):
    client_ids = ['a', 'b', 'c']

    iterator = client_id_data_source.ClientIdDataSourceIterator(client_ids)

    federated_type = computation_types.FederatedType(
        np.str_, placements.CLIENTS
    )
    self.assertEqual(iterator.federated_type, federated_type)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_client_ids(self, client_ids):
    with self.assertRaises(TypeError):
      client_id_data_source.ClientIdDataSourceIterator(client_ids)

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []

    with self.assertRaises(ValueError):
      client_id_data_source.ClientIdDataSourceIterator(client_ids)

  @parameterized.named_parameters(
      ('zero', 0),
      ('one', 1),
      ('two', 2),
  )
  def test_select_returns_client_ids_with_k(self, k):
    client_ids = ['a', 'b', 'c']
    iterator = client_id_data_source.ClientIdDataSourceIterator(client_ids)

    actual_client_ids = iterator.select(k)

    self.assertLen(actual_client_ids, k)
    for actual_client_id in actual_client_ids:
      self.assertIn(actual_client_id, client_ids)

  @parameterized.named_parameters(
      ('str', 'a'),
      ('list', []),
  )
  def test_select_raises_type_error_with_k(self, k):
    client_ids = ['a', 'b', 'c']
    iterator = client_id_data_source.ClientIdDataSourceIterator(client_ids)

    with self.assertRaises(TypeError):
      iterator.select(k)

  @parameterized.named_parameters(
      ('none', None),
      ('negative', -1),
      ('greater', 4),
  )
  def test_select_raises_value_error_with_k(self, k):
    client_ids = ['a', 'b', 'c']
    iterator = client_id_data_source.ClientIdDataSourceIterator(client_ids)

    with self.assertRaises(ValueError):
      iterator.select(k)

  def test_serializable(self):
    client_ids = ['a', 'b', 'c']
    iterator = client_id_data_source.ClientIdDataSourceIterator(client_ids)
    iterator_bytes = iterator.to_bytes()

    actual_iterator = (
        client_id_data_source.ClientIdDataSourceIterator.from_bytes(
            iterator_bytes
        )
    )

    self.assertIsNot(actual_iterator, iterator)
    self.assertEqual(actual_iterator, iterator)


class ClientIdDataSourceTest(parameterized.TestCase):

  def test_init_sets_federated_type(self):
    client_ids = ['a', 'b', 'c']

    data_source = client_id_data_source.ClientIdDataSource(client_ids)

    federated_type = computation_types.FederatedType(
        np.str_, placements.CLIENTS
    )
    self.assertEqual(data_source.federated_type, federated_type)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_client_ids(self, client_ids):
    with self.assertRaises(TypeError):
      client_id_data_source.ClientIdDataSource(client_ids)

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []

    with self.assertRaises(ValueError):
      client_id_data_source.ClientIdDataSource(client_ids)


if __name__ == '__main__':
  absltest.main()
