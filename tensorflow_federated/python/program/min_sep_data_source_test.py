# Copyright 2025, The TensorFlow Federated Authors.
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
import federated_language
import numpy as np
from tensorflow_federated.python.program import min_sep_data_source


class MinSepDataSourceIteratorTest(parameterized.TestCase):

  def test_init_sets_federated_type(self):
    client_ids = ['a', 'b', 'c']
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )
    iterator = min_sep_data_source.MinSepDataSourceIterator(
        client_ids, min_sep=10, federated_type=client_data_type
    )

    self.assertEqual(iterator.federated_type, client_data_type)

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )

    with self.assertRaises(ValueError):
      min_sep_data_source.MinSepDataSourceIterator(
          client_ids, min_sep=10, federated_type=client_data_type
      )

  def test_init_raises_value_error_with_invalid_min_sep(self):
    client_ids = []
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )

    with self.assertRaises(ValueError):
      min_sep_data_source.MinSepDataSourceIterator(
          client_ids, min_sep=0, federated_type=client_data_type
      )

  @parameterized.named_parameters(
      ('none', None),
      ('negative', -1),
  )
  def test_select_raises_value_error_with_k(self, k):
    client_ids = ['a', 'b', 'c']
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )
    iterator = min_sep_data_source.MinSepDataSourceIterator(
        client_ids, min_sep=10, federated_type=client_data_type
    )

    with self.assertRaises(ValueError):
      iterator.select(k)

  def test_select_returns_client_ids(self):
    num_clients = 1000
    client_ids = [str(i) for i in range(num_clients)]
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )

    # Create an iterator that will have 100 eligible clients per round on
    # average.
    # (1000 clients that are eligible to participate every 10th round)
    min_sep = 10
    iterator = min_sep_data_source.MinSepDataSourceIterator(
        client_ids, min_sep, federated_type=client_data_type
    )

    # Call `select` 1000 times with `k=10`. Each client should be selected for
    # an average of 10 rounds (although each client will have been eligible for
    # 100 rounds).
    # (1000 rounds * 10 clients per round / 1000 clients = 10)
    client_id_to_round_indices = {}
    num_rounds = 1000
    k = 10
    for round_index in range(num_rounds):
      client_ids_for_round = iterator.select(k)

      # Check that the number of client ids returned is equal to `k`.
      self.assertLen(client_ids_for_round, k)

      # Track which rounds clients are chosen to participate in.
      for client_id in client_ids_for_round:
        client_id_to_round_indices.setdefault(client_id, []).append(round_index)

    # Verify that the rounds that clients are chosen to participate in are
    # multiples of `min_sep` apart.
    for round_indices in client_id_to_round_indices.values():
      if len(round_indices) < 2:
        continue
      # Calculate the expected modulus from the first round index, and check
      # that the modulus is the same across all rounds for which this client was
      # chosen to participate.
      expected_modulus = round_indices[0] % min_sep
      for round_index in round_indices[1:]:
        self.assertEqual(round_index % min_sep, expected_modulus)


class MinSepDataSourceTest(absltest.TestCase):

  def test_init_sets_federated_type(self):
    client_ids = ['a', 'b', 'c']
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )
    data_source = min_sep_data_source.MinSepDataSource(
        client_ids, min_sep=10, federated_type=client_data_type
    )

    self.assertEqual(data_source.federated_type, client_data_type)

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )

    with self.assertRaises(ValueError):
      min_sep_data_source.MinSepDataSource(
          client_ids, min_sep=10, federated_type=client_data_type
      )

  def test_init_raises_value_error_with_invalid_min_sep(self):
    client_ids = []
    client_data_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )

    with self.assertRaises(ValueError):
      min_sep_data_source.MinSepDataSource(
          client_ids, min_sep=0, federated_type=client_data_type
      )


if __name__ == '__main__':
  absltest.main()
