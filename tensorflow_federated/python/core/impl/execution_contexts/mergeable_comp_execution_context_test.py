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

from absl.testing import absltest
import numpy as np

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class PartitionValueTest(absltest.TestCase):

  def test_partitions_value_with_no_clients_arguments(self):
    value = 0
    type_signature = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    num_desired_subrounds = 2
    partitioned_value = (
        mergeable_comp_execution_context._split_value_into_subrounds(
            value, type_signature, num_desired_subrounds
        )
    )
    self.assertEqual(partitioned_value, [0, 0])

  def test_wraps_value_with_empty_client_argument(self):
    value = (0, [])
    type_signature = computation_types.StructType([
        (None, computation_types.FederatedType(np.int32, placements.SERVER)),
        (None, computation_types.FederatedType(np.int32, placements.CLIENTS)),
    ])
    num_desired_subrounds = 2
    partitioned_value = (
        mergeable_comp_execution_context._split_value_into_subrounds(
            value, type_signature, num_desired_subrounds
        )
    )
    self.assertEqual(partitioned_value, [(0, [])])

  def test_replicates_all_equal_clients_argument(self):
    value = (0, 1)
    type_signature = computation_types.StructType([
        (None, computation_types.FederatedType(np.int32, placements.SERVER)),
        (
            None,
            computation_types.FederatedType(
                np.int32, placements.CLIENTS, all_equal=True
            ),
        ),
    ])
    num_desired_subrounds = 2
    partitioned_value = (
        mergeable_comp_execution_context._split_value_into_subrounds(
            value, type_signature, num_desired_subrounds
        )
    )
    self.assertEqual(partitioned_value, [(0, 1), (0, 1)])

  def test_partitions_client_placed_value_into_subrounds(self):
    value = list(range(10))
    type_signature = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    num_desired_subrounds = 5
    partitioned_value = (
        mergeable_comp_execution_context._split_value_into_subrounds(
            value, type_signature, num_desired_subrounds
        )
    )
    expected_partitioning = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    self.assertEqual(partitioned_value, expected_partitioning)

  def test_partitions_clients_placed_struct_elem_into_subrounds(self):
    value = (0, list(range(10)))
    server_placed_name = 'a'
    clients_placed_name = 'b'
    type_signature = computation_types.StructType([
        (
            server_placed_name,
            computation_types.FederatedType(np.int32, placements.SERVER),
        ),
        (
            clients_placed_name,
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ),
    ])

    num_desired_subrounds = 5
    expected_partitioning = []
    for j in range(0, 10, 2):
      expected_struct_partition = structure.Struct(
          [(server_placed_name, 0), (clients_placed_name, [j, j + 1])]
      )
      expected_partitioning.append(expected_struct_partition)
    partitioned_value = (
        mergeable_comp_execution_context._split_value_into_subrounds(
            value, type_signature, num_desired_subrounds
        )
    )
    self.assertEqual(partitioned_value, expected_partitioning)

  def test_partitions_fewer_clients_than_rounds_into_nonempty_rounds(self):
    value = [0, 1]
    type_signature = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    num_desired_subrounds = 5
    partitioned_value = (
        mergeable_comp_execution_context._split_value_into_subrounds(
            value, type_signature, num_desired_subrounds
        )
    )
    expected_partitioning = [[0], [1]]
    self.assertEqual(partitioned_value, expected_partitioning)


class RepackageResultsTest(absltest.TestCase):

  def assertRoundTripEqual(
      self, value, type_signature, expected_round_trip_value
  ):
    num_desired_subrounds = 2
    partitioned_value = (
        mergeable_comp_execution_context._split_value_into_subrounds(
            value, type_signature, num_desired_subrounds
        )
    )
    self.assertEqual(
        mergeable_comp_execution_context._repackage_partitioned_values(
            partitioned_value, type_signature
        ),
        expected_round_trip_value,
    )

  def test_roundtrip_with_no_clients_argument(self):
    value = 0
    type_signature = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    self.assertRoundTripEqual(value, type_signature, value)

  def test_roundtrip_with_named_struct(self):
    value = collections.OrderedDict(a=0)
    type_signature = computation_types.StructType(
        [('a', computation_types.FederatedType(np.int32, placements.SERVER))]
    )
    self.assertRoundTripEqual(
        value, type_signature, structure.Struct([('a', 0)])
    )

  def test_roundtrip_with_empty_clients_argument(self):
    value = (0, [])
    type_signature = computation_types.StructType([
        (None, computation_types.FederatedType(np.int32, placements.SERVER)),
        (None, computation_types.FederatedType(np.int32, placements.CLIENTS)),
    ])
    self.assertRoundTripEqual(
        value, type_signature, structure.from_container(value)
    )

  def test_roundtrip_with_nonempty_clients_argument(self):
    value = list(range(10))
    type_signature = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    self.assertRoundTripEqual(value, type_signature, value)

  def test_roundtrip_with_nonempty_tuple_clients_argument(self):
    value = tuple(range(10))
    type_signature = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    self.assertRoundTripEqual(value, type_signature, value)

  def test_roundtrip_with_all_equal_clients_argument(self):
    value = (0, 1)
    type_signature = computation_types.StructType([
        (None, computation_types.FederatedType(np.int32, placements.SERVER)),
        (
            None,
            computation_types.FederatedType(
                np.int32, placements.CLIENTS, all_equal=True
            ),
        ),
    ])
    self.assertRoundTripEqual(
        value, type_signature, structure.from_container(value)
    )


if __name__ == '__main__':
  absltest.main()
