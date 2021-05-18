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

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class InferCardinalitiesTest(absltest.TestCase):

  def test_returns_empty_dict_none_value(self):
    type_signature = computation_types.TensorType(tf.int32)
    self.assertEqual(
        cardinalities_utils.infer_cardinalities(None, type_signature), {})

  def test_raises_none_type(self):
    with self.assertRaises(TypeError):
      cardinalities_utils.infer_cardinalities(1, None)

  def test_noops_on_int(self):
    type_signature = computation_types.TensorType(tf.int32)
    cardinalities = cardinalities_utils.infer_cardinalities(1, type_signature)
    self.assertEmpty(cardinalities)

  def test_raises_federated_type_integer(self):
    federated_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=False)
    with self.assertRaises(TypeError):
      cardinalities_utils.infer_cardinalities(1, federated_type)

  def test_raises_federated_type_generator(self):

    def generator_fn():
      yield 1

    generator = generator_fn()
    federated_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=False)
    with self.assertRaises(TypeError):
      cardinalities_utils.infer_cardinalities(generator, federated_type)

  def test_passes_federated_type_tuple(self):
    tup = tuple(range(5))
    federated_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=False)
    cardinalities_utils.infer_cardinalities(tup, federated_type)
    five_client_cardinalities = cardinalities_utils.infer_cardinalities(
        tup, federated_type)
    self.assertEqual(five_client_cardinalities[placements.CLIENTS], 5)

  def test_adds_list_length_as_cardinality_at_clients(self):
    federated_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=False)
    five_clients = list(range(5))
    five_client_cardinalities = cardinalities_utils.infer_cardinalities(
        five_clients, federated_type)
    self.assertEqual(five_client_cardinalities[placements.CLIENTS], 5)

  def test_raises_conflicting_clients_sizes(self):
    federated_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=False)
    five_clients = list(range(5))
    ten_clients = list(range(10))
    tuple_of_federated_types = computation_types.StructType(
        [federated_type, federated_type])
    with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
      cardinalities_utils.infer_cardinalities([five_clients, ten_clients],
                                              tuple_of_federated_types)

  def test_adds_list_length_as_cardinality_at_new_placement(self):
    new_placement = placements.PlacementLiteral('Agg', 'Agg', False,
                                                'Intermediate aggregators')
    federated_type = computation_types.FederatedType(
        tf.int32, new_placement, all_equal=False)
    ten_aggregators = list(range(10))
    ten_aggregator_cardinalities = cardinalities_utils.infer_cardinalities(
        ten_aggregators, federated_type)
    self.assertEqual(ten_aggregator_cardinalities[new_placement], 10)

  def test_recurses_under_tuple_type(self):
    client_int = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=False)
    new_placement = placements.PlacementLiteral('Agg', 'Agg', False,
                                                'Intermediate aggregators')
    aggregator_placed_int = computation_types.FederatedType(
        tf.int32, new_placement, all_equal=False)
    five_aggregators = list(range(5))
    ten_clients = list(range(10))
    mixed_cardinalities = cardinalities_utils.infer_cardinalities(
        [ten_clients, five_aggregators],
        computation_types.StructType([client_int, aggregator_placed_int]))
    self.assertEqual(mixed_cardinalities[placements.CLIENTS], 10)
    self.assertEqual(mixed_cardinalities[new_placement], 5)

  def test_infer_cardinalities_success_structure(self):
    foo = cardinalities_utils.infer_cardinalities(
        structure.Struct([('A', [1, 2, 3]),
                          ('B',
                           structure.Struct([('C', [[1, 2], [3, 4], [5, 6]]),
                                             ('D', [True, False, True])]))]),
        computation_types.StructType([
            ('A', computation_types.FederatedType(tf.int32,
                                                  placements.CLIENTS)),
            ('B', [('C',
                    computation_types.FederatedType(
                        computation_types.SequenceType(tf.int32),
                        placements.CLIENTS)),
                   ('D',
                    computation_types.FederatedType(tf.bool,
                                                    placements.CLIENTS))])
        ]))
    self.assertDictEqual(foo, {placements.CLIENTS: 3})

  def test_infer_cardinalities_structure_failure(self):
    with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
      cardinalities_utils.infer_cardinalities(
          structure.Struct([('A', [1, 2, 3]), ('B', [1, 2])]),
          computation_types.StructType([
              ('A', computation_types.FederatedType(tf.int32,
                                                    placements.CLIENTS)),
              ('B', computation_types.FederatedType(tf.int32,
                                                    placements.CLIENTS))
          ]))

  def test_raises_invalid_non_all_equal_value_error(self):
    with self.assertRaises(cardinalities_utils.InvalidNonAllEqualValueError):
      cardinalities_utils.infer_cardinalities(
          tf.constant(5),
          computation_types.at_clients(computation_types.TensorType(tf.int32)))


class MergeCardinalitiesTest(absltest.TestCase):

  def test_raises_non_dict_arg(self):
    with self.assertRaises(TypeError):
      cardinalities_utils.merge_cardinalities({}, 1)

  def test_raises_non_placement_keyed_dict(self):
    with self.assertRaises(TypeError):
      cardinalities_utils.merge_cardinalities({'a': 1},
                                              {placements.CLIENTS: 10})
    with self.assertRaises(TypeError):
      cardinalities_utils.merge_cardinalities({placements.CLIENTS: 10},
                                              {'a': 1})

  def test_raises_merge_conflicting_cardinalities(self):
    with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
      cardinalities_utils.merge_cardinalities({placements.CLIENTS: 10},
                                              {placements.CLIENTS: 11})

  def test_noops_no_conflict(self):
    clients_placed_cardinality = {placements.CLIENTS: 10}
    noop = cardinalities_utils.merge_cardinalities(clients_placed_cardinality,
                                                   clients_placed_cardinality)
    self.assertEqual(noop, clients_placed_cardinality)

  def test_merges_different_placements(self):
    clients_placed_cardinality = {placements.CLIENTS: 10}
    server_placed_cardinality = {placements.SERVER: 1}
    merged = cardinalities_utils.merge_cardinalities(clients_placed_cardinality,
                                                     server_placed_cardinality)
    self.assertEqual(merged, {placements.CLIENTS: 10, placements.SERVER: 1})


if __name__ == '__main__':
  absltest.main()
