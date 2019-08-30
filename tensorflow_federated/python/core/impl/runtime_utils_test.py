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

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import runtime_utils
from tensorflow_federated.python.core.impl.compiler import placement_literals


class InferCardinalitiesTest(absltest.TestCase):

  def test_raises_none_value(self):
    with self.assertRaises(TypeError):
      runtime_utils.infer_cardinalities(None,
                                        computation_types.to_type(tf.int32))

  def test_raises_none_type(self):
    with self.assertRaises(TypeError):
      runtime_utils.infer_cardinalities(1, None)

  def test_noops_on_int(self):
    int_type = computation_types.to_type(tf.int32)
    cardinalities = runtime_utils.infer_cardinalities(1, int_type)
    self.assertEmpty(cardinalities)

  def test_raises_federated_type_non_list(self):
    federated_type = computation_types.FederatedType(
        tf.int32, placement_literals.CLIENTS, all_equal=False)
    with self.assertRaises(TypeError):
      runtime_utils.infer_cardinalities(1, federated_type)

  def test_adds_list_length_as_cardinality_at_clients(self):
    federated_type = computation_types.FederatedType(
        tf.int32, placement_literals.CLIENTS, all_equal=False)
    five_clients = list(range(5))
    five_client_cardinalities = runtime_utils.infer_cardinalities(
        five_clients, federated_type)
    self.assertEqual(five_client_cardinalities[placement_literals.CLIENTS], 5)

  def test_raises_conflicting_clients_sizes(self):
    federated_type = computation_types.FederatedType(
        tf.int32, placement_literals.CLIENTS, all_equal=False)
    five_clients = list(range(5))
    ten_clients = list(range(10))
    tuple_of_federated_types = computation_types.NamedTupleType(
        [federated_type, federated_type])
    with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
      runtime_utils.infer_cardinalities([five_clients, ten_clients],
                                        tuple_of_federated_types)

  def test_adds_list_length_as_cardinality_at_new_placement(self):
    new_placement = placement_literals.PlacementLiteral(
        'Agg', 'Agg', False, 'Intermediate aggregators')
    federated_type = computation_types.FederatedType(
        tf.int32, new_placement, all_equal=False)
    ten_aggregators = list(range(10))
    ten_aggregator_cardinalities = runtime_utils.infer_cardinalities(
        ten_aggregators, federated_type)
    self.assertEqual(ten_aggregator_cardinalities[new_placement], 10)

  def test_recurses_under_tuple_type(self):
    client_int = computation_types.FederatedType(
        tf.int32, placement_literals.CLIENTS, all_equal=False)
    new_placement = placement_literals.PlacementLiteral(
        'Agg', 'Agg', False, 'Intermediate aggregators')
    aggregator_placed_int = computation_types.FederatedType(
        tf.int32, new_placement, all_equal=False)
    five_aggregators = list(range(5))
    ten_clients = list(range(10))
    mixed_cardinalities = runtime_utils.infer_cardinalities(
        [ten_clients, five_aggregators],
        computation_types.to_type([client_int, aggregator_placed_int]))
    self.assertEqual(mixed_cardinalities[placement_literals.CLIENTS], 10)
    self.assertEqual(mixed_cardinalities[new_placement], 5)

  def test_infer_cardinalities_success_anonymous_tuple(self):
    foo = runtime_utils.infer_cardinalities(
        anonymous_tuple.AnonymousTuple([
            ('A', [1, 2, 3]),
            ('B',
             anonymous_tuple.AnonymousTuple([('C', [[1, 2], [3, 4], [5, 6]]),
                                             ('D', [True, False, True])]))
        ]),
        computation_types.to_type([
            ('A',
             computation_types.FederatedType(tf.int32,
                                             placement_literals.CLIENTS)),
            ('B', [('C',
                    computation_types.FederatedType(
                        computation_types.SequenceType(tf.int32),
                        placement_literals.CLIENTS)),
                   ('D',
                    computation_types.FederatedType(tf.bool,
                                                    placement_literals.CLIENTS))
                  ])
        ]))
    self.assertDictEqual(foo, {placement_literals.CLIENTS: 3})

  def test_infer_cardinalities_anonymous_tuple_failure(self):
    with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
      runtime_utils.infer_cardinalities(
          anonymous_tuple.AnonymousTuple([('A', [1, 2, 3]), ('B', [1, 2])]),
          computation_types.to_type([
              ('A',
               computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)),
              ('B',
               computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS))
          ]))


class MergeCardinalitiesTest(absltest.TestCase):

  def test_raises_non_dict_arg(self):
    with self.assertRaises(TypeError):
      runtime_utils.merge_cardinalities({}, 1)

  def test_raises_non_placement_keyed_dict(self):
    with self.assertRaises(TypeError):
      runtime_utils.merge_cardinalities({'a': 1},
                                        {placement_literals.CLIENTS: 10})
    with self.assertRaises(TypeError):
      runtime_utils.merge_cardinalities({placement_literals.CLIENTS: 10},
                                        {'a': 1})

  def test_raises_merge_conflicting_cardinalities(self):
    with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
      runtime_utils.merge_cardinalities({placement_literals.CLIENTS: 10},
                                        {placement_literals.CLIENTS: 11})

  def test_noops_no_conflict(self):
    clients_placed_cardinality = {placement_literals.CLIENTS: 10}
    noop = runtime_utils.merge_cardinalities(clients_placed_cardinality,
                                             clients_placed_cardinality)
    self.assertEqual(noop, clients_placed_cardinality)

  def test_merges_different_placement_literals(self):
    clients_placed_cardinality = {placement_literals.CLIENTS: 10}
    server_placed_cardinality = {placement_literals.SERVER: 1}
    merged = runtime_utils.merge_cardinalities(clients_placed_cardinality,
                                               server_placed_cardinality)
    self.assertEqual(merged, {
        placement_literals.CLIENTS: 10,
        placement_literals.SERVER: 1
    })


if __name__ == '__main__':
  absltest.main()
