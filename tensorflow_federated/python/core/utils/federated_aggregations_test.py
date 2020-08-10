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

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.utils import federated_aggregations


class FederatedMinTest(test.TestCase):

  def test_federated_min_single_value(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_min(value):
      return federated_aggregations.federated_min(value)

    value = call_federated_min([1.0, 2.0, 5.0])
    self.assertEqual(value, 1.0)

  def test_federated_min_on_nested_scalars(self):
    tuple_type = collections.OrderedDict([
        ('x', tf.float32),
        ('y', tf.float32),
    ])

    @computations.federated_computation(
        computation_types.FederatedType(tuple_type, placements.CLIENTS))
    def call_federated_min(value):
      return federated_aggregations.federated_min(value)

    test_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    value = call_federated_min(
        [test_type(0.0, 1.0),
         test_type(-1.0, 5.0),
         test_type(2.0, -10.0)])
    self.assertEqual(value, collections.OrderedDict([('x', -1.0),
                                                     ('y', -10.0)]))

  def test_federated_min_wrong_type(self):
    with self.assertRaisesRegex(TypeError,
                                r'Type must be int32 or float32. Got: .*'):

      @computations.federated_computation(
          computation_types.FederatedType(tf.bool, placements.CLIENTS))
      def call_federated_min(value):
        return federated_aggregations.federated_min(value)

      call_federated_min([False])

  def test_federated_min_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.* argument must be a tff.Value placed at CLIENTS'):

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def call_federated_min(value):
        return federated_aggregations.federated_min(value)

      call_federated_min([1, 2, 3])


class FederatedMaxTest(test.TestCase):

  def test_federated_max_tensor_value(self):

    @computations.federated_computation(
        computation_types.FederatedType((tf.int32, [3]), placements.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    client1 = np.array([1, -2, 3], dtype=np.int32)
    client2 = np.array([0, 7, 1], dtype=np.int32)
    value = call_federated_max([client1, client2])
    self.assertCountEqual(value, [1, 7, 3])

  def test_federated_max_single_value(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    value = call_federated_max([6.0, 4.0, 1.0, 7.0])
    self.assertEqual(value, 7.0)

  def test_federated_max_wrong_type(self):
    with self.assertRaisesRegex(TypeError,
                                r'Type must be int32 or float32. Got: .*'):

      @computations.federated_computation(
          computation_types.FederatedType(tf.bool, placements.CLIENTS))
      def call_federated_max(value):
        return federated_aggregations.federated_max(value)

      call_federated_max([True, False])

  def test_federated_max_on_nested_scalars(self):
    tuple_type = collections.OrderedDict([
        ('a', tf.int32),
        ('b', tf.int32),
    ])

    @computations.federated_computation(
        computation_types.FederatedType(tuple_type, placements.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    test_type = collections.namedtuple('NestedScalars', ['a', 'b'])
    value = call_federated_max(
        [test_type(1, 5), test_type(2, 3),
         test_type(1, 8)])
    self.assertEqual(value, collections.OrderedDict([('a', 2), ('b', 8)]))

  def test_federated_max_nested_tensor_value(self):
    tuple_type = collections.OrderedDict([
        ('a', (tf.int32, [2])),
        ('b', (tf.int32, [3])),
    ])

    @computations.federated_computation(
        computation_types.FederatedType(tuple_type, placements.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    test_type = collections.namedtuple('NestedScalars', ['a', 'b'])
    client1 = test_type(
        np.array([4, 5], dtype=np.int32), np.array([1, -2, 3], dtype=np.int32))
    client2 = test_type(
        np.array([9, 0], dtype=np.int32), np.array([5, 1, -2], dtype=np.int32))
    value = call_federated_max([client1, client2])
    self.assertCountEqual(value['a'], [9, 5])
    self.assertCountEqual(value['b'], [5, 1, 3])

  def test_federated_max_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.*argument must be a tff.Value placed at CLIENTS.*'):

      @computations.federated_computation(
          computation_types.FederatedType(tf.float32, placements.SERVER))
      def call_federated_max(value):
        return federated_aggregations.federated_max(value)

      call_federated_max([1.0, 2.0, 3.0])


class FederatedSampleTest(tf.test.TestCase):

  def test_federated_sample_single_value(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return federated_aggregations.federated_sample(value)

    value = call_federated_sample([1.0, 2.0, 5.0])
    self.assertCountEqual(value, [1.0, 2.0, 5.0])

  def test_federated_sample_on_nested_scalars(self):
    tuple_type = collections.OrderedDict([
        ('x', tf.float32),
        ('y', tf.float32),
    ])

    @computations.federated_computation(
        computation_types.FederatedType(tuple_type, placements.CLIENTS))
    def call_federated_sample(value):
      return federated_aggregations.federated_sample(value)

    x0 = 0.0
    y0 = 1.0
    x1 = -1.0
    y1 = 5.0
    test_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    value = call_federated_sample(
        [test_type(x0, y0),
         test_type(x1, y1),
         test_type(2.0, -10.0)])
    result = value._asdict()
    i0 = list(result['x']).index(x0)
    i1 = list(result['y']).index(y1)

    # Assert shuffled in unison.
    self.assertEqual(result['y'][i0], y0)
    self.assertEqual(result['x'][i1], x1)

  def test_federated_sample_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.*argument must be a tff.Value placed at CLIENTS.*'):

      @computations.federated_computation(
          computation_types.FederatedType(tf.bool, placements.SERVER))
      def call_federated_sample(value):
        return federated_aggregations.federated_sample(value)

      call_federated_sample([True, False, True, True])

  def test_federated_sample_max_size_is_100(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return federated_aggregations.federated_sample(value)

    value = call_federated_sample([1.0] * 100 + [0.0] * 100)
    self.assertLen(value, 100)
    self.assertAlmostEqual(len(np.nonzero(value)[0]), 50, delta=15)

  def test_federated_sample_preserves_nan_percentage(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return federated_aggregations.federated_sample(value)

    value = call_federated_sample([1.0] * 100 + [np.nan] * 100)
    self.assertAlmostEqual(np.count_nonzero(np.isnan(value)), 50, delta=15)

  def test_federated_sample_preserves_inf_percentage(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return federated_aggregations.federated_sample(value)

    value = call_federated_sample([1.0] * 100 + [np.inf] * 100)
    self.assertAlmostEqual(np.count_nonzero(np.isinf(value)), 50, delta=15)

  def test_federated_sample_named_tuple_type_of_ordered_dict(self):
    dict_type = computation_types.to_type(
        collections.OrderedDict([('x', tf.float32), ('y', tf.float32)]))

    @computations.federated_computation(
        computation_types.FederatedType(dict_type, placements.CLIENTS))
    def call_federated_sample(value):
      return federated_aggregations.federated_sample(value)

    x = 0.0
    y = 5.0
    test_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    value = call_federated_sample(
        [test_type(x, y),
         test_type(3.4, 5.6),
         test_type(1.0, 1.0)])
    result = value._asdict()

    self.assertIn(y, result['y'])
    self.assertIn(x, result['x'])

  def test_federated_sample_nested_named_tuples(self):
    tuple_test_type = (
        collections.OrderedDict([('x', tf.float32), ('y', tf.float32)]))
    dict_test_type = (
        computation_types.to_type(
            collections.OrderedDict([('a', tf.float32), ('b', tf.float32)])))
    nested_tuple_type = collections.OrderedDict([('tuple_1', tuple_test_type),
                                                 ('tuple_2', dict_test_type)])
    nested_test_type = collections.namedtuple('Nested', ['tuple_1', 'tuple_2'])

    @computations.federated_computation(
        computation_types.FederatedType(nested_tuple_type, placements.CLIENTS))
    def call_federated_sample(value):
      return federated_aggregations.federated_sample(value)

    tuple_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    dict_type = collections.namedtuple('NestedScalars', ['a', 'b'])
    value = call_federated_sample([
        nested_test_type(tuple_type(1.2, 2.2), dict_type(1.3, 8.8)),
        nested_test_type(tuple_type(-9.1, 3.1), dict_type(1.2, -5.4))
    ])._asdict(recursive=True)

    self.assertIn(1.2, value['tuple_1']['x'])
    self.assertIn(8.8, value['tuple_2']['b'])


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
