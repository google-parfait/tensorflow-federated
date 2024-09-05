# Copyright 2024, The TensorFlow Federated Authors.
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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning.optimizers import nest_utils


class MapAtLeavesTest(tf.test.TestCase):

  def test_raises_on_mismatched_leaves(self):
    f = lambda a, b: a + b
    x = (1.0, (1.0, 2.0))
    y = (4.0, (5.0,))
    with self.assertRaises(ValueError):
      nest_utils.map_at_leaves(f, x, y)

  def test_raises_on_mismatched_structure(self):
    f = lambda a, b: a + b
    x = (1.0, 2.0)
    y = (1.0, (2.0,))
    with self.assertRaises(ValueError):
      nest_utils.map_at_leaves(f, x, y)

  def test_scalar_single_arg_single_out(self):
    f = lambda a: 2 * a
    result = nest_utils.map_at_leaves(f, 3.0)
    self.assertEqual(result, 6.0)

  def test_scalar_single_arg_multi_out(self):
    f = lambda a: (2 * a, 3 * a)
    result = nest_utils.map_at_leaves(f, 1.0)
    self.assertEqual(result, (2.0, 3.0))

  def test_scalar_multi_arg_single_out(self):
    f = lambda a, b, c: a + b - c
    result = nest_utils.map_at_leaves(f, 1.0, 4.0, -1.0)
    self.assertEqual(result, 6.0)

  def test_scalar_multi_arg_multi_out(self):
    f = lambda a, b, c: (a + b, a - c)
    result = nest_utils.map_at_leaves(f, 1.0, 4.0, -1.0)
    self.assertEqual(result, (5.0, 2.0))

  def test_dict_single_arg_single_out(self):
    f = lambda a: 2 * a
    x = {'k1': 1.0, 'k2': np.asarray([1.0, 2.0])}
    result = nest_utils.map_at_leaves(f, x)
    expected_result = {'k1': 2.0, 'k2': np.asarray([2.0, 4.0])}
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_dict_single_arg_multi_out(self):
    f = lambda a: (2 * a, np.zeros_like(a))
    x = {'k1': 1.0, 'k2': np.asarray([1.0, 2.0])}
    result = nest_utils.map_at_leaves(f, x)
    expected_result = (
        {'k1': 2.0, 'k2': np.asarray([2.0, 4.0])},
        {'k1': 0.0, 'k2': np.asarray([0.0, 0.0])},
    )
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_dict_multi_arg_single_out(self):
    f = lambda a, b: a + b
    x = {'k1': 1.0, 'k2': np.asarray([1.0, 2.0])}
    y = {'k1': 2.0, 'k2': np.asarray([3.0, 4.0])}
    result = nest_utils.map_at_leaves(f, x, y)
    expected_result = {'k1': 3.0, 'k2': np.asarray([4.0, 6.0])}
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_dict_multi_arg_multi_out(self):
    f = lambda a, b: (a + b, a - b)
    x = {'k1': 1.0, 'k2': np.asarray([1.0, 2.0])}
    y = {'k1': 2.0, 'k2': np.asarray([3.0, 4.0])}
    result = nest_utils.map_at_leaves(f, x, y)
    expected_result = (
        {'k1': 3.0, 'k2': np.asarray([4.0, 6.0])},
        {'k1': -1.0, 'k2': np.asarray([-2.0, -2.0])},
    )
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_dict_in_tuple_single_arg_single_out(self):
    f = lambda a: 2 * a
    x = (1.0, {'k1': 1.0, 'k2': 2.0})
    result = nest_utils.map_at_leaves(f, x)
    expected_result = (2.0, {'k1': 2.0, 'k2': 4.0})
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_dict_int_tuple_single_arg_multi_out(self):
    f = lambda a: (2 * a, np.zeros_like(a))
    x = (1.0, {'k1': 1.0, 'k2': 2.0})
    result = nest_utils.map_at_leaves(f, x)
    expected_result = (
        (2.0, {'k1': 2.0, 'k2': 4.0}),
        (0.0, {'k1': 0.0, 'k2': 0.0}),
    )
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_dict_in_tuple_multi_arg_single_out(self):
    f = lambda a, b: a + b
    x = (1.0, {'k1': 1.0, 'k2': 2.0})
    y = (2.0, {'k1': 2.0, 'k2': 4.0})
    result = nest_utils.map_at_leaves(f, x, y)
    expected_result = (3.0, {'k1': 3.0, 'k2': 6.0})
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_dict_in_tuple_multi_arg_multi_out(self):
    f = lambda a, b: (a + b, a - b)
    x = (1.0, {'k1': 1.0, 'k2': 2.0})
    y = (2.0, {'k1': 2.0, 'k2': 4.0})
    result = nest_utils.map_at_leaves(f, x, y)
    expected_result = (
        (3.0, {'k1': 3.0, 'k2': 6.0}),
        (-1.0, {'k1': -1.0, 'k2': -2.0}),
    )
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_tuple_in_dict_single_arg_single_out(self):
    f = lambda a: 2 * a
    x = {'k1': (1.0, 2.0), 'k2': (3.0, 4.0, 5.0)}
    result = nest_utils.map_at_leaves(f, x)
    expected_result = {'k1': (2.0, 4.0), 'k2': (6.0, 8.0, 10.0)}
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_tuple_in_dict_single_arg_multi_out(self):
    f = lambda a: (2 * a, np.zeros_like(a))
    x = {'k1': (1.0, 2.0), 'k2': (3.0, 4.0, 5.0)}
    result = nest_utils.map_at_leaves(f, x)
    expected_result = (
        {'k1': (2.0, 4.0), 'k2': (6.0, 8.0, 10.0)},
        {'k1': (0.0, 0.0), 'k2': (0.0, 0.0, 0.0)},
    )
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_tuple_in_dict_multi_arg_single_out(self):
    f = lambda a, b: a + b
    x = {'k1': (1.0, 2.0), 'k2': (3.0, 4.0, 5.0)}
    y = {'k1': (2.0, 3.0), 'k2': (6.0, 7.0, 8.0)}
    result = nest_utils.map_at_leaves(f, x, y)
    expected_result = {'k1': (3.0, 5.0), 'k2': (9.0, 11.0, 13.0)}
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)

  def test_tuple_in_dict_multi_arg_multi_out(self):
    f = lambda a, b: (a + b, a - b)
    x = {'k1': (1.0, 2.0), 'k2': (3.0, 4.0, 5.0)}
    y = {'k1': (2.0, 3.0), 'k2': (6.0, 7.0, 8.0)}
    result = nest_utils.map_at_leaves(f, x, y)
    expected_result = (
        {'k1': (3.0, 5.0), 'k2': (9.0, 11.0, 13.0)},
        {'k1': (-1.0, -1.0), 'k2': (-3.0, -3.0, -3.0)},
    )
    tf.nest.map_structure(self.assertAllEqual, result, expected_result)


if __name__ == '__main__':
  tf.test.main()
