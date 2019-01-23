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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs.model_compression.core import utils


class StaticOrDynamicShapeTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `static_or_dynamic_shape` method."""

  def test_static_or_dynamic_shape(self):
    # Tensor with statically known shape.
    x = tf.constant([0.0, 2.0, -1.5])
    self.assertAllEqual((3,), utils.static_or_dynamic_shape(x))

    # Tensor without statically known shape.
    x = tf.squeeze(tf.where(tf.less(tf.random_uniform([10]), 0.5)))
    shape = utils.static_or_dynamic_shape(x)
    self.assertIsInstance(shape, tf.Tensor)
    x, shape = self.evaluate([x, shape])
    self.assertLen(x, shape)

    # Numpy value.
    x = np.array([0.0, 2.0, -1.5])
    self.assertAllEqual((3,), utils.static_or_dynamic_shape(x))

  @parameterized.parameters([1.0, 'str', object])
  def test_static_or_dynamic_shape_raises(self, bad_input):
    with self.assertRaises(TypeError):
      utils.static_or_dynamic_shape(bad_input)


class SplitMergeDictTest(parameterized.TestCase):
  """Tests for `split_dict_py_tf` and `merge_dicts` methods."""

  def test_split_dict_py_tf_empty(self):
    """Tests that `split_dict_py_tf` works with empty dictionary."""
    d_py, d_tf = utils.split_dict_py_tf({})
    self.assertDictEqual({}, d_py)
    self.assertDictEqual({}, d_tf)

  def test_split_dict_py_tf_basic(self):
    """Tests that `split_dict_py_tf` works with flat dictionary."""
    const = tf.constant(2.0)
    test_dict = {'py': 1.0, 'tf': const}
    expected_d_py = {'py': 1.0}
    expected_d_tf = {'tf': const}
    d_py, d_tf = utils.split_dict_py_tf(test_dict)
    self.assertDictEqual(expected_d_py, d_py)
    self.assertDictEqual(expected_d_tf, d_tf)

  def test_split_dict_py_tf_nested(self):
    """Tests that `split_dict_py_tf` works with nested dictionary."""
    const_1, const_2 = tf.constant(1.0), tf.constant(2.0)
    test_dict = {
        'nested': {
            'a': 1.0,
            'b': const_1
        },
        'py': 'string',
        'tf': const_2
    }
    expected_d_py = {
        'nested': {
            'a': 1.0,
        },
        'py': 'string',
    }
    expected_d_tf = {'nested': {'b': const_1}, 'tf': const_2}
    d_py, d_tf = utils.split_dict_py_tf(test_dict)
    self.assertDictEqual(expected_d_py, d_py)
    self.assertDictEqual(expected_d_tf, d_tf)

  @parameterized.parameters(None, [[]], 2.0, 'string')
  def test_split_dict_py_tf_raises(self, bad_input):
    """Tests that `split_dict_py_tf` raises `TypeError`."""
    with self.assertRaises(TypeError):
      utils.split_dict_py_tf(bad_input)

  # pyformat: disable
  @parameterized.parameters(
      ({}, {}, {}),
      ({'a': {'b': None}}, {'a': {'c': None}}, {'a': {'b': None, 'c': None}}),
      ({'a': 1}, {'b': None}, {'a': 1, 'b': None}),
      ({'a': 1}, {'b': 2}, {'a': 1, 'b': 2}),
      ({'a': {'aa': 11}, 'b': 2},
       {'a': {'ab': 12}},
       {'a': {'aa': 11, 'ab': 12}, 'b': 2})
      )
  # pyformat: enable
  def test_merge_dicts(self, dict1, dict2, expected_dict):
    """Tests that `merge_dicts` works as expected."""
    self.assertDictEqual(expected_dict, utils.merge_dicts(dict1, dict2))
    self.assertDictEqual(expected_dict, utils.merge_dicts(dict2, dict1))

  # pyformat: disable
  @parameterized.parameters(
      ({}),
      ({'py': 1.0, 'tf': tf.constant(1.0)}),
      ({'nested': {'a': 1.0, 'b': tf.constant(1.0)},
        'py': 'string', 'tf': tf.constant(2.0)}))
  # pyformat: enable
  def test_split_merge_identity(self, **test_dict):
    """Tests that spliting and merging amounts to identity.

    This test method tests that using the `split_dict_py_tf` and `merge_dicts`
    methods together amounts to an identity.

    Args:
      **test_dict: A dictionary to be used for the test.
    """
    new_dict = utils.merge_dicts(*utils.split_dict_py_tf(test_dict))
    self.assertDictEqual(new_dict, test_dict)

  # pyformat: disable
  @parameterized.parameters(
      ('not_a_dict', {'a': None}, TypeError),  # Not a dictionary.
      ({'a': {'b': 0}}, {'a': None}, ValueError),  # Bad structure.
      ({'a': {}}, {'b': {}}, ValueError),  # Bad structure.
      ({'a': 1.0}, {'a': 2.0}, ValueError),  # Both values are set.
      ({1: None}, {1.0: 'value'}, ValueError))  # 1 and 1.0 are not the same.
  # pyformat: enable
  def test_merge_dicts_raises(self, bad_dict1, bad_dict2, error_type):
    """Tests that `merge_dicts` raises appropriate error."""
    with self.assertRaises(error_type):
      utils.merge_dicts(bad_dict1, bad_dict2)
    with self.assertRaises(error_type):
      utils.merge_dicts(bad_dict2, bad_dict1)


if __name__ == '__main__':
  tf.test.main()
