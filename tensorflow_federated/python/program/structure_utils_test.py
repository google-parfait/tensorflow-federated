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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import test_utils


class FlattenWithNameTest(parameterized.TestCase, tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, [('', None)]),
      ('bool', True, [('', True)]),
      ('int', 1, [('', 1)]),
      ('list', [True, 1, 'a'], [('0', True), ('1', 1), ('2', 'a')]),
      ('list_empty', [], []),
      ('list_nested',
       [[True, 1], ['a']],
       [('0/0', True), ('0/1', 1), ('1/0', 'a')]),
      ('dict',
       {'a': True, 'b': 1, 'c': 'a'},
       [('a', True), ('b', 1), ('c', 'a')]),
      ('dict_empty', {}, []),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       [('x/a', True), ('x/b', 1), ('y/c', 'a')]),
      ('tensor', tf.ones([1]), [('', tf.ones([1]))]),
      ('tensor_int', tf.constant(1), [('', tf.constant(1))]),
      ('tensor_str', tf.constant('a'), [('', tf.constant('a'))]),
      ('tensor_2d', tf.ones((2, 3)), [('', tf.ones((2, 3)))]),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       [('a/0', tf.constant(True)),
        ('a/1', tf.constant(1)),
        ('b/0', tf.constant('a'))]),
      ('numpy_int', np.int32(1), [('', np.int32(1))]),
      ('numpy_2d', np.ones((2, 3)), [('', np.ones((2, 3)))]),
      ('numpy_nested',
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]},
       [('a/0', np.bool(True)), ('a/1', np.int32(1)), ('b/0', np.str_('a'))]),
      ('materializable_value_reference_tensor',
       test_utils.TestMaterializableValueReference(1),
       [('', test_utils.TestMaterializableValueReference(1))]),
      ('materializable_value_reference_sequence',
       test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       [('', test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])))]),
      ('materializable_value_reference_nested',
       {'a': [test_utils.TestMaterializableValueReference(True),
              test_utils.TestMaterializableValueReference(1)],
        'b': [test_utils.TestMaterializableValueReference('a')]},
       [('a/0', test_utils.TestMaterializableValueReference(True)),
        ('a/1', test_utils.TestMaterializableValueReference(1)),
        ('b/0', test_utils.TestMaterializableValueReference('a'))]),
      ('materializable_value_reference_and_materialized_value',
       [1, test_utils.TestMaterializableValueReference(2)],
       [('0', 1), ('1', test_utils.TestMaterializableValueReference(2))]),
  )
  # pyformat: enable
  def test_returns_result(self, structure, expected_result):
    actual_result = structure_utils.flatten_with_name(structure)

    for actual_item, expected_item in zip(actual_result, expected_result):
      actual_path, actual_value = actual_item
      expected_path, expected_value = expected_item
      self.assertEqual(actual_path, expected_path)
      self.assertAllEqual(actual_value, expected_value)


if __name__ == '__main__':
  absltest.main()
