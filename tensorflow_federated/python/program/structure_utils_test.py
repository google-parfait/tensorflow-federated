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

from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import structure_utils


class FlattenWithNameTest(parameterized.TestCase, tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, [('', None)]),
      ('bool', True, [('', True)]),
      ('int', 1, [('', 1)]),
      ('str', 'a', [('', 'a')]),
      ('tensor_int', tf.constant(1), [('', tf.constant(1))]),
      ('tensor_str', tf.constant('a'), [('', tf.constant('a'))]),
      ('tensor_2d', tf.ones((2, 3)), [('', tf.ones((2, 3)))]),
      ('numpy_int', np.int32(1), [('', np.int32(1))]),
      ('numpy_2d', np.ones((2, 3)), [('', np.ones((2, 3)))]),

      # value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       [('', program_test_utils.TestMaterializableValueReference(1))]),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       [('', program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])))]),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       [('0', True),
        ('1', program_test_utils.TestMaterializableValueReference(1)),
        ('2', 'a')]),
      ('list_empty', [], []),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       [('0/0', True),
        ('0/1', program_test_utils.TestMaterializableValueReference(1)),
        ('1/0', 'a')]),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       [('a', True),
        ('b', program_test_utils.TestMaterializableValueReference(1)),
        ('c', 'a')]),
      ('dict_empty', {}, []),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       [('x/a', True),
        ('x/b', program_test_utils.TestMaterializableValueReference(1)),
        ('y/c', 'a')]),
      ('attr',
       program_test_utils.TestAttrObject2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       [('a', True),
        ('b', program_test_utils.TestMaterializableValueReference(1))]),
      ('attr_nested',
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObject1('a')),
       [('a/a', True),
        ('a/b', program_test_utils.TestMaterializableValueReference(1)),
        ('b/a', 'a')]),
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
