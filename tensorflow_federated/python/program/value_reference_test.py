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

from tensorflow_federated.python.program import test_utils
from tensorflow_federated.python.program import value_reference


class MaterializeValueTest(parameterized.TestCase, tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, None),
      ('bool', True, True),
      ('int', 1, 1),
      ('str', 'a', 'a'),
      ('list', [True, 1, 'a'], [True, 1, 'a']),
      ('list_empty', [], []),
      ('list_nested', [[True, 1], ['a']], [[True, 1], ['a']]),
      ('dict', {'a': True, 'b': 1, 'c': 'a'}, {'a': True, 'b': 1, 'c': 'a'}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
      ('attr',
       test_utils.TestAttrObject1(True, 1),
       test_utils.TestAttrObject1(True, 1)),
      ('attr_nested',
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')},
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')}),
      ('tensor_int', tf.constant(1), tf.constant(1)),
      ('tensor_str', tf.constant('a'), tf.constant('a')),
      ('tensor_2d', tf.ones((2, 3)), tf.ones((2, 3))),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]}),
      ('numpy_int', np.int32(1), np.int32(1)),
      ('numpy_2d', np.ones((2, 3)), np.ones((2, 3))),
      ('numpy_nested',
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]},
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]}),
      ('materializable_value_reference_tensor',
       test_utils.TestMaterializableValueReference(1), 1),
      ('materializable_value_reference_sequence',
       test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),
      ('materializable_value_reference_nested',
       {'a': [test_utils.TestMaterializableValueReference(True),
              test_utils.TestMaterializableValueReference(1)],
        'b': [test_utils.TestMaterializableValueReference('a')]},
       {'a': [True, 1], 'b': ['a']}),
      ('materializable_value_reference_and_materialized_value',
       [1, test_utils.TestMaterializableValueReference(2)],
       [1, 2]),
  )
  # pyformat: enable
  @test_utils.run_sync
  async def test_returns_value(self, value, expected_value):
    actual_value = await value_reference.materialize_value(value)

    if (isinstance(actual_value, tf.data.Dataset) and
        isinstance(expected_value, tf.data.Dataset)):
      self.assertEqual(list(actual_value), list(expected_value))
    else:
      self.assertAllEqual(actual_value, expected_value)


if __name__ == '__main__':
  absltest.main()
