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
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import memory_release_manager
from tensorflow_federated.python.program import test_utils


class MemoryReleaseManagerTest(parameterized.TestCase, tf.test.TestCase):

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
  async def test_release_saves_value(self, value, expected_value):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    await release_mngr.release(value, 1)

    self.assertLen(release_mngr._values, 1)
    actual_value = release_mngr._values[1]
    if (isinstance(actual_value, tf.data.Dataset) and
        isinstance(expected_value, tf.data.Dataset)):
      self.assertEqual(list(actual_value), list(expected_value))
    else:
      self.assertAllEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
  )
  @test_utils.run_sync
  async def test_release_saves_key(self, key):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    await release_mngr.release(1, key)

    self.assertLen(release_mngr._values, 1)
    self.assertIn(key, release_mngr._values)

  @parameterized.named_parameters(
      ('list', []),
      ('dict', {}),
      ('orderd_dict', collections.OrderedDict()),
  )
  @test_utils.run_sync
  async def test_release_raises_type_error_with_key(self, key):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    with self.assertRaises(TypeError):
      await release_mngr.release(1, key)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('10', 10),
  )
  def test_values_returns_values(self, count):
    release_mngr = memory_release_manager.MemoryReleaseManager()
    for i in range(count):
      release_mngr._values[i] = i * 10

    values = release_mngr.values()

    self.assertEqual(values, {i: i * 10 for i in range(count)})

  def test_values_returns_copy(self):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    values_1 = release_mngr.values()
    values_2 = release_mngr.values()
    self.assertIsNot(values_1, values_2)


if __name__ == '__main__':
  absltest.main()
