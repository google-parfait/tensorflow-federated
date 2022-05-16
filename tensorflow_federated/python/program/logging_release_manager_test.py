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

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import logging_release_manager
from tensorflow_federated.python.program import program_test_utils


class LoggingReleaseManagerTest(parameterized.TestCase,
                                unittest.IsolatedAsyncioTestCase,
                                tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, None),
      ('bool', True, True),
      ('int', 1, 1),
      ('str', 'a', 'a'),
      ('tensor_int', tf.constant(1), tf.constant(1)),
      ('tensor_str', tf.constant('a'), tf.constant('a')),
      ('tensor_2d', tf.ones((2, 3)), tf.ones((2, 3))),
      ('numpy_int', np.int32(1), np.int32(1)),
      ('numpy_2d', np.ones((2, 3)), np.ones((2, 3))),

      # value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1), 1),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       [True, 1, 'a']),
      ('list_empty', [], []),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       [[True, 1], ['a']]),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       {'a': True, 'b': 1, 'c': 'a'}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
      ('attr',
       program_test_utils.TestAttrObject2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       program_test_utils.TestAttrObject2(True, 1)),
      ('attr_nested',
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObject1('a')),
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(True, 1),
           program_test_utils.TestAttrObject1('a'))),
  )
  # pyformat: enable
  async def test_release_logs_value(self, value, expected_value):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_info:
      await release_mngr.release(value)

      mock_info.assert_called_once()
      call = mock_info.mock_calls[0]
      _, args, _ = call
      _, actual_value = args
      if isinstance(actual_value, tf.data.Dataset):
        actual_value = list(actual_value)
      if isinstance(expected_value, tf.data.Dataset):
        expected_value = list(expected_value)
      self.assertAllEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  async def test_release_logs_key(self, key):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_info:
      await release_mngr.release(1, key)

      mock_info.assert_called_once()
      call = mock_info.mock_calls[0]
      _, args, _ = call
      _, actual_key, actual_value = args
      self.assertEqual(actual_key, key)
      self.assertEqual(actual_value, 1)

if __name__ == '__main__':
  absltest.main()
