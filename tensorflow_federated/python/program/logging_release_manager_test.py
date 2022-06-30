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

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import logging_release_manager
from tensorflow_federated.python.program import program_test_utils


class LoggingReleaseManagerTest(parameterized.TestCase,
                                unittest.IsolatedAsyncioTestCase,
                                tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, computation_types.StructType([]), None),
      ('bool', True, computation_types.TensorType(tf.bool), True),
      ('int', 1, computation_types.TensorType(tf.int32), 1),
      ('str', 'a', computation_types.TensorType(tf.string), 'a'),
      ('tensor_int',
       tf.constant(1),
       computation_types.TensorType(tf.int32),
       tf.constant(1)),
      ('tensor_str',
       tf.constant('a'),
       computation_types.TensorType(tf.string),
       tf.constant('a')),
      ('tensor_array',
       tf.ones([3], tf.int32),
       computation_types.TensorType(tf.int32, [3]),
       tf.ones([3], tf.int32)),
      ('numpy_int',
       np.int32(1),
       computation_types.TensorType(tf.int32),
       np.int32(1)),
      ('numpy_array',
       np.ones([3], int),
       computation_types.TensorType(tf.int32, [3]),
       np.ones([3], int)),

      # value references
      ('value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32),
       1),
      ('value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(tf.int32),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       computation_types.SequenceType([tf.bool, tf.int32, tf.string]),
       [True, 1, 'a']),
      ('list_empty', [], computation_types.SequenceType([]), []),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       computation_types.SequenceType([[tf.bool, tf.int32], [tf.string]]),
       [[True, 1], ['a']]),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       computation_types.SequenceType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string)]),
       {'a': True, 'b': 1, 'c': 'a'}),
      ('dict_empty', {}, computation_types.SequenceType([]), {}),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       computation_types.SequenceType([
           ('x', [('a', tf.bool), ('b', tf.int32)]),
           ('y', [('c', tf.string)])]),
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
      ('attr',
       program_test_utils.TestAttrObject2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.SequenceType([('a', tf.bool), ('b', tf.int32)]),
       program_test_utils.TestAttrObject2(True, 1)),
      ('attr_nested',
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObject1('a')),
       computation_types.SequenceType([
           ('a', [('a', tf.bool), ('b', tf.int32)]),
           ('b', [('c', tf.string)])]),
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(True, 1),
           program_test_utils.TestAttrObject1('a'))),
  )
  # pyformat: enable
  async def test_release_logs_value_and_type_signature(self, value,
                                                       type_signature,
                                                       expected_value):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_info:
      await release_mngr.release(value, type_signature)

      self.assertLen(mock_info.mock_calls, 3)
      mock_info.assert_has_calls([
          mock.call(mock.ANY),
          mock.call(mock.ANY, mock.ANY),
          mock.call(mock.ANY, type_signature),
      ])
      call = mock_info.mock_calls[1]
      _, args, kwargs = call
      _, actual_value = args
      if isinstance(actual_value, tf.data.Dataset):
        actual_value = list(actual_value)
      if isinstance(expected_value, tf.data.Dataset):
        expected_value = list(expected_value)
      self.assertAllEqual(actual_value, expected_value)
      self.assertEqual(kwargs, {})

  @parameterized.named_parameters(
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  async def test_release_logs_key(self, key):
    release_mngr = logging_release_manager.LoggingReleaseManager()
    value = 1
    type_signature = computation_types.TensorType(tf.int32)

    with mock.patch('absl.logging.info') as mock_info:
      await release_mngr.release(value, type_signature, key)

      self.assertLen(mock_info.mock_calls, 4)
      mock_info.assert_has_calls([
          mock.call(mock.ANY),
          mock.call(mock.ANY, value),
          mock.call(mock.ANY, type_signature),
          mock.call(mock.ANY, key)
      ])

if __name__ == '__main__':
  absltest.main()
