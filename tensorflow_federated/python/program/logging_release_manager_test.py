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
import tree

from tensorflow_federated.python.program import logging_release_manager
from tensorflow_federated.python.program import program_test_utils


class LoggingReleaseManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, None),
      ('bool', True, True),
      ('int', 1, 1),
      ('str', 'a', 'a'),
      ('tensor_int', tf.constant(1), tf.constant(1)),
      ('tensor_str', tf.constant('a'), tf.constant('a')),
      ('tensor_array', tf.constant([1] * 3), tf.constant([1] * 3)),
      ('numpy_int', np.int32(1), np.int32(1)),
      ('numpy_array', np.array([1] * 3, np.int32), np.array([1] * 3, np.int32)),

      # materializable value references
      ('value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       1),
      ('value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       program_test_utils.TestSerializable(1, 2)),

      # other values
      ('attrs',
       program_test_utils.TestAttrs(1, 2),
       program_test_utils.TestAttrs(1, 2)),

      # structures
      ('list',
       [
           True,
           1,
           'a',
           program_test_utils.TestMaterializableValueReference(2),
           program_test_utils.TestSerializable(3, 4),
       ],
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4),
       ]),
      ('list_empty', [], []),
      ('list_nested',
       [
           [
               True,
               1,
               'a',
               program_test_utils.TestMaterializableValueReference(2),
               program_test_utils.TestSerializable(3, 4),
           ],
           [5],
       ],
       [
           [
               True,
               1,
               'a',
               2,
               program_test_utils.TestSerializable(3, 4),
           ],
           [5],
       ]),
      ('dict',
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': program_test_utils.TestMaterializableValueReference(2),
           'e': program_test_utils.TestSerializable(3, 4),
       },
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': 2,
           'e': program_test_utils.TestSerializable(3, 4),
       }),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': program_test_utils.TestMaterializableValueReference(2),
               'e': program_test_utils.TestSerializable(3, 4),
           },
           'y': {'a': 5},
       },
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': 2,
               'e': program_test_utils.TestSerializable(3, 4),
           },
           'y': {'a': 5},
       }),
      ('named_tuple',
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2),
           e=program_test_utils.TestSerializable(3, 4),
       ),
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=2,
           e=program_test_utils.TestSerializable(3, 4),
       )),
      ('named_tuple_nested',
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2),
               e=program_test_utils.TestSerializable(3, 4),
           ),
           y=program_test_utils.TestNamedTuple2(a=5),
       ),
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=2,
               e=program_test_utils.TestSerializable(3, 4),
           ),
           y=program_test_utils.TestNamedTuple2(a=5),
       )),
  )
  # pyformat: enable
  async def test_release_logs_value(self, value, expected_value):
    release_mngr = logging_release_manager.LoggingReleaseManager()
    key = 1

    with mock.patch('absl.logging.info') as mock_info:
      await release_mngr.release(value, key=key)

      self.assertLen(mock_info.mock_calls, 3)
      mock_info.assert_has_calls([
          mock.call(mock.ANY),
          mock.call(mock.ANY, mock.ANY),
          mock.call(mock.ANY, key),
      ])
      call = mock_info.mock_calls[1]
      _, args, kwargs = call
      _, actual_value = args
      tree.assert_same_structure(actual_value, expected_value)
      actual_value = program_test_utils.to_python(actual_value)
      expected_value = program_test_utils.to_python(expected_value)
      self.assertEqual(actual_value, expected_value)
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

    with mock.patch('absl.logging.info') as mock_info:
      await release_mngr.release(value, key=key)

      self.assertLen(mock_info.mock_calls, 3)
      mock_info.assert_has_calls([
          mock.call(mock.ANY),
          mock.call(mock.ANY, value),
          mock.call(mock.ANY, key),
      ])


if __name__ == '__main__':
  absltest.main()
