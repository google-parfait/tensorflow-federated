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
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import logging_release_manager
from tensorflow_federated.python.program import program_test_utils


class LoggingReleaseManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase, tf.test.TestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, computation_types.StructWithPythonType([], list), None),
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
       tf.constant([1] * 3),
       computation_types.TensorType(tf.int32, [3]),
       tf.constant([1] * 3)),
      ('numpy_int',
       np.int32(1),
       computation_types.TensorType(tf.int32),
       np.int32(1)),
      ('numpy_array',
       np.array([1] * 3, np.int32),
       computation_types.TensorType(tf.int32, [3]),
       np.array([1] * 3, np.int32)),

      # materializable value references
      ('value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32),
       1),
      ('value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(tf.int32),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict),
       program_test_utils.TestSerializable(1, 2)),

      # other values
      ('attrs',
       program_test_utils.TestAttrs(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict),
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
       computation_types.StructWithPythonType([
           tf.bool,
           tf.int32,
           tf.string,
           tf.int32,
           computation_types.StructWithPythonType([
               ('a', tf.int32),
               ('b', tf.int32),
           ], collections.OrderedDict),
       ], list),
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4),
       ]),
      ('list_empty', [], computation_types.StructWithPythonType([], list), []),
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
       computation_types.StructWithPythonType([
           computation_types.StructWithPythonType([
               tf.bool,
               tf.int32,
               tf.string,
               tf.int32,
               computation_types.StructWithPythonType([
                   ('a', tf.int32),
                   ('b', tf.int32),
               ], collections.OrderedDict),
           ], list),
           computation_types.StructWithPythonType([tf.int32], list),
       ], list),
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
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
           ('d', tf.int32),
           ('e', computation_types.StructWithPythonType([
               ('a', tf.int32),
               ('b', tf.int32),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': 2,
           'e': program_test_utils.TestSerializable(3, 4),
       }),
      ('dict_empty',
       {},
       computation_types.StructWithPythonType([], collections.OrderedDict),
       {}),
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
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
               ('c', tf.string),
               ('d', tf.int32),
               ('e', computation_types.StructWithPythonType([
                   ('a', tf.int32),
                   ('b', tf.int32),
               ], collections.OrderedDict)),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('a', tf.int32),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
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
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
           ('d', tf.int32),
           ('e', computation_types.StructWithPythonType([
               ('a', tf.int32),
               ('b', tf.int32),
           ], collections.OrderedDict)),
       ], program_test_utils.TestNamedTuple1),
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
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
               ('c', tf.string),
               ('d', tf.int32),
               ('e', computation_types.StructWithPythonType([
                   ('a', tf.int32),
                   ('b', tf.int32),
               ], collections.OrderedDict)),
           ], program_test_utils.TestNamedTuple1)),
           ('y', computation_types.StructWithPythonType([
               ('c', tf.int32),
           ], program_test_utils.TestNamedTuple2)),
       ], program_test_utils.TestNamedTuple3),
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
  async def test_release_logs_value_and_type_signature(
      self, value, type_signature, expected_value
  ):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_info:
      await release_mngr.release(value, type_signature, key=1)

      self.assertLen(mock_info.mock_calls, 4)
      mock_info.assert_has_calls([
          mock.call(mock.ANY),
          mock.call(mock.ANY, mock.ANY),
          mock.call(mock.ANY, type_signature),
          mock.call(mock.ANY, 1),
      ])
      call = mock_info.mock_calls[1]
      _, args, kwargs = call
      _, actual_value = args
      program_test_utils.assert_types_equal(actual_value, expected_value)
      if isinstance(actual_value, tf.data.Dataset) and isinstance(
          expected_value, tf.data.Dataset
      ):
        actual_value = list(actual_value)
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
          mock.call(mock.ANY, key),
      ])

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  async def test_release_raises_type_error_with_type_signature(
      self, type_signature
  ):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with self.assertRaises(TypeError):
      await release_mngr.release(1, type_signature, key=1)


if __name__ == '__main__':
  absltest.main()
