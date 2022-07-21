# Copyright 2022, The TensorFlow Federated Authors.
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
from typing import Tuple, Union
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import release_manager


class FilteringReleaseManager(parameterized.TestCase,
                              unittest.IsolatedAsyncioTestCase):

  def test_init_does_not_raise_type_error(self):
    filter_fn = lambda _: True
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)

    try:
      release_manager.FilteringReleaseManager(mock_release_mngr, filter_fn)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_init_raises_type_error_with_release_manager(self, release_mngr):
    filter_fn = lambda _: True

    with self.assertRaises(TypeError):
      release_manager.FilteringReleaseManager(release_mngr, filter_fn)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_init_raises_type_error_with_filter_fn(self, filter_fn):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)

    with self.assertRaises(TypeError):
      release_manager.FilteringReleaseManager(mock_release_mngr, filter_fn)

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, computation_types.StructWithPythonType([], list)),
      ('bool', True, computation_types.TensorType(tf.bool)),
      ('int', 1, computation_types.TensorType(tf.int32)),
      ('str', 'a', computation_types.TensorType(tf.string)),
      ('tensor_int', tf.constant(1), computation_types.TensorType(tf.int32)),
      ('tensor_str', tf.constant('a'), computation_types.TensorType(tf.string)),
      ('tensor_array',
       tf.ones([3], tf.int32),
       computation_types.TensorType(tf.int32, [3])),
      ('numpy_int', np.int32(1), computation_types.TensorType(tf.int32)),
      ('numpy_array',
       np.ones([3], np.int32),
       computation_types.TensorType(tf.int32, [3])),

      # value references
      ('value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32)),
      ('value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(tf.int32)),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       computation_types.StructWithPythonType(
           [tf.bool, tf.int32, tf.string], list)),
      ('list_empty', [], computation_types.StructWithPythonType([], list)),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       computation_types.StructWithPythonType([
           computation_types.StructWithPythonType([tf.bool, tf.int32], list),
           computation_types.StructWithPythonType([tf.string], list)
       ], list)),
      ('dict',
       {
           'a': True,
           'b': program_test_utils.TestMaterializableValueReference(1),
           'c': 'a',
       },
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
       ], collections.OrderedDict)),
      ('dict_empty', {}, computation_types.StructWithPythonType([], list)),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': program_test_utils.TestMaterializableValueReference(1),
           },
           'y': {
               'c': 'a',
           },
       },
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', tf.string),
           ], collections.OrderedDict)),
       ], collections.OrderedDict)),
  )
  # pyformat: enable
  async def test_release_delegates_value_and_type_signature(
      self, value, type_signature):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    filter_fn = lambda _: True
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn)

    await release_mngr.release(value, type_signature, 1)

    mock_release_mngr.release.assert_called_once_with(value, type_signature, 1)

  async def test_release_filters_value_and_type_signature(self):

    def _filter_fn(path: Tuple[Union[str, int], ...]) -> bool:
      return path == (0,) or path == (3, 0) or path == (4, 'a')

    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, _filter_fn)
    value = [True, 1, 'a', [True, 1, 'a'], {'a': True, 'b': 1, 'c': 'a'}]
    type_signature = computation_types.StructWithPythonType([
        tf.bool,
        tf.int32,
        tf.string,
        computation_types.StructWithPythonType([tf.bool, tf.int32, tf.string],
                                               list),
        computation_types.StructWithPythonType([
            ('a', tf.bool),
            ('b', tf.int32),
            ('c', tf.string),
        ], collections.OrderedDict),
    ], list)

    await release_mngr.release(value, type_signature, 1)

    expected_value = [True, [True], {'a': True}]
    expected_type_signature = computation_types.StructWithPythonType([
        tf.bool,
        computation_types.StructWithPythonType([tf.bool], list),
        computation_types.StructWithPythonType([
            ('a', tf.bool),
        ], collections.OrderedDict),
    ], list)
    mock_release_mngr.release.assert_called_once_with(expected_value,
                                                      expected_type_signature,
                                                      1)

  async def test_release_converts_named_struct_types_to_ordered_dict(self):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    filter_fn = lambda _: True
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn)
    value = {
        'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a',
    }
    type_signature = computation_types.StructWithPythonType([
        ('a', tf.bool),
        ('b', tf.int32),
        ('c', tf.string),
    ], list)

    await release_mngr.release(value, type_signature, 1)

    expected_type_signature = computation_types.StructWithPythonType([
        ('a', tf.bool),
        ('b', tf.int32),
        ('c', tf.string),
    ], collections.OrderedDict)
    mock_release_mngr.release.assert_called_once_with(value,
                                                      expected_type_signature,
                                                      1)

  # pyformat: disable
  @parameterized.named_parameters(
      ('attr',
       program_test_utils.TestAttrObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.SequenceType([('a', tf.bool), ('b', tf.int32)])),
      ('namedtuple',
       program_test_utils.TestNamedtupleObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.StructType([
           ('a', tf.bool),
           ('b', tf.int32),
       ])),
  )
  # pyformat: enable
  async def test_release_raises_not_implemented_error_with_value_and_type_signature(
      self, value, type_signature):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    filter_fn = lambda _: True
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn)

    with self.assertRaises(NotImplementedError):
      await release_mngr.release(value, type_signature, 1)


class GroupingReleaseManager(parameterized.TestCase,
                             unittest.IsolatedAsyncioTestCase):

  def test_init_does_not_raise_type_error_with_release_managers(self):
    release_mngrs = [
        mock.AsyncMock(spec=release_manager.ReleaseManager),
        mock.AsyncMock(spec=release_manager.ReleaseManager),
        mock.AsyncMock(spec=release_manager.ReleaseManager),
    ]

    try:
      release_manager.GroupingReleaseManager(release_mngrs)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_release_managers(self, release_mngrs):
    with self.assertRaises(TypeError):
      release_manager.GroupingReleaseManager(release_mngrs)

  def test_init_raises_value_error_with_release_manager_empty(self):
    release_mngrs = []

    with self.assertRaises(ValueError):
      release_manager.GroupingReleaseManager(release_mngrs)

  async def test_release_delegates_value_and_type_signature(self):
    release_mngrs = [
        mock.AsyncMock(spec=release_manager.ReleaseManager),
        mock.AsyncMock(spec=release_manager.ReleaseManager),
        mock.AsyncMock(spec=release_manager.ReleaseManager),
    ]
    release_mngr = release_manager.GroupingReleaseManager(release_mngrs)
    value = 1
    type_signature = computation_types.TensorType(tf.int32)
    key = 1

    await release_mngr.release(value, type_signature, key)

    for mock_release_mngr in release_mngrs:
      mock_release_mngr.release.assert_called_once_with(value, type_signature,
                                                        key)


if __name__ == '__main__':
  absltest.main()
