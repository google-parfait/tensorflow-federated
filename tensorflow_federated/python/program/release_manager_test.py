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
import datetime
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import release_manager


class FilteringReleaseManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def test_init_does_not_raise_type_error(self):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    filter_fn = lambda _: True

    try:
      release_manager.FilteringReleaseManager(mock_release_mngr, filter_fn)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

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

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none_filter_none',
       None,
       computation_types.StructType([]),
       lambda _: True,
       None,
       computation_types.StructType([])),
      ('bool_filter_none',
       True,
       computation_types.TensorType(tf.bool),
       lambda _: True,
       True,
       computation_types.TensorType(tf.bool)),
      ('int_filter_none',
       1,
       computation_types.TensorType(tf.int32),
       lambda _: True,
       1,
       computation_types.TensorType(tf.int32)),
      ('str_filter_none',
       'a',
       computation_types.TensorType(tf.string),
       lambda _: True,
       'a',
       computation_types.TensorType(tf.string)),
      ('tensor_int_filter_none',
       tf.constant(1),
       computation_types.TensorType(tf.int32),
       lambda _: True,
       tf.constant(1),
       computation_types.TensorType(tf.int32)),
      ('tensor_array_filter_none',
       tf.constant([1] * 3),
       computation_types.TensorType(tf.int32, [3]),
       lambda _: True,
       tf.constant([1] * 3),
       computation_types.TensorType(tf.int32, [3])),
      ('numpy_int_filter_none',
       np.int32(1),
       computation_types.TensorType(tf.int32),
       lambda _: True,
       np.int32(1),
       computation_types.TensorType(tf.int32)),
      ('numpy_array_filter_none',
       np.array([1] * 1, np.int32),
       computation_types.TensorType(tf.int32, [3]),
       lambda _: True,
       np.array([1] * 1, np.int32),
       computation_types.TensorType(tf.int32, [3])),

      # materializable value references
      ('value_reference_tensor_filter_none',
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32),
       lambda _: True,
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32)),

      # serializable values
      ('serializable_value_filter_none',
       program_test_utils.TestSerializable(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict),
       lambda _: True,
       program_test_utils.TestSerializable(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict)),

      # other values
      ('attrs_filter_none',
       program_test_utils.TestAttrs(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict),
       lambda _: True,
       program_test_utils.TestAttrs(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict)),

      # structures
      ('list_filter_none',
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
       lambda _: True,
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
       ], list)),
      ('list_filter_some',
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
       lambda path: path == (1,) or path == (2,),
       [1, 'a'],
       computation_types.StructWithPythonType([tf.int32, tf.string], list)),
      ('dict_filter_none',
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
       lambda _: True,
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
       ], collections.OrderedDict)),
      ('dict_filter_some',
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
       lambda path: path == ('b',) or path == ('c',),
       {'b': 1, 'c': 'a'},
       computation_types.StructWithPythonType([
           ('b', tf.int32),
           ('c', tf.string),
       ], collections.OrderedDict)),
      ('named_tuple_filter_none',
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
       lambda _: True,
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
       ], program_test_utils.TestNamedTuple1)),
  )
  # pyformat: enable
  async def test_release_filters_and_delegates_value_and_type_signature(
      self,
      value,
      type_signature,
      filter_fn,
      expected_value,
      expected_type_signature,
  ):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn
    )
    key = 1

    await release_mngr.release(value, type_signature, key)

    mock_release_mngr.release.assert_called_once()
    call = mock_release_mngr.release.mock_calls[0]
    _, args, kwargs = call
    actual_value, actual_type_signature, actual_key = args
    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)
    self.assertEqual(actual_type_signature, expected_type_signature)
    self.assertEqual(actual_key, key)
    self.assertEqual(kwargs, {})

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, computation_types.StructType([])),
      ('bool', True, computation_types.TensorType(tf.bool)),
      ('int', 1, computation_types.TensorType(tf.int32)),
      ('str', 'a', computation_types.TensorType(tf.string)),
      ('tensor_int', tf.constant(1), computation_types.TensorType(tf.int32)),
      ('tensor_str', tf.constant('a'), computation_types.TensorType(tf.string)),
      ('tensor_array',
       tf.constant([1] * 3),
       computation_types.TensorType(tf.int32, [3])),
      ('numpy_int', np.int32(1), computation_types.TensorType(tf.int32)),
      ('numpy_array',
       np.array([1] * 3, np.int32),
       computation_types.TensorType(tf.int32, [3])),

      # materializable value references
      ('value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32)),
      ('value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(tf.int32)),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict)),

      # other values
      ('attrs',
       program_test_utils.TestAttrs(1, 2),
       computation_types.StructWithPythonType([
           ('a', tf.int32),
           ('b', tf.int32),
       ], collections.OrderedDict)),

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
       ], list)),
      ('list_empty', [], computation_types.StructWithPythonType([], list)),
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
       ], list)),
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
       ], collections.OrderedDict)),
      ('dict_empty', {}, computation_types.StructWithPythonType([], list)),
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
       ], collections.OrderedDict)),
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
       ], program_test_utils.TestNamedTuple1)),
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
       ], program_test_utils.TestNamedTuple3)),
  )
  # pyformat: enable
  async def test_release_filters_and_does_not_delegate_value_and_type_signature(
      self,
      value,
      type_signature,
  ):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    filter_fn = lambda _: False
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn
    )
    key = 1

    await release_mngr.release(value, type_signature, key)

    mock_release_mngr.release.assert_not_called()

  # pyformat: disable
  @parameterized.named_parameters(
      ('list_filter_none',
       [True, 1, 'a', [], [2]],
       computation_types.StructWithPythonType([
           tf.bool,
           tf.int32,
           tf.string,
           computation_types.StructWithPythonType([], list),
           computation_types.StructWithPythonType([tf.int32], list),
       ], list),
       lambda _: True,
       [True, 1, 'a', [2]],
       computation_types.StructWithPythonType([
           tf.bool,
           tf.int32,
           tf.string,
           computation_types.StructWithPythonType([tf.int32], list),
       ], list)),
      ('list_filter_some',
       [True, 1, 'a', [], [2]],
       computation_types.StructWithPythonType([
           tf.bool,
           tf.int32,
           tf.string,
           computation_types.StructWithPythonType([], list),
           computation_types.StructWithPythonType([tf.int32], list),
       ], list),
       lambda path: path != (4, 0),
       [True, 1, 'a'],
       computation_types.StructWithPythonType([
           tf.bool,
           tf.int32,
           tf.string,
       ], list)),
      ('dict_filter_none',
       {'a': True, 'b': 1, 'c': 'a', 'd': {}, 'e': {'a': 2}},
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
           ('d', computation_types.StructWithPythonType([
           ], collections.OrderedDict)),
           ('e', computation_types.StructWithPythonType([
               ('a', tf.int32),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       lambda _: True,
       {'a': True, 'b': 1, 'c': 'a', 'e': {'a': 2}},
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
           ('e', computation_types.StructWithPythonType([
               ('a', tf.int32),
           ], collections.OrderedDict)),
       ], collections.OrderedDict)),
      ('dict_filter_some',
       {'a': True, 'b': 1, 'c': 'a', 'd': {}, 'e': {'a': 2}},
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
           ('d', computation_types.StructWithPythonType([
           ], collections.OrderedDict)),
           ('e', computation_types.StructWithPythonType([
               ('a', tf.int32),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       lambda path: path != ('e', 'a'),
       {'a': True, 'b': 1, 'c': 'a'},
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
       ], collections.OrderedDict)),
  )
  # pyformat: enable
  async def test_release_filters_and_does_not_delegate_empty_structures(
      self,
      value,
      type_signature,
      filter_fn,
      expected_value,
      expected_type_signature,
  ):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn
    )
    key = 1

    await release_mngr.release(value, type_signature, key)

    mock_release_mngr.release.assert_called_once()
    call = mock_release_mngr.release.mock_calls[0]
    _, args, kwargs = call
    actual_value, actual_type_signature, actual_key = args
    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)
    self.assertEqual(actual_type_signature, expected_type_signature)
    self.assertEqual(actual_key, key)
    self.assertEqual(kwargs, {})

  # pyformat: disable
  @parameterized.named_parameters(
      # structures
      ('named_tuple_filter_some',
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
       lambda path: path == ('b',) or path == ('c',)),
  )
  # pyformat: enable
  async def test_release_raises_not_filterable_error(
      self, value, type_signature, filter_fn
  ):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn
    )
    key = 1

    with self.assertRaises(release_manager.NotFilterableError):
      await release_mngr.release(value, type_signature, key)

  async def test_release_raises_filter_mismatch_error(self):
    mock_release_mngr = mock.AsyncMock(spec=release_manager.ReleaseManager)
    filter_fn = lambda path: path == ('a',)
    release_mngr = release_manager.FilteringReleaseManager(
        mock_release_mngr, filter_fn
    )
    value = {'a': 1}
    type_signature = computation_types.StructWithPythonType(
        [('b', tf.int32)], collections.OrderedDict
    )
    key = 1

    with self.assertRaises(release_manager.FilterMismatchError):
      await release_mngr.release(value, type_signature, key)


class GroupingReleaseManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def test_init_does_not_raise_type_error(self):
    release_mngrs = [
        mock.AsyncMock(spec=release_manager.ReleaseManager),
        mock.AsyncMock(spec=release_manager.ReleaseManager),
        mock.AsyncMock(spec=release_manager.ReleaseManager),
    ]

    try:
      release_manager.GroupingReleaseManager(release_mngrs)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

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
      mock_release_mngr.release.assert_called_once_with(
          value, type_signature, key
      )


class PeriodicReleaseManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('int_negative', -1),
      ('int_zero', 0),
      ('timedelta_negative', datetime.timedelta(seconds=-1)),
      ('timedelta_zero', datetime.timedelta()),
  )
  def test_init_raises_value_error_with_period(self, periodicity):
    mock_release_mngr = mock.AsyncMock(
        spec=release_manager.ReleaseManager, set_spec=True
    )

    with self.assertRaises(ValueError):
      release_manager.PeriodicReleaseManager(mock_release_mngr, periodicity)

  @parameterized.named_parameters(
      ('all_releases', 1, 10, 10),
      ('some_releases', 2, 10, 5),
      ('last_release', 10, 10, 1),
      ('drops_trailing_releases', 3, 10, 3),
      ('drops_all_releases', 11, 10, 0),
  )
  async def test_release_delegates_value_and_type_signature_with_periodicity_int(
      self, periodicity, total, expected_count
  ):
    mock_release_mngr = mock.AsyncMock(
        spec=release_manager.ReleaseManager, set_spec=True
    )
    release_mngr = release_manager.PeriodicReleaseManager(
        mock_release_mngr, periodicity
    )
    value = 1
    type_signature = computation_types.TensorType(tf.int32)
    key = 1

    for _ in range(total):
      await release_mngr.release(value, type_signature, key)

    self.assertEqual(mock_release_mngr.release.call_count, expected_count)
    mock_release_mngr.release.assert_has_calls(
        [mock.call(value, type_signature, key)] * expected_count
    )

  @parameterized.named_parameters(
      (
          'all_releases',
          datetime.timedelta(seconds=1),
          [datetime.timedelta(seconds=x) for x in range(1, 11)],
          10,
      ),
      (
          'some_releases',
          datetime.timedelta(seconds=2),
          [datetime.timedelta(seconds=x) for x in range(1, 11)],
          5,
      ),
      (
          'last_release',
          datetime.timedelta(seconds=10),
          [datetime.timedelta(seconds=x) for x in range(1, 11)],
          1,
      ),
      (
          'drops_trailing_releases',
          datetime.timedelta(seconds=3),
          [datetime.timedelta(seconds=x) for x in range(1, 11)],
          3,
      ),
      (
          'drops_all_releases',
          datetime.timedelta(seconds=11),
          [datetime.timedelta(seconds=x) for x in range(1, 11)],
          0,
      ),
  )
  async def test_release_delegates_value_and_type_signature_with_periodicity_timedelta(
      self, periodicity, timedeltas, expected_count
  ):
    mock_release_mngr = mock.AsyncMock(
        spec=release_manager.ReleaseManager, set_spec=True
    )
    release_mngr = release_manager.PeriodicReleaseManager(
        mock_release_mngr, periodicity
    )
    value = 1
    type_signature = computation_types.TensorType(tf.int32)
    key = 1

    now = datetime.datetime.now()
    with mock.patch.object(datetime, 'datetime') as mock_datetime:
      mock_datetime.now.side_effect = [now + x for x in timedeltas]

      for _ in timedeltas:
        await release_mngr.release(value, type_signature, key)

    self.assertEqual(mock_release_mngr.release.call_count, expected_count)
    mock_release_mngr.release.assert_has_calls(
        [mock.call(value, type_signature, key)] * expected_count
    )


class DelayedReleaseManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('int_negative', -1),
      ('int_zero', 0),
  )
  def test_init_raises_value_error_with_bad_delay(self, delay):
    mock_release_mngr = mock.AsyncMock(
        spec=release_manager.ReleaseManager, set_spec=True
    )

    with self.assertRaises(ValueError):
      release_manager.DelayedReleaseManager(mock_release_mngr, delay)

  @parameterized.named_parameters(
      ('all_releases', 1, 10, 10),
      ('some_releases', 3, 10, 8),
      ('last_release', 10, 10, 1),
      ('drops_all_releases', 11, 10, 0),
  )
  async def test_release_delegates_value_and_type_signature_with_delay(
      self, delay, total, expected_count
  ):
    mock_release_mngr = mock.AsyncMock(
        spec=release_manager.ReleaseManager, set_spec=True
    )
    release_mngr = release_manager.DelayedReleaseManager(
        mock_release_mngr, delay
    )
    value = 1
    type_signature = computation_types.TensorType(tf.int32)
    key = 1

    for _ in range(total):
      await release_mngr.release(value, type_signature, key)

    self.assertEqual(mock_release_mngr.release.call_count, expected_count)
    mock_release_mngr.release.assert_has_calls(
        [mock.call(value, type_signature, key)] * expected_count
    )


if __name__ == '__main__':
  absltest.main()
