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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import memory_release_manager
from tensorflow_federated.python.program import program_test_utils


class MemoryReleaseManagerTest(parameterized.TestCase,
                               unittest.IsolatedAsyncioTestCase,
                               tf.test.TestCase):

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
       tf.ones([3], tf.int32),
       computation_types.TensorType(tf.int32, [3]),
       tf.ones([3], tf.int32)),
      ('numpy_int',
       np.int32(1),
       computation_types.TensorType(tf.int32),
       np.int32(1)),
      ('numpy_array',
       np.ones([3], np.int32),
       computation_types.TensorType(tf.int32, [3]),
       np.ones([3], np.int32)),

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
       computation_types.StructWithPythonType(
           [tf.bool, tf.int32, tf.string], list),
       [True, 1, 'a']),
      ('list_empty', [], computation_types.StructWithPythonType([], list), []),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       computation_types.StructWithPythonType([
           computation_types.StructWithPythonType([tf.bool, tf.int32], list),
           computation_types.StructWithPythonType([tf.string], list)
       ], list),
       [[True, 1], ['a']]),
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
       ], collections.OrderedDict),
       {'a': True, 'b': 1, 'c': 'a'}),
      ('dict_empty',
       {},
       computation_types.StructWithPythonType([], collections.OrderedDict),
       {}),
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
       ], collections.OrderedDict),
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
      ('attr',
       program_test_utils.TestAttrObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
       ], program_test_utils.TestAttrObj2),
       program_test_utils.TestAttrObj2(True, 1)),
      ('attr_nested',
       program_test_utils.TestAttrObj2(
           program_test_utils.TestAttrObj2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObj1('a')),
       computation_types.StructWithPythonType([
           ('a', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
           ], program_test_utils.TestAttrObj2)),
           ('b', computation_types.StructWithPythonType([
               ('c', tf.string),
           ], program_test_utils.TestAttrObj1)),
       ], program_test_utils.TestAttrObj2),
       program_test_utils.TestAttrObj2(
           program_test_utils.TestAttrObj2(True, 1),
           program_test_utils.TestAttrObj1('a'))),
      ('namedtuple',
       program_test_utils.TestNamedtupleObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
       ], program_test_utils.TestNamedtupleObj2),
       program_test_utils.TestNamedtupleObj2(True, 1)),
      ('namedtuple_nested',
       program_test_utils.TestNamedtupleObj2(
           program_test_utils.TestNamedtupleObj2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestNamedtupleObj1('a')),
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
           ], program_test_utils.TestNamedtupleObj2)),
           ('y', computation_types.StructWithPythonType([
               ('c', tf.string),
           ], program_test_utils.TestNamedtupleObj1)),
       ], program_test_utils.TestNamedtupleObj2),
       program_test_utils.TestNamedtupleObj2(
           program_test_utils.TestNamedtupleObj2(True, 1),
           program_test_utils.TestNamedtupleObj1('a'))),
  )
  # pyformat: enable
  async def test_release_saves_value_and_type_signature(self, value,
                                                        type_signature,
                                                        expected_value):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    await release_mngr.release(value, type_signature, 1)

    self.assertLen(release_mngr._values, 1)
    actual_value, actual_type_signature = release_mngr._values[1]
    program_test_utils.assert_types_equal(actual_value, expected_value)
    if (isinstance(actual_value, tf.data.Dataset) and
        isinstance(expected_value, tf.data.Dataset)):
      actual_value = list(actual_value)
      expected_value = list(expected_value)
    self.assertAllEqual(actual_value, expected_value)
    self.assertEqual(actual_type_signature, type_signature)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  async def test_release_raises_type_error_with_type_signature(
      self, type_signature):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    with self.assertRaises(TypeError):
      await release_mngr.release(1, type_signature, 1)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
  )
  async def test_release_does_not_raise_type_error_with_key(self, key):
    release_mngr = memory_release_manager.MemoryReleaseManager()
    value = 1
    type_signature = computation_types.TensorType(tf.int32)

    try:
      await release_mngr.release(value, type_signature, key)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('list', []),)
  async def test_release_raises_type_error_with_key(self, key):
    release_mngr = memory_release_manager.MemoryReleaseManager()
    value = 1
    type_signature = computation_types.TensorType(tf.int32)

    with self.assertRaises(TypeError):
      await release_mngr.release(value, type_signature, key)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('10', 10),
  )
  def test_values_returns_values_and_type_signatures(self, count):
    expected_values = collections.OrderedDict([
        (i, (i, computation_types.TensorType(tf.int32))) for i in range(count)
    ])
    release_mngr = memory_release_manager.MemoryReleaseManager()
    release_mngr._values = expected_values

    actual_values = release_mngr.values()

    self.assertEqual(actual_values, expected_values)

  def test_values_returns_copy(self):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    values_1 = release_mngr.values()
    values_2 = release_mngr.values()
    self.assertIsNot(values_1, values_2)


if __name__ == '__main__':
  absltest.main()
