# Copyright 2020, The TensorFlow Federated Authors.
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

import os
import os.path
import shutil
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import tensorboard_release_manager


class TensorBoardReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_new_dir_with_summary_dir_str(self):
    summary_dir = self.create_tempdir()
    summary_dir = summary_dir.full_path
    shutil.rmtree(summary_dir)
    self.assertFalse(os.path.exists(summary_dir))

    tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    self.assertTrue(os.path.exists(summary_dir))

  def test_creates_new_dir_with_summary_dir_path_like(self):
    summary_dir = self.create_tempdir()
    shutil.rmtree(summary_dir)
    self.assertFalse(os.path.exists(summary_dir))

    tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    self.assertTrue(os.path.exists(summary_dir))

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_summary_dir(self, summary_dir):
    with self.assertRaises(TypeError):
      tensorboard_release_manager.TensorBoardReleaseManager(
          summary_dir=summary_dir)

  def test_raises_value_error_with_summary_dir_empty(self):
    with self.assertRaises(ValueError):
      tensorboard_release_manager.TensorBoardReleaseManager(summary_dir='')


class TensorBoardReleaseManagerReleaseTest(parameterized.TestCase,
                                           unittest.IsolatedAsyncioTestCase,
                                           tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('bool', True, [('', True)]),
      ('int', 1, [('', 1)]),
      ('tensor_int', tf.constant(1), [('', tf.constant(1))]),
      ('numpy_int', np.int32(1), [('', np.int32(1))]),

      # value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       [('', 1)]),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       [('0', True), ('1', 1)]),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       [('0/0', True), ('0/1', 1)]),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       [('a', True), ('b', 1)]),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       [('x/a', True), ('x/b', 1)]),
      ('attr',
       program_test_utils.TestAttrObject2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       [('a', True), ('b', 1)]),
      ('attr_nested',
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObject1('a')),
       [('a/a', True), ('a/b', 1)]),
  )
  # pyformat: enable
  async def test_writes_value_scalar(self, value, expected_calls):
    summary_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    with mock.patch.object(tf.summary, 'scalar') as mock_scalar:
      await release_mngr.release(value, 1)

      self.assertEqual(len(mock_scalar.mock_calls), len(expected_calls))
      for call, expected_args in zip(mock_scalar.mock_calls, expected_calls):
        _, actual_args, _ = call
        actual_name, actual_value = actual_args
        expected_name, expected_value = expected_args
        self.assertEqual(actual_name, expected_name)
        self.assertEqual(actual_value, expected_value)

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('tensor_2d', tf.ones((2, 3)), [('', tf.ones((2, 3)))]),
      ('numpy_2d', np.ones((2, 3)), [('', np.ones((2, 3)))]),

      # value references
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       [('', [1, 2, 3])]),
  )
  # pyformat: enable
  async def test_writes_value_histogram(self, value, expected_calls):
    summary_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    with mock.patch.object(tf.summary, 'histogram') as mock_histogram:
      await release_mngr.release(value, 1)

      self.assertEqual(len(mock_histogram.mock_calls), len(expected_calls))
      for call, expected_args in zip(mock_histogram.mock_calls, expected_calls):
        _, actual_args, _ = call
        actual_name, actual_value = actual_args
        expected_name, expected_value = expected_args
        self.assertEqual(actual_name, expected_name)
        self.assertAllEqual(actual_value, expected_value)

  async def test_writes_value_scalar_and_histogram(self):
    summary_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    patched_scalar = mock.patch.object(tf.summary, 'scalar')
    patched_histogram = mock.patch.object(tf.summary, 'histogram')
    with patched_scalar as mock_scalar, patched_histogram as mock_histogram:
      await release_mngr.release([1, tf.ones([1])], 1)

      mock_scalar.assert_called_once_with('0', 1, step=1)
      mock_histogram.assert_called_once_with('1', tf.ones([1]), step=1)

  @parameterized.named_parameters(
      # materialized values
      ('none', None),
      ('str', 'a'),
      ('tensor_str', tf.constant('a')),

      # structures
      ('list_empty', []),
      ('dict_empty', {}),
  )
  async def test_does_not_write_value(self, value):
    summary_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    patch_scalar = mock.patch.object(tf.summary, 'scalar')
    patch_histogram = mock.patch.object(tf.summary, 'histogram')
    with patch_scalar as mock_scalar, patch_histogram as mock_histogram:
      await release_mngr.release(value, 1)

      mock_scalar.assert_not_called()
      mock_histogram.assert_not_called()

  @parameterized.named_parameters(
      ('negative_1', -1),
      ('0', 0),
      ('1', 1),
  )
  async def test_does_not_raise_with_key(self, key):
    summary_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    try:
      await release_mngr.release(1, key)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_key(self, key):
    summary_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorBoardReleaseManager(
        summary_dir=summary_dir)

    with self.assertRaises(TypeError):
      await release_mngr.release(1, key)


if __name__ == '__main__':
  absltest.main()
