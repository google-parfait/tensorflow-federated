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

import os.path
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import tensorboard_release_manager
from tensorflow_federated.python.program import test_utils


class TensorboardReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_root_dir(self):
    temp_dir = self.create_tempdir()
    summary_dir = os.path.join(temp_dir, 'test')
    self.assertFalse(os.path.exists(summary_dir))

    tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=summary_dir)

    self.assertTrue(os.path.exists(summary_dir))

  def test_does_not_raise_type_error_with_root_dir_str(self):
    try:
      tensorboard_release_manager.TensorboardReleaseManager(summary_dir='/tmp')
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_does_not_raise_type_error_with_summary_dir_path_like(self):
    temp_dir = self.create_tempdir()

    try:
      tensorboard_release_manager.TensorboardReleaseManager(
          summary_dir=temp_dir)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_summary_dir(self, summary_dir):
    with self.assertRaises(TypeError):
      tensorboard_release_manager.TensorboardReleaseManager(
          summary_dir=summary_dir)

  def test_raises_value_error_with_summary_dir_empty(self):
    with self.assertRaises(ValueError):
      tensorboard_release_manager.TensorboardReleaseManager(summary_dir='')


class TensorboardReleaseManagerReleaseTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('bool', True, [('', True)]),
      ('int', 1, [('', 1)]),
      ('list', [True, 1, 'a'], [('0', True), ('1', 1)]),
      ('list_nested', [[True, 1], ['a']], [('0/0', True), ('0/1', 1)]),
      ('dict',
       {'a': True, 'b': 1, 'c': 'a'},
       [('a', True), ('b', 1)]),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       [('x/a', True), ('x/b', 1)]),
      ('attr',
       test_utils.TestAttrObject1(True, 1),
       [('a', True), ('b', 1)]),
      ('attr_nested',
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')},
       [('a/0/a', True), ('a/0/b', 1)]),
      ('tensor_int', tf.constant(1), [('', tf.constant(1))]),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       [('a/0', True), ('a/1', 1)]),
      ('numpy_int', np.int32(1), [('', np.int32(1))]),
      ('numpy_nested',
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]},
       [('a/0', True), ('a/1', 1)]),
      ('server_array_reference',
       test_utils.TestServerArrayReference(1),
       [('', 1)]),
      ('server_array_reference_nested',
       {'a': [test_utils.TestServerArrayReference(True),
              test_utils.TestServerArrayReference(1)],
        'b': [test_utils.TestServerArrayReference('a')]},
       [('a/0', True), ('a/1', 1)]),
      ('materialized_values_and_value_references',
       [1, test_utils.TestServerArrayReference(2)],
       [('0', 1), ('1', 2)]),
  )
  # pyformat: enable
  def test_writes_value_scalar(self, value, expected_names_and_values):
    temp_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'scalar') as mock_scalar:
      release_mngr.release(value, 1)

      calls = []
      for name, value in expected_names_and_values:
        call = mock.call(name, value, step=1)
        calls.append(call)
      mock_scalar.assert_has_calls(calls)

  @parameterized.named_parameters(
      ('tensor_2d', tf.ones((2, 3)), [('', tf.ones((2, 3)))]),
      ('numpy_2d', np.ones((2, 3)), [('', np.ones((2, 3)))]),
  )
  def test_writes_value_histogram(self, value, expected_names_and_values):
    temp_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'histogram') as mock_histogram:
      release_mngr.release(value, 1)

      calls = []
      for name, value in expected_names_and_values:
        mock.call(name, value, step=1)
      mock_histogram.assert_has_calls(calls)

  def test_writes_value_scalar_and_histogram(self):
    temp_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    patch_scalar = mock.patch.object(tf.summary, 'scalar')
    patch_histogram = mock.patch.object(tf.summary, 'histogram')
    with patch_scalar as mock_scalar, patch_histogram as mock_histogram:
      release_mngr.release([1, tf.ones([1])], 1)
      mock_scalar.assert_called_once_with('0', 1, step=1)
      mock_histogram.assert_called_once_with('1', tf.ones([1]), step=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list_empty', []),
      ('dict_empty', {}),
      ('tensor_str', tf.constant('a')),
  )
  def test_does_not_write_value(self, value):
    temp_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    patch_scalar = mock.patch.object(tf.summary, 'scalar')
    patch_histogram = mock.patch.object(tf.summary, 'histogram')
    with patch_scalar as mock_scalar, patch_histogram as mock_histogram:
      release_mngr.release(value, 1)
      mock_scalar.assert_not_called()
      mock_histogram.assert_not_called()

  @parameterized.named_parameters(
      ('negative_1', -1),
      ('0', 0),
      ('1', 1),
  )
  def test_does_not_raise_with_key(self, key):
    temp_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    try:
      release_mngr.release(1, key)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_key(self, key):
    temp_dir = self.create_tempdir()
    release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with self.assertRaises(TypeError):
      release_mngr.release(1, key)


if __name__ == '__main__':
  absltest.main()
