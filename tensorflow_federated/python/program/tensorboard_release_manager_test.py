# Copyright 2020, Google LLC.
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
import tensorflow as tf

from tensorflow_federated.python.program import tensorboard_release_manager


class TensorBoardManagerInitTest(parameterized.TestCase):

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


class TensorBoardManagerReleaseTest(parameterized.TestCase):

  def test_release_writes_scalar_int(self):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'scalar') as mock_scalar:
      tensorboard_release_mngr.release(1, 1)
      mock_scalar.assert_called_once_with('', 1, step=1)

  def test_release_writes_scalar_list(self):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'scalar') as mock_scalar:
      tensorboard_release_mngr.release([1, 2, 3], 1)
      mock_scalar.assert_has_calls([
          mock.call('0', 1, step=1),
          mock.call('1', 2, step=1),
          mock.call('2', 3, step=1),
      ])

  def test_release_writes_scalar_dict(self):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'scalar') as mock_scalar:
      tensorboard_release_mngr.release({'a': 1, 'b': 2, 'c': 3}, 1)
      mock_scalar.assert_has_calls([
          mock.call('a', 1, step=1),
          mock.call('b', 2, step=1),
          mock.call('c', 3, step=1),
      ])

  def test_release_writes_scalar_nested(self):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'scalar') as mock_scalar:
      tensorboard_release_mngr.release([1, [2, 2], {'a': 3}], 1)
      mock_scalar.assert_has_calls([
          mock.call('0', 1, step=1),
          mock.call('1/0', 2, step=1),
          mock.call('1/1', 2, step=1),
          mock.call('2/a', 3, step=1),
      ])

  def test_release_writes_histogram_tensor(self):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'histogram') as mock_histogram:
      tensorboard_release_mngr.release(tf.ones([1]), 1)
      mock_histogram.assert_has_calls([
          mock.call('', tf.ones([1]), step=1),
      ])

  def test_release_writes_histogram_nested(self):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'histogram') as mock_histogram:
      tensorboard_release_mngr.release([tf.ones([1]), [tf.ones([1])]], 1)
      mock_histogram.assert_has_calls([
          mock.call('0', tf.ones([1]), step=1),
          mock.call('1/0', tf.ones([1]), step=1),
      ])

  def test_release_writes_scalar_int_and_histogram_tensor(self):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    patch_scalar = mock.patch.object(tf.summary, 'scalar')
    patch_histogram = mock.patch.object(tf.summary, 'histogram')
    with patch_scalar as mock_scalar, patch_histogram as mock_histogram:
      tensorboard_release_mngr.release([1, tf.ones([1])], 1)
      mock_scalar.assert_called_once_with('0', 1, step=1)
      mock_histogram.assert_called_once_with('1', tf.ones([1]), step=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_release_does_not_write_value(self, value):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with mock.patch.object(tf.summary, 'scalar') as mock_scalar:
      tensorboard_release_mngr.release(value, 1)
      mock_scalar.assert_not_called()

  @parameterized.named_parameters(
      ('negative_1', -1),
      ('0', 0),
      ('1', 1),
  )
  def test_release_does_not_raise_with_key(self, key):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    try:
      tensorboard_release_mngr.release(1, key)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_release_raises_type_error_with_key(self, key):
    temp_dir = self.create_tempdir()
    tensorboard_release_mngr = tensorboard_release_manager.TensorboardReleaseManager(
        summary_dir=temp_dir)

    with self.assertRaises(TypeError):
      tensorboard_release_mngr.release(1, key)


if __name__ == '__main__':
  absltest.main()
