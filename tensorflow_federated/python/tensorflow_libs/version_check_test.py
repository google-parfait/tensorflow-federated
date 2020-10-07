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

import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import version_check


class _FakeTFModule():
  """A class that fakes the tensorflow.version.VERSION attribute."""

  def __init__(self, version: str):
    self._version = version

  @property
  def version(self) -> '_FakeTFModule':
    # This is a bit of trickery to get `self.version.VERSION` to give what we
    # want.
    return self

  @property
  def VERSION(self) -> str:  # pylint: disable=invalid-name
    return self._version


class VersionCheckTest(tf.test.TestCase):

  def test_is_tf_release(self):
    mock_tf_module = _FakeTFModule('2.2.2')
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.2.1', mock_tf_module))
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.1.0', mock_tf_module))
    self.assertFalse(
        version_check.is_tensorflow_version_newer('2.2.4', mock_tf_module))
    self.assertFalse(
        version_check.is_tensorflow_version_newer('2.3.0', mock_tf_module))

  def test_is_tf_release_candidate(self):
    # Release candidates behave the same as regular releases.
    mock_tf_module = _FakeTFModule('2.2.2-rc2')
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.2.1', mock_tf_module))
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.1.0', mock_tf_module))
    self.assertFalse(
        version_check.is_tensorflow_version_newer('2.2.4', mock_tf_module))
    self.assertFalse(
        version_check.is_tensorflow_version_newer('2.3.0', mock_tf_module))

  def test_is_tf_nightly(self):
    # TF-nightly modules are always true.
    mock_tf_module = _FakeTFModule('2.2.2-dev202004016')
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.2.4', mock_tf_module))
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.3.0', mock_tf_module))
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.2.0', mock_tf_module))
    self.assertTrue(
        version_check.is_tensorflow_version_newer('2.1.0', mock_tf_module))


if __name__ == '__main__':
  tf.test.main()
