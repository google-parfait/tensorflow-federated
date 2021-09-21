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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.program import logging_release_manager


class RoundLoggingReleaseManagerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('none_none', None, 1),
      ('none_1', None, 1),
      ('int_none', 1, None),
      ('int_1', 1, 1),
      ('list_none', [1, 2, 3], None),
      ('list_1', [1, 2, 3], 1),
  )
  def test_release_logs_with_value_and_key(self, value, key):
    logging_release_mngr = logging_release_manager.RoundLoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_logging_info:
      logging_release_mngr.release(value, key)
      mock_logging_info.assert_called_once()

  @parameterized.named_parameters(
      ('bool', 1, True),
      ('str', 1, 'a'),
  )
  def test_raises_type_error_with_key(self, value, key):
    logging_release_mngr = logging_release_manager.RoundLoggingReleaseManager()

    with self.assertRaises(TypeError):
      logging_release_mngr.release(value, key)


if __name__ == '__main__':
  absltest.main()
