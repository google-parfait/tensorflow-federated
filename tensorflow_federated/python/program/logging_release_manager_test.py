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


class LoggingReleaseManagerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('none_none', None, None),
      ('none_int', None, 1),
      ('none_list', None, [1, 2, 3]),
      ('int_none', 1, None),
      ('int_int', 1, 1),
      ('int_list', 1, [1, 2, 3]),
      ('list_none', [1, 2, 3], None),
      ('list_int', [1, 2, 3], 1),
      ('list_list', [1, 2, 3], [1, 2, 3]),
  )
  def test_release_logs_value_and_key(self, value, key):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_info:
      release_mngr.release(value, key)
      mock_info.assert_called_once()

if __name__ == '__main__':
  absltest.main()
