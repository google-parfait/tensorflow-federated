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

import time
from unittest import mock

import portpicker

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.simulation import server_utils


class ServerUtilsTest(test.TestCase):

  @mock.patch('absl.logging.info')
  def test_server_context_shuts_down_under_keyboard_interrupt(
      self, mock_logging_info):

    ex = eager_tf_executor.EagerTFExecutor()

    with server_utils.server_context(ex, 1,
                                     portpicker.pick_unused_port()) as server:
      time.sleep(1)
      raise KeyboardInterrupt

    mock_logging_info.assert_has_calls([
        mock.call('Server stopped by KeyboardInterrupt.'),
        mock.call('Shutting down server.')
    ])

  @mock.patch('absl.logging.info')
  def test_server_context_shuts_down_uncaught_exception(self,
                                                        mock_logging_info):

    ex = eager_tf_executor.EagerTFExecutor()

    with self.assertRaises(TypeError):
      with server_utils.server_context(ex, 1,
                                       portpicker.pick_unused_port()) as server:
        time.sleep(1)
        raise TypeError

    mock_logging_info.assert_called_once_with('Shutting down server.')


if __name__ == '__main__':
  test.main()
