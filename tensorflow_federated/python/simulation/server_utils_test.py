# Lint as: python3
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
"""Tests for server_utils."""
import multiprocessing
import signal
import time

from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core import framework
from tensorflow_federated.python.simulation import server_utils


class ServerUtilsTest(common_test.TestCase):

  def test_server_runs(self):
    ex = framework.EagerExecutor()

    def noarg_run_server():
      server_utils.run_server(ex, 1, 8888)

    process = multiprocessing.Process(target=noarg_run_server)
    process.start()
    time.sleep(1)
    process.terminate()
    counter = 0
    while process.exitcode is None:
      time.sleep(1)
      counter += 1
      if counter > 10:
        raise AssertionError('Exitcode not propagated.')
    self.assertEqual(process.exitcode, -signal.SIGTERM)


if __name__ == '__main__':
  common_test.main()
