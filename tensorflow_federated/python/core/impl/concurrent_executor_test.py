# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for concurrent_executor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from tensorflow_federated.python.core.impl import concurrent_executor


class ConcurrentExecutorTest(absltest.TestCase):

  def test_something(self):
    # TODO(b/134543154): Actually test something.

    concurrent_executor.ConcurrentExecutor([])


if __name__ == '__main__':
  absltest.main()
