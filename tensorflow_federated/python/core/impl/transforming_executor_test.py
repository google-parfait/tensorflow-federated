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
"""Tests for transforming_executor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import tensorflow as tf

from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import transforming_executor


class TransformingExecutorTest(absltest.TestCase):

  def test_something(self):
    # TODO(b/134543154): Actually test something.

    transforming_executor.TransformingExecutor(eager_executor.EagerExecutor())


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
