# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for reference_executor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computations

from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import reference_executor


class ReferenceExecutorTest(absltest.TestCase):

  def test_something(self):
    executor = reference_executor.ReferenceExecutor()
    computation_proto = computation_impl.ComputationImpl.get_proto(
        computations.tf_computation(lambda: tf.constant(10)))
    self.assertRaisesRegexp(
        NotImplementedError,
        'The reference executor is not implemented yet.',
        executor.execute,
        computation_proto)


if __name__ == '__main__':
  absltest.main()
