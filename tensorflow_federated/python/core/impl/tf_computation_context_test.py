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
"""Tests for tf_computation_context.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

import unittest

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import tf_computation_context


class TensorFlowComputationContextTest(unittest.TestCase):

  def test_invoke_federated_computation_fails(self):
    @computations.federated_computation(
        types.FederatedType(tf.int32, placements.SERVER, True))
    def foo(x):
      return intrinsics.federated_broadcast(x)
    context = tf_computation_context.TensorFlowComputationContext()
    with self.assertRaisesRegexp(
        ValueError,
        'Only TF computations can be invoked in a TF computation context.'):
      context.invoke(foo, None)

  def test_invoke_tf_computation(self):
    foo = computations.tf_computation(lambda: tf.constant(10))
    context = tf_computation_context.TensorFlowComputationContext()

    # TODO(b/113112885): Adjust the test logic after implementing this.
    with self.assertRaises(NotImplementedError):
      context.invoke(foo, None)


if __name__ == '__main__':
  unittest.main()
